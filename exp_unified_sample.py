"""
통합 해석 가능성 실험 — 일부 추론 (Sample Mode)

[起] 텍스트 의미 고착 분석:
  - CSS    : Sentence-BERT 기반 변형 이미지 출력 텍스트 vs 원본의 코사인 의미 유사도
  - ROUGE-L: 원본 출력 텍스트 대비 변형 이미지 출력의 어휘 중첩도

[承] 내부 기작 분석:
  - ViT 패치 코사인 유사도 : 원본 vs HPF+DC 시각 인코더 마지막 레이어 패치 피처 붕괴 측정
  - Attention IoU          : 원본 vs LPF LLM 어텐션 상위 20% 패치 집합 일치율
  - Attention Heatmap      : LLM 어텐션 활성화 패치를 원본 이미지에 오버레이 시각화

배치당 처리 흐름 (7 Pass):
  Pass 1 : Forward original (output_attentions=True, ViT hook) → saliency_orig, vit_orig
  Pass 2 : Generate original                                   → text_orig, pred_orig
  Pass 3 : Forward LPF (output_attentions=True)                → saliency_lpf
  Pass 4 : Generate LPF                                        → text_lpf, pred_lpf
  Pass 5 : model.vision_tower(pixel_values_hpf_dc)             → vit_hpf_dc
  Pass 6 : Generate HPF+DC                                     → text_hpf_dc, pred_hpf_dc
  Pass 7 : Generate Masked (top-20% patches zeroed)            → text_masked, pred_masked

하드웨어 (2x L40S):
  GPU 0 : FakeVLM bfloat16 eager (~14GB 상주, 피크 ~21GB at BATCH=8)
  GPU 1 : Sentence-BERT (~0.5GB)

실행 예시:
  python exp_unified_sample.py
  python exp_unified_sample.py --n_samples 500
"""

import argparse
import gc
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rouge_score import rouge_scorer as rouge_scorer_lib
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    LlamaTokenizer,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
WEIGHT_PATH = "/workspace/fakevlm_analysis/weights/fakeVLM"
DATA_JSON   = "/workspace/fakevlm_analysis/data/FakeClue/data_json/test.json"
IMG_DIR     = "/workspace/fakevlm_analysis/data/FakeClue/test"
RESULTS_DIR = "/workspace/fakevlm_analysis/results_unified"

# ---------------------------------------------------------------------------
# 하이퍼파라미터 (Sample Mode)
# ---------------------------------------------------------------------------
N_SAMPLES_DEFAULT  = 200
BATCH_SIZE         = 8
NUM_WORKERS        = 4
MAX_NEW_TOKENS     = 128

FFT_RADIUS         = 30
MASK_RATIO         = 0.20
GRID_SIZE          = 24
NUM_PATCHES        = GRID_SIZE * GRID_SIZE  # 576

# LLaVA-1.5 expanded sequence: [BOS] + [576 visual] + [text tokens]
VISUAL_START       = 1
VISUAL_END         = 577

LLM_ATTN_LAYERS    = 4
SAVE_HEATMAP       = True
CHECKPOINT_INTERVAL = 50

SBERT_MODEL        = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# 로거
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FakeClueDataset(Dataset):
    def __init__(self, json_path: str, img_dir: str):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_dir = img_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            logger.warning("Image load failed [%d]: %s", idx, e)
            image = Image.new("RGB", (336, 336), color=(128, 128, 128))
        prompt = item["conversations"][0]["value"]
        label  = int(item["label"])
        cate   = item.get("cate", "unknown")
        return image, prompt, label, cate, idx, item["image"]


def collate_fn(batch):
    images, prompts, labels, cates, indices, paths = zip(*batch)
    return list(images), list(prompts), list(labels), list(cates), list(indices), list(paths)


# ---------------------------------------------------------------------------
# 모델 / 프로세서 로더
# ---------------------------------------------------------------------------
def load_processor(model_path: str) -> LlavaProcessor:
    """transformers 4.40.0 호환: 컴포넌트 직접 조립."""
    tokenizer       = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_model(model_path: str, device: str = "cuda:0") -> LlavaForConditionalGeneration:
    """
    output_attentions=True 지원을 위해 attn_implementation="eager" 고정.
    Flash Attention 2 및 SDPA는 어텐션 가중치 행렬을 반환하지 않으므로 사용 불가.
    """
    return LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device).eval()


# ---------------------------------------------------------------------------
# FFT 필터 (custom_experiment_fft.py 로직 그대로 적용)
# ---------------------------------------------------------------------------
def apply_fft_filter(
    image: Image.Image,
    filter_type: str,
    radius: int = FFT_RADIUS,
) -> Image.Image:
    if filter_type == "original":
        return image

    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    filtered_channels = []

    for ch in range(3):
        channel = img_np[:, :, ch]
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        f_shift = np.fft.fftshift(np.fft.fft2(channel))

        y_idx, x_idx = np.ogrid[:rows, :cols]
        dist_sq = (x_idx - ccol) ** 2 + (y_idx - crow) ** 2
        circle_mask = (dist_sq <= radius ** 2).astype(np.float32)

        if filter_type == "lpf":
            mask = circle_mask
            img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))

        elif filter_type == "hpf":
            mask = 1.0 - circle_mask
            img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))

        elif filter_type == "hpf_dc":
            dc_offset = channel.mean()
            mask = 1.0 - circle_mask
            img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))
            img_back = img_back + dc_offset

        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        filtered_channels.append(np.clip(img_back, 0, 255).astype(np.uint8))

    return Image.fromarray(np.stack(filtered_channels, axis=2), mode="RGB")


# ---------------------------------------------------------------------------
# 패치 마스킹 (custom_experiment_hook.py 로직 그대로 적용)
# ---------------------------------------------------------------------------
def mask_top_attention_patches(
    image: Image.Image,
    saliency: torch.Tensor,
    grid_size: int = GRID_SIZE,
    mask_ratio: float = MASK_RATIO,
) -> Image.Image:
    img_array = np.array(image.convert("RGB"))
    h, w, _ = img_array.shape
    top_k = max(1, int(grid_size * grid_size * mask_ratio))
    top_indices = torch.topk(saliency, top_k).indices

    patch_h = h // grid_size
    patch_w = w // grid_size
    masked = img_array.copy()

    for idx in top_indices:
        row, col = divmod(int(idx.item()), grid_size)
        y1, y2 = row * patch_h, min((row + 1) * patch_h, h)
        x1, x2 = col * patch_w, min((col + 1) * patch_w, w)
        masked[y1:y2, x1:x2] = 0

    return Image.fromarray(masked, mode="RGB")


# ---------------------------------------------------------------------------
# 예측 파싱
# ---------------------------------------------------------------------------
def parse_prediction(text: str) -> int:
    response = text.split("?")[-1].strip().lower()
    for part in response.split(".")[:2]:
        if "real" in part:
            return 1
        if "fake" in part:
            return 0
    return -1


# ---------------------------------------------------------------------------
# ViT 특징 추출 + LLM Saliency 동시 추출 (Pass 1 / Pass 3)
# ---------------------------------------------------------------------------
def extract_saliency_and_vit(
    model: LlavaForConditionalGeneration,
    inputs: dict,
    num_layers_avg: int = LLM_ATTN_LAYERS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    단일 forward pass에서 LLM Attention Saliency와 ViT 마지막 레이어 패치 피처를 동시에 추출.

    ViT 피처 추출 방법:
      model.vision_tower.vision_model.encoder.layers[-1]에 forward hook 등록.
      CLIPEncoderLayer 출력의 out[0]이 hidden_states [B, 577, D].

    LLM Saliency 추출 방법 (custom_experiment_hook.py 동일):
      LLM 마지막 num_layers_avg개 레이어의 마지막 텍스트 토큰(-1)이
      비주얼 토큰(1:577)에 주는 어텐션 가중치 레이어·헤드 평균.

    Returns
    -------
    saliency : [B, 576] float32 CPU
    vit_feat : [B, 576, D] float32 CPU (CLS 토큰 제외)
    """
    captured: Dict[str, torch.Tensor] = {}

    def vit_hook(module, inp, out):
        captured["feat"] = out[0].detach().cpu().float()

    last_vit_layer = model.vision_tower.vision_model.encoder.layers[-1]
    hook_handle = last_vit_layer.register_forward_hook(vit_hook)

    with torch.no_grad():
        fwd_out = model(**inputs, output_attentions=True)

    hook_handle.remove()

    # LLM Attention Saliency
    selected = fwd_out.attentions[-num_layers_avg:]
    attn_to_visual = torch.stack(
        [layer[:, :, -1, VISUAL_START:VISUAL_END].detach().cpu().float()
         for layer in selected]
    )
    saliency = attn_to_visual.mean(dim=(0, 2))  # [B, 576]

    del fwd_out, selected, attn_to_visual
    torch.cuda.empty_cache()

    vit_feat = captured["feat"][:, 1:, :]  # CLS 제외 → [B, 576, D]
    return saliency, vit_feat


def extract_vit_only(
    model: LlavaForConditionalGeneration,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """LLM forward 없이 ViT 피처만 추출 (Pass 5 전용). [B, 576, D] float32 CPU."""
    with torch.no_grad():
        vt_out = model.vision_tower(pixel_values)
    feat = vt_out.last_hidden_state[:, 1:, :].detach().cpu().float()
    torch.cuda.empty_cache()
    return feat


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------
def compute_css_batch(
    text_model: SentenceTransformer,
    texts_orig: List[str],
    texts_variants: Dict[str, List[str]],
) -> Dict[str, List[float]]:
    """Sentence-BERT 코사인 유사도를 배치 단위로 계산."""
    embs_orig = text_model.encode(
        texts_orig, convert_to_tensor=True, show_progress_bar=False
    ).float()

    results = {}
    for cond, texts in texts_variants.items():
        embs_var = text_model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        ).float()
        sims = F.cosine_similarity(embs_orig, embs_var, dim=-1)
        results[cond] = sims.cpu().tolist()
    return results


def compute_rouge_l_batch(
    scorer,
    texts_orig: List[str],
    texts_variants: Dict[str, List[str]],
) -> Dict[str, List[float]]:
    """ROUGE-L F-measure를 배치 단위로 계산."""
    results = {}
    for cond, texts in texts_variants.items():
        scores = [
            scorer.score(t_orig, t_var)["rougeL"].fmeasure
            for t_orig, t_var in zip(texts_orig, texts)
        ]
        results[cond] = scores
    return results


def compute_attn_iou_batch(
    saliency_orig: torch.Tensor,
    saliency_lpf: torch.Tensor,
    mask_ratio: float = MASK_RATIO,
) -> List[float]:
    """Attention IoU: 원본 vs LPF 상위 20% 패치 집합 Jaccard 유사도."""
    k = max(1, int(NUM_PATCHES * mask_ratio))
    ious = []
    for i in range(saliency_orig.shape[0]):
        s_orig = set(torch.topk(saliency_orig[i], k).indices.tolist())
        s_lpf  = set(torch.topk(saliency_lpf[i],  k).indices.tolist())
        iou = len(s_orig & s_lpf) / len(s_orig | s_lpf)
        ious.append(float(iou))
    return ious


def compute_vit_cosine_sim_batch(
    vit_orig:   torch.Tensor,
    vit_hpf_dc: torch.Tensor,
) -> List[float]:
    """원본 vs HPF+DC ViT 패치 피처의 코사인 유사도 배치 평균. [B]"""
    sims = F.cosine_similarity(vit_orig, vit_hpf_dc, dim=-1)  # [B, 576]
    return sims.mean(dim=-1).tolist()


# ---------------------------------------------------------------------------
# Attention 히트맵 시각화
# ---------------------------------------------------------------------------
def save_heatmap(
    image: Image.Image,
    saliency: torch.Tensor,
    save_path: str,
    grid_size: int = GRID_SIZE,
    img_size: int = 336,
) -> None:
    heat = saliency.numpy().reshape(grid_size, grid_size)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    orig_np = np.array(image.resize((img_size, img_size)))

    heat_resized = np.array(
        Image.fromarray((heat * 255).astype(np.uint8)).resize(
            (img_size, img_size), resample=Image.BILINEAR
        )
    ) / 255.0
    colored = (cm.jet(heat_resized)[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(orig_np)
    axes[1].imshow(colored, alpha=0.5)
    axes[1].set_title("LLM Attention Heatmap")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 분석 요약 계산
# ---------------------------------------------------------------------------
def compute_analysis_summary(results: List[dict]) -> dict:
    total = len(results)
    summary: dict = {"total": total}

    pred_keys = {
        "original": "pred_original",
        "lpf":      "pred_lpf",
        "hpf_dc":   "pred_hpf_dc",
        "masked":   "pred_masked",
    }

    for cond, pk in pred_keys.items():
        valid = [r for r in results if r[pk] != -1]
        if valid:
            acc = sum(1 for r in valid if r[pk] == r["label"]) / len(valid)
            summary[f"acc_{cond}"]          = round(acc, 4)
            summary[f"valid_{cond}"]        = len(valid)
            summary[f"undetermined_{cond}"] = total - len(valid)
        else:
            summary[f"acc_{cond}"] = None

    metric_keys = [
        "css_orig_vs_lpf", "css_orig_vs_hpf_dc", "css_orig_vs_masked",
        "rouge_l_orig_vs_lpf", "rouge_l_orig_vs_hpf_dc", "rouge_l_orig_vs_masked",
        "attn_iou_orig_lpf", "vit_cosine_sim_orig_hpf_dc",
    ]
    for mk in metric_keys:
        vals = [r[mk] for r in results if mk in r and r[mk] is not None]
        if vals:
            summary[f"{mk}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{mk}_std"]  = round(float(np.std(vals)),  4)

    valid_bias = [r for r in results if r["pred_original"] != -1 and r["pred_masked"] != -1]
    if valid_bias:
        bias_cases = [r for r in valid_bias if r["language_bias"]]
        summary["language_bias_rate"]  = round(len(bias_cases) / len(valid_bias), 4)
        summary["language_bias_count"] = len(bias_cases)
        summary["language_bias_valid"] = len(valid_bias)

    categories = sorted(set(r["cate"] for r in results))
    by_cate: dict = {}
    for cate in categories:
        cat_rs = [r for r in results if r["cate"] == cate]
        cat_sum: dict = {"count": len(cat_rs)}
        for cond, pk in pred_keys.items():
            valid = [r for r in cat_rs if r[pk] != -1]
            if valid:
                cat_sum[f"acc_{cond}"] = round(
                    sum(1 for r in valid if r[pk] == r["label"]) / len(valid), 4
                )
        for mk in ["css_orig_vs_hpf_dc", "css_orig_vs_masked",
                   "attn_iou_orig_lpf", "vit_cosine_sim_orig_hpf_dc"]:
            vals = [r[mk] for r in cat_rs if mk in r and r[mk] is not None]
            if vals:
                cat_sum[f"{mk}_mean"] = round(float(np.mean(vals)), 4)
        by_cate[cate] = cat_sum
    summary["by_category"] = by_cate

    return summary


def _print_summary(summary: dict) -> None:
    sep = "=" * 65
    logger.info(sep)
    logger.info("Unified Experiment Summary")
    logger.info("  Total samples           : %d", summary["total"])
    logger.info("  Accuracy [original]     : %s", summary.get("acc_original"))
    logger.info("  Accuracy [lpf]          : %s", summary.get("acc_lpf"))
    logger.info("  Accuracy [hpf_dc]       : %s", summary.get("acc_hpf_dc"))
    logger.info("  Accuracy [masked]       : %s", summary.get("acc_masked"))
    logger.info("  --- [起] Text Analysis ---")
    logger.info("  CSS (orig vs lpf)      mean : %.4f", summary.get("css_orig_vs_lpf_mean", 0))
    logger.info("  CSS (orig vs hpf_dc)   mean : %.4f", summary.get("css_orig_vs_hpf_dc_mean", 0))
    logger.info("  CSS (orig vs masked)   mean : %.4f", summary.get("css_orig_vs_masked_mean", 0))
    logger.info("  ROUGE-L (orig vs lpf)     mean : %.4f", summary.get("rouge_l_orig_vs_lpf_mean", 0))
    logger.info("  ROUGE-L (orig vs hpf_dc)  mean : %.4f", summary.get("rouge_l_orig_vs_hpf_dc_mean", 0))
    logger.info("  ROUGE-L (orig vs masked)  mean : %.4f", summary.get("rouge_l_orig_vs_masked_mean", 0))
    logger.info("  --- [承] Mechanistic Analysis ---")
    logger.info("  Attention IoU (orig vs lpf)     mean : %.4f", summary.get("attn_iou_orig_lpf_mean", 0))
    logger.info("  ViT cosine sim (orig vs hpf_dc) mean : %.4f", summary.get("vit_cosine_sim_orig_hpf_dc_mean", 0))
    logger.info("  Language Bias Rate               : %.4f", summary.get("language_bias_rate", 0))
    logger.info(sep)


# ---------------------------------------------------------------------------
# 통합 실험 루프
# ---------------------------------------------------------------------------
def run_unified_experiment(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    text_model: SentenceTransformer,
    rouge_scorer_inst,
    dataloader: DataLoader,
    checkpoint_path: str,
    heatmap_dir: Optional[str],
    main_device: str = "cuda:0",
) -> List[dict]:
    results: List[dict] = []
    top_k = max(1, int(NUM_PATCHES * MASK_RATIO))
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Unified Experiment", unit="batch"):
            images, prompts, labels, cates, indices, paths = batch
            batch_size = len(images)

            lpf_images    = [apply_fft_filter(img, "lpf")    for img in images]
            hpf_dc_images = [apply_fft_filter(img, "hpf_dc") for img in images]

            # ------------------------------------------------------------------
            # Pass 1 + Pass 2: original → saliency_orig, vit_orig, texts_orig
            # ------------------------------------------------------------------
            inputs_orig = processor(
                text=prompts, images=images,
                return_tensors="pt", padding=True,
            ).to(main_device, torch.bfloat16)

            saliency_orig, vit_orig = extract_saliency_and_vit(model, inputs_orig)

            out_orig   = model.generate(**inputs_orig, max_new_tokens=MAX_NEW_TOKENS)
            texts_orig = processor.batch_decode(out_orig, skip_special_tokens=True)
            del inputs_orig, out_orig
            torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Pass 3 + Pass 4: LPF → saliency_lpf, texts_lpf
            # ------------------------------------------------------------------
            inputs_lpf = processor(
                text=prompts, images=lpf_images,
                return_tensors="pt", padding=True,
            ).to(main_device, torch.bfloat16)

            saliency_lpf, _ = extract_saliency_and_vit(model, inputs_lpf)

            out_lpf   = model.generate(**inputs_lpf, max_new_tokens=MAX_NEW_TOKENS)
            texts_lpf = processor.batch_decode(out_lpf, skip_special_tokens=True)
            del inputs_lpf, out_lpf
            torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Pass 5 + Pass 6: HPF+DC → vit_hpf_dc, texts_hpf_dc
            # ------------------------------------------------------------------
            inputs_hpf = processor(
                text=prompts, images=hpf_dc_images,
                return_tensors="pt", padding=True,
            ).to(main_device, torch.bfloat16)

            vit_hpf_dc = extract_vit_only(model, inputs_hpf["pixel_values"])

            out_hpf      = model.generate(**inputs_hpf, max_new_tokens=MAX_NEW_TOKENS)
            texts_hpf_dc = processor.batch_decode(out_hpf, skip_special_tokens=True)
            del inputs_hpf, out_hpf
            torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Pass 7: Masked → texts_masked
            # ------------------------------------------------------------------
            masked_images = [
                mask_top_attention_patches(images[i], saliency_orig[i])
                for i in range(batch_size)
            ]
            inputs_mask = processor(
                text=prompts, images=masked_images,
                return_tensors="pt", padding=True,
            ).to(main_device, torch.bfloat16)

            out_mask     = model.generate(**inputs_mask, max_new_tokens=MAX_NEW_TOKENS)
            texts_masked = processor.batch_decode(out_mask, skip_special_tokens=True)
            del inputs_mask, out_mask
            torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # 지표 계산
            # ------------------------------------------------------------------
            variants = {
                "lpf":    texts_lpf,
                "hpf_dc": texts_hpf_dc,
                "masked": texts_masked,
            }
            css_scores   = compute_css_batch(text_model, texts_orig, variants)
            rouge_scores = compute_rouge_l_batch(rouge_scorer_inst, texts_orig, variants)
            attn_ious    = compute_attn_iou_batch(saliency_orig, saliency_lpf)
            vit_sims     = compute_vit_cosine_sim_batch(vit_orig, vit_hpf_dc)

            # ------------------------------------------------------------------
            # 결과 기록
            # ------------------------------------------------------------------
            for i in range(batch_size):
                p_orig   = parse_prediction(texts_orig[i])
                p_lpf    = parse_prediction(texts_lpf[i])
                p_hpf_dc = parse_prediction(texts_hpf_dc[i])
                p_masked = parse_prediction(texts_masked[i])

                record = {
                    "idx":           int(indices[i]),
                    "image":         paths[i],
                    "label":         int(labels[i]),
                    "cate":          cates[i],
                    "pred_original": p_orig,
                    "pred_lpf":      p_lpf,
                    "pred_hpf_dc":   p_hpf_dc,
                    "pred_masked":   p_masked,
                    "text_original": texts_orig[i],
                    "text_lpf":      texts_lpf[i],
                    "text_hpf_dc":   texts_hpf_dc[i],
                    "text_masked":   texts_masked[i],
                    "css_orig_vs_lpf":            float(css_scores["lpf"][i]),
                    "css_orig_vs_hpf_dc":         float(css_scores["hpf_dc"][i]),
                    "css_orig_vs_masked":         float(css_scores["masked"][i]),
                    "rouge_l_orig_vs_lpf":        float(rouge_scores["lpf"][i]),
                    "rouge_l_orig_vs_hpf_dc":     float(rouge_scores["hpf_dc"][i]),
                    "rouge_l_orig_vs_masked":     float(rouge_scores["masked"][i]),
                    "attn_iou_orig_lpf":          float(attn_ious[i]),
                    "vit_cosine_sim_orig_hpf_dc": float(vit_sims[i]),
                    "language_bias": bool(p_orig != -1 and p_orig == p_masked),
                    "top_attn_indices_orig": (
                        torch.topk(saliency_orig[i], top_k).indices.tolist()
                    ),
                    "attn_max":  float(saliency_orig[i].max()),
                    "attn_mean": float(saliency_orig[i].mean()),
                }
                results.append(record)

                if heatmap_dir is not None:
                    label_str = "real" if int(labels[i]) == 1 else "fake"
                    hmap_path = os.path.join(
                        heatmap_dir,
                        f"{int(indices[i]):05d}_{cates[i]}_{label_str}.png",
                    )
                    try:
                        save_heatmap(images[i], saliency_orig[i], hmap_path)
                    except Exception as e:
                        logger.warning("Heatmap save failed [%d]: %s", int(indices[i]), e)

            if len(results) % CHECKPOINT_INTERVAL < batch_size:
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(
                    "[CHECKPOINT] %d samples | VRAM: %.2f GB alloc",
                    len(results), torch.cuda.memory_allocated(0) / 1e9,
                )

    return results


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main(n_samples: Optional[int] = None) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    heatmap_dir = os.path.join(RESULTS_DIR, "heatmaps") if SAVE_HEATMAP else None
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)

    results_path    = os.path.join(RESULTS_DIR, "results_sample.json")
    checkpoint_path = os.path.join(RESULTS_DIR, "results_sample_partial.json")
    analysis_path   = os.path.join(RESULTS_DIR, "analysis_sample.json")

    n_gpu        = torch.cuda.device_count()
    main_device  = "cuda:0"
    sbert_device = "cuda:1" if n_gpu >= 2 else "cuda:0"
    logger.info(
        "GPUs: %d | FakeVLM → %s | Sentence-BERT → %s",
        n_gpu, main_device, sbert_device,
    )

    logger.info("Loading processor from %s", WEIGHT_PATH)
    processor = load_processor(WEIGHT_PATH)

    logger.info("Loading FakeVLM model (eager attn)")
    model = load_model(WEIGHT_PATH, main_device)
    logger.info(
        "FakeVLM loaded. VRAM: %.2f GB alloc / %.2f GB reserved",
        torch.cuda.memory_allocated(0) / 1e9,
        torch.cuda.memory_reserved(0) / 1e9,
    )

    logger.info("Loading Sentence-BERT '%s' on %s", SBERT_MODEL, sbert_device)
    text_model = SentenceTransformer(SBERT_MODEL, device=sbert_device)

    rouge_scorer_inst = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)

    logger.info("Loading dataset from %s", DATA_JSON)
    full_dataset = FakeClueDataset(json_path=DATA_JSON, img_dir=IMG_DIR)

    if n_samples is not None and n_samples < len(full_dataset):
        import torch.utils.data as tud
        dataset = tud.Subset(full_dataset, list(range(n_samples)))
        logger.info("Sample mode: %d / %d samples", n_samples, len(full_dataset))
    else:
        dataset = full_dataset
        logger.info("Full mode: %d samples", len(full_dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    results = run_unified_experiment(
        model=model,
        processor=processor,
        text_model=text_model,
        rouge_scorer_inst=rouge_scorer_inst,
        dataloader=dataloader,
        checkpoint_path=checkpoint_path,
        heatmap_dir=heatmap_dir,
        main_device=main_device,
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("[SAVED] results → %s (%d samples)", results_path, len(results))

    summary = compute_analysis_summary(results)
    with open(analysis_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("[SAVED] analysis → %s", analysis_path)

    _print_summary(summary)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="통합 해석 가능성 실험 - 일부 추론 (Sample Mode)")
    parser.add_argument(
        "--n_samples", type=int, default=N_SAMPLES_DEFAULT,
        help=f"추론할 샘플 수 (default: {N_SAMPLES_DEFAULT})",
    )
    args = parser.parse_args()
    main(n_samples=args.n_samples)

"""
통합 해석 가능성 실험 — 전체 추론 (Full Mode)

sample.py와 동일한 2-GPU 병렬·경로·processor/ViT 구조로 동작.
FakeClue test set 전체에 대해 FFT(LPF, HPF+DC) 및 Attention 마스킹 통합 수행.

[1] 텍스트 의미 고착 분석: CSS, ROUGE-L
[2] 내부 기작 분석:
  - ViT 패치 코사인 유사도  : 원본 vs HPF+DC 피처 붕괴 측정
  - Attention IoU (LPF)    : 원본 vs LPF LLM 어텐션 상위 20% 패치 집합 일치율
  - Attention IoU (HPF+DC) : 원본 vs HPF+DC — 노이즈 환경 Semantic Anchoring 여부 측정
[3] 계층화 통계: by_label(Real/Fake) × by_category × by_category_label

배치당 8-Pass, 2-GPU ThreadPoolExecutor 병렬. OOM → Graceful degradation.

실행:
  python exp_unified_full.py
  python exp_unified_full.py --no_visualization  # 실험만 수행, 시각화 생략
"""

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_BASE_DIR = Path(__file__).resolve().parent

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
# 경로 설정 (스크립트 기준 상대 경로, sample과 동일)
# ---------------------------------------------------------------------------
WEIGHT_PATH = str(_BASE_DIR / "weights" / "fakeVLM")
DATA_JSON   = str(_BASE_DIR / "data" / "FakeClue" / "data_json" / "test.json")
IMG_DIR     = str(_BASE_DIR / "data" / "FakeClue" / "test")
RESULTS_DIR = str(_BASE_DIR / "results_unified")

# ---------------------------------------------------------------------------
# 하이퍼파라미터 (Full Mode — sample과 동일 배치/워커 전략)
# ---------------------------------------------------------------------------
BATCH_SIZE         = 48
NUM_WORKERS        = 12
FFT_THREADS        = 12
MAX_NEW_TOKENS     = 128

FFT_RADIUS         = 30
MASK_RATIO         = 0.20
GRID_SIZE          = 24
NUM_PATCHES        = GRID_SIZE * GRID_SIZE  # 576

VISUAL_START       = 1
VISUAL_END         = 577

LLM_ATTN_LAYERS    = 4
SAVE_HEATMAP       = False  # 전체는 기본 비활성화; 대표 heatmap만 sanity에서 저장
CHECKPOINT_INTERVAL = 100

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
# 모델 / 프로세서 로더 (sample과 동일)
# ---------------------------------------------------------------------------
def load_processor(model_path: str) -> LlavaProcessor:
    """transformers 4.40.0+ 호환: patch_size 필요 (num_image_tokens 계산용)."""
    tokenizer       = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    patch_size = 14
    return LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=patch_size,
    )


def load_model(model_path: str, device: str = "cuda:0") -> LlavaForConditionalGeneration:
    """
    output_attentions=True 지원을 위해 attn_implementation="eager" 고정.
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
    ViT 경로: model.model.vision_tower (sample과 동일).
    Returns: saliency [B, 576], vit_feat [B, 576, D] float32 CPU.
    """
    captured: Dict[str, torch.Tensor] = {}

    def vit_hook(module, inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        captured["feat"] = feat.detach().cpu().float()

    last_vit_layer = model.model.vision_tower.vision_model.encoder.layers[-1]
    hook_handle = last_vit_layer.register_forward_hook(vit_hook)

    with torch.no_grad():
        fwd_out = model(**inputs, output_attentions=True)

    hook_handle.remove()

    selected = fwd_out.attentions[-num_layers_avg:]
    attn_to_visual = torch.stack(
        [layer[:, :, -1, VISUAL_START:VISUAL_END].detach().cpu().float()
         for layer in selected]
    )
    saliency = attn_to_visual.mean(dim=(0, 2))  # [B, 576]

    del fwd_out, selected, attn_to_visual
    torch.cuda.empty_cache()

    vit_feat = captured["feat"][:, 1:, :]  # [B, 576, D]
    return saliency, vit_feat


def extract_vit_only(
    model: LlavaForConditionalGeneration,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """LLM forward 없이 ViT 피처만 추출 (Pass 5 전용). model.model.vision_tower 사용."""
    with torch.no_grad():
        vt_out = model.model.vision_tower(pixel_values)
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
# 분석 요약 계산 헬퍼
# ---------------------------------------------------------------------------
_PRED_KEYS = {
    "original": "pred_original",
    "lpf":      "pred_lpf",
    "hpf_dc":   "pred_hpf_dc",
    "masked":   "pred_masked",
}
_METRIC_KEYS = [
    "css_orig_vs_lpf", "css_orig_vs_hpf_dc", "css_orig_vs_masked",
    "rouge_l_orig_vs_lpf", "rouge_l_orig_vs_hpf_dc", "rouge_l_orig_vs_masked",
    "attn_iou_orig_lpf", "attn_iou_orig_hpf_dc", "vit_cosine_sim_orig_hpf_dc",
]


def _summarize_subset(rs: List[dict]) -> dict:
    """주어진 샘플 부분집합에 대해 정확도·지표·bias 통계 산출."""
    total = len(rs)
    sub: dict = {"count": total}

    for cond, pk in _PRED_KEYS.items():
        valid = [r for r in rs if r[pk] != -1]
        if valid:
            sub[f"acc_{cond}"]          = round(sum(1 for r in valid if r[pk] == r["label"]) / len(valid), 4)
            sub[f"valid_{cond}"]        = len(valid)
            sub[f"undetermined_{cond}"] = total - len(valid)
        else:
            sub[f"acc_{cond}"] = None

    for mk in _METRIC_KEYS:
        vals = [r[mk] for r in rs if mk in r and r[mk] is not None]
        if vals:
            sub[f"{mk}_mean"] = round(float(np.mean(vals)), 4)
            sub[f"{mk}_std"]  = round(float(np.std(vals)),  4)

    valid_bias = [r for r in rs if r["pred_original"] != -1 and r["pred_masked"] != -1]
    if valid_bias:
        bias_cases                = [r for r in valid_bias if r["language_bias"]]
        sub["language_bias_rate"]  = round(len(bias_cases) / len(valid_bias), 4)
        sub["language_bias_count"] = len(bias_cases)
        sub["language_bias_valid"] = len(valid_bias)

    return sub


# ---------------------------------------------------------------------------
# 분석 요약 계산 (by_label / by_category / by_category_label 3단 계층화)
# ---------------------------------------------------------------------------
def compute_analysis_summary(results: List[dict]) -> dict:
    """
    계층화 통계:
      [전체]          summary 루트
      [라벨별]        by_label.real / by_label.fake
      [카테고리별]    by_category.<cate>
      [카테고리×라벨] by_category.<cate>.by_label.real / .fake
    """
    summary = _summarize_subset(results)
    summary["total"] = len(results)

    # 라벨별 (Real vs Fake 대조군)
    label_map = {1: "real", 0: "fake"}
    by_label: dict = {}
    for lv, lname in label_map.items():
        sub_rs = [r for r in results if r["label"] == lv]
        if sub_rs:
            by_label[lname] = _summarize_subset(sub_rs)
    summary["by_label"] = by_label

    # 카테고리별 + 카테고리×라벨
    categories = sorted(set(r["cate"] for r in results))
    by_cate: dict = {}
    for cate in categories:
        cat_rs   = [r for r in results if r["cate"] == cate]
        cat_sum  = _summarize_subset(cat_rs)

        # 카테고리 내 라벨별 세분화
        cat_by_label: dict = {}
        for lv, lname in label_map.items():
            cl_rs = [r for r in cat_rs if r["label"] == lv]
            if cl_rs:
                cat_by_label[lname] = _summarize_subset(cl_rs)
        cat_sum["by_label"] = cat_by_label

        by_cate[cate] = cat_sum
    summary["by_category"] = by_cate

    return summary


def _print_summary(summary: dict) -> None:
    sep = "=" * 70
    logger.info(sep)
    logger.info("Unified Experiment Summary (Full Mode)")
    logger.info("  Total samples           : %d", summary["total"])
    logger.info("  Accuracy [original]     : %s", summary.get("acc_original"))
    logger.info("  Accuracy [lpf]          : %s", summary.get("acc_lpf"))
    logger.info("  Accuracy [hpf_dc]       : %s", summary.get("acc_hpf_dc"))
    logger.info("  Accuracy [masked]       : %s", summary.get("acc_masked"))
    logger.info("  Language Bias Rate      : %.4f", summary.get("language_bias_rate", 0))
    logger.info("  --- [1] Text Semantic Analysis ---")
    logger.info("  CSS (orig vs lpf)          mean : %.4f ± %.4f",
                summary.get("css_orig_vs_lpf_mean", 0), summary.get("css_orig_vs_lpf_std", 0))
    logger.info("  CSS (orig vs hpf_dc)       mean : %.4f ± %.4f",
                summary.get("css_orig_vs_hpf_dc_mean", 0), summary.get("css_orig_vs_hpf_dc_std", 0))
    logger.info("  CSS (orig vs masked)       mean : %.4f ± %.4f",
                summary.get("css_orig_vs_masked_mean", 0), summary.get("css_orig_vs_masked_std", 0))
    logger.info("  ROUGE-L (orig vs lpf)      mean : %.4f ± %.4f",
                summary.get("rouge_l_orig_vs_lpf_mean", 0), summary.get("rouge_l_orig_vs_lpf_std", 0))
    logger.info("  ROUGE-L (orig vs hpf_dc)   mean : %.4f ± %.4f",
                summary.get("rouge_l_orig_vs_hpf_dc_mean", 0), summary.get("rouge_l_orig_vs_hpf_dc_std", 0))
    logger.info("  ROUGE-L (orig vs masked)   mean : %.4f ± %.4f",
                summary.get("rouge_l_orig_vs_masked_mean", 0), summary.get("rouge_l_orig_vs_masked_std", 0))
    logger.info("  --- [2] Mechanistic Analysis ---")
    logger.info("  Attn IoU (orig vs lpf)        mean : %.4f ± %.4f",
                summary.get("attn_iou_orig_lpf_mean", 0), summary.get("attn_iou_orig_lpf_std", 0))
    logger.info("  Attn IoU (orig vs hpf_dc)     mean : %.4f ± %.4f",
                summary.get("attn_iou_orig_hpf_dc_mean", 0), summary.get("attn_iou_orig_hpf_dc_std", 0))
    logger.info("  ViT cos sim (orig vs hpf_dc)  mean : %.4f ± %.4f",
                summary.get("vit_cosine_sim_orig_hpf_dc_mean", 0), summary.get("vit_cosine_sim_orig_hpf_dc_std", 0))
    logger.info("  --- [3] Label Stratification ---")
    for lname in ("real", "fake"):
        bl = summary.get("by_label", {}).get(lname, {})
        if bl:
            logger.info(
                "  [%s] n=%d | acc_orig=%.4f | bias=%.4f | attn_iou_hpf=%.4f | vit_sim=%.4f",
                lname, bl.get("count", 0),
                bl.get("acc_original") or 0.0,
                bl.get("language_bias_rate") or 0.0,
                bl.get("attn_iou_orig_hpf_dc_mean") or 0.0,
                bl.get("vit_cosine_sim_orig_hpf_dc_mean") or 0.0,
            )
    logger.info("  --- [4] Category Summary ---")
    for cate, cs in summary.get("by_category", {}).items():
        logger.info(
            "  [%s] n=%d | acc_orig=%.4f | acc_masked=%.4f | bias=%.4f",
            cate, cs.get("count", 0),
            cs.get("acc_original") or 0.0,
            cs.get("acc_masked") or 0.0,
            cs.get("language_bias_rate") or 0.0,
        )
        for lname in ("real", "fake"):
            cl = cs.get("by_label", {}).get(lname, {})
            if cl:
                logger.info(
                    "    [%s/%s] n=%d | acc_orig=%.4f | attn_iou_hpf=%.4f | vit_sim=%.4f",
                    cate, lname, cl.get("count", 0),
                    cl.get("acc_original") or 0.0,
                    cl.get("attn_iou_orig_hpf_dc_mean") or 0.0,
                    cl.get("vit_cosine_sim_orig_hpf_dc_mean") or 0.0,
                )
    logger.info(sep)


# ---------------------------------------------------------------------------
# 단일 GPU 배치 처리 (8-Pass: HPF+DC 어텐션 추적 포함 + OOM 방어)
# ---------------------------------------------------------------------------
def process_batch_half(
    half_images:  List,
    half_prompts: List[str],
    half_labels:  List,
    half_cates:   List[str],
    half_indices: List,
    half_paths:   List[str],
    model:        LlavaForConditionalGeneration,
    processor:    LlavaProcessor,
    device:       str,
    rouge_scorer_inst,
) -> Optional[dict]:
    """
    배치 절반을 지정된 GPU에서 8-Pass 처리.
    Pass 5를 extract_saliency_and_vit 로 변경해 saliency_hpf_dc를 추출,
    attn_iou_orig_hpf_dc 지표를 통해 노이즈 환경의 어텐션 행동을 분석함.
    OOM 발생 시 None 반환 → 호출 측에서 해당 배치를 건너뜀.
    """
    try:
        batch_size = len(half_images)
        top_k      = max(1, int(NUM_PATCHES * MASK_RATIO))

        with ThreadPoolExecutor(max_workers=max(1, FFT_THREADS // 2)) as fft_pool:
            lpf_images    = list(fft_pool.map(lambda img: apply_fft_filter(img, "lpf"),    half_images))
            hpf_dc_images = list(fft_pool.map(lambda img: apply_fft_filter(img, "hpf_dc"), half_images))

        # Pass 1 + 2: original → saliency_orig, vit_orig, texts_orig
        inputs_orig = processor(
            text=half_prompts, images=half_images,
            return_tensors="pt", padding=True,
        ).to(device, torch.bfloat16)
        saliency_orig, vit_orig = extract_saliency_and_vit(model, inputs_orig)
        out_orig   = model.generate(**inputs_orig, max_new_tokens=MAX_NEW_TOKENS)
        texts_orig = processor.batch_decode(out_orig, skip_special_tokens=True)
        del inputs_orig, out_orig
        torch.cuda.empty_cache()

        # Pass 3 + 4: LPF → saliency_lpf, texts_lpf
        inputs_lpf = processor(
            text=half_prompts, images=lpf_images,
            return_tensors="pt", padding=True,
        ).to(device, torch.bfloat16)
        saliency_lpf, _ = extract_saliency_and_vit(model, inputs_lpf)
        out_lpf   = model.generate(**inputs_lpf, max_new_tokens=MAX_NEW_TOKENS)
        texts_lpf = processor.batch_decode(out_lpf, skip_special_tokens=True)
        del inputs_lpf, out_lpf
        torch.cuda.empty_cache()

        # Pass 5 + 6: HPF+DC → saliency_hpf_dc (어텐션 추적), vit_hpf_dc, texts_hpf_dc
        # [핵심 변경] extract_vit_only → extract_saliency_and_vit
        # 노이즈 환경에서 LLM 어텐션이 의미론적 영역에 고착(Semantic Anchoring)되는지 측정
        inputs_hpf = processor(
            text=half_prompts, images=hpf_dc_images,
            return_tensors="pt", padding=True,
        ).to(device, torch.bfloat16)
        saliency_hpf_dc, vit_hpf_dc = extract_saliency_and_vit(model, inputs_hpf)
        out_hpf      = model.generate(**inputs_hpf, max_new_tokens=MAX_NEW_TOKENS)
        texts_hpf_dc = processor.batch_decode(out_hpf, skip_special_tokens=True)
        del inputs_hpf, out_hpf
        torch.cuda.empty_cache()

        # Pass 7: Masked → texts_masked
        masked_images = [
            mask_top_attention_patches(half_images[i], saliency_orig[i])
            for i in range(batch_size)
        ]
        inputs_mask = processor(
            text=half_prompts, images=masked_images,
            return_tensors="pt", padding=True,
        ).to(device, torch.bfloat16)
        out_mask     = model.generate(**inputs_mask, max_new_tokens=MAX_NEW_TOKENS)
        texts_masked = processor.batch_decode(out_mask, skip_special_tokens=True)
        del inputs_mask, out_mask
        torch.cuda.empty_cache()

        rouge_scores    = compute_rouge_l_batch(rouge_scorer_inst, texts_orig, {
            "lpf": texts_lpf, "hpf_dc": texts_hpf_dc, "masked": texts_masked,
        })
        attn_ious_lpf    = compute_attn_iou_batch(saliency_orig, saliency_lpf)
        attn_ious_hpf_dc = compute_attn_iou_batch(saliency_orig, saliency_hpf_dc)
        vit_sims         = compute_vit_cosine_sim_batch(vit_orig, vit_hpf_dc)

        return {
            "texts_orig":      texts_orig,
            "texts_lpf":       texts_lpf,
            "texts_hpf_dc":    texts_hpf_dc,
            "texts_masked":    texts_masked,
            "saliency_orig":   saliency_orig,
            "saliency_hpf_dc": saliency_hpf_dc,
            "rouge_scores":    rouge_scores,
            "attn_ious_lpf":   attn_ious_lpf,
            "attn_ious_hpf_dc": attn_ious_hpf_dc,
            "vit_sims":        vit_sims,
            "labels":          half_labels,
            "cates":           half_cates,
            "indices":         half_indices,
            "paths":           half_paths,
            "images":          half_images,
            "top_k":           top_k,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(
                "[OOM] %s — batch skipped (indices=%s). Clearing cache.",
                device, list(half_indices),
            )
            torch.cuda.empty_cache()
            gc.collect()
            return None
        raise


# ---------------------------------------------------------------------------
# 통합 실험 루프 (2-GPU 데이터 병렬, sample과 동일)
# ---------------------------------------------------------------------------
def run_unified_experiment(
    model0:       LlavaForConditionalGeneration,
    model1:       LlavaForConditionalGeneration,
    processor:    LlavaProcessor,
    text_model:   SentenceTransformer,
    dataloader:   DataLoader,
    checkpoint_path: str,
    heatmap_dir:  Optional[str],
    heatmap_per_cate_label: int = 4,
) -> List[dict]:
    """
    배치를 절반으로 나눠 model0(cuda:0), model1(cuda:1)에서 동시 실행.
    heatmap_dir이 있으면 카테고리·라벨별로 대표 heatmap_per_cate_label개만 저장.
    """
    results: List[dict] = []
    model0.eval()
    model1.eval()
    rouge0 = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
    rouge1 = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
    heatmap_counts: Dict[Tuple[str, int], int] = defaultdict(int)

    with ThreadPoolExecutor(max_workers=2) as gpu_pool:
        for batch in tqdm(dataloader, desc="Unified Experiment [2-GPU Full]", unit="batch"):
            images, prompts, labels, cates, indices, paths = batch
            batch_size = len(images)
            mid = batch_size // 2
            if mid == 0:
                mid = batch_size

            future0 = gpu_pool.submit(
                process_batch_half,
                images[:mid], prompts[:mid], labels[:mid],
                cates[:mid], indices[:mid], paths[:mid],
                model0, processor, "cuda:0", rouge0,
            )
            future1 = gpu_pool.submit(
                process_batch_half,
                images[mid:], prompts[mid:], labels[mid:],
                cates[mid:], indices[mid:], paths[mid:],
                model1, processor, "cuda:1", rouge1,
            ) if mid < batch_size else None

            # OOM이면 None 반환됨 → 해당 절반 건너뜀 (Graceful degradation)
            r0 = future0.result()
            r1 = future1.result() if future1 is not None else None
            half_results = [hr for hr in (r0, r1) if hr is not None]
            if not half_results:
                logger.warning("[SKIP] Both halves OOM — skipping batch.")
                continue

            all_texts_orig   = []
            all_texts_lpf    = []
            all_texts_hpf_dc = []
            all_texts_masked = []
            for hr in half_results:
                all_texts_orig   += hr["texts_orig"]
                all_texts_lpf    += hr["texts_lpf"]
                all_texts_hpf_dc += hr["texts_hpf_dc"]
                all_texts_masked += hr["texts_masked"]

            css_scores = compute_css_batch(text_model, all_texts_orig, {
                "lpf": all_texts_lpf, "hpf_dc": all_texts_hpf_dc, "masked": all_texts_masked,
            })

            global_i = 0
            for hr in half_results:
                sub_size         = len(hr["labels"])
                saliency_orig    = hr["saliency_orig"]
                saliency_hpf_dc  = hr["saliency_hpf_dc"]
                top_k            = hr["top_k"]

                for j in range(sub_size):
                    cate, label = hr["cates"][j], int(hr["labels"][j])
                    key = (cate, label)
                    do_heatmap = (
                        heatmap_dir is not None
                        and heatmap_counts[key] < heatmap_per_cate_label
                    )
                    if do_heatmap:
                        heatmap_counts[key] += 1

                    p_orig   = parse_prediction(hr["texts_orig"][j])
                    p_lpf    = parse_prediction(hr["texts_lpf"][j])
                    p_hpf_dc = parse_prediction(hr["texts_hpf_dc"][j])
                    p_masked = parse_prediction(hr["texts_masked"][j])

                    record = {
                        "idx":           int(hr["indices"][j]),
                        "image":         hr["paths"][j],
                        "label":         int(hr["labels"][j]),
                        "cate":          hr["cates"][j],
                        "pred_original": p_orig,
                        "pred_lpf":      p_lpf,
                        "pred_hpf_dc":   p_hpf_dc,
                        "pred_masked":   p_masked,
                        "text_original": hr["texts_orig"][j],
                        "text_lpf":      hr["texts_lpf"][j],
                        "text_hpf_dc":   hr["texts_hpf_dc"][j],
                        "text_masked":   hr["texts_masked"][j],
                        "css_orig_vs_lpf":            float(css_scores["lpf"][global_i]),
                        "css_orig_vs_hpf_dc":         float(css_scores["hpf_dc"][global_i]),
                        "css_orig_vs_masked":         float(css_scores["masked"][global_i]),
                        "rouge_l_orig_vs_lpf":        float(hr["rouge_scores"]["lpf"][j]),
                        "rouge_l_orig_vs_hpf_dc":     float(hr["rouge_scores"]["hpf_dc"][j]),
                        "rouge_l_orig_vs_masked":     float(hr["rouge_scores"]["masked"][j]),
                        "attn_iou_orig_lpf":          float(hr["attn_ious_lpf"][j]),
                        "attn_iou_orig_hpf_dc":       float(hr["attn_ious_hpf_dc"][j]),
                        "vit_cosine_sim_orig_hpf_dc": float(hr["vit_sims"][j]),
                        "language_bias": bool(p_orig != -1 and p_orig == p_masked),
                        "top_attn_indices_orig": (
                            torch.topk(saliency_orig[j], top_k).indices.tolist()
                        ),
                        "top_attn_indices_hpf_dc": (
                            torch.topk(saliency_hpf_dc[j], top_k).indices.tolist()
                        ),
                        "attn_max":      float(saliency_orig[j].max()),
                        "attn_mean":     float(saliency_orig[j].mean()),
                        "attn_hpf_max":  float(saliency_hpf_dc[j].max()),
                        "attn_hpf_mean": float(saliency_hpf_dc[j].mean()),
                    }
                    results.append(record)

                    if do_heatmap:
                        label_str = "real" if label == 1 else "fake"
                        # 원본 어텐션 heatmap
                        try:
                            save_heatmap(
                                hr["images"][j], saliency_orig[j],
                                os.path.join(
                                    heatmap_dir,
                                    f"repr_{int(hr['indices'][j]):05d}_{cate}_{label_str}_orig.png",
                                ),
                            )
                        except Exception as e:
                            logger.warning("Heatmap(orig) save failed [%d]: %s", int(hr["indices"][j]), e)
                        # HPF+DC 어텐션 heatmap (노이즈 환경 semantic anchoring 시각화)
                        try:
                            save_heatmap(
                                hr["images"][j], saliency_hpf_dc[j],
                                os.path.join(
                                    heatmap_dir,
                                    f"repr_{int(hr['indices'][j]):05d}_{cate}_{label_str}_hpf.png",
                                ),
                            )
                        except Exception as e:
                            logger.warning("Heatmap(hpf) save failed [%d]: %s", int(hr["indices"][j]), e)

                    global_i += 1

            prev_len = len(results) - sum(len(hr["labels"]) for hr in half_results)
            if prev_len // CHECKPOINT_INTERVAL < len(results) // CHECKPOINT_INTERVAL:
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(
                    "[CHECKPOINT] %d samples | GPU0: %.2f GB | GPU1: %.2f GB",
                    len(results),
                    torch.cuda.memory_allocated(0) / 1e9,
                    torch.cuda.memory_allocated(1) / 1e9,
                )

    return results


# ---------------------------------------------------------------------------
# 결과 종합 시각화: matplotlib 요약 PNG (6종)
# ---------------------------------------------------------------------------
_COND_COLORS = {
    "original": "#2ecc71",
    "lpf":      "#3498db",
    "hpf_dc":   "#9b59b6",
    "masked":   "#e74c3c",
}
_CONDS = ["original", "lpf", "hpf_dc", "masked"]


def _bar_with_err(ax, xs, ys, errs, colors, labels=None, ylim=(0, 1.05)):
    """에러바 포함 막대 그리기 헬퍼."""
    for i, (x, y, err, c) in enumerate(zip(xs, ys, errs, colors)):
        ax.bar(x, y, color=c, edgecolor="black",
               yerr=err if err else None, capsize=4,
               label=labels[i] if labels else None)
    ax.set_ylim(*ylim)


def plot_summary_png(summary: dict, save_dir: str) -> None:
    """분석 요약 6종 PNG 저장."""
    os.makedirs(save_dir, exist_ok=True)
    by_cate  = summary.get("by_category") or {}
    by_label = summary.get("by_label") or {}
    cats     = list(by_cate.keys())

    # ── 1) 전체 정확도 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    accs = [summary.get(f"acc_{c}") or 0.0 for c in _CONDS]
    ax.bar(range(4), accs, color=[_COND_COLORS[c] for c in _CONDS], edgecolor="black")
    ax.set_xticks(range(4))
    ax.set_xticklabels(_CONDS)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prediction Accuracy by Condition (Overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_accuracy.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── 2) 텍스트·메커니즘 지표 (2×3 grid, attn_iou_hpf_dc 추가) ─────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cond_labels = ["orig vs LPF", "orig vs HPF+DC", "orig vs Masked"]
    # CSS
    ax = axes[0, 0]
    mk = ["css_orig_vs_lpf", "css_orig_vs_hpf_dc", "css_orig_vs_masked"]
    _bar_with_err(ax, range(3),
                  [summary.get(f"{k}_mean", 0) for k in mk],
                  [summary.get(f"{k}_std", 0)  for k in mk],
                  ["#3498db", "#9b59b6", "#e74c3c"])
    ax.set_xticks(range(3)); ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Cosine Similarity"); ax.set_title("CSS (Text Semantic Sim.)")
    # ROUGE-L
    ax = axes[0, 1]
    mk = ["rouge_l_orig_vs_lpf", "rouge_l_orig_vs_hpf_dc", "rouge_l_orig_vs_masked"]
    _bar_with_err(ax, range(3),
                  [summary.get(f"{k}_mean", 0) for k in mk],
                  [summary.get(f"{k}_std", 0)  for k in mk],
                  ["#3498db", "#9b59b6", "#e74c3c"])
    ax.set_xticks(range(3)); ax.set_xticklabels(cond_labels)
    ax.set_ylabel("ROUGE-L F"); ax.set_title("ROUGE-L (Lexical Overlap)")
    # Language Bias Rate (Real vs Fake)
    ax = axes[0, 2]
    br_vals  = [by_label.get(l, {}).get("language_bias_rate") or 0.0 for l in ("real", "fake")]
    ax.bar([0, 1], br_vals, color=["#27ae60", "#e74c3c"], edgecolor="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Real", "Fake"])
    ax.set_ylabel("Bias Rate"); ax.set_title("Language Bias Rate (Real vs Fake)")
    ax.set_ylim(0, 1.05)
    # Attention IoU (LPF vs HPF+DC)
    ax = axes[1, 0]
    akeys = ["attn_iou_orig_lpf", "attn_iou_orig_hpf_dc"]
    _bar_with_err(ax, [0, 1],
                  [summary.get(f"{k}_mean", 0) for k in akeys],
                  [summary.get(f"{k}_std", 0)  for k in akeys],
                  ["#16a085", "#d35400"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["orig vs LPF", "orig vs HPF+DC"])
    ax.set_ylabel("IoU"); ax.set_title("Attention IoU (Semantic Anchoring)")
    # ViT cosine sim
    ax = axes[1, 1]
    ax.bar([0], [summary.get("vit_cosine_sim_orig_hpf_dc_mean", 0)],
           yerr=[summary.get("vit_cosine_sim_orig_hpf_dc_std", 0)],
           color="#c0392b", edgecolor="black", capsize=4)
    ax.set_xticks([0]); ax.set_xticklabels(["orig vs HPF+DC"])
    ax.set_ylabel("Cosine Similarity"); ax.set_title("ViT Patch Sim. (Feature Collapse)")
    ax.set_ylim(0, 1.05)
    # Attn IoU Real vs Fake 비교 (HPF+DC)
    ax = axes[1, 2]
    hpf_by_label = [
        by_label.get(l, {}).get("attn_iou_orig_hpf_dc_mean") or 0.0
        for l in ("real", "fake")
    ]
    ax.bar([0, 1], hpf_by_label, color=["#27ae60", "#e74c3c"], edgecolor="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Real", "Fake"])
    ax.set_ylabel("Attn IoU (HPF+DC)"); ax.set_title("Attn IoU HPF+DC: Real vs Fake")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_metrics.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── 3) 카테고리별 정확도 ────────────────────────────────────────────────
    if cats:
        fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.5), 4))
        x, w = np.arange(len(cats)), 0.2
        for i, cond in enumerate(_CONDS):
            accs_c = [by_cate[c].get(f"acc_{cond}") or 0.0 for c in cats]
            ax.bar(x + i * w, accs_c, width=w, color=_COND_COLORS[cond], label=cond, edgecolor="black")
        ax.set_xticks(x + w * 1.5); ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.legend(); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Category")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "summary_by_category.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── 4) 카테고리별 Attn IoU (LPF vs HPF+DC) ──────────────────────────
    if cats:
        fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.5), 4))
        x, w = np.arange(len(cats)), 0.35
        ax.bar(x,       [by_cate[c].get("attn_iou_orig_lpf_mean") or 0.0    for c in cats],
               width=w, color="#16a085", label="Attn IoU orig vs LPF",    edgecolor="black")
        ax.bar(x + w,   [by_cate[c].get("attn_iou_orig_hpf_dc_mean") or 0.0 for c in cats],
               width=w, color="#d35400", label="Attn IoU orig vs HPF+DC", edgecolor="black")
        ax.set_xticks(x + w / 2); ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.legend(); ax.set_ylabel("Attention IoU")
        ax.set_title("Attention IoU by Category (LPF vs HPF+DC)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "summary_attn_iou_by_category.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── 5) 카테고리×라벨별 bias rate 히트맵 ──────────────────────────────
    if cats:
        lnames = ["real", "fake"]
        data = np.array([
            [by_cate[c].get("by_label", {}).get(l, {}).get("language_bias_rate") or 0.0
             for l in lnames]
            for c in cats
        ])
        fig, ax = plt.subplots(figsize=(max(4, len(cats) * 0.8), 3))
        im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_yticks(range(2));  ax.set_yticklabels(lnames)
        plt.colorbar(im, ax=ax, label="Language Bias Rate")
        for i in range(len(cats)):
            for j, l in enumerate(lnames):
                ax.text(i, j, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Language Bias Rate: Category × Label")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "summary_bias_heatmap.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── 6) 카테고리×라벨별 attn_iou_hpf_dc 히트맵 ───────────────────────
    if cats:
        lnames = ["real", "fake"]
        data = np.array([
            [by_cate[c].get("by_label", {}).get(l, {}).get("attn_iou_orig_hpf_dc_mean") or 0.0
             for l in lnames]
            for c in cats
        ])
        fig, ax = plt.subplots(figsize=(max(4, len(cats) * 0.8), 3))
        im = ax.imshow(data.T, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_yticks(range(2));  ax.set_yticklabels(lnames)
        plt.colorbar(im, ax=ax, label="Attn IoU (orig vs HPF+DC)")
        for i in range(len(cats)):
            for j, l in enumerate(lnames):
                ax.text(i, j, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Attn IoU (HPF+DC): Category × Label — Semantic Anchoring")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "summary_attn_iou_hpf_heatmap.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    logger.info("[SAVED] summary PNGs → %s", save_dir)


# ---------------------------------------------------------------------------
# Sanity check: 카테고리별 real/fake 4장씩 → Original, LPF, HPF+DC, Masked 4열 저장
# ---------------------------------------------------------------------------
def mask_image_by_indices(
    image: Image.Image,
    top_indices: List[int],
    grid_size: int = GRID_SIZE,
) -> Image.Image:
    """top_attn_indices_orig로 패치를 0으로 마스킹한 이미지 반환."""
    img_array = np.array(image.convert("RGB"))
    h, w, _ = img_array.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    masked = img_array.copy()
    for idx in top_indices:
        row, col = divmod(idx, grid_size)
        y1, y2 = row * patch_h, min((row + 1) * patch_h, h)
        x1, x2 = col * patch_w, min((col + 1) * patch_w, w)
        masked[y1:y2, x1:x2] = 0
    return Image.fromarray(masked, mode="RGB")


def _make_attn_overlay(
    image: Image.Image,
    top_indices: List[int],
    grid_size: int = GRID_SIZE,
    img_size: int = 336,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    top_attn_indices 기반으로 히트맵을 만들고 원본 위에 오버레이한 numpy 배열 반환.
    저장된 인덱스는 순서 = saliency 크기 순이 아닌 set이므로,
    top-k 패치에 균일한 heat를 할당해 위치 인식에 집중.
    """
    img_size_sq = img_size
    heat = np.zeros(grid_size * grid_size, dtype=np.float32)
    for idx in top_indices:
        if 0 <= idx < len(heat):
            heat[idx] = 1.0
    heat = heat.reshape(grid_size, grid_size)
    heat_img = np.array(
        Image.fromarray((heat * 255).astype(np.uint8)).resize(
            (img_size_sq, img_size_sq), resample=Image.BILINEAR
        )
    ) / 255.0
    colored = (cm.jet(heat_img)[:, :, :3] * 255).astype(np.uint8)
    orig_np = np.array(image.resize((img_size_sq, img_size_sq)))
    overlay = (orig_np.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return overlay


def save_sanity_check_images(
    results: List[dict],
    img_dir: str,
    save_dir: str,
    per_label_count: int = 4,
) -> None:
    """
    전체 카테고리에서 real/fake별로 per_label_count개씩 뽑아
    5열 패널로 저장:
      [Original] [LPF] [HPF+DC] [Masked(black)] [Attn Heatmap Overlay]
    논문 Figure 삽입용으로 Masked 자리에 단순 검은 마스크 대신
    Saliency 위치를 컬러로 보여주는 오버레이를 5번째 열에 추가.
    """
    os.makedirs(save_dir, exist_ok=True)
    picked: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in results:
        key = (r["cate"], int(r["label"]))
        if len(picked[key]) < per_label_count:
            picked[key].append(r)

    for (cate, label), recs in picked.items():
        label_str = "real" if label == 1 else "fake"
        for i, rec in enumerate(recs):
            img_path = os.path.join(img_dir, rec["image"])
            try:
                orig = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning("Sanity image load failed %s: %s", rec["image"], e)
                continue

            lpf_img    = apply_fft_filter(orig, "lpf")
            hpf_img    = apply_fft_filter(orig, "hpf_dc")
            top_idx    = rec.get("top_attn_indices_orig") or []
            masked_img = mask_image_by_indices(orig, top_idx)
            overlay_np = _make_attn_overlay(orig, top_idx)

            # 예측 정보
            p_orig   = rec.get("pred_original", -1)
            p_masked = rec.get("pred_masked", -1)
            pred_str = lambda p: {1: "Real", 0: "Fake"}.get(p, "?")
            label_gt = "Real" if label == 1 else "Fake"

            fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
            panel_data = [
                (orig,       f"Original\n(pred:{pred_str(p_orig)} GT:{label_gt})"),
                (lpf_img,    "LPF\n(저주파만)"),
                (hpf_img,    "HPF+DC\n(노이즈+DC)"),
                (masked_img, f"Masked(top {int(MASK_RATIO*100)}%)\n(pred:{pred_str(p_masked)})"),
                (overlay_np, "Attn Heatmap\n(Top-K 패치 위치)"),
            ]
            for ax, (img_or_arr, title) in zip(axes, panel_data):
                if isinstance(img_or_arr, np.ndarray):
                    ax.imshow(img_or_arr)
                else:
                    ax.imshow(img_or_arr)
                ax.set_title(title, fontsize=8)
                ax.axis("off")

            # CSS/ROUGE-L 수치를 subtitle로
            css_h   = rec.get("css_orig_vs_hpf_dc", 0.0)
            css_m   = rec.get("css_orig_vs_masked", 0.0)
            iou_h   = rec.get("attn_iou_orig_hpf_dc", 0.0)
            vit_s   = rec.get("vit_cosine_sim_orig_hpf_dc", 0.0)
            plt.suptitle(
                f"{cate} | {label_str} | idx={rec['idx']} | "
                f"CSS↓hpf={css_h:.3f} CSS↓mask={css_m:.3f} | "
                f"AttnIoU↓hpf={iou_h:.3f} ViTSim={vit_s:.3f}",
                fontsize=8,
            )
            plt.tight_layout()
            out_path = os.path.join(
                save_dir,
                f"sanity_{cate}_{label_str}_{i:02d}_idx{rec['idx']}.png",
            )
            plt.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)

    logger.info("[SAVED] sanity check images (5-panel) → %s", save_dir)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main(no_visualization: bool = False) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path    = os.path.join(RESULTS_DIR, "results_full.json")
    checkpoint_path = os.path.join(RESULTS_DIR, "results_full_partial.json")
    analysis_path   = os.path.join(RESULTS_DIR, "analysis_full.json")
    viz_dir         = os.path.join(RESULTS_DIR, "visualization")
    sanity_dir      = os.path.join(RESULTS_DIR, "sanity_check")
    heatmap_dir     = os.path.join(RESULTS_DIR, "heatmaps_repr")

    n_gpu = torch.cuda.device_count()
    logger.info("GPUs detected: %d", n_gpu)
    if n_gpu < 2:
        logger.warning("GPU가 1개뿐입니다. model0만 사용하며 단일 GPU로 실행됩니다.")

    logger.info("Loading processor from %s", WEIGHT_PATH)
    processor = load_processor(WEIGHT_PATH)

    logger.info("Loading FakeVLM model0 → cuda:0")
    model0 = load_model(WEIGHT_PATH, "cuda:0")
    logger.info(
        "model0 loaded. VRAM GPU0: %.2f GB alloc / %.2f GB reserved",
        torch.cuda.memory_allocated(0) / 1e9,
        torch.cuda.memory_reserved(0) / 1e9,
    )

    if n_gpu >= 2:
        logger.info("Loading FakeVLM model1 → cuda:1")
        model1 = load_model(WEIGHT_PATH, "cuda:1")
        logger.info(
            "model1 loaded. VRAM GPU1: %.2f GB alloc / %.2f GB reserved",
            torch.cuda.memory_allocated(1) / 1e9,
            torch.cuda.memory_reserved(1) / 1e9,
        )
        sbert_device = "cuda:1"
    else:
        model1 = model0
        sbert_device = "cuda:0"

    logger.info("Loading Sentence-BERT '%s' on %s", SBERT_MODEL, sbert_device)
    text_model = SentenceTransformer(SBERT_MODEL, device=sbert_device)

    logger.info("Loading dataset from %s", DATA_JSON)
    dataset = FakeClueDataset(json_path=DATA_JSON, img_dir=IMG_DIR)
    logger.info("Full dataset: %d samples", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    # 대표 attention heatmap 저장용 (카테고리·라벨별 4개씩)
    heatmap_dir_run = heatmap_dir if not no_visualization else None
    if heatmap_dir_run:
        os.makedirs(heatmap_dir_run, exist_ok=True)

    results = run_unified_experiment(
        model0=model0,
        model1=model1,
        processor=processor,
        text_model=text_model,
        dataloader=dataloader,
        checkpoint_path=checkpoint_path,
        heatmap_dir=heatmap_dir_run,
        heatmap_per_cate_label=4,
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("[SAVED] results → %s (%d samples)", results_path, len(results))

    summary = compute_analysis_summary(results)
    with open(analysis_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("[SAVED] analysis → %s", analysis_path)

    _print_summary(summary)

    # 시각화: 요약 PNG, sanity check 이미지
    if not no_visualization:
        plot_summary_png(summary, viz_dir)
        save_sanity_check_images(results, IMG_DIR, sanity_dir, per_label_count=4)
        logger.info("Visualization done: summary PNGs, sanity check 4-panel images, representative heatmaps.")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="통합 해석 가능성 실험 - 전체 추론 (Full Mode)")
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="실험만 수행하고 요약 PNG / sanity 이미지 / heatmap 저장 생략",
    )
    args = parser.parse_args()
    main(no_visualization=args.no_visualization)

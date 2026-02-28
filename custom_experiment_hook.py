"""
LLM 비주얼 토큰 어텐션 기반 패치 마스킹 실험 (FakeVLM Language Bias 분석).

[설계 원칙 — Gem AC 검토 반영]

기존 설계(CLIP Layer 23 CLS 어텐션)의 구조적 결함:
  LLaVA-1.5는 CLIP의 CLS 토큰을 버리고 576개 공간 패치만 MLP 어댑터로 투사한다.
  따라서 CLIP CLS 어텐션은 LLM이 "fake/real"을 생성할 때 참조하는 시각 영역과
  직접적 연관이 없다.

수정된 Saliency Map 추출 방법 (LLM Decoder Attention):
  model(**inputs, output_attentions=True) 단일 Forward Pass 실행.
  LLM 마지막 디코더 레이어의 어텐션에서
  "마지막 텍스트 토큰(위치 -1) → 576개 비주얼 패치 토큰(위치 1-576)"
  방향의 어텐션 가중치를 Saliency Map으로 사용한다.

  입력 시퀀스 구조 (expanded after _merge_input_ids_with_image_features):
    [0]       BOS
    [1..576]  비주얼 패치 토큰 (CLIP → MLP 어댑터 투사, 576개)
    [577..586] 텍스트 토큰 "Does the image looks real/fake?" (10토큰)
  총 587 토큰, 패딩 없음 (동일 프롬프트 반복으로 모든 배치 동일 길이)

Forward Pass vs. Generate 분리:
  Pass 1 (output_attentions=True): Saliency 추출 전용, generate 없음
  Pass 2 (generate, standard SDPA): 원본 이미지 텍스트 예측
  Pass 3 (generate, standard SDPA): 마스킹 이미지 텍스트 예측

출력:
  - results_hook.json
  - results_hook_partial.json (체크포인트)
"""

import os
import sys
import gc
import json
import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlavaProcessor,
    LlamaTokenizer,
    CLIPImageProcessor,
    LlavaForConditionalGeneration,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
WEIGHT_PATH = "/workspace/fakevlm_analysis/weights/fakeVLM"
DATA_JSON   = "/workspace/fakevlm_analysis/data/FakeClue/data_json/test.json"
IMG_DIR     = "/workspace/fakevlm_analysis/data/FakeClue/test"
RESULTS_DIR = "/workspace/fakevlm_analysis"

# ---------------------------------------------------------------------------
# 하이퍼파라미터
# ---------------------------------------------------------------------------
# !! BATCH_SIZE는 FFT 실험(32)보다 훨씬 작게 유지해야 함 !!
#
# output_attentions=True의 VRAM 구조 (FFT 실험과의 근본적 차이):
#   - 32개 레이어 어텐션 행렬이 Forward 완료 전까지 동시에 메모리에 상주
#     → [BATCH, 32heads, 587seq, 587seq] bfloat16 × 32레이어
#   - Softmax float32 캐스팅 추가 발생 (1레이어당 BATCH×0.04GB)
#
# BATCH별 피크 VRAM 추정 (GPU 44.39GB):
#   BATCH=32 → 14GB(모델) + 21.5GB(어텐션) + 5GB(활성화) ≈ 42GB → OOM ✗
#   BATCH=16 → 14GB + 10.9GB + 3GB ≈ 28GB → 가능
#   BATCH=8  → 14GB +  5.4GB + 2GB ≈ 21GB → 안전 ✓
BATCH_SIZE     = 16
NUM_WORKERS    = 24
MAX_NEW_TOKENS = 128
MASK_RATIO     = 0.20  # 상위 20% 어텐션 패치 마스킹

# CLIP-ViT-L/14@336: 336/14 = 24, 24×24 = 576 패치
GRID_SIZE    = 24
NUM_PATCHES  = GRID_SIZE * GRID_SIZE  # 576

# 입력 시퀀스 내 비주얼 토큰 위치 (expanded sequence 기준)
VISUAL_START = 1    # inclusive
VISUAL_END   = 577  # exclusive (1 + 576)

# LLM 어텐션 평균화에 사용할 마지막 N개 레이어 수
LLM_ATTN_LAYERS = 4

CHECKPOINT_INTERVAL = 100

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
# 프로세서 / 모델 로더 (transformers 4.40.0 호환)
# ---------------------------------------------------------------------------
def load_processor(model_path: str) -> LlavaProcessor:
    tokenizer       = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_model(model_path: str) -> LlavaForConditionalGeneration:
    """
    output_attentions=True 시 Flash Attention 2 및 SDPA는 어텐션 가중치 반환 불가.
    attn_implementation="eager"를 명시하여 SDPA 폴백 경고를 제거하고
    full attention weight 행렬을 안정적으로 수집한다.
    """
    return LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda").eval()


# ---------------------------------------------------------------------------
# LLM Decoder Attention 기반 Saliency Map 추출
# ---------------------------------------------------------------------------
def extract_llm_visual_saliency(
    model: LlavaForConditionalGeneration,
    inputs: dict,
    num_layers_avg: int = LLM_ATTN_LAYERS,
) -> torch.Tensor:
    """
    LLM 디코더 레이어의 어텐션에서 비주얼 패치 Saliency Map을 추출한다.

    방법:
      model(**inputs, output_attentions=True) 호출 →
      LLM 마지막 num_layers_avg개 레이어의 어텐션 텐서 수집 →
      마지막 텍스트 토큰(position=-1)이 비주얼 토큰(positions 1-576)에
      주는 어텐션 가중치를 레이어·헤드 평균.

    output_attentions=True 시 LlamaSdpaAttention은 자동으로 eager 모드로 폴백.
    (transformers 4.40.0 경고 메시지 발생하나 기능적으로 정상)

    Parameters
    ----------
    model          : LlavaForConditionalGeneration (eval mode, cuda)
    inputs         : processor 출력 (input_ids, attention_mask, pixel_values)
    num_layers_avg : 마지막 N개 레이어 어텐션 평균화

    Returns
    -------
    saliency : [batch, NUM_PATCHES] float32 CPU 텐서 (헤드·레이어 평균)
    """
    with torch.no_grad():
        fwd_out = model(**inputs, output_attentions=True)

    # fwd_out.attentions: tuple of 32 tensors, each [batch, heads, seq, seq]
    # seq = 587 (expanded: 1 BOS + 576 visual + 10 question tokens)
    num_layers = len(fwd_out.attentions)
    selected_layers = fwd_out.attentions[-num_layers_avg:]

    # 마지막 텍스트 토큰(-1)이 비주얼 토큰(1:577)에 주는 어텐션
    # [num_layers_avg, batch, heads, NUM_PATCHES]
    attn_to_visual = torch.stack(
        [layer[:, :, -1, VISUAL_START:VISUAL_END].detach().cpu().float()
         for layer in selected_layers]
    )

    # 레이어·헤드 평균: [batch, NUM_PATCHES]
    saliency = attn_to_visual.mean(dim=(0, 2))

    del fwd_out, selected_layers, attn_to_visual
    torch.cuda.empty_cache()

    return saliency


# ---------------------------------------------------------------------------
# 패치 마스킹
# ---------------------------------------------------------------------------
def mask_top_attention_patches(
    image: Image.Image,
    saliency: torch.Tensor,  # [NUM_PATCHES] float32
    grid_size: int = GRID_SIZE,
    mask_ratio: float = MASK_RATIO,
) -> Image.Image:
    """
    LLM Saliency Map 기준 상위 활성화 패치를 검정 픽셀(0)으로 마스킹한다.
    saliency: [NUM_PATCHES] — extract_llm_visual_saliency의 단일 샘플 출력
    """
    img_array = np.array(image.convert("RGB"))
    h, w, _ = img_array.shape
    num_patches = grid_size * grid_size

    top_k = max(1, int(num_patches * mask_ratio))
    top_indices = torch.topk(saliency, top_k).indices

    patch_h = h // grid_size
    patch_w = w // grid_size
    masked = img_array.copy()

    for idx in top_indices:
        row, col = divmod(int(idx.item()), grid_size)
        y1 = row * patch_h
        y2 = min((row + 1) * patch_h, h)
        x1 = col * patch_w
        x2 = min((col + 1) * patch_w, w)
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
# Dataset / collate
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
# 메인 실험 루프
# ---------------------------------------------------------------------------
def main():
    output_path = os.path.join(RESULTS_DIR, "results_hook.json")
    if os.path.exists(output_path):
        logger.info("[SKIP] %s already exists.", output_path)
        return

    logger.info("Loading processor from %s", WEIGHT_PATH)
    processor = load_processor(WEIGHT_PATH)

    logger.info("Loading model from %s (BATCH_SIZE=%d, NUM_WORKERS=%d)", WEIGHT_PATH, BATCH_SIZE, NUM_WORKERS)
    model = load_model(WEIGHT_PATH)
    logger.info(
        "Model loaded. VRAM: %.2f GB allocated / %.2f GB reserved",
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )

    dataset = FakeClueDataset(json_path=DATA_JSON, img_dir=IMG_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        prefetch_factor=2,
    )

    results = []

    with torch.no_grad():
        for images, prompts, labels, cates, indices, paths in tqdm(
            dataloader, desc="Hook Experiment [LLM Attention Saliency]", unit="batch"
        ):
            batch_size = len(images)

            # ---------------------------------------------------------------
            # Pass 1: output_attentions=True forward → Saliency Map 추출
            # LlamaSdpaAttention이 eager 모드로 자동 폴백
            # ---------------------------------------------------------------
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to("cuda", torch.bfloat16)

            # [batch, 576] float32 CPU 텐서
            saliency_batch = extract_llm_visual_saliency(model, inputs, LLM_ATTN_LAYERS)

            # ---------------------------------------------------------------
            # Pass 2: 원본 이미지 → 텍스트 예측
            # ---------------------------------------------------------------
            out_orig = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            texts_orig = processor.batch_decode(out_orig, skip_special_tokens=True)
            preds_orig = [parse_prediction(t) for t in texts_orig]

            del inputs, out_orig
            torch.cuda.empty_cache()

            # ---------------------------------------------------------------
            # 마스킹 적용 후 Pass 3
            # ---------------------------------------------------------------
            masked_images = [
                mask_top_attention_patches(img, saliency_batch[i])
                for i, img in enumerate(images)
            ]

            inputs_masked = processor(
                text=prompts,
                images=masked_images,
                return_tensors="pt",
                padding=True,
            ).to("cuda", torch.bfloat16)

            out_masked = model.generate(**inputs_masked, max_new_tokens=MAX_NEW_TOKENS)
            texts_masked = processor.batch_decode(out_masked, skip_special_tokens=True)
            preds_masked = [parse_prediction(t) for t in texts_masked]

            del inputs_masked, out_masked
            torch.cuda.empty_cache()

            # ---------------------------------------------------------------
            # 결과 기록
            # ---------------------------------------------------------------
            for i in range(batch_size):
                p_orig, p_mask = preds_orig[i], preds_masked[i]
                # Language Bias: 마스킹 전후 동일 예측 유지 (미결정 제외)
                is_bias = bool(p_orig != -1 and p_orig == p_mask)

                results.append(
                    {
                        "idx":           int(indices[i]),
                        "image":         paths[i],
                        "label":         int(labels[i]),
                        "cate":          cates[i],
                        "pred_original": p_orig,
                        "pred_masked":   p_mask,
                        "text_original": texts_orig[i],
                        "text_masked":   texts_masked[i],
                        "language_bias": is_bias,
                        "attn_max":      float(saliency_batch[i].max()),
                        "attn_mean":     float(saliency_batch[i].mean()),
                    }
                )

            # 중간 저장 + VRAM 로그 (CHECKPOINT_INTERVAL 경계 통과 시)
            if len(results) % CHECKPOINT_INTERVAL < BATCH_SIZE:
                partial_path = os.path.join(RESULTS_DIR, "results_hook_partial.json")
                with open(partial_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(
                    "[CHECKPOINT] %d samples saved. VRAM: %.2f GB allocated",
                    len(results), torch.cuda.memory_allocated() / 1e9,
                )

    # 최종 저장
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "[SAVED] %s (%d samples). VRAM: %.2f GB allocated",
        output_path, len(results), torch.cuda.memory_allocated() / 1e9,
    )

    # Language Bias 통계 요약
    valid      = [r for r in results if r["pred_original"] != -1 and r["pred_masked"] != -1]
    bias_cases = [r for r in valid if r["language_bias"]]
    bias_rate  = len(bias_cases) / len(valid) if valid else 0.0

    correct_orig  = [r for r in valid if r["pred_original"] == r["label"]]
    correct_mask  = [r for r in valid if r["pred_masked"]   == r["label"]]
    true_bias     = [r for r in bias_cases if r["pred_original"] == r["label"]]
    wrong_bias    = [r for r in bias_cases if r["pred_original"] != r["label"]]

    logger.info("=" * 60)
    logger.info("Language Bias Summary (LLM Decoder Attention Masking)")
    logger.info("  Total results           : %d", len(results))
    logger.info("  Valid (both parsed)     : %d", len(valid))
    logger.info("  Accuracy (original)     : %.4f", len(correct_orig) / len(valid) if valid else 0)
    logger.info("  Accuracy (masked)       : %.4f", len(correct_mask) / len(valid) if valid else 0)
    logger.info("  Bias cases (same pred)  : %d / %d", len(bias_cases), len(valid))
    logger.info("  Language Bias Rate      : %.4f", bias_rate)
    logger.info("  [Correct + Bias]        : %d (%.2f%%)",
                len(true_bias), len(true_bias) / len(valid) * 100 if valid else 0)
    logger.info("  [Wrong + Bias]          : %d (%.2f%%)",
                len(wrong_bias), len(wrong_bias) / len(valid) * 100 if valid else 0)
    logger.info("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
FFT 기반 주파수 넉아웃 실험 (FakeVLM 내부 기작 분석).

조건:
  - original : 원본 이미지 (필터 미적용)
  - lpf      : 저주파 통과 필터 (Radius=30, 나이퀴스트 대비 ~17.9%)
  - hpf      : 고주파 통과 필터 (Radius=30)

출력:
  - results_fft_original.json  (label, prediction, cate 포함)
  - results_fft_lpf.json
  - results_fft_hpf.json
  - fft_sanity_check/          (각 조건별 샘플 이미지 5장)

하드웨어 활용:
  - BATCH_SIZE=16 (L40S 48GB: 모델 14GB + 배치 KV캐시 ~6GB + 활성화 ~6GB = ~26GB 예상)
  - NUM_WORKERS=16 (32-core CPU 활용)
  - Flash Attention 2 (설치된 경우 자동 활성화)
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
SANITY_DIR  = os.path.join(RESULTS_DIR, "fft_sanity_check")

# ---------------------------------------------------------------------------
# 하이퍼파라미터
# ---------------------------------------------------------------------------
BATCH_SIZE     = 32    # VRAM ~33GB 목표 (모델14GB + KV캐시11GB + 활성화8GB)
NUM_WORKERS    = 16
MAX_NEW_TOKENS = 128
FFT_RADIUS     = 30
# hpf_dc: HPF 후 DC Offset 복원 → CLIP 정규화 분포 이탈 방지 (Gem AC 지적 반영)
# hpf    : DC 제거 순수 HPF → "시각 완전 박탈" 대조군 (언어 편향 측정용으로 재분류)
CONDITIONS     = ["original", "lpf", "hpf", "hpf_dc"]
SANITY_N       = 5

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
# FFT 필터 함수
# ---------------------------------------------------------------------------
def apply_fft_filter(
    image: Image.Image,
    filter_type: str = "original",
    radius: int = FFT_RADIUS,
) -> Image.Image:
    """
    RGB 이미지에 FFT 기반 주파수 필터를 채널별로 독립 적용한다.

    Radius 30 / 해상도 336 → 나이퀴스트 주파수 168 대비 17.9%.

    filter_type:
      'lpf'    : 저주파만 유지 (색상·형태 보존, 텍스처 제거)
      'hpf'    : DC 제거 순수 HPF → 근암흑 이미지 (OOD, 언어 편향 대조군)
      'hpf_dc' : HPF + 원본 채널 평균(DC) 복원 → 회색 배경 위 엣지·노이즈 패턴
                 CLIP 정규화 통계 범위 유지, 공정한 고주파 탐지 실험 가능
    """
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
            # np.real(): 실입력 신호의 IFFT 결과는 이론상 순실수.
            # np.abs() 사용 시 음수값 전파 정류(full-wave rectification) 발생 → np.real() 필수
            img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))

        elif filter_type == "hpf":
            # DC 성분 제거 → 픽셀값이 0 근방으로 붕괴 (OOD, CLIP 활성화 붕괴)
            mask = 1.0 - circle_mask
            img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))

        elif filter_type == "hpf_dc":
            # HPF 후 원본 채널 평균(DC 성분) 복원
            # np.real() 사용으로 고주파 성분의 부호(위상) 보존 — Gem AC 지적 수용
            # np.abs() 사용 시: 음수 AC 성분 → 양수로 뒤집힘 → 인위적 하모닉스 생성
            dc_offset = channel.mean()
            mask = 1.0 - circle_mask
            img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))
            img_back = img_back + dc_offset

        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        filtered_channels.append(np.clip(img_back, 0, 255).astype(np.uint8))

    return Image.fromarray(np.stack(filtered_channels, axis=2), mode="RGB")


# ---------------------------------------------------------------------------
# Sanity Check 시각화
# ---------------------------------------------------------------------------
def save_fft_sanity_check(data: list, img_dir: str, n: int = SANITY_N) -> None:
    """
    각 필터 조건의 처리 결과를 시각적으로 확인하기 위한 샘플 이미지를 저장한다.
    4열: original | LPF | HPF (OOD) | HPF+DC (공정 실험용)
    """
    os.makedirs(SANITY_DIR, exist_ok=True)
    saved = 0
    for i, item in enumerate(data[:50]):
        img_path = os.path.join(img_dir, item["image"])
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            continue

        W = 336
        orig   = img.resize((W, W))
        lpf    = apply_fft_filter(img, "lpf").resize((W, W))
        hpf    = apply_fft_filter(img, "hpf").resize((W, W))
        hpf_dc = apply_fft_filter(img, "hpf_dc").resize((W, W))

        # 4열 연결: original | LPF | HPF(OOD) | HPF+DC
        combined = Image.new("RGB", (W * 4, W))
        combined.paste(orig,   (0,      0))
        combined.paste(lpf,    (W,      0))
        combined.paste(hpf,    (W * 2,  0))
        combined.paste(hpf_dc, (W * 3,  0))

        label_str = "fake" if item["label"] == 0 else "real"
        fname = f"sanity_{i:04d}_{label_str}_{item['cate']}.png"
        combined.save(os.path.join(SANITY_DIR, fname))
        saved += 1
        if saved >= n:
            break

    logger.info(
        "[SANITY] %d images saved to %s (col: orig | LPF | HPF-OOD | HPF-DC)",
        saved, SANITY_DIR,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FakeClueDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        img_dir: str,
        filter_type: str = "original",
        radius: int = FFT_RADIUS,
    ):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_dir  = img_dir
        self.filter_type = filter_type
        self.radius   = radius

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

        filtered = apply_fft_filter(image, self.filter_type, self.radius)
        prompt   = item["conversations"][0]["value"]
        label    = int(item["label"])
        cate     = item.get("cate", "unknown")
        return filtered, prompt, label, cate, idx


def collate_fn(batch):
    images, prompts, labels, cates, indices = zip(*batch)
    return list(images), list(prompts), list(labels), list(cates), list(indices)


# ---------------------------------------------------------------------------
# 프로세서 로더 (transformers 4.40.0 호환)
# ---------------------------------------------------------------------------
def load_processor(model_path: str) -> LlavaProcessor:
    """
    processor_config.json의 image_token 필드가 transformers 4.40.0의
    LlavaProcessor.__init__ 시그니처와 호환되지 않으므로 컴포넌트 직접 조립.
    """
    tokenizer       = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# 예측 파싱
# ---------------------------------------------------------------------------
def parse_prediction(text: str) -> int:
    """
    모델 출력 텍스트에서 real/fake 키워드 파싱.
    Returns: 1=real, 0=fake, -1=미결정
    """
    response = text.split("?")[-1].strip().lower()
    for part in response.split(".")[:2]:
        if "real" in part:
            return 1
        if "fake" in part:
            return 0
    return -1


# ---------------------------------------------------------------------------
# 추론 루프
# ---------------------------------------------------------------------------
def run_inference(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    dataloader: DataLoader,
    condition: str,
) -> list:
    results = []
    model.eval()

    with torch.no_grad():
        for images, prompts, labels, cates, indices in tqdm(
            dataloader, desc=f"Inference [{condition}]", unit="batch"
        ):
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to("cuda", torch.bfloat16)

            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            decoded_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

            for text, label, cate, idx in zip(decoded_texts, labels, cates, indices):
                results.append(
                    {
                        "idx":             int(idx),
                        "label":           int(label),
                        "cate":            cate,
                        "prediction":      parse_prediction(text),
                        "prediction_text": text,
                    }
                )

            del inputs, output_ids
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 모델 로더 (Flash Attention 2 지원 포함)
# ---------------------------------------------------------------------------
def load_model(model_path: str) -> LlavaForConditionalGeneration:
    """
    Flash Attention 2가 설치된 경우 자동 활성화하여 처리량을 향상시킨다.
    미설치 시 표준 Scaled Dot-Product Attention(SDPA)로 폴백한다.
    """
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    try:
        import flash_attn  # noqa: F401
        kwargs["use_flash_attention_2"] = True
        logger.info("Flash Attention 2 enabled.")
    except ImportError:
        logger.info("Flash Attention 2 not available. Using SDPA.")

    return LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs).to("cuda").eval()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    logger.info("Loading processor from %s", WEIGHT_PATH)
    processor = load_processor(WEIGHT_PATH)

    logger.info("Loading model (BATCH_SIZE=%d, NUM_WORKERS=%d)", BATCH_SIZE, NUM_WORKERS)
    model = load_model(WEIGHT_PATH)
    logger.info(
        "Model loaded. VRAM: %.2f GB allocated / %.2f GB reserved",
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )

    # Sanity check: 실험 시작 전 첫 5개 샘플 필터 결과 시각화
    with open(DATA_JSON) as f:
        raw_data = json.load(f)
    if not os.path.exists(SANITY_DIR):
        save_fft_sanity_check(raw_data, IMG_DIR)

    for condition in CONDITIONS:
        output_path = os.path.join(RESULTS_DIR, f"results_fft_{condition}.json")

        if os.path.exists(output_path):
            logger.info("[SKIP] %s already exists.", output_path)
            continue

        logger.info("Starting condition: %s (radius=%d)", condition, FFT_RADIUS)

        dataset = FakeClueDataset(
            json_path=DATA_JSON,
            img_dir=IMG_DIR,
            filter_type=condition,
            radius=FFT_RADIUS,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=False,
            prefetch_factor=2,
        )

        results = run_inference(model, processor, dataloader, condition)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        valid   = sum(1 for r in results if r["prediction"] != -1)
        undeter = len(results) - valid
        logger.info(
            "[SAVED] %s | total=%d valid=%d undetermined=%d",
            output_path, len(results), valid, undeter,
        )

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            "VRAM after condition [%s]: %.2f GB", condition,
            torch.cuda.memory_allocated() / 1e9,
        )

    logger.info("FFT frequency knockout experiment complete.")


if __name__ == "__main__":
    main()

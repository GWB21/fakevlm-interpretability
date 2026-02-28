"""
Hook Experiment (Experiment 2) Sanity Check 이미지 생성기.

각 카테고리마다 1장의 PNG를 생성한다:
  - 행(row) : fake 5개 → real 5개  (총 10행)
  - 열(col) : Original | Saliency Heatmap | Heatmap Overlay | Masked (Top 20%)
  - 행 레이블: Label / pred_orig→pred_masked / Language Bias 여부

추가 정보:
  - 사이컬럼: attn_max·attn_mean 수치 표시
  - 빨간 테두리 = FAKE / 파란 테두리 = REAL
  - 별(★) = Language Bias 발생 케이스

실행 전 조건: custom_experiment_hook.py 완료 후 실행 (VRAM 동시 점유 방지)
출력: hook_sanity_by_category/<cate>_hook_sanity.png
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
from matplotlib.gridspec import GridSpec
from transformers import (
    LlavaProcessor,
    LlamaTokenizer,
    CLIPImageProcessor,
    LlavaForConditionalGeneration,
)

# ------------------------------------------------------------------
# 설정 (hook.py 설정과 반드시 일치)
# ------------------------------------------------------------------
WEIGHT_PATH  = "/workspace/fakevlm_analysis/weights/fakeVLM"
DATA_JSON    = "/workspace/fakevlm_analysis/data/FakeClue/data_json/test.json"
IMG_DIR      = "/workspace/fakevlm_analysis/data/FakeClue/test"
RESULTS_JSON = "/workspace/fakevlm_analysis/results_hook.json"
OUT_DIR      = "hook_sanity_by_category"

N_PER_LABEL  = 5
SEED         = 42
IMG_SIZE     = 336
GRID_SIZE    = 24           # 24×24 = 576 패치
VISUAL_START = 1
VISUAL_END   = 577
LLM_ATTN_LAYERS = 4
MASK_RATIO   = 0.20

FAKE_BG = "#ffe0e0"
REAL_BG = "#e0f0ff"
random.seed(SEED)


# ------------------------------------------------------------------
# 모델 / 프로세서
# ------------------------------------------------------------------
def load_processor(path):
    tok = LlamaTokenizer.from_pretrained(path, use_fast=False)
    img_proc = CLIPImageProcessor.from_pretrained(path)
    return LlavaProcessor(image_processor=img_proc, tokenizer=tok)


def load_model(path):
    return LlavaForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda").eval()


# ------------------------------------------------------------------
# Saliency 추출 (단일 샘플)
# ------------------------------------------------------------------
@torch.no_grad()
def get_saliency(model, processor, image: Image.Image, prompt: str) -> np.ndarray:
    """
    Returns:
        saliency [24, 24] float32 (레이어·헤드 평균, CPU numpy)
    """
    inputs = processor(
        text=[prompt], images=[image], return_tensors="pt", padding=True,
    ).to("cuda", torch.bfloat16)

    fwd = model(**inputs, output_attentions=True)

    # 마지막 LLM_ATTN_LAYERS 레이어, 마지막 텍스트 토큰(-1) → 비주얼 패치(1:577)
    selected = fwd.attentions[-LLM_ATTN_LAYERS:]
    attn_vis = torch.stack(
        [layer[0, :, -1, VISUAL_START:VISUAL_END].detach().cpu().float()
         for layer in selected]
    )  # [LLM_ATTN_LAYERS, heads, 576]
    saliency = attn_vis.mean(dim=(0, 1))  # [576]

    del fwd, selected, attn_vis, inputs
    torch.cuda.empty_cache()

    return saliency.reshape(GRID_SIZE, GRID_SIZE).numpy()


# ------------------------------------------------------------------
# 시각화 헬퍼
# ------------------------------------------------------------------
def saliency_to_heatmap(sal_2d: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """[24,24] → [size,size,3] uint8 jet colormap."""
    sal_norm = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-9)
    sal_pil  = Image.fromarray((sal_norm * 255).astype(np.uint8)).resize(
        (size, size), Image.NEAREST
    )
    rgba = mcm.jet(np.array(sal_pil) / 255.0)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def saliency_overlay(orig: np.ndarray, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    """원본 이미지 위에 반투명 heatmap 합성."""
    return np.clip(
        (1 - alpha) * orig.astype(np.float32) + alpha * heatmap.astype(np.float32),
        0, 255
    ).astype(np.uint8)


def masked_image(orig_pil: Image.Image, sal_2d: np.ndarray) -> np.ndarray:
    """상위 MASK_RATIO 패치를 검정으로 마스킹 + 패치 그리드 오버레이."""
    arr = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    flat = sal_2d.flatten()
    top_k = max(1, int(len(flat) * MASK_RATIO))
    top_idx = np.argsort(flat)[::-1][:top_k]

    patch_h = IMG_SIZE // GRID_SIZE
    patch_w = IMG_SIZE // GRID_SIZE
    result  = arr.copy()

    # 마스킹 + 그리드 선
    for idx in top_idx:
        r, c = divmod(int(idx), GRID_SIZE)
        y1, y2 = r * patch_h, min((r + 1) * patch_h, IMG_SIZE)
        x1, x2 = c * patch_w, min((c + 1) * patch_w, IMG_SIZE)
        result[y1:y2, x1:x2] = 0

    # 비마스킹 패치에 연한 그리드
    for r in range(GRID_SIZE + 1):
        y = r * patch_h
        if 0 <= y < IMG_SIZE:
            result[y, :] = np.clip(result[y, :].astype(int) + 30, 0, 255)
    for c in range(GRID_SIZE + 1):
        x = c * patch_w
        if 0 <= x < IMG_SIZE:
            result[:, x] = np.clip(result[:, x].astype(int) + 30, 0, 255)

    return result


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------
def main():
    # ---- 결과 JSON 로드 (pred / language_bias 어노테이션용) ----
    with open(RESULTS_JSON) as f:
        results_list = json.load(f)
    results_map = {r["idx"]: r for r in results_list}

    # ---- 데이터 JSON 로드 ----
    with open(DATA_JSON) as f:
        data = json.load(f)

    # image 경로 → dataset index 매핑 (results_hook.json의 idx와 일치)
    image_to_idx = {item["image"]: i for i, item in enumerate(data)}

    # 카테고리 × 레이블 분류
    from collections import defaultdict
    by_cate: dict = defaultdict(lambda: {"fake": [], "real": []})
    for item in data:
        lbl = "real" if item["label"] == 1 else "fake"
        by_cate[item["cate"]][lbl].append(item)

    # ---- 모델 로드 ----
    print("Loading processor...")
    processor = load_processor(WEIGHT_PATH)
    print("Loading model (eager attn)...")
    model = load_model(WEIGHT_PATH)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    Path(OUT_DIR).mkdir(exist_ok=True)

    PRED_LABEL = {1: "REAL", 0: "FAKE", -1: "?"}

    for cate in sorted(by_cate.keys()):
        pool   = by_cate[cate]
        fakes  = random.sample(pool["fake"], min(N_PER_LABEL, len(pool["fake"])))
        reals  = random.sample(pool["real"], min(N_PER_LABEL, len(pool["real"])))
        samples = [("fake", it) for it in fakes] + [("real", it) for it in reals]

        n_rows = len(samples)
        n_cols = 4  # Original | Heatmap | Overlay | Masked

        # ---- figure ----
        cell_w   = IMG_SIZE / 100
        cell_h   = IMG_SIZE / 100
        pad_top  = 0.70
        pad_left = 1.30
        pad_bot  = 0.35

        fig_w = pad_left + n_cols * cell_w
        fig_h = pad_top  + n_rows * cell_h + pad_bot

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Hook Experiment — Category: {cate.upper()}  "
            f"[fake={len(pool['fake'])} / real={len(pool['real'])}]  "
            f"Mask ratio={int(MASK_RATIO*100)}%  LLM layers avg={LLM_ATTN_LAYERS}",
            fontsize=13, fontweight="bold", y=1.0 - 0.03 / fig_h,
        )

        gs = GridSpec(
            n_rows, n_cols + 1,
            figure=fig,
            left=pad_left / fig_w,
            right=0.99,
            top=1.0 - pad_top / fig_h,
            bottom=pad_bot / fig_h,
            wspace=0.04, hspace=0.07,
        )

        # 열 제목
        col_titles = ["Original", "Saliency Heatmap", "Heatmap Overlay", f"Masked (Top {int(MASK_RATIO*100)}%)"]
        for c, t in enumerate(col_titles):
            ax = fig.add_subplot(gs[0, c + 1])
            ax.set_title(t, fontsize=9, fontweight="bold", pad=3)
            ax.axis("off")

        for r, (lbl_str, item) in enumerate(samples):
            # --- 이미지 로드 ---
            img_path = os.path.join(IMG_DIR, item["image"])
            try:
                orig_pil = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            except OSError:
                orig_pil = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

            prompt  = item["conversations"][0]["value"]
            ds_idx  = image_to_idx.get(item["image"], -1)
            res     = results_map.get(ds_idx, {})

            pred_o = res.get("pred_original", -1)
            pred_m = res.get("pred_masked",   -1)
            is_bias  = res.get("language_bias", False)
            attn_max  = res.get("attn_max",  0.0)
            attn_mean = res.get("attn_mean", 0.0)

            # --- Saliency 추출 ---
            sal_2d = get_saliency(model, processor, orig_pil, prompt)

            # --- 열별 이미지 생성 ---
            heatmap   = saliency_to_heatmap(sal_2d)
            overlay   = saliency_overlay(np.array(orig_pil), heatmap)
            masked    = masked_image(orig_pil, sal_2d)

            col_imgs = [np.array(orig_pil), heatmap, overlay, masked]
            bg  = FAKE_BG if lbl_str == "fake" else REAL_BG
            ec  = "#cc0000" if lbl_str == "fake" else "#0066cc"
            tc  = "#c00000" if lbl_str == "fake" else "#0055aa"

            # ---- 행 레이블 셀 ----
            bias_mark = " ★" if is_bias else ""
            ax_lbl = fig.add_subplot(gs[r, 0])
            ax_lbl.set_facecolor(bg)
            ax_lbl.text(
                0.5, 0.62,
                f"{lbl_str.upper()}{bias_mark}",
                ha="center", va="center", fontsize=8, fontweight="bold", color=tc,
            )
            ax_lbl.text(
                0.5, 0.38,
                f"{PRED_LABEL[pred_o]}→{PRED_LABEL[pred_m]}",
                ha="center", va="center", fontsize=7,
                color="#228800" if pred_o == pred_m else "#aa0000",
            )
            ax_lbl.text(
                0.5, 0.15,
                f"max={attn_max:.4f}\nmean={attn_mean:.5f}",
                ha="center", va="center", fontsize=5.5, color="#444444",
            )
            ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(0, 1)
            ax_lbl.axis("off")

            # ---- 이미지 4열 ----
            for c, img_arr in enumerate(col_imgs):
                ax = fig.add_subplot(gs[r, c + 1])
                ax.imshow(img_arr)
                ax.set_facecolor(bg)
                for sp in ax.spines.values():
                    sp.set_edgecolor(ec); sp.set_linewidth(1.5)
                ax.set_xticks([]); ax.set_yticks([])

        # 범례
        fake_p  = mpatches.Patch(color=FAKE_BG, label="FAKE")
        real_p  = mpatches.Patch(color=REAL_BG, label="REAL")
        star_p  = mpatches.Patch(color="gold",  label="★ Language Bias")
        fig.legend(handles=[fake_p, real_p, star_p], loc="lower right",
                   fontsize=8, framealpha=0.8)

        out_path = os.path.join(OUT_DIR, f"{cate}_hook_sanity.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_path}  (VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB)")

    print(f"\n완료: {OUT_DIR}/ 에 {len(by_cate)}개 파일 저장됨.")


if __name__ == "__main__":
    main()

"""
카테고리별 Real/Fake Sanity Check 이미지 생성기.

각 카테고리마다 1장의 PNG를 생성한다:
  - 행(row) : fake 5개 → real 5개 (총 10행)
  - 열(col) : original | LPF | HPF-OOD | HPF+DC (4열)

출력: sanity_by_category/<cate>_sanity.png
"""

import json
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from custom_experiment_fft import apply_fft_filter

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
DATA_JSON  = "data/FakeClue/data_json/test.json"
IMG_DIR    = "data/FakeClue/test"
OUT_DIR    = "sanity_by_category"
N_PER_LABEL = 5         # 레이블당 샘플 수
SEED        = 42
IMG_SIZE    = 224        # 각 셀 해상도 (픽셀)
RADIUS      = 30

CONDITIONS  = ["original", "lpf", "hpf", "hpf_dc"]
COL_TITLES  = ["Original", "LPF (r=30)", "HPF-OOD", "HPF+DC"]

# 배경 색 (행 구분용)
FAKE_BG = "#ffe0e0"   # 연한 빨강
REAL_BG = "#e0f0ff"   # 연한 파랑

random.seed(SEED)


# ------------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------------
def load_rgb(item: dict) -> Image.Image:
    path = os.path.join(IMG_DIR, item["image"])
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))


def make_row_images(item: dict) -> list[Image.Image]:
    """original / lpf / hpf / hpf_dc 순서로 PIL 이미지 반환."""
    orig = load_rgb(item)
    imgs = [apply_fft_filter(orig, cond, RADIUS) for cond in CONDITIONS]
    return [img.resize((IMG_SIZE, IMG_SIZE)) for img in imgs]


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------
def main():
    with open(DATA_JSON) as f:
        data = json.load(f)

    # 카테고리 × 레이블 분류
    from collections import defaultdict
    by_cate: dict[str, dict[str, list]] = defaultdict(lambda: {"fake": [], "real": []})
    for item in data:
        label_str = "real" if item["label"] == 1 else "fake"
        by_cate[item["cate"]][label_str].append(item)

    Path(OUT_DIR).mkdir(exist_ok=True)

    for cate in sorted(by_cate.keys()):
        pool = by_cate[cate]
        fakes = random.sample(pool["fake"], min(N_PER_LABEL, len(pool["fake"])))
        reals = random.sample(pool["real"], min(N_PER_LABEL, len(pool["real"])))
        samples = [("fake", item) for item in fakes] + \
                  [("real", item) for item in reals]

        n_rows  = len(samples)          # 최대 10
        n_cols  = len(CONDITIONS)       # 4

        # --- figure 설정 ---
        cell_w = IMG_SIZE / 100         # inch (100 dpi 기준)
        cell_h = IMG_SIZE / 100
        pad_top    = 0.55               # 제목용 상단 여백
        pad_left   = 1.1               # 행 레이블용 좌측 여백
        pad_bottom = 0.25

        fig_w = pad_left + n_cols * cell_w
        fig_h = pad_top  + n_rows * cell_h + pad_bottom

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        fig.patch.set_facecolor("white")

        # 전체 제목
        fig.suptitle(
            f"Category: {cate.upper()}   "
            f"[fake={len(pool['fake'])} / real={len(pool['real'])}]",
            fontsize=14, fontweight="bold", y=1.0 - 0.04 / fig_h,
        )

        # GridSpec: 좌측 레이블 열 + 4개 이미지 열
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(
            n_rows, n_cols + 1,
            figure=fig,
            left=pad_left / fig_w,
            right=0.99,
            top=1.0 - pad_top / fig_h,
            bottom=pad_bottom / fig_h,
            wspace=0.04,
            hspace=0.06,
        )

        # 열 제목 (첫 행 위에)
        for c, title in enumerate(COL_TITLES):
            ax_title = fig.add_subplot(gs[0, c + 1])
            ax_title.set_title(title, fontsize=9, fontweight="bold", pad=3)
            ax_title.axis("off")

        for r, (label_str, item) in enumerate(samples):
            row_imgs = make_row_images(item)
            bg_color = FAKE_BG if label_str == "fake" else REAL_BG

            # 좌측 레이블 셀
            ax_lbl = fig.add_subplot(gs[r, 0])
            ax_lbl.set_facecolor(bg_color)
            ax_lbl.text(
                0.5, 0.5,
                f"{label_str.upper()}\n#{r if label_str=='fake' else r - N_PER_LABEL}",
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="#c00000" if label_str == "fake" else "#0055aa",
            )
            ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(0, 1)
            ax_lbl.axis("off")

            # 이미지 4열
            for c, img in enumerate(row_imgs):
                ax = fig.add_subplot(gs[r, c + 1])
                ax.imshow(np.array(img))
                ax.set_facecolor(bg_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(
                        "#cc0000" if label_str == "fake" else "#0066cc"
                    )
                    spine.set_linewidth(1.2)
                ax.set_xticks([]); ax.set_yticks([])

        # 범례
        fake_patch = mpatches.Patch(color=FAKE_BG, label="FAKE")
        real_patch = mpatches.Patch(color=REAL_BG, label="REAL")
        fig.legend(
            handles=[fake_patch, real_patch],
            loc="lower right", fontsize=8,
            framealpha=0.8,
        )

        out_path = os.path.join(OUT_DIR, f"{cate}_sanity.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_path}")

    print(f"\n완료: {OUT_DIR}/ 에 {len(by_cate)}개 파일 저장됨.")


if __name__ == "__main__":
    main()

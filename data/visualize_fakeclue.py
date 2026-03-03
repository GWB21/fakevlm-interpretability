"""
FakeClue 테스트셋 시각화 스크립트
- 전체 5000개 이미지에 대해 이미지 + Fake/Real 라벨 + Reasoning GT 전체 표시
- 카테고리별 페이지로 저장 (data/FakeClue_visualized/)
- 통계 요약 차트 생성
"""

import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

# ── 경로 설정 ────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "FakeClue"
JSON_PATH  = DATA_DIR / "data_json" / "test.json"
IMG_ROOT   = DATA_DIR / "test"
OUT_DIR    = BASE_DIR / "FakeClue_visualized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 레이아웃 설정 ─────────────────────────────────────────────────
COLS        = 3          # 한 행에 이미지 수
ROWS        = 3          # 한 페이지에 행 수  (3×3 = 9개/페이지)
PER_PAGE    = COLS * ROWS
FIG_W       = 22
FIG_H       = 48         # 텍스트 전체 표시를 위해 충분한 높이
REASON_WRAP = 68         # reasoning 줄바꿈 폭(문자)
# REASON_LINES 제한 없음 — 전체 표시

LABEL_COLOR = {0: "#E53935", 1: "#43A047"}   # 0=FAKE(빨), 1=REAL(초)
LABEL_TEXT  = {0: "FAKE", 1: "REAL"}

CATE_ORDER  = ["deepfake", "human", "animal", "object", "scene", "satellite", "doc"]

# ── 데이터 로드 ───────────────────────────────────────────────────
print("📂 test.json 로드 중...")
with open(JSON_PATH, encoding="utf-8") as f:
    raw = json.load(f)
print(f"  총 {len(raw):,}개 샘플")


def get_reasoning(sample: dict) -> str:
    for turn in sample["conversations"]:
        if turn["from"] == "gpt":
            return turn["value"].strip()
    return "(reasoning 없음)"


def wrap_reason(text: str) -> str:
    """reasoning 전체 텍스트를 줄바꿈만 적용해 반환 (잘림 없음)"""
    lines = textwrap.wrap(text, width=REASON_WRAP)
    return "\n".join(lines)


# ── 카테고리별로 그룹핑 ───────────────────────────────────────────
by_cate = defaultdict(list)
for s in raw:
    by_cate[s["cate"]].append(s)


# ── 통계 집계 ─────────────────────────────────────────────────────
def collect_stats(data):
    total   = len(data)
    n_fake  = sum(1 for d in data if d["label"] == 0)
    n_real  = total - n_fake
    cate_stats = {}
    for cate, items in by_cate.items():
        f = sum(1 for i in items if i["label"] == 0)
        r = len(items) - f
        cate_stats[cate] = {"total": len(items), "fake": f, "real": r}
    return total, n_fake, n_real, cate_stats

total, n_fake, n_real, cate_stats = collect_stats(raw)

# ── ① 통계 요약 차트 저장 ─────────────────────────────────────────
print("\n📊 통계 요약 차트 생성 중...")

fig_stat, axes = plt.subplots(1, 3, figsize=(20, 7))
fig_stat.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")

# (a) 전체 Fake/Real 파이
ax = axes[0]
ax.pie(
    [n_fake, n_real],
    labels=["FAKE", "REAL"],
    colors=["#E53935", "#43A047"],
    autopct="%1.1f%%",
    startangle=140,
    textprops={"color": "white", "fontsize": 13, "fontweight": "bold"},
    wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
)
ax.set_title(
    f"전체 Fake / Real 비율\n(총 {total:,}개)",
    color="white", fontsize=14, fontweight="bold", pad=15,
)

# (b) 카테고리별 샘플 수 누적 막대
ax = axes[1]
cates = [c for c in CATE_ORDER if c in cate_stats]
fakes = [cate_stats[c]["fake"] for c in cates]
reals = [cate_stats[c]["real"] for c in cates]
x = np.arange(len(cates))
b1 = ax.bar(x, fakes, color="#E53935", label="FAKE", width=0.55, edgecolor="#1a1a2e")
b2 = ax.bar(x, reals, bottom=fakes, color="#43A047", label="REAL", width=0.55, edgecolor="#1a1a2e")
ax.set_xticks(x)
ax.set_xticklabels(cates, rotation=30, ha="right", color="white", fontsize=11)
ax.set_ylabel("샘플 수", color="white", fontsize=12)
ax.set_title("카테고리별 Fake / Real 분포", color="white", fontsize=14, fontweight="bold")
ax.tick_params(colors="white")
ax.spines[:].set_color("#444")
ax.yaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=11)
# 수치 표시
for i, (f, r) in enumerate(zip(fakes, reals)):
    ax.text(i, f / 2, str(f), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(i, f + r / 2, str(r), ha="center", va="center", color="white", fontsize=9, fontweight="bold")

# (c) 카테고리별 Fake 비율
ax = axes[2]
fake_ratios = [cate_stats[c]["fake"] / cate_stats[c]["total"] * 100 for c in cates]
colors_bar = ["#E53935" if r > 50 else "#43A047" for r in fake_ratios]
bars = ax.barh(cates, fake_ratios, color=colors_bar, edgecolor="#1a1a2e", height=0.6)
ax.axvline(50, color="white", linestyle="--", linewidth=1.2, alpha=0.6)
ax.set_xlim(0, 100)
ax.set_xlabel("Fake 비율 (%)", color="white", fontsize=12)
ax.set_title("카테고리별 Fake 비율", color="white", fontsize=14, fontweight="bold")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
for bar, ratio in zip(bars, fake_ratios):
    ax.text(ratio + 1, bar.get_y() + bar.get_height() / 2,
            f"{ratio:.1f}%", va="center", color="white", fontsize=11, fontweight="bold")

fig_stat.suptitle(
    "FakeClue Test Set — 데이터셋 통계 요약",
    color="white", fontsize=18, fontweight="bold", y=1.02,
)
plt.tight_layout()
stat_path = OUT_DIR / "00_statistics_summary.png"
fig_stat.savefig(stat_path, dpi=150, bbox_inches="tight", facecolor=fig_stat.get_facecolor())
plt.close(fig_stat)
print(f"  저장: {stat_path}")


# ── ② 카테고리별 이미지 시각화 ───────────────────────────────────
def render_page(samples, cate: str, page_idx: int, total_pages: int):
    """samples 목록(최대 PER_PAGE)을 한 페이지 figure로 렌더링 후 반환"""
    n = len(samples)
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#1a1a2e")

    # 제목
    fig.suptitle(
        f"[{cate.upper()}]  Page {page_idx + 1} / {total_pages}  "
        f"(Fake {cate_stats[cate]['fake']:,} / Real {cate_stats[cate]['real']:,} / "
        f"Total {cate_stats[cate]['total']:,})",
        color="white", fontsize=15, fontweight="bold", y=0.995,
    )

    for idx, sample in enumerate(samples):
        row, col = divmod(idx, COLS)
        # subplot 위치: 각 셀을 세로 2분할(이미지 | 텍스트)
        # GridSpec(ROWS*2, COLS) 사용
        pass

    # GridSpec: 각 행 = 이미지(2) + 텍스트(7) 비율 — 텍스트 영역을 충분히 확보
    height_ratios = []
    for _ in range(ROWS):
        height_ratios += [2, 7]
    gs = GridSpec(
        ROWS * 2, COLS,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.04, wspace=0.08,
        top=0.97, bottom=0.01, left=0.01, right=0.99,
    )

    for idx, sample in enumerate(samples):
        row, col = divmod(idx, COLS)
        img_row  = row * 2
        txt_row  = row * 2 + 1

        label    = sample["label"]
        cate_tag = sample["cate"]
        reason   = get_reasoning(sample)
        img_path = IMG_ROOT / sample["image"]

        # ─ 이미지 subplot ─
        ax_img = fig.add_subplot(gs[img_row, col])
        try:
            img = Image.open(img_path).convert("RGB")
            ax_img.imshow(img)
        except Exception:
            ax_img.set_facecolor("#333")
            ax_img.text(0.5, 0.5, "이미지 없음", ha="center", va="center",
                        color="white", transform=ax_img.transAxes)

        # 라벨 배지 (이미지 위)
        lc = LABEL_COLOR[label]
        lt = LABEL_TEXT[label]
        ax_img.text(
            0.5, 0.97, lt,
            transform=ax_img.transAxes,
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(facecolor=lc, edgecolor="none",
                      boxstyle="round,pad=0.3", alpha=0.92),
        )
        ax_img.axis("off")

        # 이미지 테두리 색
        for spine in ax_img.spines.values():
            spine.set_edgecolor(lc)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # ─ 텍스트 subplot ─
        ax_txt = fig.add_subplot(gs[txt_row, col])
        ax_txt.set_facecolor("#0f3460")
        ax_txt.axis("off")

        wrapped = wrap_reason(reason)
        ax_txt.text(
            0.5, 0.98, wrapped,
            transform=ax_txt.transAxes,
            ha="center", va="top",
            fontsize=9.5, color="#e0e0e0",
            fontfamily="monospace",
            linespacing=1.45,
        )

    return fig


print("\n🖼  카테고리별 이미지 시각화 시작...")
for cate in CATE_ORDER:
    if cate not in by_cate:
        continue
    items    = by_cate[cate]
    n_pages  = (len(items) + PER_PAGE - 1) // PER_PAGE
    cate_dir = OUT_DIR / f"cat_{cate}"
    cate_dir.mkdir(exist_ok=True)

    print(f"  [{cate}] {len(items):,}개 → {n_pages}페이지")
    for pg in range(n_pages):
        chunk = items[pg * PER_PAGE : (pg + 1) * PER_PAGE]
        fig   = render_page(chunk, cate, pg, n_pages)
        out_path = cate_dir / f"{cate}_page{pg + 1:03d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        if (pg + 1) % 5 == 0 or (pg + 1) == n_pages:
            print(f"    → {pg + 1}/{n_pages} 페이지 저장 완료")


# ── ③ 텍스트 통계 리포트 저장 ────────────────────────────────────
report_lines = [
    "=" * 60,
    "FakeClue Test Set — 데이터 통계 리포트",
    "=" * 60,
    f"총 샘플 수  : {total:,}",
    f"  FAKE     : {n_fake:,}  ({n_fake/total*100:.1f}%)",
    f"  REAL     : {n_real:,}  ({n_real/total*100:.1f}%)",
    "",
    f"{'카테고리':<12} {'총계':>6} {'FAKE':>6} {'REAL':>6} {'Fake%':>7}",
    "-" * 42,
]
for c in CATE_ORDER:
    if c not in cate_stats:
        continue
    s = cate_stats[c]
    report_lines.append(
        f"{c:<12} {s['total']:>6,} {s['fake']:>6,} {s['real']:>6,} {s['fake']/s['total']*100:>6.1f}%"
    )
report_lines += [
    "=" * 60,
    f"저장 경로: {OUT_DIR}",
]

report_path = OUT_DIR / "statistics_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n" + "\n".join(report_lines))
print(f"\n✅ 완료! 결과물: {OUT_DIR}")

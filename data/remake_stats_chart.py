"""
FakeClue 통계 요약 차트 재생성 — 한글 폰트 적용
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

# ── 한글 폰트 등록 ────────────────────────────────────────────────
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지

# ── 경로 설정 ─────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "FakeClue" / "data_json" / "test.json"
OUT_PATH  = BASE_DIR / "FakeClue_visualized" / "00_statistics_summary.png"

CATE_ORDER = ["deepfake", "human", "animal", "object", "scene", "satellite", "doc"]

# ── 데이터 로드 & 집계 ────────────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    raw = json.load(f)

by_cate = defaultdict(list)
for d in raw:
    by_cate[d["cate"]].append(d)

total  = len(raw)
n_fake = sum(1 for d in raw if d["label"] == 0)
n_real = total - n_fake

cate_stats = {
    c: {
        "total": len(items),
        "fake":  sum(1 for i in items if i["label"] == 0),
        "real":  sum(1 for i in items if i["label"] == 1),
    }
    for c, items in by_cate.items()
}

# ── 차트 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")

cates = [c for c in CATE_ORDER if c in cate_stats]

# ① 전체 Fake/Real 파이차트
ax = axes[0]
ax.pie(
    [n_fake, n_real],
    labels=["FAKE", "REAL"],
    colors=["#E53935", "#43A047"],
    autopct="%1.1f%%",
    startangle=140,
    textprops={"color": "white", "fontsize": 14, "fontweight": "bold"},
    wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
)
ax.set_title(
    f"전체 Fake / Real 비율\n(총 {total:,}개)",
    color="white", fontsize=15, fontweight="bold", pad=18,
)

# ② 카테고리별 누적 막대
ax = axes[1]
fakes = [cate_stats[c]["fake"] for c in cates]
reals = [cate_stats[c]["real"] for c in cates]
x = np.arange(len(cates))
ax.bar(x, fakes, color="#E53935", label="FAKE", width=0.55, edgecolor="#1a1a2e")
ax.bar(x, reals, bottom=fakes, color="#43A047", label="REAL", width=0.55, edgecolor="#1a1a2e")
ax.set_xticks(x)
ax.set_xticklabels(cates, rotation=30, ha="right", color="white", fontsize=12)
ax.set_ylabel("샘플 수", color="white", fontsize=13)
ax.set_title("카테고리별 Fake / Real 분포", color="white", fontsize=15, fontweight="bold")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=12)
for i, (f, r) in enumerate(zip(fakes, reals)):
    ax.text(i, f / 2, str(f), ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    ax.text(i, f + r / 2, str(r), ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")

# ③ 카테고리별 Fake 비율 수평 막대
ax = axes[2]
fake_ratios = [cate_stats[c]["fake"] / cate_stats[c]["total"] * 100 for c in cates]
bar_colors  = ["#E53935" if r > 50 else "#43A047" for r in fake_ratios]
bars = ax.barh(cates, fake_ratios, color=bar_colors, edgecolor="#1a1a2e", height=0.6)
ax.axvline(50, color="white", linestyle="--", linewidth=1.4, alpha=0.7)
ax.set_xlim(0, 100)
ax.set_xlabel("Fake 비율 (%)", color="white", fontsize=13)
ax.set_title("카테고리별 Fake 비율", color="white", fontsize=15, fontweight="bold")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white", labelsize=12)
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
for bar, ratio in zip(bars, fake_ratios):
    ax.text(ratio + 1, bar.get_y() + bar.get_height() / 2,
            f"{ratio:.1f}%", va="center", color="white",
            fontsize=12, fontweight="bold")

fig.suptitle(
    "FakeClue Test Set — 데이터셋 통계 요약",
    color="white", fontsize=19, fontweight="bold", y=1.02,
)
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✅ 저장 완료: {OUT_PATH}")

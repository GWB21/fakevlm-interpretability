"""
AC Summary 시각화: metrics_by_set_gt.csv 기반
- type=cate_label 만 사용, n 가중 평균
- Overall (N=5000) + Real vs Fake 비대칭 비교
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

# 한글 폰트
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
try:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "results_unified" / "metrics_by_set_gt" / "metrics_by_set_gt.csv"
OUT_DIR = BASE / "results_unified" / "metrics_by_set_gt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["Original", "LPF", "HPF+DC", "Masked"]
CONDITION_KEYS = ["Original", "LPF", "HPF+DC", "Masked"]  # CSV 컬럼 접미사와 동일
METRICS = ["ACC", "CSS", "ROUGE_L"]


def load_rows():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["n"] = int(r["n"])
            rows.append(r)
    return rows


def weighted_avg(rows, metric_prefix):
    """metric_prefix e.g. 'ACC', 'CSS', 'ROUGE_L'. 조건별 가중 평균."""
    total_n = sum(r["n"] for r in rows)
    if total_n == 0:
        return {c: 0.0 for c in CONDITION_KEYS}, total_n
    out = {}
    for c in CONDITION_KEYS:
        key = f"{metric_prefix}_{c}"
        val = sum(r["n"] * float(r[key]) for r in rows) / total_n
        out[c] = val
    return out, total_n


def main():
    all_rows = load_rows()
    cate_label = [r for r in all_rows if r["type"] == "cate_label"]
    real_rows = [r for r in cate_label if r["set"].endswith("_real")]
    fake_rows = [r for r in cate_label if r["set"].endswith("_fake")]

    # Overall
    overall_acc, n_overall = weighted_avg(cate_label, "ACC")
    overall_css, _ = weighted_avg(cate_label, "CSS")
    overall_rl, _ = weighted_avg(cate_label, "ROUGE_L")
    n_real = sum(r["n"] for r in real_rows)
    n_fake = sum(r["n"] for r in fake_rows)

    # Real / Fake
    real_acc, _ = weighted_avg(real_rows, "ACC")
    real_css, _ = weighted_avg(real_rows, "CSS")
    fake_acc, _ = weighted_avg(fake_rows, "ACC")
    fake_css, _ = weighted_avg(fake_rows, "CSS")

    # ---- 1) Overall (N=5000): ACC, CSS, ROUGE_L by condition ----
    fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(CONDITIONS))
    width = 0.6
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

    for ax, (metric, overall_vals) in zip(axes, [
        ("Accuracy (ACC)", overall_acc),
        ("CSS (vs GT)", overall_css),
        ("ROUGE_L (vs GT)", overall_rl),
    ]):
        vals = [overall_vals[c] for c in CONDITION_KEYS]
        bars = ax.bar(x, vals, width=width, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_KEYS, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig1.suptitle(f"Overall (type=cate_label, N={n_overall:,}) — Weighted Average", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1.savefig(OUT_DIR / "ac_summary_overall.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"💾 {OUT_DIR / 'ac_summary_overall.png'}")

    # ---- 2) Real vs Fake: ACC & CSS 비교 (비대칭 강조) ----
    fig2, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(CONDITIONS))
    width = 0.35

    # ACC
    ax = axes[0]
    vals_fake = [fake_acc[c] for c in CONDITION_KEYS]
    vals_real = [real_acc[c] for c in CONDITION_KEYS]
    ax.bar(x - width / 2, vals_fake, width=width, label=f"Fake (n={n_fake:,})", color="#3498db", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, vals_real, width=width, label=f"Real (n={n_real:,})", color="#e74c3c", edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Accuracy (ACC)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_KEYS, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Real vs Fake — Asymmetry (ACC)", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5)

    # CSS
    ax = axes[1]
    vals_fake = [fake_css[c] for c in CONDITION_KEYS]
    vals_real = [real_css[c] for c in CONDITION_KEYS]
    ax.bar(x - width / 2, vals_fake, width=width, label=f"Fake (n={n_fake:,})", color="#3498db", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, vals_real, width=width, label=f"Real (n={n_real:,})", color="#e74c3c", edgecolor="black", linewidth=0.6)
    ax.set_ylabel("CSS (vs GT)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_KEYS, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Real vs Fake — Asymmetry (CSS)", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5)

    fig2.suptitle("Label-stratified weighted average (cate_label only)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "ac_summary_real_vs_fake.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"💾 {OUT_DIR / 'ac_summary_real_vs_fake.png'}")

    # ---- 3) 한 장에 Overall + Real/Fake ACC 요약 (선택) ----
    fig3, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONDITIONS))
    width = 0.25
    ax.bar(x - width, [overall_acc[c] for c in CONDITION_KEYS], width=width, label=f"Overall (N={n_overall:,})", color="#2ecc71", edgecolor="black", linewidth=0.6)
    ax.bar(x, [fake_acc[c] for c in CONDITION_KEYS], width=width, label=f"Fake (n={n_fake:,})", color="#3498db", edgecolor="black", linewidth=0.6)
    ax.bar(x + width, [real_acc[c] for c in CONDITION_KEYS], width=width, label=f"Real (n={n_real:,})", color="#e74c3c", edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Accuracy (ACC)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_KEYS, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("ACC: Overall vs Fake vs Real (weighted avg, cate_label)", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "ac_summary_acc_overall_fake_real.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"💾 {OUT_DIR / 'ac_summary_acc_overall_fake_real.png'}")

    # ---- 텍스트 요약 저장 (report와 동일 형식) ----
    report_path = OUT_DIR / "ac_summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AC Summary — type=cate_label 가중 평균 (중복 집계 배제)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"1. Overall (N={n_overall:,})\n")
        f.write("-" * 50 + "\n")
        f.write("  ACC:     " + "  ".join(f"{c}={overall_acc[c]:.4f}" for c in CONDITION_KEYS) + "\n")
        f.write("  CSS:     " + "  ".join(f"{c}={overall_css[c]:.4f}" for c in CONDITION_KEYS) + "\n")
        f.write("  ROUGE_L: " + "  ".join(f"{c}={overall_rl[c]:.4f}" for c in CONDITION_KEYS) + "\n\n")
        f.write(f"2. Fake (n={n_fake:,})\n")
        f.write("-" * 50 + "\n")
        f.write("  ACC: " + "  ".join(f"{c}={fake_acc[c]:.4f}" for c in CONDITION_KEYS) + "\n")
        f.write("  CSS: " + "  ".join(f"{c}={fake_css[c]:.4f}" for c in CONDITION_KEYS) + "\n\n")
        f.write(f"3. Real (n={n_real:,})\n")
        f.write("-" * 50 + "\n")
        f.write("  ACC: " + "  ".join(f"{c}={real_acc[c]:.4f}" for c in CONDITION_KEYS) + "\n")
        f.write("  CSS: " + "  ".join(f"{c}={real_css[c]:.4f}" for c in CONDITION_KEYS) + "\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"💾 {report_path}")
    print("✅ AC Summary 시각화 완료.")


if __name__ == "__main__":
    main()

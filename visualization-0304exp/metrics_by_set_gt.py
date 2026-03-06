"""
21개 Set × 4 조건별 ACC / CSS(vs GT) / ROUGE_L(vs GT) 집계 및 시각화.

- Set: 14 (카테고리–라벨) + 7 (카테고리별) = 21개
- 조건: Original, LPF, HPF+DC, Masked
- GT: data/FakeClue/data_json/test.json 의 gpt 응답
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 한글 폰트
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
try:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
# results_full.json 및 test.json 경로 (스크립트 기준: visualization-0304exp)
# FakeClue test.json 은 fakevlm-interpretability/data/FakeClue/data_json/ 에 있음
FAKEVLM_ROOT = BASE.parent  # fakevlm-interpretability
RESULTS_JSON = BASE / "results_unified" / "jsonresults" / "results_full.json"
GT_JSON = FAKEVLM_ROOT / "data" / "FakeClue" / "data_json" / "test.json"
OUT_DIR = BASE / "results_unified" / "metrics_by_set_gt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["Original", "LPF", "HPF+DC", "Masked"]
TEXT_KEYS = ["text_original", "text_lpf", "text_hpf_dc", "text_masked"]
PRED_KEYS = ["pred_original", "pred_lpf", "pred_hpf_dc", "pred_masked"]
# 영어 GT/출력에 적합한 널리 쓰이는 모델 (다운로드 실패 시 all-MiniLM-L6-v2 로 변경 가능)
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


def strip_prompt(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if "Does the image looks real/fake?" in text:
        text = text.split("Does the image looks real/fake?")[-1].strip()
    if text.startswith("\n"):
        text = text.lstrip("\n").strip()
    return text[:8000]  # cap length for encoder


def get_gt_captions(test_path: Path) -> list[str]:
    with open(test_path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data:
        for c in item["conversations"]:
            if c["from"] == "gpt":
                out.append(strip_prompt(c["value"]))
                break
        else:
            out.append("")
    return out


def compute_css_vs_gt(
    model: SentenceTransformer,
    model_texts: dict[str, list[str]],
    gt_texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> dict[str, list[float]]:
    """model_texts: {condition: [text per sample]}. Returns {condition: [cos_sim per sample]}."""
    n = len(gt_texts)
    gt_embs = model.encode(gt_texts, show_progress_bar=True, batch_size=batch_size)
    gt_embs = np.array(gt_embs)
    if gt_embs.ndim == 1:
        gt_embs = gt_embs.reshape(1, -1)
    results = {}
    for cond, texts in model_texts.items():
        if len(texts) != n:
            continue
        embs = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
        embs = np.array(embs)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        nrm_gt = np.linalg.norm(gt_embs, axis=1, keepdims=True)
        nrm_emb = np.linalg.norm(embs, axis=1, keepdims=True)
        sims = (gt_embs * embs).sum(axis=1) / (nrm_gt.ravel() * nrm_emb.ravel() + 1e-9)
        results[cond] = sims.tolist()
    return results


def compute_rouge_l_vs_gt(
    scorer,
    model_texts: dict[str, list[str]],
    gt_texts: list[str],
) -> dict[str, list[float]]:
    """reference=GT, hypothesis=model. Returns {condition: [rouge_l f per sample]}."""
    n = len(gt_texts)
    results = {}
    for cond, texts in model_texts.items():
        if len(texts) != n:
            continue
        scores = []
        for gt, hyp in tqdm(zip(gt_texts, texts), total=n, desc=f"ROUGE-L {cond}", leave=False):
            gt = gt or " "
            hyp = hyp or " "
            s = scorer.score(gt, hyp)["rougeL"].fmeasure
            scores.append(s)
        results[cond] = scores
    return results


def main():
    # HuggingFace 캐시를 워크스페이스 내로 (쓰기 권한 확보)
    cache_dir = BASE / ".cache_sbert"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import os
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir))

    print("📂 GT 캡션 로드:", GT_JSON)
    gt_list = get_gt_captions(GT_JSON)
    print(f"  {len(gt_list):,}개")

    print("📂 results_full.json 로드:", RESULTS_JSON)
    with open(RESULTS_JSON, encoding="utf-8") as f:
        results = json.load(f)
    print(f"  {len(results):,}개")

    n = len(results)
    assert len(gt_list) >= n, "GT 개수 부족"
    gt_list = gt_list[:n]

    # 모델 출력만 추출 (프롬프트 제거)
    model_texts = {}
    for cond, key in zip(CONDITIONS, TEXT_KEYS):
        model_texts[cond] = [strip_prompt(r.get(key, "")) for r in results]

    # CSS (vs GT)
    print("🔢 CSS (model vs GT) 계산 중...")
    sbert = SentenceTransformer(SBERT_MODEL, cache_folder=str(cache_dir))
    css_by_cond = compute_css_vs_gt(sbert, model_texts, gt_list)
    del sbert

    # ROUGE-L (vs GT)
    print("🔢 ROUGE-L (model vs GT) 계산 중...")
    rouge_scorer_inst = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_by_cond = compute_rouge_l_vs_gt(rouge_scorer_inst, model_texts, gt_list)

    # Per-sample ACC (이미 있음)
    acc_by_cond = {}
    for cond, key in zip(CONDITIONS, PRED_KEYS):
        acc_by_cond[cond] = [
            1.0 if r.get(key) == r["label"] else 0.0
            for r in results
        ]

    # 21 sets 정의
    cates = ["deepfake", "human", "animal", "object", "scene", "satellite", "doc"]
    set_specs = []  # (set_name, set_type, mask_fn)
    for c in cates:
        for label in (0, 1):
            label_str = "fake" if label == 0 else "real"
            set_specs.append((f"{c}_{label_str}", "cate_label", lambda r, c=c, label=label: r["cate"] == c and r["label"] == label))
    for c in cates:
        set_specs.append((f"{c}_all", "cate", lambda r, c=c: r["cate"] == c))

    # 집계: 각 set × condition -> ACC mean, CSS mean, ROUGE_L mean, n
    rows = []
    for set_name, set_type, mask_fn in set_specs:
        indices = [i for i in range(n) if mask_fn(results[i])]
        if not indices:
            rows.append({
                "set": set_name, "type": set_type,
                **{f"ACC_{c}": np.nan for c in CONDITIONS},
                **{f"CSS_{c}": np.nan for c in CONDITIONS},
                **{f"ROUGE_L_{c}": np.nan for c in CONDITIONS},
                "n": 0,
            })
            continue
        row = {"set": set_name, "type": set_type, "n": len(indices)}
        for cond in CONDITIONS:
            acc_vals = [acc_by_cond[cond][i] for i in indices]
            css_vals = [css_by_cond[cond][i] for i in indices]
            rouge_vals = [rouge_by_cond[cond][i] for i in indices]
            row[f"ACC_{cond}"] = np.mean(acc_vals)
            row[f"CSS_{cond}"] = np.mean(css_vals)
            row[f"ROUGE_L_{cond}"] = np.mean(rouge_vals)
        rows.append(row)

    # CSV 저장
    csv_path = OUT_DIR / "metrics_by_set_gt.csv"
    col_names = ["set", "type", "n"] + [f"ACC_{c}" for c in CONDITIONS] + [f"CSS_{c}" for c in CONDITIONS] + [f"ROUGE_L_{c}" for c in CONDITIONS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(col_names)
        for row in rows:
            w.writerow([row.get(k) for k in col_names])
    print(f"💾 CSV 저장: {csv_path}")

    # 텍스트 보고서
    report_lines = [
        "=" * 80,
        "21 Set × 4 Condition  ACC / CSS(vs GT) / ROUGE_L(vs GT)",
        "=" * 80,
    ]
    for row in rows:
        report_lines.append(f"\n[{row['set']}] type={row['type']} n={int(row['n'])}")
        for cond in CONDITIONS:
            a, c, rl = row.get(f"ACC_{cond}", 0), row.get(f"CSS_{cond}", 0), row.get(f"ROUGE_L_{cond}", 0)
            report_lines.append(f"  {cond:12}  ACC={a:.4f}  CSS={c:.4f}  ROUGE_L={rl:.4f}")
    report_path = OUT_DIR / "report_metrics_by_set_gt.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"💾 보고서 저장: {report_path}")

    # ---------- 시각화 ----------
    fig_bg = "#1a1a2e"
    panel_bg = "#16213e"
    text_color = "white"

    # 1) 요약 테이블 이미지 (21 set × 4 condition)
    cols = [f"ACC_{c}" for c in CONDITIONS] + [f"CSS_{c}" for c in CONDITIONS] + [f"ROUGE_L_{c}" for c in CONDITIONS]
    tab_data = []
    for row in rows:
        tab_data.append([row["set"], str(int(row["n"]))] + [f"{row.get(k, 0):.4f}" for k in cols])
    col_labels = ["set", "n"] + cols
    fig1, ax1 = plt.subplots(figsize=(20, 14), facecolor=fig_bg)
    ax1.set_facecolor(panel_bg)
    ax1.axis("off")
    table = ax1.table(
        cellText=tab_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=[panel_bg] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for (i, j), cell in table.get_celld().items():
        cell.set_text_props(color=text_color)
        cell.set_facecolor(panel_bg)
    ax1.set_title("21 Set × 4 Condition  ACC / CSS(vs GT) / ROUGE_L(vs GT)", color=text_color, fontsize=14, pad=20)
    plt.tight_layout()
    fig1.savefig(OUT_DIR / "table_summary.png", dpi=150, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig1)
    print("💾 시각화: table_summary.png")

    # 2) 히트맵 3개: Set × Condition for ACC, CSS, ROUGE_L
    set_order = [r["set"] for r in rows]
    cond_order = CONDITIONS
    for metric, title in [("ACC", "ACC"), ("CSS", "CSS (vs GT)"), ("ROUGE_L", "ROUGE-L (vs GT)")]:
        mat = np.array([[r[f"{metric}_{c}"] for c in cond_order] for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=fig_bg)
        ax.set_facecolor(panel_bg)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn" if metric == "ACC" else "RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(cond_order, color=text_color, rotation=25, ha="right")
        ax.set_yticks(range(len(set_order)))
        ax.set_yticklabels(set_order, color=text_color, fontsize=9)
        ax.set_title(title, color=text_color, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(colors=text_color)
        for i in range(len(set_order)):
            for j in range(len(cond_order)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black" if 0.4 < v < 0.7 else "white", fontsize=8, fontweight="bold")
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"heatmap_{metric.lower()}.png", dpi=150, bbox_inches="tight", facecolor=fig_bg)
        plt.close(fig)
    print("💾 시각화: heatmap_acc.png, heatmap_css.png, heatmap_rouge_l.png")

    # 3) 히스토그램: Set별 ACC (4 conditions 막대)
    x = np.arange(len(set_order))
    width = 0.2
    fig3, ax3 = plt.subplots(figsize=(18, 8), facecolor=fig_bg)
    ax3.set_facecolor(panel_bg)
    for i, cond in enumerate(CONDITIONS):
        offset = (i - 1.5) * width
        vals = np.array([r[f"ACC_{cond}"] for r in rows], dtype=float)
        ax3.bar(x + offset, vals, width, label=cond, alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(set_order, rotation=45, ha="right", color=text_color, fontsize=9)
    ax3.set_ylabel("ACC", color=text_color, fontsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.legend(facecolor=panel_bg, labelcolor=text_color, fontsize=10)
    ax3.tick_params(colors=text_color)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")
    ax3.set_title("ACC by Set (4 conditions)", color=text_color, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "histogram_acc_by_set.png", dpi=150, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig3)
    print("💾 시각화: histogram_acc_by_set.png")

    # 4) 히스토그램: Set별 CSS (4 conditions)
    fig4, ax4 = plt.subplots(figsize=(18, 8), facecolor=fig_bg)
    ax4.set_facecolor(panel_bg)
    for i, cond in enumerate(CONDITIONS):
        offset = (i - 1.5) * width
        vals = np.array([r[f"CSS_{cond}"] for r in rows], dtype=float)
        ax4.bar(x + offset, vals, width, label=cond, alpha=0.9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(set_order, rotation=45, ha="right", color=text_color, fontsize=9)
    ax4.set_ylabel("CSS (vs GT)", color=text_color, fontsize=11)
    ax4.set_ylim(0, 1.05)
    ax4.legend(facecolor=panel_bg, labelcolor=text_color, fontsize=10)
    ax4.tick_params(colors=text_color)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#444")
    ax4.set_title("CSS (vs GT) by Set (4 conditions)", color=text_color, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig4.savefig(OUT_DIR / "histogram_css_by_set.png", dpi=150, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig4)
    print("💾 시각화: histogram_css_by_set.png")

    # 5) 히스토그램: Set별 ROUGE_L (4 conditions)
    fig5, ax5 = plt.subplots(figsize=(18, 8), facecolor=fig_bg)
    ax5.set_facecolor(panel_bg)
    for i, cond in enumerate(CONDITIONS):
        offset = (i - 1.5) * width
        vals = np.array([r[f"ROUGE_L_{cond}"] for r in rows], dtype=float)
        ax5.bar(x + offset, vals, width, label=cond, alpha=0.9)
    ax5.set_xticks(x)
    ax5.set_xticklabels(set_order, rotation=45, ha="right", color=text_color, fontsize=9)
    ax5.set_ylabel("ROUGE-L (vs GT)", color=text_color, fontsize=11)
    ax5.set_ylim(0, 1.05)
    ax5.legend(facecolor=panel_bg, labelcolor=text_color, fontsize=10)
    ax5.tick_params(colors=text_color)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#444")
    ax5.set_title("ROUGE-L (vs GT) by Set (4 conditions)", color=text_color, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig5.savefig(OUT_DIR / "histogram_rouge_l_by_set.png", dpi=150, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig5)
    print("💾 시각화: histogram_rouge_l_by_set.png")

    print(f"\n✅ 완료. 결과물: {OUT_DIR}")


if __name__ == "__main__":
    main()

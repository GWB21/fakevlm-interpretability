"""
FakeVLM 실험 결과 통합 분석 스크립트.

입력:
  - results_fft_original.json  (주파수 실험 - 원본)
  - results_fft_lpf.json       (주파수 실험 - 저주파 통과)
  - results_fft_hpf.json       (주파수 실험 - 고주파 통과)
  - results_hook.json          (어텐션 마스킹 실험)

출력:
  - performance_degradation.png  (조건별 Accuracy / F1 히스토그램 차트)
  - analysis_report.txt          (정량 분석 텍스트 보고서)
"""

import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
RESULTS_DIR  = "/workspace/fakevlm_analysis"
CHART_PATH   = os.path.join(RESULTS_DIR, "performance_degradation.png")
REPORT_PATH  = os.path.join(RESULTS_DIR, "analysis_report.txt")

# hpf    : DC 제거 순수 HPF (OOD 이미지 — "시각 완전 박탈" 대조군)
# hpf_dc : HPF + DC Offset 복원 (공정한 고주파 탐지 실험, Gem AC 지적 반영)
FFT_CONDITIONS = ["original", "lpf", "hpf", "hpf_dc"]

# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------

def load_json(path: str) -> list | None:
    """JSON 파일을 로드한다. 파일이 없을 경우 None 반환."""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}", file=sys.stderr)
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_metrics(results: list) -> dict:
    """
    결과 리스트에서 Accuracy / F1 / Confusion Matrix를 계산한다.
    prediction == -1 (미결정) 항목은 제외하여 산출한다.

    label:      0=fake, 1=real
    prediction: 0=fake, 1=real, -1=undetermined
    """
    valid = [r for r in results if r.get("prediction", -1) != -1]
    if not valid:
        return {
            "accuracy": None, "f1": None,
            "n_total": len(results), "n_valid": 0, "n_undetermined": len(results),
            "confusion": None,
        }

    y_true = [r["label"] for r in valid]
    y_pred = [r["prediction"] for r in valid]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": round(acc, 6),
        "f1": round(f1, 6),
        "n_total": len(results),
        "n_valid": len(valid),
        "n_undetermined": len(results) - len(valid),
        "confusion": cm,  # [[TN, FP], [FN, TP]]
    }


def compute_hook_metrics(results: list) -> dict:
    """
    Hook 실험 결과에서 마스킹 전/후 지표 및 Language Bias 통계를 계산한다.
    """
    valid_original = [
        r for r in results if r.get("pred_original", -1) != -1
    ]
    valid_both = [
        r for r in results
        if r.get("pred_original", -1) != -1 and r.get("pred_masked", -1) != -1
    ]

    # 마스킹 전 지표
    if valid_original:
        y_true_o = [r["label"] for r in valid_original]
        y_pred_o = [r["pred_original"] for r in valid_original]
        acc_o = accuracy_score(y_true_o, y_pred_o)
        f1_o  = f1_score(y_true_o, y_pred_o, zero_division=0)
        cm_o  = confusion_matrix(y_true_o, y_pred_o, labels=[0, 1]).tolist()
    else:
        acc_o = f1_o = None
        cm_o = None

    # 마스킹 후 지표
    if valid_both:
        y_true_m = [r["label"] for r in valid_both]
        y_pred_m = [r["pred_masked"] for r in valid_both]
        acc_m = accuracy_score(y_true_m, y_pred_m)
        f1_m  = f1_score(y_true_m, y_pred_m, zero_division=0)
        cm_m  = confusion_matrix(y_true_m, y_pred_m, labels=[0, 1]).tolist()
    else:
        acc_m = f1_m = None
        cm_m = None

    # Language Bias 통계 (valid_both 기준)
    bias_cases    = [r for r in valid_both if r.get("language_bias", False)]
    correct_bias  = [r for r in bias_cases if r["pred_original"] == r["label"]]
    wrong_bias    = [r for r in bias_cases if r["pred_original"] != r["label"]]

    bias_rate = len(bias_cases) / len(valid_both) if valid_both else None

    return {
        "original": {
            "accuracy": round(acc_o, 6) if acc_o is not None else None,
            "f1": round(f1_o, 6) if f1_o is not None else None,
            "n_valid": len(valid_original),
            "confusion": cm_o,
        },
        "masked": {
            "accuracy": round(acc_m, 6) if acc_m is not None else None,
            "f1": round(f1_m, 6) if f1_m is not None else None,
            "n_valid": len(valid_both),
            "confusion": cm_m,
        },
        "language_bias": {
            "n_valid_both": len(valid_both),
            "n_bias": len(bias_cases),
            "bias_rate": round(bias_rate, 6) if bias_rate is not None else None,
            "n_correct_bias": len(correct_bias),
            "n_wrong_bias": len(wrong_bias),
        },
    }


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_performance(
    fft_metrics: dict,
    hook_metrics: dict,
    output_path: str,
) -> None:
    """
    FFT 조건별 및 Hook 마스킹 전/후 Accuracy / F1-Score 비교 차트를 생성한다.

    layout: 두 개의 서브플롯
      (좌) FFT 조건별 성능 (Original / LPF / HPF)
      (우) Hook 마스킹 전후 성능 (Before / After Masking)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 서브플롯 1: FFT 조건별 ---
    ax = axes[0]
    cond_label_map = {
        "original": "Original",
        "lpf":      "LPF\n(R=30)",
        "hpf":      "HPF-OOD\n(R=30)",
        "hpf_dc":   "HPF+DC\n(R=30)",
    }
    labels_fft = [cond_label_map.get(c, c) for c in FFT_CONDITIONS]
    acc_fft = [
        fft_metrics[c]["accuracy"] if fft_metrics[c]["accuracy"] is not None else 0.0
        for c in FFT_CONDITIONS
    ]
    f1_fft = [
        fft_metrics[c]["f1"] if fft_metrics[c]["f1"] is not None else 0.0
        for c in FFT_CONDITIONS
    ]

    x = np.arange(len(labels_fft))
    width = 0.35
    bars_acc = ax.bar(x - width / 2, acc_fft, width, label="Accuracy", color="#4C72B0")
    bars_f1  = ax.bar(x + width / 2, f1_fft,  width, label="F1-Score",  color="#DD8452")

    ax.set_title("FFT Frequency Knockout: Performance Degradation", fontsize=11)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_fft)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 값 레이블
    for bar in bars_acc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_f1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # --- 서브플롯 2: Hook 마스킹 전후 ---
    ax2 = axes[1]
    labels_hook = ["Before Masking", "After Masking"]
    acc_hook = [
        hook_metrics["original"]["accuracy"] or 0.0,
        hook_metrics["masked"]["accuracy"] or 0.0,
    ]
    f1_hook = [
        hook_metrics["original"]["f1"] or 0.0,
        hook_metrics["masked"]["f1"] or 0.0,
    ]

    x2 = np.arange(len(labels_hook))
    bars_acc2 = ax2.bar(x2 - width / 2, acc_hook, width, label="Accuracy", color="#4C72B0")
    bars_f12  = ax2.bar(x2 + width / 2, f1_hook,  width, label="F1-Score",  color="#DD8452")

    bias_rate = hook_metrics["language_bias"]["bias_rate"]
    bias_str  = f"Language Bias Rate: {bias_rate:.4f}" if bias_rate is not None else "Language Bias Rate: N/A"
    ax2.set_title(f"Attention Masking: Before vs After\n({bias_str})", fontsize=11)
    ax2.set_ylabel("Score")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_hook)
    ax2.set_ylim(0, 1.15)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_acc2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_f12:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle(
        "FakeVLM Mechanistic Interpretability Analysis",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Chart: {output_path}")


# ---------------------------------------------------------------------------
# 텍스트 보고서 생성
# ---------------------------------------------------------------------------

def generate_report(
    fft_metrics: dict,
    hook_metrics: dict,
) -> str:
    """
    실험 결과 정량 분석 텍스트 보고서를 생성하여 문자열로 반환한다.
    """
    lines = []

    def sep(char="=", width=70):
        lines.append(char * width)

    def head(title):
        sep()
        lines.append(f"  {title}")
        sep()

    head("FakeVLM Mechanistic Interpretability Analysis Report")
    lines.append("")

    # -----------------------------------------------------------------------
    # 섹션 1: FFT 주파수 넉아웃
    # -----------------------------------------------------------------------
    lines.append("SECTION 1: FFT Frequency Knockout Experiment")
    sep("-")
    lines.append(
        "Hypothesis: If the model relies on frequency artifacts (high-frequency\n"
        "components), HPF should preserve or improve accuracy while LPF degrades it.\n"
        "Conversely, reliance on semantic content predicts HPF degradation."
    )
    lines.append("")

    condition_labels = {
        "original": "Original",
        "lpf":      "LPF (R=30)",
        "hpf":      "HPF-OOD (R=30)",   # DC 제거 → 시각 박탈 대조군
        "hpf_dc":   "HPF+DC (R=30)",    # DC 복원 → 공정 고주파 탐지 실험
    }
    for cond in FFT_CONDITIONS:
        m = fft_metrics[cond]
        label = condition_labels[cond]
        lines.append(f"  [{label}]")
        if m["accuracy"] is not None:
            lines.append(f"    Accuracy      : {m['accuracy']:.6f}")
            lines.append(f"    F1-Score      : {m['f1']:.6f}")
            lines.append(f"    Valid samples : {m['n_valid']} / {m['n_total']}")
            lines.append(f"    Undetermined  : {m['n_undetermined']}")
            if m["confusion"]:
                cm = m["confusion"]
                lines.append(
                    f"    Confusion Matrix (label: 0=fake, 1=real):"
                )
                lines.append(f"      TN={cm[0][0]}  FP={cm[0][1]}")
                lines.append(f"      FN={cm[1][0]}  TP={cm[1][1]}")
        else:
            lines.append("    Result file not available.")
        lines.append("")

    # 조건별 성능 변화 해석
    orig_acc = fft_metrics["original"]["accuracy"]
    lpf_acc  = fft_metrics["lpf"]["accuracy"]
    hpf_acc  = fft_metrics["hpf"]["accuracy"]

    hpf_dc_acc = fft_metrics["hpf_dc"]["accuracy"]

    if orig_acc is not None and lpf_acc is not None:
        lines.append("  Performance Delta (vs. Original):")
        lines.append(f"    LPF    - Original: {lpf_acc - orig_acc:+.6f}")
        if hpf_dc_acc is not None:
            lines.append(f"    HPF+DC - Original: {hpf_dc_acc - orig_acc:+.6f}  [fair high-freq probe]")
        if fft_metrics["hpf"]["accuracy"] is not None:
            lines.append(
                f"    HPF-OOD- Original: {fft_metrics['hpf']['accuracy'] - orig_acc:+.6f}"
                "  [visual deprivation control, NOT a fair freq. test]"
            )
        lines.append("")

        lines.append("  [NOTE] HPF-OOD removes DC component, producing near-black OOD images.")
        lines.append("  CLIP activation collapse expected. HPF-OOD results should be interpreted")
        lines.append("  ONLY as 'language bias under complete visual deprivation', NOT as evidence")
        lines.append("  for/against high-frequency artifact dependency.")
        lines.append("")

        # Primary interpretation based on LPF and HPF+DC
        if orig_acc is not None and lpf_acc is not None and hpf_dc_acc is not None:
            if abs(lpf_acc - orig_acc) < 0.02 and abs(hpf_dc_acc - orig_acc) < 0.02:
                interp = (
                    "Both LPF and HPF+DC show minimal performance change (<2%).\n"
                    "Strong evidence that FakeVLM relies on low-frequency semantic\n"
                    "features rather than high-frequency artifacts. Supports Hypothesis 1."
                )
            elif hpf_dc_acc > lpf_acc and hpf_dc_acc > orig_acc - 0.02:
                interp = (
                    "HPF+DC performance is comparable to original while LPF degrades.\n"
                    "Suggests the model leverages high-frequency patterns for detection."
                )
            elif lpf_acc > orig_acc - 0.02 and hpf_dc_acc < orig_acc - 0.05:
                interp = (
                    "LPF maintains performance while HPF+DC degrades (>5%).\n"
                    "The model depends on low-frequency semantic content; high-frequency\n"
                    "texture removal alone disrupts detection."
                )
            else:
                interp = (
                    "Moderate or mixed frequency sensitivity. Both LPF and HPF+DC\n"
                    "show non-trivial changes. Further analysis with multiple radii recommended."
                )
        elif orig_acc is not None and lpf_acc is not None:
            if abs(lpf_acc - orig_acc) < 0.02:
                interp = (
                    "LPF shows minimal degradation (<2%). FakeVLM appears robust to\n"
                    "high-frequency removal, suggesting semantic (low-freq) dependence.\n"
                    "Run HPF+DC condition for complete evidence."
                )
            else:
                interp = (
                    f"LPF degrades performance by {abs(lpf_acc - orig_acc):.3f}.\n"
                    "Run HPF+DC condition to determine frequency dependency direction."
                )
        else:
            interp = "Insufficient data for interpretation."

        lines.append("  Interpretation (LPF + HPF+DC primary analysis):")
        for ln in interp.split("\n"):
            lines.append(f"    {ln}")
    lines.append("")

    # -----------------------------------------------------------------------
    # 섹션 2: Attention 마스킹 (Language Bias)
    # -----------------------------------------------------------------------
    lines.append("SECTION 2: Attention Masking Experiment (Language Bias Analysis)")
    sep("-")
    lines.append(
        "Hypothesis: If the model exhibits language bias, masking the top-20%\n"
        "visually attended patches will not change the prediction output,\n"
        "indicating the model's response is driven by language priors rather\n"
        "than visual evidence."
    )
    lines.append("")

    m_orig   = hook_metrics["original"]
    m_masked = hook_metrics["masked"]
    m_bias   = hook_metrics["language_bias"]

    lines.append("  [Before Masking (1st inference)]")
    if m_orig["accuracy"] is not None:
        lines.append(f"    Accuracy      : {m_orig['accuracy']:.6f}")
        lines.append(f"    F1-Score      : {m_orig['f1']:.6f}")
        lines.append(f"    Valid samples : {m_orig['n_valid']}")
        if m_orig["confusion"]:
            cm = m_orig["confusion"]
            lines.append("    Confusion Matrix:")
            lines.append(f"      TN={cm[0][0]}  FP={cm[0][1]}")
            lines.append(f"      FN={cm[1][0]}  TP={cm[1][1]}")
    else:
        lines.append("    Result file not available.")
    lines.append("")

    lines.append("  [After Masking (2nd inference, top-20% attention patches zeroed)]")
    if m_masked["accuracy"] is not None:
        lines.append(f"    Accuracy      : {m_masked['accuracy']:.6f}")
        lines.append(f"    F1-Score      : {m_masked['f1']:.6f}")
        lines.append(f"    Valid samples : {m_masked['n_valid']}")
        if m_masked["confusion"]:
            cm = m_masked["confusion"]
            lines.append("    Confusion Matrix:")
            lines.append(f"      TN={cm[0][0]}  FP={cm[0][1]}")
            lines.append(f"      FN={cm[1][0]}  TP={cm[1][1]}")
        delta_acc = (m_masked["accuracy"] or 0.0) - (m_orig["accuracy"] or 0.0)
        delta_f1  = (m_masked["f1"] or 0.0) - (m_orig["f1"] or 0.0)
        lines.append(f"    Accuracy Delta: {delta_acc:+.6f}")
        lines.append(f"    F1 Delta      : {delta_f1:+.6f}")
    else:
        lines.append("    Result file not available.")
    lines.append("")

    lines.append("  [Language Bias Statistics]")
    if m_bias["bias_rate"] is not None:
        n_valid  = m_bias["n_valid_both"]
        n_bias   = m_bias["n_bias"]
        rate     = m_bias["bias_rate"]
        n_corr   = m_bias["n_correct_bias"]
        n_wrong  = m_bias["n_wrong_bias"]

        lines.append(f"    Samples with both predictions parsed : {n_valid}")
        lines.append(
            f"    Language Bias cases (pred unchanged)  : {n_bias} / {n_valid}"
        )
        lines.append(f"    Language Bias Rate                  : {rate:.6f}")
        lines.append(
            f"    Breakdown:"
        )
        lines.append(
            f"      Correct prediction + bias preserved : {n_corr}"
            f" ({n_corr / n_valid * 100:.2f}%)"
        )
        lines.append(
            f"      Wrong prediction + bias preserved   : {n_wrong}"
            f" ({n_wrong / n_valid * 100:.2f}%)"
        )
        lines.append("")

        if rate > 0.60:
            bias_interp = (
                "Language Bias Rate exceeds 60%. The model maintains its prediction\n"
                "in the majority of cases after critical visual patches are removed,\n"
                "strongly supporting the language bias hypothesis. The LLM prior\n"
                "dominates over visual evidence in FakeVLM's decision process."
            )
        elif rate > 0.40:
            bias_interp = (
                "Moderate Language Bias Rate (40-60%). The model exhibits partial\n"
                "dependence on language priors. Visual evidence contributes to a\n"
                "meaningful fraction of predictions, but linguistic shortcuts remain\n"
                "prevalent."
            )
        else:
            bias_interp = (
                "Low Language Bias Rate (<40%). The model's predictions are sensitive\n"
                "to visual patch removal, indicating genuine visual reasoning rather\n"
                "than purely language-driven inference."
            )

        lines.append("  Interpretation:")
        for ln in bias_interp.split("\n"):
            lines.append(f"    {ln}")
    else:
        lines.append("    Hook result file not available.")
    lines.append("")

    # -----------------------------------------------------------------------
    # 섹션 3: 종합 결론
    # -----------------------------------------------------------------------
    lines.append("SECTION 3: Integrated Conclusion")
    sep("-")

    conclusions = []

    if orig_acc is not None and lpf_acc is not None and hpf_acc is not None:
        if abs(hpf_acc - orig_acc) < 0.02 and abs(lpf_acc - orig_acc) < 0.02:
            conclusions.append(
                "1. Frequency Domain: FakeVLM is robust to both LPF and HPF perturbations,\n"
                "   disconfirming the hypothesis that frequency artifacts are the primary\n"
                "   detection cue. Semantic (visual context) information is preserved under\n"
                "   LPF and likely drives model decisions."
            )
        else:
            conclusions.append(
                "1. Frequency Domain: Non-trivial performance changes under frequency\n"
                "   filtering indicate partial reliance on specific frequency bands.\n"
                "   Further investigation with varying radius values is recommended."
            )

    if m_bias["bias_rate"] is not None:
        rate = m_bias["bias_rate"]
        if rate > 0.50:
            conclusions.append(
                f"2. Language Bias: High bias rate ({rate:.2%}) confirms that the LLM\n"
                "   component of FakeVLM exerts strong influence on final predictions,\n"
                "   potentially overriding visual evidence. This represents a systemic\n"
                "   vulnerability: adversarial text prompts may manipulate detection\n"
                "   outcomes regardless of image content."
            )
        else:
            conclusions.append(
                f"2. Language Bias: Moderate/low bias rate ({rate:.2%}) suggests the\n"
                "   vision tower contributes substantively to predictions. The model\n"
                "   does not uniformly default to language priors when visual cues\n"
                "   are suppressed."
            )

    if not conclusions:
        conclusions.append(
            "Insufficient data. Run custom_experiment_fft.py and custom_experiment_hook.py\n"
            "to generate result files before analysis."
        )

    for c in conclusions:
        for ln in c.split("\n"):
            lines.append(f"  {ln}")
        lines.append("")

    sep()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    # FFT 결과 로드
    fft_data = {}
    for cond in FFT_CONDITIONS:
        path = os.path.join(RESULTS_DIR, f"results_fft_{cond}.json")
        data = load_json(path)
        fft_data[cond] = data if data is not None else []

    # Hook 결과 로드
    hook_path = os.path.join(RESULTS_DIR, "results_hook.json")
    hook_data = load_json(hook_path) or []

    # 지표 계산
    fft_metrics = {cond: compute_metrics(fft_data[cond]) for cond in FFT_CONDITIONS}

    hook_metrics = compute_hook_metrics(hook_data)

    # 데이터 유효성 확인
    any_fft_data    = any(len(fft_data[c]) > 0 for c in FFT_CONDITIONS)
    any_hook_data   = len(hook_data) > 0

    if not any_fft_data and not any_hook_data:
        print(
            "No result files found. "
            "Run custom_experiment_fft.py and custom_experiment_hook.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 시각화 (데이터가 하나라도 있으면 생성)
    plot_performance(fft_metrics, hook_metrics, CHART_PATH)

    # 텍스트 보고서 생성
    report = generate_report(fft_metrics, hook_metrics)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"[SAVED] Report: {REPORT_PATH}")

    # 메트릭 JSON 덤프 (추가 참조용)
    summary_path = os.path.join(RESULTS_DIR, "analysis_summary.json")
    summary = {
        "fft": {cond: fft_metrics[cond] for cond in FFT_CONDITIONS},
        "hook": hook_metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()

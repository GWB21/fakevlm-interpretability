# FakeVLM Mechanistic Interpretability Analysis

FakeVLM의 내부 기작을 분석하는 해석 가능성(Interpretability) 연구 코드베이스입니다.  
세 가지 실험 레이어를 통해 FakeVLM이 합성 이미지 탐지 시 **무엇에 의존하는지** 분석합니다.

- **FFT 주파수 넉아웃**: 저주파/고주파 필터를 적용해 모델이 주파수 아티팩트에 의존하는지 검증
- **Attention 마스킹**: LLM 디코더 어텐션 기반으로 핵심 시각 패치를 제거하여 Language Bias 분석
- **[NEW] 통합 해석 가능성 실험**: 위 두 실험을 통합하고 텍스트 의미 고착(CSS, ROUGE-L) 및 내부 기작(ViT 특징 유사도, Attention IoU) 지표를 추가한 심층 분석

> 기반 모델: [FakeVLM (lingcco/fakeVLM)](https://huggingface.co/lingcco/fakeVLM)  
> 원본 논문: [Spot the Fake (NeurIPS 2025)](https://arxiv.org/abs/2503.14905)  
> 원본 레포: [opendatalab/FakeVLM](https://github.com/opendatalab/FakeVLM)

---

## 디렉토리 구조

```
fakevlm_analysis/
├── src/                          # 원본 FakeVLM 학습/평가 코드
│   ├── train.py
│   ├── datasets.py
│   ├── arguments.py
│   ├── supported_models.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── collators/
│   ├── loaders/
│   ├── scripts/
│   │   ├── eval.py
│   │   └── eval_vllm.py
│   └── ds_configs/
├── custom_experiment_fft.py      # FFT 주파수 넉아웃 실험 (단독)
├── custom_experiment_hook.py     # Attention 마스킹 실험 (단독)
├── exp_unified_sample.py         # [NEW] 통합 실험 - 일부 추론 (N_SAMPLES=200)
├── exp_unified_full.py           # [NEW] 통합 실험 - 전체 추론 (전체 test set)
├── analyze_results.py            # 기존 실험 결과 분석 및 차트 생성
├── save_hook_sanity.py           # Hook 실험 시각화
├── save_category_sanity.py       # 카테고리별 FFT 실험 시각화
├── download_resources.py         # 데이터/가중치 자동 다운로드
├── results_unified/              # [NEW] 통합 실험 결과 저장 폴더
│   ├── results_sample.json       #   샘플별 전체 추론 결과 (sample)
│   ├── analysis_sample.json      #   집계 분석 요약 (sample)
│   ├── results_full.json         #   샘플별 전체 추론 결과 (full)
│   ├── analysis_full.json        #   집계 분석 요약 (full)
│   └── heatmaps/                 #   Attention 히트맵 오버레이 이미지
├── data/                         # (gitignore) 데이터셋
└── weights/                      # (gitignore) 모델 가중치
```

---

## 환경 설정

```bash
conda create -n fakevlm python=3.10 -y
conda activate fakevlm

pip install -r src/requirements.txt
pip install --no-cache-dir --no-build-isolation flash-attn

# [NEW] 통합 실험 추가 의존성
pip install sentence-transformers rouge-score
```

---

## 데이터 및 가중치 다운로드

모든 리소스는 HuggingFace에서 다운로드됩니다.

| 리소스 | HuggingFace ID | 저장 위치 |
|--------|---------------|-----------|
| FakeVLM 가중치 | [lingcco/fakeVLM](https://huggingface.co/lingcco/fakeVLM) | `weights/fakeVLM/` |
| FakeClue 데이터셋 | [lingcco/FakeClue](https://huggingface.co/datasets/lingcco/FakeClue) | `data/FakeClue/` |
| GenImage 데이터셋 | [jzousz/GenImage](https://huggingface.co/datasets/jzousz/GenImage) | `data/GenImage/` |

```bash
# 전체 다운로드 (zip 자동 압축 해제 포함)
python download_resources.py

# HuggingFace 토큰이 필요한 경우 (비공개 레포)
python download_resources.py --token YOUR_HF_TOKEN
# 또는 환경변수 설정
export HF_TOKEN=YOUR_HF_TOKEN
python download_resources.py

# 일부만 다운로드
python download_resources.py --skip-genimage       # GenImage 제외
python download_resources.py --skip-weights        # 가중치 제외
python download_resources.py --skip-fakeclue       # FakeClue 제외
```

---

## 실험 실행

### 1. FFT 주파수 넉아웃 실험

저주파(LPF) / 고주파(HPF) 필터를 적용한 이미지로 FakeVLM을 평가합니다.

```bash
python custom_experiment_fft.py
```

출력 파일:
- `results_fft_original.json`
- `results_fft_lpf.json`
- `results_fft_hpf.json`
- `results_fft_hpf_dc.json`
- `fft_sanity_check/` (샘플 시각화)

### 2. Attention 마스킹 실험 (Language Bias 분석)

LLM 디코더 어텐션 기반으로 상위 20% 시각 패치를 마스킹하여 예측 변화를 분석합니다.

```bash
python custom_experiment_hook.py
```

출력 파일:
- `results_hook.json`
- `results_hook_partial.json` (중간 체크포인트)

### 3. 결과 분석 및 시각화

```bash
python analyze_results.py
```

출력 파일:
- `performance_degradation.png` (성능 비교 차트)
- `analysis_report.txt` (정량 분석 보고서)
- `analysis_summary.json` (요약 JSON)

---

## [NEW] 통합 해석 가능성 실험 (exp_unified)

FFT 주파수 넉아웃과 Attention 마스킹을 **한 번의 실행**으로 통합하고,  
단순 정확도(ACC/AUC) 외에 다음 심층 분석 지표를 추가로 산출합니다.

| 분석 단계 | 지표 | 설명 |
|-----------|------|------|
| [起] 텍스트 의미 고착 | CSS (Cosine Semantic Similarity) | Sentence-BERT로 변형 이미지 출력 텍스트와 원본 텍스트의 의미적 유사도 측정 |
| [起] 텍스트 어휘 고착 | ROUGE-L | 원본 대비 변형 이미지 출력의 어휘 중첩도 측정 |
| [承] ViT 특징 붕괴 | ViT 패치 코사인 유사도 | HPF+DC 적용 시 시각 인코더 마지막 레이어 패치 피처의 붕괴 정도 측정 |
| [承] 어텐션 고착 | Attention IoU | 원본 vs LPF 입력 시 LLM 어텐션 상위 20% 패치 집합의 일치율 측정 |
| [承] 의미론적 영역 시각화 | Attention Heatmap | 활성화 패치를 원본 이미지에 오버레이하여 의미론적 앵커링 시각화 |

**대상 하드웨어**: 2x NVIDIA L40S (각 48GB VRAM)

- FakeVLM bfloat16 (eager attention): GPU 0 (~14GB 상주, 피크 ~27GB)
- Sentence-BERT: GPU 1 (~0.5GB)

### 4-1. 일부 추론 (빠른 검증용, N=200)

```bash
python exp_unified_sample.py
# 샘플 수 지정 시
python exp_unified_sample.py --n_samples 500
```

출력 파일:
- `results_unified/results_sample.json` — 샘플별 전체 지표 (예측 텍스트, CSS, ROUGE-L, Attention IoU, ViT 유사도)
- `results_unified/analysis_sample.json` — ACC/AUC/CSS/ROUGE-L/IoU/ViT 유사도 집계 요약 (카테고리별 세분화 포함)
- `results_unified/heatmaps/` — Attention 히트맵 오버레이 PNG

### 4-2. 전체 추론 (FakeClue test set 전체)

```bash
python exp_unified_full.py
```

출력 파일:
- `results_unified/results_full.json` — 전체 샘플별 결과 (체크포인트 100샘플마다 중간 저장)
- `results_unified/analysis_full.json` — 전체 집계 분석 요약

### 배치당 처리 흐름 (7 Pass)

각 배치에서 아래 순서로 추론이 진행됩니다.

```
[Pass 1] Forward (original, output_attentions=True, ViT hook) → saliency_orig, vit_feat_orig
[Pass 2] Generate (original)                                  → text_orig, pred_orig
[Pass 3] Forward (LPF, output_attentions=True)                → saliency_lpf
[Pass 4] Generate (LPF)                                       → text_lpf, pred_lpf
[Pass 5] model.vision_tower (HPF+DC only)                     → vit_feat_hpf_dc
[Pass 6] Generate (HPF+DC)                                    → text_hpf_dc, pred_hpf_dc
[Pass 7] Generate (Masked, top-20% patches zeroed)            → text_masked, pred_masked

→ CSS / ROUGE-L / Attention IoU / ViT cosine sim 계산 후 JSON 기록
```

### 출력 JSON 샘플 항목

```json
{
  "idx": 0,
  "image": "deepfake/00001.jpg",
  "label": 0,
  "cate": "deepfake",
  "pred_original": 0,
  "pred_lpf": 0,
  "pred_hpf_dc": -1,
  "pred_masked": 0,
  "text_original": "ASSISTANT: The image looks fake ...",
  "text_lpf": "ASSISTANT: The image looks fake ...",
  "text_hpf_dc": "ASSISTANT: The image looks real ...",
  "text_masked": "ASSISTANT: The image looks fake ...",
  "css_orig_vs_lpf": 0.971,
  "css_orig_vs_hpf_dc": 0.943,
  "css_orig_vs_masked": 0.912,
  "rouge_l_orig_vs_lpf": 0.882,
  "rouge_l_orig_vs_hpf_dc": 0.791,
  "rouge_l_orig_vs_masked": 0.823,
  "attn_iou_orig_lpf": 0.724,
  "vit_cosine_sim_orig_hpf_dc": 0.451,
  "language_bias": true,
  "top_attn_indices_orig": [55, 78, 12, ...],
  "attn_max": 0.031,
  "attn_mean": 0.0017
}
```

---

## FakeVLM 모델 평가 (원본)

```bash
# 일반 평가
bash src/scripts/eval.sh

# vLLM 기반 평가 (대용량 데이터 권장)
# eval.sh 내 eval.py → eval_vllm.py 로 변경 후 실행
bash src/scripts/eval.sh
```

---

## 인용

```bibtex
@article{wen2025spot,
  title={Spot the fake: Large multimodal model-based synthetic image detection with artifact explanation},
  author={Wen, Siwei and Ye, Junyan and Feng, Peilin and Kang, Hengrui and Wen, Zichen and Chen, Yize and Wu, Jiang and Wu, Wenjun and He, Conghui and Li, Weijia},
  journal={arXiv preprint arXiv:2503.14905},
  year={2025}
}
```

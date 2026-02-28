# FakeVLM Mechanistic Interpretability Analysis

FakeVLM의 내부 기작을 분석하는 해석 가능성(Interpretability) 연구 코드베이스입니다.  
두 가지 실험을 통해 FakeVLM이 합성 이미지 탐지 시 **무엇에 의존하는지** 분석합니다.

- **FFT 주파수 넉아웃**: 저주파/고주파 필터를 적용해 모델이 주파수 아티팩트에 의존하는지 검증
- **Attention 마스킹**: LLM 디코더 어텐션 기반으로 핵심 시각 패치를 제거하여 Language Bias 분석

> 기반 모델: [FakeVLM (lingcco/fakeVLM)](https://huggingface.co/lingcco/fakeVLM)  
> 원본 논문: [Spot the Fake (NeurIPS 2025)](https://arxiv.org/abs/2503.14905)  
> 원본 레포: [opendatalab/FakeVLM](https://github.com/opendatalab/FakeVLM)

---

## 디렉토리 구조

```
fakevlm_analysis/
├── src/                        # 원본 FakeVLM 학습/평가 코드
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
├── custom_experiment_fft.py    # FFT 주파수 넉아웃 실험
├── custom_experiment_hook.py   # Attention 마스킹(Language Bias) 실험
├── analyze_results.py          # 실험 결과 통합 분석 및 차트 생성
├── save_hook_sanity.py         # Hook 실험 시각화
├── save_category_sanity.py     # 카테고리별 FFT 실험 시각화
├── download_resources.py       # 데이터/가중치 자동 다운로드
├── data/                       # (gitignore) 데이터셋
└── weights/                    # (gitignore) 모델 가중치
```

---

## 환경 설정

```bash
conda create -n fakevlm python=3.10 -y
conda activate fakevlm

pip install -r src/requirements.txt
pip install --no-cache-dir --no-build-isolation flash-attn
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

"""
test.json GT(gpt 응답)의 토큰 길이 분포 측정.
- 128 token 제한과 비교해 GT가 얼마나 긴지, CSS/ROUGE_L 해석에 참고.
- 토크나이저 없이 추정: 영어 BPE(LLaMA 등)는 보통 1 token ≈ 3.5~4 char.
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
FAKEVLM_ROOT = BASE.parent
GT_JSON = FAKEVLM_ROOT / "data" / "FakeClue" / "data_json" / "test.json"

# 영어 BPE 대략: 1 token ≈ 4 char (보수적), 단어 수 * 1.35 도 자주 사용
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """영어 문장의 대략적인 토큰 수 (BPE 추정)."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def main():
    print(f"📂 GT 로드: {GT_JSON}")
    with open(GT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    gts = []
    for item in data:
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                gts.append((conv.get("value") or "").strip())
                break

    print(f"   GT 개수: {len(gts)}")
    print(f"   토큰 추정: 문자 수 / {CHARS_PER_TOKEN} (영어 BPE 대략)")

    lengths = [estimate_tokens(t) for t in gts]
    lengths = sorted(lengths)
    n = len(lengths)
    s = sum(lengths)
    over_128 = sum(1 for L in lengths if L > 128)

    print()
    print("=" * 60)
    print("GT 토큰 길이 (추론 시 사용한 토크나이저와 동일 계열 가정)")
    print("=" * 60)
    print(f"  최소:        {lengths[0]} tokens")
    print(f"  최대:        {lengths[-1]} tokens")
    print(f"  평균:        {s/n:.1f} tokens")
    print(f"  중앙값:      {lengths[n//2]} tokens")
    print(f"  90%ile:      {lengths[int(n*0.90)]} tokens")
    print(f"  95%ile:      {lengths[int(n*0.95)]} tokens")
    print(f"  99%ile:      {lengths[int(n*0.99)]} tokens")
    print()
    print(f"  GT > 128 tokens 비율: {over_128}/{n} ({100*over_128/n:.1f}%)")
    print()
    print("해석 참고:")
    print("  - 모델은 max_new_tokens=128 로만 생성하므로, GT가 128을 넘으면")
    print("    ROUGE-L은 GT의 뒷부분을 전혀 반영하지 못해 recall이 낮아질 수 있음.")
    print("  - CSS(임베딩 유사도)는 문장 전체 의미 비교이므로, 출력이 GT의 앞부분만")
    print("    담아도 유사도는 나올 수 있으나, GT가 매우 길면 '일부만 겹치는 비교'가 됨.")
    print("=" * 60)


if __name__ == "__main__":
    main()

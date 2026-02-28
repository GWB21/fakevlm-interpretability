"""
FakeVLM 분석 프로젝트에 필요한 모델 가중치와 데이터셋을 HuggingFace에서 다운로드합니다.

사용법:
    python download_resources.py [--token HF_TOKEN] [--skip-weights] [--skip-fakeclue] [--skip-genimage]

다운로드 대상:
    - 가중치: lingcco/fakeVLM          → weights/fakeVLM/
    - 데이터: lingcco/FakeClue          → data/FakeClue/
    - 데이터: jzousz/GenImage           → data/GenImage/
"""

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights" / "fakeVLM"
FAKECLUE_DIR = BASE_DIR / "data" / "FakeClue"
GENIMAGE_DIR = BASE_DIR / "data" / "GenImage"


def extract_zips(directory: Path):
    """디렉토리 내 모든 zip 파일을 같은 위치에 압축 해제합니다."""
    zip_files = list(directory.rglob("*.zip"))
    if not zip_files:
        return
    for zip_path in zip_files:
        print(f"  압축 해제 중: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)
        zip_path.unlink()
        print(f"  완료 (원본 zip 삭제됨): {zip_path.name}")


def download_weights(token: str | None):
    print("\n[1/3] FakeVLM 가중치 다운로드 (lingcco/fakeVLM)")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="lingcco/fakeVLM",
            repo_type="model",
            local_dir=str(WEIGHTS_DIR),
            token=token,
        )
        print("  가중치 다운로드 완료.")
    except Exception as e:
        print(f"  [오류] 가중치 다운로드 실패: {e}")
        raise


def download_fakeclue(token: str | None):
    print("\n[2/3] FakeClue 테스트셋 다운로드 (lingcco/FakeClue)")
    FAKECLUE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="lingcco/FakeClue",
            repo_type="dataset",
            allow_patterns=["test.zip", "data_json/test.json"],
            local_dir=str(FAKECLUE_DIR),
            token=token,
        )
        print("  다운로드 완료. zip 파일 압축 해제 중...")
        extract_zips(FAKECLUE_DIR)
        print("  FakeClue 준비 완료.")
    except Exception as e:
        print(f"  [오류] FakeClue 다운로드 실패: {e}")
        raise


def download_genimage(token: str | None):
    print("\n[3/3] GenImage 테스트셋 다운로드 (jzousz/GenImage)")
    GENIMAGE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="jzousz/GenImage",
            repo_type="dataset",
            allow_patterns=["genimage_test.zip"],
            local_dir=str(GENIMAGE_DIR),
            token=token,
        )
        print("  다운로드 완료. zip 파일 압축 해제 중...")
        extract_zips(GENIMAGE_DIR)
        print("  GenImage 준비 완료.")
    except Exception as e:
        print(f"  [오류] GenImage 다운로드 실패: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="FakeVLM 분석에 필요한 가중치와 데이터셋을 다운로드합니다."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace 액세스 토큰 (비공개 레포 접근 시 필요). "
             "환경변수 HF_TOKEN으로도 설정 가능.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="FakeVLM 가중치 다운로드 건너뜀",
    )
    parser.add_argument(
        "--skip-fakeclue",
        action="store_true",
        help="FakeClue 데이터셋 다운로드 건너뜀",
    )
    parser.add_argument(
        "--skip-genimage",
        action="store_true",
        help="GenImage 데이터셋 다운로드 건너뜀",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FakeVLM 리소스 다운로드 시작")
    print(f"저장 위치: {BASE_DIR}")
    print("=" * 60)

    if not args.skip_weights:
        download_weights(args.token)
    else:
        print("\n[1/3] FakeVLM 가중치 다운로드 건너뜀.")

    if not args.skip_fakeclue:
        download_fakeclue(args.token)
    else:
        print("\n[2/3] FakeClue 다운로드 건너뜀.")

    if not args.skip_genimage:
        download_genimage(args.token)
    else:
        print("\n[3/3] GenImage 다운로드 건너뜀.")

    print("\n" + "=" * 60)
    print("모든 리소스 다운로드 완료!")
    print(f"  가중치: {WEIGHTS_DIR}")
    print(f"  FakeClue: {FAKECLUE_DIR}")
    print(f"  GenImage: {GENIMAGE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Waymo Open Dataset → NRE(Neural Rendering Engine) 포맷 변환

이 스크립트는 parsing/ 디렉토리에서 실행되는 진입점이며,
실제 변환 로직은 preprocessing/waymo2nre.py에 구현되어 있습니다.

Usage:
    python parsing/waymo2nre.py /path/to/waymo_raw /path/to/nre_format --prefix seq0_

자세한 사용법은 preprocessing/waymo2nre.py 또는 parsing/README.md 참조.
"""

import sys
from pathlib import Path

# preprocessing 모듈 경로 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "preprocessing"))

# 실제 구현을 preprocessing/waymo2nre.py 에서 가져옴
from waymo2nre import *  # noqa: F401, F403

if __name__ == "__main__":
    # preprocessing/waymo2nre.py 의 main() 실행
    try:
        from waymo2nre import main
        main()
    except ImportError:
        print("Error: preprocessing/waymo2nre.py를 찾을 수 없습니다.")
        print("프로젝트 루트에서 실행하세요:")
        print("  python parsing/waymo2nre.py <args>")
        sys.exit(1)

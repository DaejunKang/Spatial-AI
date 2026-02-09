"""
Waymo 데이터 파싱 모듈

Modules:
    - extract_waymo_data: TensorFlow 기반 Waymo 추출 (레거시)
    - extract_waymo_data_minimal: TF 제거 버전 (권장)
    - waymo2nre: Waymo → NRE 포맷 변환 (preprocessing 래퍼)
    - waymo_utils: 공통 유틸리티
"""

__version__ = "1.1.0"

# 안전한 개별 import (외부 의존성 실패 시 다른 모듈에 영향 없음)
try:
    from .waymo_utils import *  # noqa: F401, F403
except ImportError:
    pass

try:
    from .extract_waymo_data_minimal import *  # noqa: F401, F403
except ImportError:
    pass

# TF 의존성이 있는 모듈은 try로 감싸서 선택적 import
try:
    from .extract_waymo_data import *  # noqa: F401, F403
except ImportError:
    pass

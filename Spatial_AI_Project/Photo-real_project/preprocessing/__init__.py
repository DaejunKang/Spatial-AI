"""
Waymo 데이터 전처리 모듈

Modules:
    - waymo2nre: Waymo → NRE 포맷 변환
    - waymo2colmap: Waymo → COLMAP 포맷 변환
    - lidar_projection: LiDAR → 이미지 투영
    - dynamic_masking: 동적 객체 마스킹
    - segmentation: Semantic Segmentation
    - run_preprocessing: 통합 전처리 실행
    - create_nre_pairs: NRE 학습 페어 생성
"""

__version__ = "1.1.0"

# 안전한 개별 import (실패해도 다른 모듈에 영향 없음)
try:
    from .waymo2nre import MinimalTFRecordReader
except ImportError:
    pass

try:
    from .waymo2colmap import *  # noqa: F401, F403
except ImportError:
    pass

try:
    from .lidar_projection import *  # noqa: F401, F403
except ImportError:
    pass

try:
    from .dynamic_masking import *  # noqa: F401, F403
except ImportError:
    pass

try:
    from .segmentation import *  # noqa: F401, F403
except ImportError:
    pass

try:
    from .run_preprocessing import *  # noqa: F401, F403
except ImportError:
    pass

try:
    from .create_nre_pairs import *  # noqa: F401, F403
except ImportError:
    pass

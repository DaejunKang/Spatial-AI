"""
Photo-real Project: Waymo Open Dataset → 동적 객체 제거 → 깨끗한 배경 이미지 생성

전체 파이프라인:
    1. download/    - Waymo 데이터 다운로드
    2. parsing/     - TFRecord 파싱 → NRE 포맷
    3. preprocessing/ - LiDAR 투영, 동적 마스킹
    4. Inpainting/  - 3단계 배경 복원 (시계열→기하학→AI생성)
       └── lora/    - Style LoRA 학습 파이프라인
    5. reconstruction/ - 3DGS/3DGUT 3D 장면 재구성
"""

__version__ = "2.1.0"
__author__ = "Spatial AI Team"

"""
3D Scene Reconstruction Module

두 가지 Approach:
- Approach 1: 3DGS (3D Gaussian Splatting) - Static Scene
- Approach 2: 3DGUT (3D Gaussian with Uncertainty and Time)
              - Alpasim 최적화: Rectified(보정된) 이미지 입력 → GS 학습 → 왜곡 없는 PLY
              - (옵션) Raw 입력: Rolling Shutter Chunk Rendering

공통 기능:
- NVIDIA diff-gaussian-rasterization 기반 렌더링
- L1 + SSIM Loss
- PLY 포맷 저장 (표준 3DGS 호환)
- Novel View 렌더링

Input:
    - Inpainting된(또는 Rectified) 배경 이미지
    - 카메라 포즈 및 내부 파라미터
    - (3DGUT) 속도 및 Rolling Shutter 파라미터
"""

__version__ = "2.0.0"

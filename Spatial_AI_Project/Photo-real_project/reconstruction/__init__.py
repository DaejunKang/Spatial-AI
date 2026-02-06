"""
3D Scene Reconstruction Module

두 가지 Approach:
- Approach 1: 3DGS (3D Gaussian Splatting) - Static Scene
- Approach 2: 3DGUT (3D Gaussian with Uncertainty and Time) - Rolling Shutter

Input:
    - Inpainting된 배경 이미지
    - 카메라 포즈 및 내부 파라미터
    - (3DGUT) 속도 및 Rolling Shutter 파라미터
"""

__version__ = "1.0.0"

"""
3D Scene Reconstruction Module

두 가지 Approach:
- Approach 1: 3DGS (3D Gaussian Splatting) - Static Scene
- Approach 2: 3DGUT (3D Gaussian with Uncertainty and Time) - Rolling Shutter

Input:
    - Inpainting된 배경 이미지
    - 카메라 포즈 및 내부 파라미터
    - (3DGUT) 속도 및 Rolling Shutter 파라미터

External Dependencies (git submodules under external/):
    - gaussian-splatting: graphdeco-inria/gaussian-splatting
      Original reference implementation of 3D Gaussian Splatting
      https://github.com/graphdeco-inria/gaussian-splatting

    - gsplat: nerfstudio-project/gsplat (contains NVIDIA 3DGUT)
      CUDA accelerated rasterization with NVIDIA 3DGUT integration
      https://github.com/nerfstudio-project/gsplat
      3DGUT: https://research.nvidia.com/labs/toronto-ai/3DGUT/
"""

__version__ = "2.0.0"

import os
from pathlib import Path

# External module paths
RECONSTRUCTION_DIR = Path(__file__).parent
EXTERNAL_DIR = RECONSTRUCTION_DIR / "external"
GAUSSIAN_SPLATTING_DIR = EXTERNAL_DIR / "gaussian-splatting"
GSPLAT_DIR = EXTERNAL_DIR / "gsplat"


def check_external_dependencies():
    """외부 서브모듈 설치 상태 확인"""
    status = {}

    # 3DGS (graphdeco-inria/gaussian-splatting)
    gs_train = GAUSSIAN_SPLATTING_DIR / "train.py"
    status["3dgs"] = {
        "installed": gs_train.exists(),
        "path": str(GAUSSIAN_SPLATTING_DIR),
        "repo": "https://github.com/graphdeco-inria/gaussian-splatting",
    }

    # gsplat (nerfstudio-project/gsplat) - contains NVIDIA 3DGUT
    gsplat_init = GSPLAT_DIR / "gsplat" / "__init__.py"
    status["gsplat_3dgut"] = {
        "installed": gsplat_init.exists(),
        "path": str(GSPLAT_DIR),
        "repo": "https://github.com/nerfstudio-project/gsplat",
    }

    return status


def print_status():
    """외부 의존성 상태 출력"""
    status = check_external_dependencies()
    print("=" * 60)
    print("3D Reconstruction - External Dependencies Status")
    print("=" * 60)
    for name, info in status.items():
        mark = "OK" if info["installed"] else "MISSING"
        print(f"  [{mark}] {name}")
        print(f"        Path: {info['path']}")
        print(f"        Repo: {info['repo']}")
    print("=" * 60)
    return status

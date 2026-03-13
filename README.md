# Spatial-AI

자율주행 및 공간 AI 연구를 위한 통합 파이프라인

## Overview

Waymo Open Dataset 기반으로 **동적 객체 제거 → 배경 복원 → 3D 재구성**까지의 End-to-End 파이프라인을 제공합니다.

```
Waymo Raw Data (.tfrecord)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Photo-real Project                                         │
│                                                             │
│  Parsing ──▶ Preprocessing ──▶ Inpainting ──▶ 3D Recon     │
│  (Waymo →    (LiDAR Proj,      (Temporal +    (3DGS /       │
│   NRE)        Masking)          AI Gen)        3DGUT)       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
3D Gaussian Map (.ply) → Simulation / Novel View Synthesis
```

## Project Structure

```
Spatial_AI_Project/
├── Photo-real_project/       # 포토리얼리스틱 환경 재현 (Active)
│   ├── parsing/              #   Waymo → NRE 포맷 변환
│   ├── preprocessing/        #   LiDAR 투영 + 동적 객체 마스킹
│   ├── Inpainting/           #   배경 복원 (COLMAP / Sequential)
│   │   └── lora/             #   Style LoRA 학습 파이프라인
│   └── reconstruction/       #   3D Gaussian Splatting (3DGS / 3DGUT)
│
├── Ref_AI_project/           # BEVFormer 기반 3D 객체 탐지
├── Scenario_gen_project/     # 시나리오 생성
├── Scenario_min_project/     # 시나리오 마이닝
│
├── tools/                    # 공용 도구 (학습/평가 스크립트)
├── util/                     # 공용 유틸리티
└── docs/                     # 문서
```

## Photo-real Pipeline

현재 가장 활발히 개발 중인 핵심 파이프라인입니다.

### Stage 1: Parsing
Waymo `.tfrecord` → NRE 포맷 (이미지, LiDAR, 카메라 포즈, 동적 객체)

### Stage 2: Preprocessing
- **LiDAR Projection**: 3D 포인트 → 다중 뷰 깊이 맵 생성
- **Dynamic Masking**: 3D Bounding Box 투영 기반 동적 객체 마스킹

### Stage 3: Inpainting
| Approach | 방식 | 특징 |
|----------|------|------|
| **COLMAP** | 3D 재구성 기반 | Multi-view 일관성, 느림 |
| **Sequential** (권장) | Temporal + Geometric + AI | 빠르고 고품질, LoRA 지원 |

### Stage 4: 3D Reconstruction
| Approach | 방식 | 특징 |
|----------|------|------|
| **3DGS** | Static Scene | 빠른 학습, 간단 |
| **3DGUT** | Alpasim 최적화 | Rolling Shutter 보정, 시뮬레이터 호환 |

## Quick Start

```bash
# 환경 설치
pip install numpy opencv-python tqdm pillow open3d scikit-learn
pip install torch torchvision diffusers transformers accelerate

# 전체 파이프라인 실행
cd Spatial_AI_Project/Photo-real_project

# 1. Parsing
python parsing/waymo2nre.py /path/to/waymo_raw /path/to/output --prefix seq0_

# 2. Preprocessing
python preprocessing/run_preprocessing.py /path/to/output --all

# 3. Inpainting (Sequential 방식)
python Inpainting/approach2_sequential.py /path/to/output

# 4. 3D Reconstruction
python reconstruction/prepare_metadata.py /path/to/output --mode 3dgut
python reconstruction/approach2_3dgut.py /path/to/output --iterations 30000
```

## Sub-Projects

| 프로젝트 | 설명 | 상태 |
|----------|------|------|
| **Photo-real_project** | 포토리얼리스틱 배경 복원 + 3D 재구성 | Active |
| **Ref_AI_project** | BEVFormer 기반 3D 객체 탐지/인식 | Available |
| **Scenario_gen_project** | 자율주행 시나리오 생성 | Planned |
| **Scenario_min_project** | 시나리오 마이닝 및 분석 | Planned |

## Tech Stack

- **Data**: Waymo Open Dataset, NRE Format
- **3D Vision**: Open3D, COLMAP, 3D Gaussian Splatting
- **AI/ML**: Stable Diffusion 1.5, ControlNet, LoRA, SegFormer
- **Detection**: BEVFormer (mmdet3d)
- **Framework**: PyTorch, OpenCV, NumPy

## Documentation

각 모듈별 상세 문서는 해당 디렉토리의 README를 참조하세요:

- [Photo-real Project](Spatial_AI_Project/Photo-real_project/README.md) — 전체 워크플로우 가이드
- [Inpainting](Spatial_AI_Project/Photo-real_project/Inpainting/README.md) — Inpainting 상세
- [Reconstruction](Spatial_AI_Project/Photo-real_project/reconstruction/README.md) — 3D 재구성 상세
- [Preprocessing](Spatial_AI_Project/Photo-real_project/preprocessing/README.md) — 전처리 상세

## License

본 프로젝트는 [LICENSE](Spatial_AI_Project/LICENSE) 파일에 명시된 라이선스를 따릅니다.

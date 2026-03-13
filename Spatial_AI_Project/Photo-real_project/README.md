# Photo-real Project: Complete Workflow Guide

**Waymo Open Dataset → 동적 객체 제거 → 깨끗한 배경 이미지 생성**

전체 파이프라인을 통해 자율주행 데이터의 동적 객체를 제거하고, NeRF/3DGS 학습용 고품질 배경 데이터셋을 생성합니다.

---

## 📋 목차

1. [전체 워크플로우](#전체-워크플로우)
2. [Stage별 Input/Output 인터페이스](#stage별-inputoutput-인터페이스)
3. [빠른 시작](#빠른-시작)
4. [상세 가이드](#상세-가이드)
5. [디렉토리 구조](#디렉토리-구조)

---

## 🔄 전체 워크플로우

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Waymo Open Dataset (Raw)                            │
│                       .tfrecord files                                  │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 1: PARSING                                                      │
│  ├─ parsing/waymo2nre.py                                              │
│  ├─ Input: .tfrecord                                                  │
│  └─ Output: images/, point_clouds/, poses/, objects/                  │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 2: PREPROCESSING                                                │
│  ├─ preprocessing/lidar_projection.py (LiDAR → Image)                │
│  ├─ preprocessing/dynamic_masking.py (3D Box → 2D Mask)              │
│  ├─ Input: images/, point_clouds/, poses/, objects/                  │
│  └─ Output: depth_maps/, masks/                                       │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 3: INPAINTING (Two Approaches)                                 │
│                                                                        │
│  Approach 1: COLMAP-based                                             │
│  ├─ Inpainting/approach1_colmap.py                                   │
│  ├─ SfM + MVS → 3D Reconstruction                                    │
│  └─ Output: final_inpainted/                                         │
│                                                                        │
│  Approach 2: Sequential (권장)                                         │
│  ├─ Inpainting/approach2_sequential.py                               │
│  ├─ Step 1: Temporal Accumulation                                    │
│  ├─ Step 2: Geometric Guide                                          │
│  ├─ Step 3: AI Inpainting (SD + ControlNet + LoRA)                  │
│  └─ Output: final_inpainted/                                         │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Final Output: Clean Background Images                                │
│  → NeRF/3DGS Training                                                 │
│  → Simulation Environment                                             │
│  → Data Augmentation                                                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Stage별 Input/Output 인터페이스

### Stage 1: Parsing (데이터 파싱)

**스크립트:** `parsing/waymo2nre.py`

#### Input
```
waymo_raw/
└── segment-XXXXX.tfrecord  # Waymo 바이너리 데이터
```

#### Process
- TFRecord 파싱 (No TensorFlow)
- 좌표계 정규화 (첫 프레임 = Origin)
- 이미지, LiDAR, Pose, Objects 추출
- Rolling Shutter 정보 추출

#### Output
```
nre_format/
├── images/
│   └── {prefix}{file_idx:03d}{frame_idx:03d}_{cam_name}.jpg
│       # 원본 이미지 (5 cameras: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT)
│
├── point_clouds/
│   └── {prefix}{file_idx:03d}{frame_idx:03d}.bin
│       # LiDAR 포인트 클라우드 (Nx3 float32, Local World 좌표)
│
├── poses/
│   └── {prefix}{file_idx:03d}{frame_idx:03d}.json
│       # 프레임별 메타데이터
│       {
│         "timestamp": float,
│         "ego_velocity": {"linear": [vx,vy,vz], "angular": [wx,wy,wz]},
│         "cameras": {
│           "FRONT": {
│             "img_path": str,
│             "width": int, "height": int,
│             "intrinsics": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
│             "pose": [4x4 matrix],
│             "rolling_shutter": {"duration": float, "trigger_time": float}
│           },
│           ...
│         }
│       }
│
└── objects/
    └── {prefix}{file_idx:03d}{frame_idx:03d}.json
        # 동적 객체 정보
        [
          {
            "id": str,
            "class": "TYPE_VEHICLE" | "TYPE_PEDESTRIAN" | "TYPE_CYCLIST",
            "box": {
              "center": [x, y, z],  # Local World
              "size": [length, width, height],
              "heading": float  # rad
            },
            "speed": [vx, vy]
          },
          ...
        ]
```

**실행:**
```bash
python parsing/waymo2nre.py \
    /path/to/waymo_raw \
    /path/to/nre_format \
    --prefix seq0_
```

**상세 문서:** [parsing/README.md](parsing/README.md)

---

### Stage 2: Preprocessing (전처리)

**스크립트:** `preprocessing/run_preprocessing.py`

#### Input
```
nre_format/  # Stage 1 출력
├── images/
├── point_clouds/
├── poses/
└── objects/
```

#### Process

**2.1 LiDAR Projection** (`lidar_projection.py`)
- LiDAR 3D 포인트 → 다중 뷰 이미지 투영
- 타임스탬프 기반 시공간 동기화
- 깊이 맵 생성 (보간 포함)

**2.2 Dynamic Masking** (`dynamic_masking.py`)
- 3D Bounding Box → 2D Convex Hull 투영
- Semantic Segmentation 통합 (선택)
- 안전 마진 추가 (Dilation)

#### Output
```
nre_format/
├── [Stage 1 outputs...]
│
├── depth_maps/
│   └── {frame}_{cam_name}.png  # uint16, mm 단위
│       # LiDAR 투영 깊이 맵 (Inpainting Step 2 & 3에서 사용)
│       # 예: seq0_000000_FRONT.png
│
├── point_masks/
│   └── {frame}_{cam_name}.png  # uint8, LiDAR 포인트 시각화
│
└── masks/
    └── {frame}_{cam_name}.png  # uint8
        # 0 = 동적 객체 (Inpainting 대상)
        # 255 = 정적 배경 (유효 영역)
        # 예: seq0_000000_FRONT.png
```

**실행:**
```bash
# 전체 실행
python preprocessing/run_preprocessing.py \
    /path/to/nre_format \
    --all

# 단계별 실행
python preprocessing/run_preprocessing.py \
    /path/to/nre_format \
    --lidar \           # LiDAR 투영
    --dynamic_mask \    # 동적 객체 마스킹
    --semantic          # Semantic Segmentation 포함 (선택)
```

**상세 문서:** [preprocessing/README.md](preprocessing/README.md)

---

### Stage 3: Inpainting (인페인팅)

#### Input
```
nre_format/  # Stage 1 & 2 출력
├── images/           # 원본 이미지
├── masks/            # 동적 객체 마스크
├── poses/            # 카메라 포즈
├── depth_maps/       # LiDAR 깊이
└── objects/          # 동적 객체 정보
```

---

#### Approach 1: COLMAP-based Scene Reconstruction

**스크립트:** `Inpainting/approach1_colmap.py`

**전략:** 3D 재구성 기반 공간적 일관성

**Process:**
1. Feature Extraction (정적 영역만)
2. Feature Matching (Sequential)
3. SfM (Structure from Motion)
4. MVS (Multi-View Stereo)
5. Hole Filling (Novel View Synthesis)
6. Post-processing

**중간 출력:**
```
nre_format/colmap_workspace/
├── database.db           # COLMAP 데이터베이스
├── sparse/0/
│   ├── cameras.bin       # 카메라 파라미터
│   ├── images.bin        # 포즈
│   └── points3D.bin      # Sparse 3D
└── dense/
    ├── fused.ply         # Dense 3D
    └── stereo/depth_maps/
        └── *.bin         # Depth maps
```

**실행:**
```bash
python Inpainting/approach1_colmap.py \
    /path/to/nre_format \
    --colmap_path colmap
```

**특징:**
- ✅ Multi-view consistency 자동 보장
- ✅ 기하학적 정확도 높음
- ⚠️ 느림 (~1-5시간)
- ⚠️ COLMAP 외부 의존성

---

#### Approach 2: Sequential Multi-Stage Pipeline (권장)

**스크립트:** `Inpainting/approach2_sequential.py`

**전략:** 시계열 누적 → 기하학적 가이드 → AI 생성

**Process:**

**Step 1: Temporal Accumulation** (`step1_temporal_accumulation.py`)
- 여러 프레임의 정적 배경을 3D로 누적
- Voxel downsampling (5cm)
- 기준 프레임에 재투영

**중간 출력:**
```
nre_format/step1_warped/
└── *.png  # 시계열 누적 결과 (70-85% 복원)
```

---

**Step 2: Geometric Guide** (`step2_geometric_guide.py`)
- Step 1 실패 영역 탐지
- RANSAC 평면 추정
- 구멍 영역 depth 예측

**중간 출력:**
```
nre_format/
├── step2_depth_guide/
│   └── *.png  # uint16, 기하학적 depth 가이드
└── step2_hole_masks/
    └── *.png  # uint8, 최종 구멍 마스크
```

---

**Step 3: AI Inpainting** (`step3_final_inpainting.py`)
- Stable Diffusion 1.5
- ControlNet (Depth)
- LoRA (Waymo 특화, 선택)

**모델:**
- Stable Diffusion 1.5: 4GB (사전학습)
- ControlNet Depth: 1.5GB (사전학습)
- LoRA: 10MB (선택적 학습)

**중간 출력:**
```
nre_format/step3_final_inpainted/
└── *.jpg  # AI 생성 최종 결과
```

---

**실행:**
```bash
# 전체 파이프라인
python Inpainting/approach2_sequential.py \
    /path/to/nre_format \
    --voxel_size 0.05 \
    --sample_interval 5 \
    --ground_ratio 0.6 \
    --lora_path ./trained_lora.safetensors  # 선택

# 단계별 실행
python Inpainting/step1_temporal_accumulation.py --data_root /path/to/nre_format
python Inpainting/step2_geometric_guide.py --data_root /path/to/nre_format
python Inpainting/step3_final_inpainting.py --data_root /path/to/nre_format
```

**특징:**
- ✅ 빠름 (~10-30분)
- ✅ 고품질 텍스처
- ✅ 100% 완전성
- ✅ LoRA 도메인 특화
- ⚠️ GPU 필수

---

#### 최종 Output (공통)
```
nre_format/final_inpainted/
└── *.jpg
    # 동적 객체가 완전히 제거된 깨끗한 배경 이미지
    # NeRF/3DGS 학습, Simulation, Data Augmentation에 활용
```

**상세 문서:** [Inpainting/README.md](Inpainting/README.md)

---

## 🚀 빠른 시작

### 전체 파이프라인 한 번에 실행

```bash
#!/bin/bash
# complete_pipeline.sh

DATA_ROOT="/path/to/nre_format"
WAYMO_RAW="/path/to/waymo_raw"

# Stage 1: Parsing
python parsing/waymo2nre.py \
    $WAYMO_RAW \
    $DATA_ROOT \
    --prefix seq0_

# Stage 2: Preprocessing
python preprocessing/run_preprocessing.py \
    $DATA_ROOT \
    --all

# Stage 3: Inpainting (Approach 2)
python Inpainting/approach2_sequential.py \
    $DATA_ROOT \
    --voxel_size 0.05 \
    --sample_interval 5

echo "Pipeline complete! Check $DATA_ROOT/final_inpainted/"
```

---

## 📖 상세 가이드

### 환경 설정

```bash
# 기본 패키지
pip install numpy opencv-python tqdm pillow

# Waymo
pip install waymo-open-dataset-tf-2-11-0

# Preprocessing
pip install open3d scikit-learn

# Inpainting (Approach 2)
pip install torch torchvision diffusers transformers accelerate

# COLMAP (Approach 1)
sudo apt-get install colmap  # Ubuntu
brew install colmap          # macOS
```

---

### 모델 다운로드 (Approach 2 Step 3)

**자동 다운로드 (권장):**
```python
# 첫 실행 시 자동 다운로드됨 (~15-20분)
python Inpainting/approach2_sequential.py /path/to/data
```

**수동 다운로드:**
```python
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

# SD 1.5 (~4GB)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# ControlNet Depth (~1.5GB)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16
)
```

---

### LoRA 학습 (선택적)

```bash
# 방법 1: 통합 학습 스크립트 (권장)
python Inpainting/lora/train_style_lora.py \
    --data_root /path/to/waymo_nre_format \
    --output_dir ./lora_output \
    --trigger_word "WaymoStyle road" \
    --max_train_steps 1000 \
    --lora_rank 16

# 방법 2: Gradio UI 사용
python Inpainting/lora/lora_ui.py --port 7860
# → 브라우저에서 http://localhost:7860 접속

# 학습된 LoRA 사용
python Inpainting/approach2_sequential.py \
    /path/to/data \
    --lora_path ./lora_output/pytorch_lora_weights.safetensors

# LoRA 전/후 비교
python Inpainting/lora/lora_inference.py compare \
    --lora_path ./lora_output/pytorch_lora_weights.safetensors \
    --output_dir ./comparison
```

---

## 📁 디렉토리 구조

```
Photo-real_project/
│
├── parsing/                      # Stage 1: Parsing
│   ├── waymo2nre.py              # Waymo → NRE 변환 (권장)
│   ├── extract_waymo_data.py     # TensorFlow 버전
│   ├── extract_waymo_data_minimal.py  # TF 제거 버전
│   ├── waymo_utils.py            # 공통 유틸리티
│   └── README.md                 # 상세 가이드
│
├── preprocessing/                # Stage 2: Preprocessing
│   ├── lidar_projection.py       # LiDAR → Image 투영
│   ├── dynamic_masking.py        # 동적 객체 마스킹
│   ├── segmentation.py           # Semantic Segmentation
│   ├── run_preprocessing.py      # 통합 실행
│   └── README.md                 # 상세 가이드
│
├── Inpainting/                   # Stage 3: Inpainting
│   ├── approach1_colmap.py       # COLMAP 기반 (3D 재구성)
│   ├── approach2_sequential.py   # Sequential 통합 실행
│   ├── step1_temporal_accumulation.py  # 시계열 누적
│   ├── step2_geometric_guide.py        # 기하학적 가이드
│   ├── step3_final_inpainting.py       # AI 최종 생성
│   ├── lora/                     # Style LoRA Training Pipeline
│   │   ├── __init__.py           # 패키지 초기화
│   │   ├── train_style_lora.py   # LoRA 학습 스크립트
│   │   ├── lora_inference.py     # 추론 & 품질 평가
│   │   ├── lora_ui.py            # Gradio 기반 통합 UI
│   │   └── training_dataset_builder.py  # 학습 데이터셋 빌더
│   └── README.md                 # 상세 가이드 (모델 정보 포함)
│
├── download/
│   └── download_waymo.py         # Waymo 다운로드
│
├── dataset.py                    # 데이터셋 관리
├── reconstruction.py             # 3D 재구성 (USD 변환)
└── README.md                     # 📖 이 문서
```

---

## 📊 성능 비교

| Stage | Approach | 처리 시간 (100 frames) | 메모리 | 특징 |
|-------|----------|----------------------|--------|------|
| **Parsing** | waymo2nre | ~5분 | 2GB RAM | TF 불필요 |
| **Preprocessing** | LiDAR + Masking | ~10분 | 4GB RAM | GPU 선택 |
| **Inpainting** | COLMAP | 1-5시간 | 8-16GB RAM | Multi-view 일관성 |
| **Inpainting** | Sequential | 10-30분 | 6GB VRAM | 빠르고 고품질 |

---

## 🎯 사용 사례별 권장 Workflow

### 1. NeRF/3DGS 학습용 데이터 준비
```bash
# Approach 2 Sequential (빠르고 고품질)
parsing/waymo2nre.py → preprocessing/run_preprocessing.py → Inpainting/approach2_sequential.py
```
**이유:** 빠른 처리, 고품질 텍스처, 2D 이미지만 필요

---

### 2. 3D 재구성 + Novel View Synthesis
```bash
# Approach 1 COLMAP (기하학적 정확도)
parsing/waymo2nre.py → preprocessing/run_preprocessing.py → Inpainting/approach1_colmap.py
```
**이유:** Multi-view consistency, 3D 모델 활용 가능

---

### 3. 대규모 데이터 증강
```bash
# Approach 2 + LoRA 학습
1. 소량 데이터로 LoRA 학습
2. 전체 데이터에 적용
```
**이유:** 도메인 특화로 품질 향상, 대규모 처리 효율적

---

## 📝 추가 리소스

- **Parsing 상세:** [parsing/README.md](parsing/README.md)
- **Preprocessing 상세:** [preprocessing/README.md](preprocessing/README.md)
- **Inpainting 상세 (모델 포함):** [Inpainting/README.md](Inpainting/README.md)
- **Waymo 변환 가이드:** [README_WAYMO_CONVERSION.md](README_WAYMO_CONVERSION.md)

---

## 🤝 기여 및 문의

- GitHub Issues: https://github.com/DaejunKang/Spatial-AI/issues
- 문서 업데이트: Pull Request 환영

---

**최종 업데이트:** 2026-02-09  
**버전:** 2.1

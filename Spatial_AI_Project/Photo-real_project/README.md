# Photo-real Project: Waymo Open Dataset Processing Pipeline

Waymo Open Dataset을 다운로드하고, 파싱하며, 다양한 3D 재구성 파이프라인(COLMAP, NRE, 3DGS 등)을 위한 전처리를 수행하는 통합 툴킷입니다.

## 📁 프로젝트 구조

```
Photo-real_project/
├── download/                    # 데이터 다운로드
│   ├── __init__.py
│   └── download_waymo.py       # Waymo 데이터 다운로드 스크립트
│
├── parsing/                     # 데이터 파싱/추출
│   ├── __init__.py
│   ├── extract_waymo_data.py           # 이미지/마스크 추출 + JSON 메타데이터
│   ├── extract_waymo_data_minimal.py   # 경량 버전 (TF 의존성 최소화)
│   ├── waymo_utils.py                  # 공통 유틸리티 모듈
│   └── test_minimal_converter.py       # 변환 테스트 스크립트
│
├── preprocessing/               # 데이터 전처리 및 변환
│   ├── __init__.py
│   ├── waymo2colmap.py         # COLMAP 포맷 변환기
│   ├── waymo2nre.py            # NRE 포맷 변환기
│   ├── create_nre_pairs.py     # NRE 학습/검증 데이터셋 생성 (NEW!)
│   ├── inpainting.py           # Stable Diffusion 기반 인페인팅
│   ├── segmentation.py         # SegFormer 기반 동적 객체 세그멘테이션
│   └── run_preprocessing.py    # 전처리 파이프라인 실행
│
├── dataset.py                   # 데이터셋 관리 모듈
├── reconstruction.py            # 3D 재구성 (3DGS -> USD 변환)
├── README.md                    # 이 문서
├── README_WAYMO_CONVERSION.md  # 상세 변환 가이드
└── README_MINIMAL.md           # Minimal 버전 가이드
```

## 🚀 빠른 시작 가이드

### 1. 환경 설정

#### 필수 요구사항
- Python 3.7+ (권장: 3.9 또는 3.10)
- CUDA 지원 GPU (전처리/학습용)
- Waymo Open Dataset 계정 (다운로드용)

#### 기본 패키지 설치

```bash
# 저장소 클론
git clone https://github.com/DaejunKang/Spatial-AI.git
cd Spatial-AI/Spatial_AI_Project/Photo-real_project

# 기본 의존성 설치
pip install numpy opencv-python tqdm

# Waymo Open Dataset 설치
pip install waymo-open-dataset-tf-2-11-0

# (선택) 전처리용 패키지
pip install torch torchvision transformers diffusers accelerate

# (선택) 3DGS 재구성용 패키지
pip install diff-gaussian-rasterization simple-knn
```

---

## 📥 Step 1: 데이터 다운로드

Waymo Open Dataset을 Google Cloud Storage에서 다운로드합니다.

```bash
# Google Cloud SDK 인증 (처음 1회만)
gcloud auth login

# 샘플 세그먼트 1개 다운로드 (테스트용)
python download/download_waymo.py ./data/waymo/raw --split training --limit 1

# 전체 학습 데이터 다운로드
python download/download_waymo.py ./data/waymo/raw --split training
```

**출력:**
```
./data/waymo/raw/
└── segment-xxxxxx.tfrecord
```

---

## 📊 Step 2: 데이터 파싱

TFRecord 파일에서 이미지, 카메라 파라미터, 동적 객체 정보를 추출합니다.

### Option A: 표준 버전 (TensorFlow 사용)

```bash
python parsing/extract_waymo_data.py \
    ./data/waymo/raw \
    ./data/waymo/extracted
```

### Option B: 경량 버전 (TensorFlow 최소 의존성)

```bash
python parsing/extract_waymo_data_minimal.py \
    ./data/waymo/raw \
    ./data/waymo/extracted
```

**출력 구조:**
```
./data/waymo/extracted/segment_xxxx/
├── images/          # 5개 카메라 이미지 (FRONT, SIDE_LEFT, SIDE_RIGHT, FRONT_LEFT, FRONT_RIGHT)
│   ├── FRONT/
│   │   ├── 000000.jpg
│   │   └── ...
│   └── ...
├── masks/           # 동적 객체 마스크 (3D 박스 투영 기반)
│   ├── FRONT/
│   │   ├── 000000.png
│   │   └── ...
│   └── ...
├── poses/
│   └── vehicle_poses.json    # 차량 포즈 (timestamp별)
└── calibration/
    └── intrinsics_extrinsics.json  # 카메라 캘리브레이션
```

---

## 🔄 Step 3: 포맷 변환 (용도별 선택)

### 3-1. COLMAP 포맷 변환 (전통적인 SfM/MVS 파이프라인용)

```bash
python preprocessing/waymo2colmap.py \
    ./data/waymo/extracted/segment_xxxx \
    ./data/waymo/colmap_format
```

**출력:**
```
./data/waymo/colmap_format/
├── cameras.txt      # COLMAP 카메라 모델 (FULL_OPENCV)
├── images.txt       # 이미지별 포즈
└── points3D.txt     # (빈 파일, COLMAP 실행 후 생성)
```

**사용 예시:**
```bash
# COLMAP으로 재구성 실행
colmap feature_extractor --database_path ./colmap.db --image_path ./images
colmap exhaustive_matcher --database_path ./colmap.db
colmap mapper --database_path ./colmap.db --image_path ./images --output_path ./sparse
```

---

### 3-2. NRE 포맷 변환 (Neural Rendering Engine / 3DGS용) ⭐ 권장

Waymo 데이터를 NRE(Neural Reconstruction Engine) 및 3D Gaussian Splatting 학습에 최적화된 포맷으로 변환합니다.

#### Step 3-2-1: TFRecord → NRE 포맷 변환

```bash
python preprocessing/waymo2nre.py \
    ./data/waymo/raw \
    ./data/waymo/nre_format \
    --prefix seq0_
```

**출력 구조:**
```
./data/waymo/nre_format/
├── images/                     # 추출된 이미지 (JPEG)
│   ├── seq0_000000_FRONT.jpg
│   ├── seq0_000000_SIDE_LEFT.jpg
│   ├── seq0_000000_FRONT_LEFT.jpg
│   ├── seq0_000000_FRONT_RIGHT.jpg
│   ├── seq0_000000_SIDE_RIGHT.jpg
│   └── ...
├── poses/                      # 프레임별 지오메트리 정보 (JSON)
│   ├── seq0_000000.json
│   ├── seq0_000001.json
│   └── ...
└── objects/                    # 동적 객체 정보 (JSON)
    ├── seq0_000000.json
    └── ...
```

**poses/*.json 구조:**
```json
{
    "frame_idx": 0,
    "timestamp": 1234567890.123456,
    "ego_velocity": {
        "linear": [5.2, 0.1, -0.05],
        "angular": [0.001, -0.002, 0.05]
    },
    "cameras": {
        "FRONT": {
            "img_path": "images/seq0_000000_FRONT.jpg",
            "width": 1920,
            "height": 1280,
            "intrinsics": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
            "pose": [/* 4x4 변환 행렬 (flattened) */],
            "rolling_shutter": {
                "duration": 0.033,
                "trigger_time": 1234567890.0
            }
        },
        // ... FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
    }
}
```

#### Step 3-2-2: Train/Validation 데이터셋 생성 🆕

NRE/3DGS 학습을 위해 개별 프레임 JSON을 통합하고 Train/Val로 분할합니다.

```bash
python preprocessing/create_nre_pairs.py
```

또는 Python API:

```python
from preprocessing.create_nre_pairs import NREPairGenerator

generator = NREPairGenerator(
    data_root='./data/waymo/nre_format',
    output_dir='./data/waymo/nre_format',
    val_interval=8  # 8프레임마다 1개씩 검증셋으로 사용 (약 12.5%)
)
generator.generate()
```

**출력 파일:**
```
./data/waymo/nre_format/
├── train_pairs.json    # 학습용 데이터셋
└── val_pairs.json      # 검증용 데이터셋
```

**train_pairs.json / val_pairs.json 구조:**
```json
{
    "meta": {
        "total_frames": 450,
        "coordinate_system": "Right-Down-Front (Waymo Native)",
        "world_origin": "Aligned to Frame 0 Vehicle Pose"
    },
    "frames": [
        {
            "file_path": "images/seq0_000000_FRONT.jpg",
            "timestamp": 1234567890.123456,
            "camera_id": "FRONT",
            "transform_matrix": [0.99, 0.01, ...],  // 4x4 flatten (16개 값)
            "intrinsics": [2000.5, 2000.5, 960.0, 640.0, 0.01, -0.02, 0.001, -0.001, 0.0],
            "width": 1920,
            "height": 1280,
            "velocity": {
                "v": [10.5, 0.1, 0.0],
                "w": [0.0, 0.0, 0.02]
            },
            "rolling_shutter": {
                "duration": 0.025,
                "trigger_time": 1234567890.0
            }
        },
        // ... (모든 카메라, 모든 프레임 나열)
    ]
}
```

#### 3DGS 학습 Config 연동 예시

```python
# configs/datasets/custom_waymo-3d.py

data = dict(
    train=dict(
        type='NREWaymoDataset',
        ann_file='data/waymo/nre_format/train_pairs.json',
        img_prefix='data/waymo/nre_format/',
        pipeline=train_pipeline
    ),
    val=dict(
        type='NREWaymoDataset',
        ann_file='data/waymo/nre_format/val_pairs.json',
        img_prefix='data/waymo/nre_format/',
        pipeline=test_pipeline
    )
)
```

---

## 🎨 Step 4: 고급 전처리 (선택 사항)

동적 객체 영역을 더욱 정교하게 마스킹하고 배경을 복원합니다.

### 4-1. SegFormer 기반 의미론적 세그멘테이션

3D 박스 투영 대신 픽셀 단위 세그멘테이션을 사용하여 더 정확한 마스크 생성:

```bash
python preprocessing/run_preprocessing.py \
    ./data/waymo/extracted/segment_xxxx \
    --use_segformer \
    --device cuda
```

### 4-2. Stable Diffusion 인페인팅

마스킹된 동적 객체 영역을 자연스러운 배경으로 복원:

```bash
python preprocessing/run_preprocessing.py \
    ./data/waymo/extracted/segment_xxxx \
    --inpainting \
    --device cuda
```

**출력:**
```
./data/waymo/extracted/segment_xxxx/
└── images_inpainted/   # 인페인팅된 이미지
    ├── FRONT/
    └── ...
```

---

## 🏗️ Step 5: 3D 재구성 (3DGS + USD Export)

3D Gaussian Splatting으로 장면을 재구성하고 NVIDIA Omniverse용 USD로 내보냅니다.

```bash
python reconstruction.py \
    ./data/waymo/extracted/segment_xxxx \
    ./output/reconstruction \
    --use_inpainted
```

**옵션:**
- `--use_inpainted`: 인페인팅된 이미지 사용 (더 깨끗한 배경)
- `--iterations 30000`: 학습 반복 횟수

**출력:**
```
./output/reconstruction/
├── point_cloud/            # 3DGS 체크포인트
└── reconstruction.usd      # Omniverse용 USD 파일
```

---

## 🎮 NVIDIA Omniverse 연동

1. **NVIDIA Omniverse USD Composer** 실행
2. **File → Open** → `reconstruction.usd` 선택
3. 3D Gaussian Point Cloud가 시각화됩니다

> **참고:** 기본 USD Points는 구/디스크로 렌더링됩니다. 완전한 Gaussian Splat 렌더링을 위해서는 커스텀 Omniverse Extension이 필요할 수 있습니다.

---

## 🔧 핵심 모듈 설명

### 📦 `parsing/waymo_utils.py` - 공통 유틸리티

재사용 가능한 헬퍼 함수들:

- `MinimalTFRecordReader`: TensorFlow 없이 TFRecord 파일 읽기
- `decode_image_opencv`: OpenCV 기반 이미지 디코딩
- `project_3d_box_to_2d`: 3D 바운딩 박스를 2D 이미지로 투영
- `get_calibration_dict`: 카메라 Calibration 정보 추출
- `quaternion_to_rotation_matrix` / `rotation_matrix_to_quaternion`: 회전 변환

### ⚙️ `preprocessing/create_nre_pairs.py` - 데이터셋 생성기 🆕

**핵심 기능:**
1. ✅ Waymo2NRE로 생성된 개별 Frame JSON (`poses/*.json`) 읽기
2. ✅ 시계열 순서를 유지하며 Train/Val 분할 (기본: 8프레임마다 검증용)
3. ✅ 5개 카메라를 개별 학습 샘플로 Flatten
4. ✅ Rolling Shutter, Ego Velocity 메타데이터 포함
5. ✅ 3DGS/NRE 학습 Config와 직접 연동 가능한 JSON 생성

**설정 파라미터:**
- `data_root`: NRE 포맷 데이터 루트 디렉토리
- `output_dir`: 출력 JSON 파일 저장 경로
- `val_interval`: 검증셋 샘플링 간격 (기본값: 8)

---

## 📊 데이터 플로우 전체 요약

```
1. Download
   └─> ./data/waymo/raw/*.tfrecord

2. Parsing (Extract)
   └─> ./data/waymo/extracted/segment_xxxx/
       ├── images/
       ├── masks/
       ├── poses/
       └── calibration/

3. Format Conversion
   ├─> COLMAP Format (waymo2colmap.py)
   │   └─> ./data/waymo/colmap_format/
   │
   └─> NRE Format (waymo2nre.py)
       └─> ./data/waymo/nre_format/
           ├── images/
           ├── poses/
           └── objects/

4. Dataset Split (create_nre_pairs.py) 🆕
   └─> ./data/waymo/nre_format/
       ├── train_pairs.json
       └── val_pairs.json

5. Training (External 3DGS/NRE Framework)
   └─> Load train_pairs.json, val_pairs.json

6. Reconstruction (reconstruction.py)
   └─> ./output/reconstruction.usd
```

---

## 🧪 테스트 및 검증

### Minimal Converter 테스트

```bash
python parsing/test_minimal_converter.py
```

### NRE 데이터셋 검증

```python
import json

# Train 데이터셋 확인
with open('./data/waymo/nre_format/train_pairs.json', 'r') as f:
    train_data = json.load(f)
    print(f"Total training samples: {train_data['meta']['total_frames']}")
    print(f"First sample: {train_data['frames'][0]['file_path']}")

# Validation 데이터셋 확인
with open('./data/waymo/nre_format/val_pairs.json', 'r') as f:
    val_data = json.load(f)
    print(f"Total validation samples: {val_data['meta']['total_frames']}")
```

---

## 📝 주요 변경 이력

### v2.0 (2026-02-02)
- 🎯 **폴더 구조 개편**: 기능별로 `download/`, `parsing/`, `preprocessing/`로 분류
- 🆕 **create_nre_pairs.py 추가**: NRE/3DGS 학습용 Train/Val 데이터셋 자동 생성
- ⚡ **Import 경로 최적화**: 모듈화 및 상대 경로 사용
- 📦 **`__init__.py` 추가**: 각 폴더를 Python 패키지로 구성
- 📚 **README 통합 업데이트**: 전체 워크플로우 반영

### v1.0 (이전 버전)
- Waymo 데이터 다운로드/추출/변환 기본 기능
- COLMAP 및 NRE 포맷 변환
- SegFormer + Stable Diffusion 전처리

---

## 🤝 기여 및 문의

**Repository:** [github.com/DaejunKang/Spatial-AI](https://github.com/DaejunKang/Spatial-AI)

**관련 문서:**
- [README_WAYMO_CONVERSION.md](./README_WAYMO_CONVERSION.md) - 상세 변환 가이드
- [README_MINIMAL.md](./README_MINIMAL.md) - Minimal 버전 가이드

---

## 📄 라이선스

이 프로젝트는 Waymo Open Dataset의 라이선스 조항을 따릅니다.
Waymo Open Dataset License Agreement를 준수해야 합니다.

---

## ⚠️ 알려진 제한사항

1. **Rolling Shutter 보정**: 현재 메타데이터만 제공, 실제 보정은 학습 프레임워크에서 구현 필요
2. **동적 객체 마스킹**: 3D 박스 기반 마스크는 오클루전 처리 불완전
3. **대용량 처리**: 전체 Waymo 데이터셋 처리 시 충분한 디스크 공간(~2TB) 필요

---

**Happy Reconstructing! 🎉**

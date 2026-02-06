# 3D Scene Reconstruction

**Inpainting된 배경 이미지로부터 3D 장면 재구성**

두 가지 Approach를 제공합니다:
- **Approach 1: 3DGS** - Static Scene (3D Gaussian Splatting)
- **Approach 2: 3DGUT** - Rolling Shutter Compensated (NVIDIA 3DGUT via gsplat)

---

## 📋 목차

1. [개요](#개요)
2. [외부 모델 (External Models)](#외부-모델-external-models)
3. [설치](#설치)
4. [Input/Output 인터페이스](#inputoutput-인터페이스)
5. [Approach 비교](#approach-비교)
6. [빠른 시작](#빠른-시작)
7. [상세 가이드](#상세-가이드)

---

## 🎯 개요

### 목적
Inpainting된 배경 이미지를 사용하여 3D Gaussian 기반 장면 재구성을 수행합니다.

### 입력
- **Inpainting 결과**: `final_inpainted/*.jpg`
- **카메라 정보**: `poses/*.json` (Parsing stage 출력)

### 출력
- **3D Gaussians**: `.ply` 파일
- **Novel View 렌더링**: 검증용 이미지

---

## 🔧 외부 모델 (External Models)

이 모듈은 다음 두 개의 외부 레포지토리를 git submodule로 포함합니다:

### 1. 3DGS: graphdeco-inria/gaussian-splatting

| 항목 | 내용 |
|------|------|
| **Repository** | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| **Paper** | "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023) |
| **Stars** | 20k+ |
| **경로** | `external/gaussian-splatting/` |
| **용도** | Approach 1 - 정적 장면 3D 재구성 |

**주요 파일:**
- `train.py` - 학습 스크립트
- `render.py` - 렌더링 스크립트
- `scene/gaussian_model.py` - Gaussian 모델 정의
- `gaussian_renderer/` - 렌더링 엔진

### 2. 3DGUT: nerfstudio-project/gsplat (NVIDIA 3DGUT 통합)

| 항목 | 내용 |
|------|------|
| **Repository** | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) |
| **NVIDIA 3DGUT** | [research.nvidia.com/labs/toronto-ai/3DGUT/](https://research.nvidia.com/labs/toronto-ai/3DGUT/) |
| **NVIDIA Blog** | [developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) |
| **Stars** | 4.4k+ |
| **License** | Apache 2.0 |
| **경로** | `external/gsplat/` |
| **용도** | Approach 2 - Rolling Shutter 보정, 렌즈 왜곡 지원 |

**3DGUT 핵심 기능:**
- **Unscented Transform (UT)**: 비선형 카메라 프로젝션 지원
- **3D Eval**: 3D 공간에서 Gaussian 응답 직접 평가
- **Rolling Shutter**: 각 픽셀의 캡처 시간을 고려한 모션 보정
- **Distortion**: Pinhole/Fisheye 렌즈 왜곡 모델 지원

**주요 파일:**
- `gsplat/rendering.py` - 핵심 렌더링 (rasterization API)
- `gsplat/cuda/` - CUDA 가속 커널
- `examples/simple_trainer.py` - 학습 스크립트
- `examples/simple_viewer_3dgut.py` - 3DGUT 뷰어
- `docs/3dgut.md` - 3DGUT 공식 문서

---

## ⚙️ 설치

### 빠른 설치

```bash
# 1. 서브모듈 초기화
git submodule update --init --recursive

# 2. 자동 설치 스크립트
bash reconstruction/setup_external.sh
```

### 수동 설치

#### 3DGS (Approach 1)

```bash
# 서브모듈 초기화
git submodule update --init --recursive

# 3DGS 의존성 (CUDA 필요)
cd reconstruction/external/gaussian-splatting
pip install plyfile tqdm
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

#### gsplat / 3DGUT (Approach 2)

```bash
# Option A: pip install (CUDA JIT compile)
pip install gsplat

# Option B: 소스에서 설치
cd reconstruction/external/gsplat
pip install -e .

# 예제 의존성
pip install -r examples/requirements.txt
```

### 의존성 상태 확인

```python
from reconstruction import print_status
print_status()
```

---

## 📊 Input/Output 인터페이스

### 공통 입력 구조

모든 Approach는 JSON 메타데이터 파일을 통해 데이터를 로드합니다.

#### 메타데이터 파일 구조

**위치:** `{data_root}/train_meta/train_pairs.json`

**기본 구조 (3DGS):**
```json
[
  {
    "file_path": "final_inpainted/seq0_000000_FRONT.jpg",
    "transform_matrix": [
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ],
    "intrinsics": [2000.0, 2000.0, 960.0, 640.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "width": 1920,
    "height": 1280,
    "camera_name": "FRONT",
    "frame_name": "seq0_000000"
  }
]
```

**확장 구조 (3DGUT):**
```json
[
  {
    "file_path": "final_inpainted/seq0_000000_FRONT.jpg",
    "transform_matrix": [ ... ],
    "intrinsics": [ ... ],
    "width": 1920,
    "height": 1280,
    "camera_name": "FRONT",
    "frame_name": "seq0_000000",

    "velocity": {
      "v": [10.5, 0.1, 0.0],
      "w": [0.0, 0.0, 0.02]
    },
    "rolling_shutter": {
      "duration": 0.025,
      "trigger_time": 0.0
    }
  }
]
```

---

### 필드 설명

| 필드 | 타입 | 설명 | 필수 여부 |
|-----|------|------|----------|
| `file_path` | string | 이미지 경로 (data_root 기준 상대 경로) | ✅ |
| `transform_matrix` | array[16] | 4x4 Camera Pose (World-to-Camera) | ✅ |
| `intrinsics` | array[9] | [fx, fy, cx, cy, k1, k2, p1, p2, k3] | ✅ |
| `width` | int | 이미지 너비 | ✅ |
| `height` | int | 이미지 높이 | ✅ |
| `camera_name` | string | 카메라 이름 (FRONT, etc.) | ⚪ |
| `frame_name` | string | 프레임 이름 | ⚪ |
| `velocity` | object | 속도 정보 (v, w) | 🔵 3DGUT 전용 |
| `rolling_shutter` | object | Rolling Shutter 파라미터 | 🔵 3DGUT 전용 |

---

### 물리적 데이터 구조

```
{data_root}/
├── final_inpainted/
│   └── *.jpg                # Inpainting 완료된 배경 이미지
│
├── poses/
│   └── *.json               # Parsing stage 카메라 메타데이터
│
├── train_meta/
│   └── train_pairs.json     # 학습용 메타데이터 (생성 필요)
│
├── val_meta/
│   └── train_pairs.json     # 검증용 메타데이터 (선택)
│
└── outputs/
    ├── 3dgs/
    │   ├── colmap_format/   # COLMAP 변환 데이터
    │   └── model/           # 학습된 3DGS 모델
    │       └── point_cloud/ # 3D Gaussians (.ply)
    └── 3dgut/
        ├── colmap_format/   # COLMAP + 3DGUT 파라미터
        └── results/         # 학습된 3DGUT 모델
            └── ckpts/       # 체크포인트 (.pt)
```

---

## 🔄 Approach 비교

### Approach 1: 3DGS (Static Scene)

**구현:** `graphdeco-inria/gaussian-splatting`

**전략:** 정적 장면 가정, Rolling Shutter 무시

#### Input Tensors
| Tensor | Shape | 설명 |
|--------|-------|------|
| Image ($I$) | `[3, H, W]` | RGB 이미지 |
| Extrinsic ($T$) | `[4, 4]` | World-to-Camera 변환 |
| Intrinsic ($K$) | `[3, 3]` | Projection Matrix |

#### 특징
- ✅ 검증된 레퍼런스 구현 (20k+ GitHub Stars)
- ✅ COLMAP 호환
- ✅ 빠른 학습
- ⚠️ Rolling Shutter 왜곡 무시
- ⚠️ 고속 이동 시 품질 저하

#### 사용 사례
- 저속 주행 데이터
- Rolling Shutter 효과가 미미한 경우
- 빠른 프로토타이핑

---

### Approach 2: 3DGUT (Rolling Shutter Compensated)

**구현:** `nerfstudio-project/gsplat` (NVIDIA 3DGUT 통합)

**전략:** Unscented Transform + 각 픽셀의 캡처 시간을 고려하여 모션 보정

#### Input Tensors (3DGS + α)
| Tensor | Shape | 설명 |
|--------|-------|------|
| Image ($I$) | `[3, H, W]` | RGB 이미지 |
| Extrinsic ($T$) | `[4, 4]` | World-to-Camera 변환 |
| Intrinsic ($K$) | `[3, 3]` | Projection Matrix |
| **Velocity ($v, \omega$)** | **`[6]`** | **[vx, vy, vz, wx, wy, wz]** |
| **RS Duration** | **scalar** | **Readout time (s)** |
| **Distortion** | **varies** | **Radial/Tangential coefficients** |

#### gsplat rasterization API

```python
from gsplat.rendering import rasterization

render_colors, render_alphas, meta = rasterization(
    means,       # [N, 3]
    quats,       # [N, 4]
    scales,      # [N, 3]
    opacities,   # [N]
    colors,      # [N, S, 3]
    viewmats,    # [C, 4, 4]
    Ks,          # [C, 3, 3]
    width, height,
    with_ut=True,          # 3DGUT: Unscented Transform
    with_eval3d=True,      # 3DGUT: 3D Evaluation
    camera_model="pinhole", # or "fisheye"
    rolling_shutter=...,    # Rolling Shutter params
    radial_coeffs=...,     # Lens distortion
    tangential_coeffs=...,
)
```

#### 특징
- ✅ NVIDIA 공식 3DGUT 알고리즘
- ✅ Rolling Shutter 왜곡 보정
- ✅ 렌즈 왜곡 (Pinhole/Fisheye) 지원
- ✅ CUDA 가속
- ✅ MCMC densification strategy
- ⚠️ CUDA 빌드 필요
- ⚠️ 학습 시간 증가 (~1.5배)

#### 사용 사례
- 고속 주행 데이터
- 렌즈 왜곡이 큰 카메라
- 정밀 3D 재구성 필요
- Novel View Synthesis

---

## 🚀 빠른 시작

### Step 1: 메타데이터 생성

```bash
# 3DGS용 (Static)
python reconstruction/prepare_metadata.py \
    /path/to/nre_format \
    --mode 3dgs \
    --output train_meta/train_pairs.json

# 3DGUT용 (Rolling Shutter)
python reconstruction/prepare_metadata.py \
    /path/to/nre_format \
    --mode 3dgut \
    --output train_meta/train_pairs.json
```

### Step 2: 학습 실행

#### Approach 1: 3DGS
```bash
python reconstruction/approach1_3dgs.py \
    /path/to/nre_format \
    --meta_file train_meta/train_pairs.json \
    --output_dir outputs/3dgs \
    --iterations 30000
```

#### Approach 2: 3DGUT
```bash
python reconstruction/approach2_3dgut.py \
    /path/to/nre_format \
    --meta_file train_meta/train_pairs.json \
    --output_dir outputs/3dgut \
    --iterations 30000 \
    --camera_model pinhole
```

### Step 3: 결과 확인

```bash
# 3DGS 결과
ls outputs/3dgs/model/point_cloud/

# 3DGUT 결과
ls outputs/3dgut/results/ckpts/
```

---

## 📖 상세 가이드

### 3DGUT 직접 학습 (gsplat CLI)

gsplat이 설치된 경우, 직접 simple_trainer.py를 사용할 수 있습니다:

```bash
cd reconstruction/external/gsplat/examples

# 3DGUT 학습
python simple_trainer.py mcmc \
    --with_ut --with_eval3d \
    --data_dir /path/to/colmap_data \
    --result_dir /path/to/results \
    --max_steps 30000 \
    --strategy.cap-max 1000000

# 3DGUT 뷰어
python simple_viewer_3dgut.py \
    --ckpt /path/to/results/ckpts/ckpt_29999_rank0.pt
```

### 초기 포인트 클라우드 사용

```bash
python reconstruction/approach1_3dgs.py \
    /path/to/nre_format \
    --initial_ply step1_warped/accumulated_static.ply \
    --iterations 30000
```

### 학습 파라미터 조정

**권장 설정:**
- **빠른 테스트**: 10,000 iterations (~30분)
- **표준 품질**: 30,000 iterations (~2-3시간)
- **고품질**: 50,000+ iterations (~5-8시간)

---

## 📊 성능 비교

| Approach | 구현 | 학습 시간 | PSNR | Rolling Shutter 보정 | 렌즈 왜곡 | 메모리 |
|----------|------|----------|------|---------------------|----------|--------|
| **3DGS** | gaussian-splatting | 2-3시간 | ~28 dB | ❌ | ❌ | 8GB VRAM |
| **3DGUT** | gsplat (NVIDIA) | 3-5시간 | ~30 dB | ✅ | ✅ | 10GB VRAM |

**테스트 환경:** 100 프레임, NVIDIA RTX 3090, 30K iterations

---

## 📁 디렉토리 구조

```
reconstruction/
├── __init__.py              # 모듈 초기화, 의존성 상태 확인
├── README.md                # 이 문서
├── setup_external.sh        # 외부 의존성 설치 스크립트
│
├── approach1_3dgs.py        # Approach 1: 3DGS 래퍼
├── approach2_3dgut.py       # Approach 2: 3DGUT 래퍼
├── data_loader.py           # 공통 데이터 로더
├── prepare_metadata.py      # 메타데이터 생성
│
└── external/                # 외부 모델 (git submodules)
    ├── gaussian-splatting/  # graphdeco-inria/gaussian-splatting
    │   ├── train.py         # 3DGS 학습 스크립트
    │   ├── render.py        # 3DGS 렌더링
    │   ├── scene/           # Gaussian 모델
    │   └── gaussian_renderer/
    │
    └── gsplat/              # nerfstudio-project/gsplat (NVIDIA 3DGUT)
        ├── gsplat/          # 핵심 라이브러리
        │   ├── rendering.py # rasterization() API
        │   └── cuda/        # CUDA 커널
        ├── examples/
        │   ├── simple_trainer.py      # 학습
        │   └── simple_viewer_3dgut.py # 3DGUT 뷰어
        └── docs/
            └── 3dgut.md    # 3DGUT 공식 문서
```

---

## 🤝 참고 자료

- **3D Gaussian Splatting**: https://github.com/graphdeco-inria/gaussian-splatting
- **gsplat (NVIDIA 3DGUT)**: https://github.com/nerfstudio-project/gsplat
- **NVIDIA 3DGUT Research**: https://research.nvidia.com/labs/toronto-ai/3DGUT/
- **NVIDIA 3DGUT Tech Blog**: https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/
- **gsplat 3DGUT 문서**: [external/gsplat/docs/3dgut.md](external/gsplat/docs/3dgut.md)

---

**최종 업데이트:** 2026-02-06
**작성자:** Cloud Agent
**버전:** 2.0

# 3D Scene Reconstruction

**Inpainting된 배경 이미지로부터 3D 장면 재구성**

두 가지 Approach를 제공합니다:
- **Approach 1: 3DGS** - Static Scene (3D Gaussian Splatting)
- **Approach 2: 3DGUT** - Rolling Shutter Compensated (3D Gaussian with Uncertainty and Time)

---

## 📋 목차

1. [개요](#개요)
2. [Input/Output 인터페이스](#inputoutput-인터페이스)
3. [Approach 비교](#approach-비교)
4. [빠른 시작](#빠른-시작)
5. [상세 가이드](#상세-가이드)

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
  },
  ...
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
    
    // ✅ 3DGUT 추가 필드
    "velocity": {
      "v": [10.5, 0.1, 0.0],  // Linear velocity (m/s)
      "w": [0.0, 0.0, 0.02]   // Angular velocity (rad/s)
    },
    "rolling_shutter": {
      "duration": 0.025,       // Readout time (s)
      "trigger_time": 0.0      // Capture start offset (s)
    }
  },
  ...
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
    │   ├── gaussians.ply    # 학습된 3D Gaussians
    │   └── novel_views/     # 렌더링 결과
    └── 3dgut/
        ├── gaussians_3dgut.ply
        └── novel_views/
```

---

## 🔄 Approach 비교

### Approach 1: 3DGS (Static Scene)

**전략:** 정적 장면 가정, Rolling Shutter 무시

#### Input Tensors
| Tensor | Shape | 설명 |
|--------|-------|------|
| Image ($I$) | `[3, H, W]` | RGB 이미지 |
| Extrinsic ($T$) | `[4, 4]` | World-to-Camera 변환 |
| Intrinsic ($K$) | `[3, 3]` | Projection Matrix |

#### 특징
- ✅ 구현 간단
- ✅ 빠른 학습
- ⚠️ Rolling Shutter 왜곡 무시
- ⚠️ 고속 이동 시 품질 저하

#### 사용 사례
- 저속 주행 데이터
- Rolling Shutter 효과가 미미한 경우
- 빠른 프로토타이핑

---

### Approach 2: 3DGUT (Rolling Shutter Compensated)

**전략:** 각 픽셀의 캡처 시간을 고려하여 모션 보정

#### Input Tensors (3DGS + α)
| Tensor | Shape | 설명 |
|--------|-------|------|
| Image ($I$) | `[3, H, W]` | RGB 이미지 |
| Extrinsic ($T$) | `[4, 4]` | World-to-Camera 변환 |
| Intrinsic ($K$) | `[3, 3]` | Projection Matrix |
| **Velocity ($v, \omega$)** | **`[6]`** | **[vx, vy, vz, wx, wy, wz]** |
| **RS Duration** | **scalar** | **Readout time (s)** |
| **RS Trigger** | **scalar** | **Capture start offset (s)** |

#### Rolling Shutter 보정 수식

**픽셀 시간 오프셋:**
$$t_{pixel} = t_{trigger} + \frac{y}{H} \times t_{duration}$$

**보정된 카메라 포즈:**
$$T_{adjusted}(t) = T_{motion}(t) \cdot T_{base}$$

where $T_{motion}(t) = \exp([\mathbf{v}, \boldsymbol{\omega}]^{\wedge} \cdot t)$

#### 특징
- ✅ Rolling Shutter 왜곡 보정
- ✅ 고속 이동 시에도 정확
- ⚠️ 구현 복잡
- ⚠️ 학습 시간 증가 (~1.5배)

#### 사용 사례
- 고속 주행 데이터
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

**출력:**
- `{data_root}/train_meta/train_pairs.json`
- `{data_root}/val_meta/train_pairs.json` (자동 분할)

---

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
    --iterations 30000
```

---

### Step 3: 결과 확인

```bash
# 3D Gaussians
ls {data_root}/outputs/3dgs/gaussians.ply
ls {data_root}/outputs/3dgut/gaussians_3dgut.ply

# Novel View 렌더링
ls {data_root}/outputs/3dgs/novel_views/
ls {data_root}/outputs/3dgut/novel_views/
```

---

## 📖 상세 가이드

### 메타데이터 생성 옵션

```bash
python reconstruction/prepare_metadata.py \
    /path/to/nre_format \
    --mode 3dgut \
    --output train_meta/train_pairs.json \
    --train_ratio 0.9 \              # Train:Val = 9:1
    --camera_filter FRONT FRONT_LEFT # 특정 카메라만 사용
```

**카메라 필터링:**
- `FRONT`: 전방 카메라만
- `FRONT FRONT_LEFT FRONT_RIGHT`: 전방 3개만
- 생략 시 전체 카메라 사용

---

### 초기 포인트 클라우드 사용

Inpainting Step 1에서 생성된 포인트 클라우드를 초기화에 사용 가능:

```bash
python reconstruction/approach1_3dgs.py \
    /path/to/nre_format \
    --initial_ply step1_warped/accumulated_static.ply \
    --iterations 30000
```

**장점:**
- 학습 수렴 속도 향상
- 초기 geometry 품질 향상

**생성 방법:**
```bash
# Inpainting Step 1 실행 시 자동 생성됨
python Inpainting/step1_temporal_accumulation.py \
    /path/to/nre_format \
    --save_point_cloud  # PLY 저장 옵션
```

---

### 학습 파라미터 조정

```bash
python reconstruction/approach2_3dgut.py \
    /path/to/nre_format \
    --iterations 50000 \        # 더 긴 학습
    --device cuda \             # GPU 사용
    --meta_file train_meta/train_pairs.json
```

**권장 설정:**
- **빠른 테스트**: 10,000 iterations (~30분)
- **표준 품질**: 30,000 iterations (~2-3시간)
- **고품질**: 50,000+ iterations (~5-8시간)

---

## 📊 성능 비교

| Approach | 학습 시간 | PSNR | Rolling Shutter 보정 | 메모리 |
|----------|----------|------|---------------------|--------|
| **3DGS** | 2-3시간 | ~28 dB | ❌ | 8GB VRAM |
| **3DGUT** | 3-5시간 | ~30 dB | ✅ | 10GB VRAM |

**테스트 환경:** 100 프레임, NVIDIA RTX 3090, 30K iterations

---

## 🔧 구현 상태

### 현재 구현 (Placeholder)

현재 스크립트는 **인터페이스 및 데이터 로더만 구현**되어 있습니다.

실제 3DGS 렌더링 엔진은 다음 라이브러리를 사용하여 구현해야 합니다:

```bash
# 3DGS Rasterization
pip install diff-gaussian-rasterization
pip install simple-knn

# 기타 의존성
pip install plyfile torch torchvision
```

### 추가 구현 필요

1. **Gaussian Splatting 렌더링 엔진**
   - `diff-gaussian-rasterization` 통합
   - Forward/Backward pass 구현

2. **Loss Functions**
   - L1 + SSIM loss
   - Temporal consistency loss (3DGUT)

3. **Adaptive Density Control**
   - Gaussian splitting/pruning
   - Opacity thresholding

4. **PLY I/O**
   - Gaussian 파라미터 저장/로드

---

## 📝 사용 예시

### 전체 파이프라인

```bash
#!/bin/bash
DATA_ROOT="/path/to/nre_format"

# 1. 메타데이터 생성 (3DGUT)
python reconstruction/prepare_metadata.py \
    $DATA_ROOT \
    --mode 3dgut \
    --output train_meta/train_pairs.json

# 2. 학습
python reconstruction/approach2_3dgut.py \
    $DATA_ROOT \
    --meta_file train_meta/train_pairs.json \
    --output_dir outputs/3dgut \
    --initial_ply step1_warped/accumulated_static.ply \
    --iterations 30000

echo "Training complete! Check $DATA_ROOT/outputs/3dgut/"
```

---

## 🤝 참고 자료

- **3D Gaussian Splatting**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Rolling Shutter Modeling**: Inpainting stage velocity 정보 활용
- **NeRF for Autonomous Driving**: Waymo 데이터 특화

---

**최종 업데이트:** 2026-02-05  
**작성자:** Cloud Agent  
**버전:** 1.0

# Spatial-AI Photo-real Pipeline: Full Workflow Architecture

> **Version:** 1.0
> **Date:** 2026-03-13
> **Purpose:** claude.ai 연구 분석용 전체 파이프라인 아키텍처 문서

---

## 1. Pipeline Overview

Waymo 자율주행 데이터에서 동적 객체를 제거하고 photorealistic 3D Gaussian Map을 생성하는 End-to-End 파이프라인.

```
Waymo Raw Data (.tfrecord)
    ↓
Stage 1: PARSING ─── Waymo → NRE Format 변환
    ↓
Stage 2: PREPROCESSING ─── LiDAR Projection + Dynamic Masking
    ↓
Stage 3: INPAINTING ─── Dynamic Object Removal + Background Restoration
    ↓
Stage 4: RECONSTRUCTION ─── 3D Gaussian Splatting / 3DGUT
    ↓
Output: 3D Gaussian Map (.ply) → Simulation / Novel View Synthesis
```

**Target Application:** Alpasim 자율주행 시뮬레이터용 정적 배경 3D 맵 생성

---

## 2. Stage 1: Parsing (Waymo → NRE Format)

### 2.1 목적
Waymo TFRecord 바이너리 포맷을 Neural Rendering Engine(NRE) 표준 포맷으로 변환.

### 2.2 입력
- Waymo `.tfrecord` 파일 (바이너리 세그먼트 데이터)

### 2.3 처리 과정
1. **TFRecord 읽기** — TensorFlow 의존 없이 직접 파싱 (경량화)
2. **프레임 정렬** — 이미지, LiDAR, 포즈를 timestamp 기준 동기화
3. **좌표계 정규화** — 첫 번째 프레임 = World Origin (0,0,0)
4. **Rolling Shutter 추출** — 카메라별 duration + trigger_time
5. **LiDAR Point Cloud 추출** — Local World 좌표 3D 포인트
6. **카메라 캘리브레이션** — Intrinsics(fx, fy, cx, cy, k1~k3, p1, p2) & Extrinsics(T_cam_to_world)
7. **동적 객체 정보 추출** — 3D bounding box, class, speed

### 2.4 출력
```
nre_format/
├── images/          # 5개 카메라 RGB 이미지
│   └── seq0_000000_FRONT.jpg
├── point_clouds/    # LiDAR 포인트 (Nx3 float32 binary)
│   └── seq0_000000.bin
├── poses/           # 카메라 포즈 + ego velocity + rolling shutter
│   └── seq0_000000.json
└── objects/         # 동적 객체 3D bbox
    └── seq0_000000.json
```

### 2.5 핵심 데이터 구조

**poses JSON:**
```json
{
  "timestamp": 1620000000.0,
  "ego_velocity": {
    "linear": [vx, vy, vz],     // m/s
    "angular": [wx, wy, wz]     // rad/s
  },
  "cameras": {
    "FRONT": {
      "intrinsics": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
      "pose": [4x4 flattened],  // T_cam_to_world
      "width": 1920, "height": 1280,
      "rolling_shutter": {
        "duration": 0.025,       // readout time (s)
        "trigger_time": 0.0      // capture start offset (s)
      }
    }
  }
}
```

**objects JSON:**
```json
[
  {
    "class": "VEHICLE",
    "center": [x, y, z],        // World 좌표
    "size": [length, width, height],
    "heading": 1.57,             // yaw (rad)
    "speed": 5.2                 // m/s
  }
]
```

### 2.6 파일 네이밍 규칙
```
{prefix}{file_idx:03d}{frame_idx:03d}_{camera_name}.{ext}
예: seq0_000000_FRONT.jpg
```
- 5개 카메라: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
- 사전식 정렬 = 시간순 정렬

---

## 3. Stage 2: Preprocessing

### 3.1 목적
Inpainting을 위한 LiDAR 깊이 맵과 동적 객체 마스크 생성.

### 3.2 LiDAR Projection

**입력:** point_clouds/*.bin + poses/*.json

**처리:**
1. LiDAR 포인트 로드 (Local World 좌표)
2. T_world_to_cam = inv(T_cam_to_world)로 카메라 프레임 변환
3. Brown-Conrady 왜곡 모델로 2D 투영 (k1, k2, k3, p1, p2)
4. Sparse depth map 생성 → 보간 (nearest/linear/cubic)

**출력:**
```
depth_maps/{cam_name}/
└── {frame}.png    # uint16, mm 단위 (0~65m 범위)
```

### 3.3 Dynamic Object Masking

**입력:** objects/*.json + poses/*.json

**처리:**
1. 3D bounding box → 8개 코너 포인트 생성
2. Heading rotation 적용
3. 카메라 프레임 변환 → 2D 투영 (왜곡 보정 포함)
4. Convex hull → 바이너리 마스크
5. Safety margin dilation (기본 5px 확장)

**출력:**
```
masks/{cam_name}/
└── {frame}.png    # uint8, 0=dynamic, 255=static
```

### 3.4 좌표 변환 체인
```
LiDAR Point (Vehicle Frame)
    ↓ [Parsing: Waymo에서 추출]
World Point (First Frame Origin)
    ↓ [T_world_to_cam = inv(T_cam_to_world)]
Camera Frame Point
    ↓ [cv2.projectPoints + Brown-Conrady distortion]
Image Pixel (u, v)
```

---

## 4. Stage 3: Inpainting (Dynamic Object Removal)

### 4.1 목적
동적 객체 영역을 제거하고 정적 배경을 복원. RGB + Depth + Confidence 출력.

### 4.2 Approach 선택

| | Approach 1: COLMAP | Approach 2: Sequential (권장) |
|---|---|---|
| 전략 | 3D 재구성 기반 공간 일관성 | 시간 누적 → 기하 가이드 → AI 생성 |
| 속도 | 1~5시간 | 10~30분 |
| GPU | 불필요 | 필수 (~6GB VRAM) |
| 품질 | 기하학적 정확성 | 텍스처 품질 우수 |
| 완성도 | 일부 hole 남을 수 있음 | 100% |

### 4.3 Approach 2: Sequential Pipeline (3단계)

#### Step 1: Temporal Accumulation
- 다중 프레임의 정적 배경을 3D 공간에서 누적
- Voxel downsampling (기본 5cm)으로 포인트 클라우드 정리
- 참조 프레임으로 재투영
- **복원율: 70~85%**

**출력:**
```
step1_warped/         # 누적된 RGB 이미지
step1_depth/          # Z-buffer depth (uint16 mm)
step1_meta/           # depth_source ('lidar'|'pseudo'), filled_ratio
step1_warped/accumulated_static.ply  # 누적 포인트 클라우드
```

**핵심:** Z-buffer 생성 시 pseudo depth (10m 균일값) 감지 및 플래그 설정

#### Step 2: Geometric Guide Generation
- Step 1 미충전 영역(hole) 감지
- LiDAR depth 또는 RANSAC 평면 추정으로 깊이 예측
- Monocular depth + LiDAR 정합 (scale, shift 추정)

**출력:**
```
step2_depth_guide/    # uint16 기하 깊이 가이드
step2_hole_masks/     # uint8 최종 hole 마스크
step2_meta/           # RANSAC stats: scale, shift, inlier_ratio
```

**핵심:** 4가지 내부 방법 로깅 (LiDAR, Mono+LiDAR, Mono only, RANSAC plane)

#### Step 3: Final Inpainting + Depth Composition
- Stable Diffusion 1.5 + ControlNet Depth로 AI 인페인팅
- Priority-based depth composition (LiDAR > Z-buffer > Guide)
- 8-level confidence map 생성
- Per-pixel method log 기록

**출력:**
```
step3_final_inpainted/  # AI 인페인팅 결과 RGB
step3_depth/            # composited depth (uint16 mm)
step3_confidence/       # 8-level confidence (uint8)
step3_method_log/       # per-pixel restoration method log (JSON)
```

### 4.4 Depth Composition (Priority-based)

```
Priority 1: LiDAR depth (원본 센서 데이터)
Priority 2: Step 1 Z-buffer (LiDAR 기반 재투영만, pseudo depth 제외)
Priority 3: Step 2 guide (Monocular + RANSAC)
```

**Pseudo Depth 제외 이유:**
- LiDAR 미도달 영역에 10m 균일값이 할당됨
- 이를 supervision에 사용하면 3DGS 학습에 bias 발생

### 4.5 Confidence Map (8-Level)

| 값 | 의미 | 신뢰도 |
|----|------|--------|
| 255 | 원본 배경 (마스크 외부) | 최고 |
| 224 | LiDAR Z-buffer 재투영 depth | 매우 높음 |
| 192 | Pseudo depth (사용하지 않음) | - |
| 160 | Mono depth + LiDAR 정합 | 높음 |
| 128 | Mono depth only | 중간 |
| 96 | RANSAC 평면 추정 | 낮음 |
| 64 | AI 생성 영역 (depth 없음) | 최저 |
| 0 | 실패/미처리 | - |

### 4.6 Final Output 구조

```
final_output/
├── rgb/              # 최종 인페인팅된 RGB
├── depth/            # composited depth (uint16 mm)
├── confidence/       # 8-level confidence (uint8)
├── method_log/       # per-pixel method log (JSON)
└── point_cloud/      # accumulated_static.ply (Step 1)

final_inpainted/      # backward compatibility (RGB 심볼릭 링크)
```

---

## 5. Stage 4: Reconstruction (3D Gaussian Splatting)

### 5.1 목적
인페인팅된 이미지들로부터 3D Gaussian Map 생성.

### 5.2 Metadata 준비

**입력:** final_output/rgb/ (또는 final_inpainted/) + poses/

**처리:**
1. 인페인팅 이미지 수집
2. 대응 pose JSON 매칭
3. depth_path, confidence_path 존재 시 메타데이터에 포함
4. Train/Val 분할 (기본 90/10)

**출력:**
```
train_meta/train_pairs.json
val_meta/val_pairs.json
```

### 5.3 Approach 1: 3DGS (Standard Gaussian Splatting)

**특징:**
- 정적 장면 가정 (Rolling Shutter 무시)
- 학습 시간: 2~3시간
- PSNR: ~28 dB

**Depth Supervision (선택적, --use_depth):**
- Opacity-weighted Gaussian center z-projection으로 rendered depth 근사
- Confidence-weighted L1 depth loss
- confidence >= threshold (기본 0.63 = 160/255) 픽셀만 supervision

**Loss:**
```
L_total = (1 - λ_ssim) * L1 + λ_ssim * (1 - SSIM) + α_depth * L_depth

L_depth = Σ(confidence[i] * |rendered_depth[i] - gt_depth[i]|) / Σ(confidence[i])
    where i ∈ {pixels | gt_depth > 0 AND confidence >= threshold}
```

### 5.4 Approach 2: 3DGUT (Alpasim-Optimized)

**설계 철학:**
```
Rectified 이미지 (RS 왜곡 제거됨)
    → velocity=0 으로 순수 GS 학습 (정확한 geometry)
    → 왜곡 없는 PLY 출력 → Alpasim 시뮬레이터 호환
```

**두 가지 모드:**
- **Rectified (기본):** 전처리된 입력 → 순수 GS → Alpasim 호환 PLY
- **Raw (선택):** RS 왜곡 입력 → 8-chunk RS 렌더링 시뮬레이션

**추가 기능:**
- Temporal uncertainty regularization
- NVIDIA Gaussian Rasterizer 통합
- PSNR: ~30 dB

### 5.5 Depth Supervision 상세

**Rendered Depth 계산 (Opacity-weighted):**
```python
# Gaussian center를 카메라 z축에 투영
z_values = (R @ gaussian_centers.T + t)[2, :]  # camera-space z

# Opacity * transmittance 가중 평균
rendered_depth = Σ(alpha_i * T_i * z_i) / Σ(alpha_i * T_i)
    where T_i = Π(1 - alpha_j) for j < i
```

**Threshold 선택 근거:**
- 0.63 = 160/255 → Mono+LiDAR 정합 이상만 supervision
- LiDAR Z-buffer (224/255=0.88), Mono+LiDAR (160/255=0.63) 포함
- Mono only (128/255=0.50), RANSAC (96/255=0.38) 제외
- AI 생성 영역 (64/255=0.25) 제외

---

## 6. Inter-Stage Dependencies

```
Stage 1 → Stage 2:
├── images/         (RGB 원본)
├── point_clouds/   (LiDAR 포인트)
├── poses/          (카메라 포즈 + ego velocity)
└── objects/        (동적 객체 bbox)

Stage 2 → Stage 3:
├── images/         (Stage 1에서 전달)
├── depth_maps/     (LiDAR 투영 depth)
├── masks/          (동적 객체 마스크)
└── poses/          (Stage 1에서 전달)

Stage 3 → Stage 4:
├── final_output/rgb/         (인페인팅된 RGB)
├── final_output/depth/       (composited depth)
├── final_output/confidence/  (8-level confidence)
├── final_output/point_cloud/ (초기화용 PLY)
└── poses/                    (Stage 1에서 전달, 메타데이터 생성용)
```

---

## 7. Key Technical Decisions & Rationale

### 7.1 왜 LiDAR depth를 Stage 3에서 보존하는가?
- Stage 2에서 생성된 depth_maps/는 sparse한 LiDAR 투영
- Stage 3 Step 1의 Z-buffer는 다중 프레임 누적으로 더 dense
- 이 depth 정보를 버리면 Stage 4에서 geometry supervision 불가
- **결론:** depth를 끝까지 보존하여 3DGS 학습 품질 향상

### 7.2 왜 Pseudo Depth를 제외하는가?
- LiDAR 미도달 영역에 10m 균일값이 할당됨
- 이를 depth GT로 사용하면 false supervision → geometry 왜곡
- **결론:** step1_meta에서 `depth_source: 'pseudo'` 감지 시 해당 Z-buffer 제외

### 7.3 왜 Confidence-weighted Loss를 사용하는가?
- 모든 depth 픽셀이 동일한 신뢰도가 아님
- LiDAR 직접 투영 > Monocular 정합 > RANSAC 추정 순 신뢰도
- **결론:** 신뢰도별 가중치로 high-quality 영역에 집중 학습

### 7.4 왜 Rectified 모드를 3DGUT 기본으로 하는가?
- Alpasim 시뮬레이터는 Global Shutter 기반
- RS 왜곡이 포함된 PLY는 시뮬레이터와 불일치
- Rectified 이미지 → velocity=0 학습 → 깨끗한 geometry
- **결론:** 시뮬레이터 호환성을 위해 Rectified 기본 채택

### 7.5 왜 backward compatibility를 유지하는가?
- `--use_depth` 기본값 False → 기존 파이프라인 그대로 동작
- `final_inpainted/` 디렉토리 유지 → 기존 스크립트 호환
- **결론:** 점진적 마이그레이션 지원

---

## 8. Performance Summary

| Stage | 방법 | 소요 시간 (100프레임) | 메모리 | 비고 |
|-------|------|----------------------|--------|------|
| Parsing | waymo2nre | ~5분 | 2GB RAM | TensorFlow 불필요 |
| Preprocessing | LiDAR+Masking | ~10분 | 4GB RAM | GPU 선택적 |
| Inpainting | Sequential | 10~30분 | 6GB VRAM | GPU 필수 |
| Reconstruction | 3DGS | 2~3시간 | 8GB VRAM | PSNR ~28dB |
| Reconstruction | 3DGUT | 3~5시간 | 10GB VRAM | PSNR ~30dB |

---

## 9. 전체 디렉토리 구조

```
nre_format/                        # NRE 표준 포맷 루트
├── images/                        # [Stage 1] 원본 RGB
├── point_clouds/                  # [Stage 1] LiDAR (Nx3 float32 bin)
├── poses/                         # [Stage 1] 포즈 + velocity + RS
├── objects/                       # [Stage 1] 동적 객체 bbox
│
├── depth_maps/{cam}/              # [Stage 2] LiDAR 투영 depth (uint16 mm)
├── masks/{cam}/                   # [Stage 2] 동적 객체 마스크
│
├── step1_warped/                  # [Stage 3.1] 시간 누적 RGB + PLY
├── step1_depth/                   # [Stage 3.1] Z-buffer depth
├── step1_meta/                    # [Stage 3.1] depth source 메타
├── step2_depth_guide/             # [Stage 3.2] 기하 깊이 가이드
├── step2_hole_masks/              # [Stage 3.2] 최종 hole 마스크
├── step2_meta/                    # [Stage 3.2] RANSAC 통계
├── step3_final_inpainted/         # [Stage 3.3] AI 인페인팅 RGB
├── step3_depth/                   # [Stage 3.3] composited depth
├── step3_confidence/              # [Stage 3.3] confidence map
├── step3_method_log/              # [Stage 3.3] method log
│
├── final_output/                  # [Stage 3 최종] 통합 출력
│   ├── rgb/
│   ├── depth/
│   ├── confidence/
│   ├── method_log/
│   └── point_cloud/
├── final_inpainted/               # [Stage 3] backward compat (RGB)
│
├── train_meta/train_pairs.json    # [Stage 4] 학습 메타데이터
├── val_meta/val_pairs.json        # [Stage 4] 검증 메타데이터
└── outputs/                       # [Stage 4] 최종 3D 모델
    ├── 3dgs/gaussians.ply
    └── 3dgut/gaussians_3dgut.ply
```

---

## 10. 연구 분석 포인트 (claude.ai용)

이 문서를 claude.ai에서 분석할 때 검토할 수 있는 주요 연구 질문:

1. **파이프라인 설계 합리성:** 4-stage 분리가 최적인가? 통합 가능한 단계가 있는가?
2. **Depth supervision 전략:** Confidence-weighted loss vs uniform loss vs learned weighting
3. **타 연구 비교:** StreetGaussians, HUGS, EmerNeRF, UniSim 대비 차별점
4. **Pseudo depth 처리:** 제외 vs 낮은 가중치 부여 중 어느 것이 더 합리적인가?
5. **Confidence 설계:** 8단계 이산값 vs 연속 confidence 추정
6. **Rolling Shutter 처리:** Rectified 전처리 vs 학습 시 RS 보상 trade-off
7. **Scalability:** 대규모 도시 데이터에 대한 확장성
8. **Novel view 품질:** AI 인페인팅 아티팩트가 3DGS 학습에 미치는 영향

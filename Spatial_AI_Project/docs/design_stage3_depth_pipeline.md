# Design Document: Stage 3 Depth Pipeline 재설계

**문서 버전:** 2.0
**작성일:** 2026-03-12
**최종 수정:** 2026-03-13
**상태:** Reviewed — 리뷰 반영 완료

---

## 1. 문제 정의

### 1.1 현재 상태

Stage 3(Inpainting)은 동적 객체가 제거된 **RGB 이미지만** 출력한다.
Stage 4(3D Reconstruction)는 이 RGB만으로 photometric loss(L1 + SSIM)를 사용하여 3D Gaussian Map을 학습한다.

### 1.2 문제점

동적 객체가 제거된 영역에 대한 **기하학적 정보(depth)가 최종 출력에서 누락**된다.

```
현재 데이터 흐름:

Stage 2                    Stage 3                         Stage 4
────────                   ────────                        ────────
depth_maps/   ──→  Step 2: dense depth 생성  ──→  ❌ 폐기
(LiDAR sparse)     (step2_depth_guide/)              │
                          │                          │
                          ▼                          │
                   Step 3: ControlNet 조건으로 소비   │
                          │                          │
                          ▼                          │
                   RGB만 출력  ──────────────────────→  photometric loss만 사용
```

**구체적 손실 지점:**

| 단계 | 생성되는 depth | 현재 처리 | 문제 |
|------|---------------|-----------|------|
| Step 1 Temporal Accumulation | Z-buffer (재투영 깊이) | 계산 후 폐기 | 복원 영역의 실측 기반 depth 손실 |
| Step 2 Geometric Guide | Dense depth (Depth Anything + LiDAR 정합) | ControlNet 조건으로만 소비 | `step2_depth_guide/`에 방치 |
| Step 3 AI Inpainting | 없음 (RGB만 출력) | — | 최종 출력에 depth 없음 |
| Orchestrator | `final_inpainted/`에 RGB만 복사 | — | Stage 4에 depth 전달 경로 없음 |

### 1.3 코드 리뷰에서 추가 발견된 문제

| 문제 | 위치 | 영향 |
|------|------|------|
| **Pseudo depth fallback** | step1 line 173 | `depth_maps/` 없으면 10m 균일값 사용 → 이 기반의 Z-buffer는 기하학적으로 무의미 |
| **Step 2 hole_masks 미사용** | step2 → step3 | `step2_hole_masks/` 생성하지만 Step 3이 무시, 자체적으로 target_mask를 재계산 |
| **Depth 절대 스케일 소실** | step3 line 87 | uint16 → 0-255 normalize 과정에서 metric depth 정보 완전 소실 |
| **Step 2 방법 선택 로깅 없음** | step2 lines 256-282 | 4가지 fallback 경로 중 어떤 것이 사용됐는지 기록 없음 → 디버깅/품질 추적 불가 |
| **accumulated_static.ply 미활용** | step1 line 116 | 생성 후 후속 단계에서 사용하지 않음 |

### 1.4 결과적 영향

- Stage 4에서 depth supervision 불가 → 기하학적 정확도가 photometric 단서에만 의존
- 특히 텍스처가 균일한 영역(도로, 벽면)에서 depth ambiguity 발생
- 초기 포인트 클라우드 없이 랜덤 50K점으로 시작 시 수렴 불안정

---

## 2. 설계 목표

1. **Stage 3 출력에 depth를 포함**하여 RGB + Depth 쌍으로 제공
2. 복원 방법별 **신뢰도(confidence)를 세분화하여 기록** → Stage 4에서 가중치로 활용
3. **Stage 4에서 depth supervision**을 선택적으로 활용할 수 있는 경로 확보
4. **Rasterizer depth 출력** 추가 → depth loss 계산의 전제 조건 확보
5. 기존 RGB-only 파이프라인과의 **하위 호환성 유지**

---

## 3. 상세 설계

### 3.1 Stage 3 Output 재정의

현재:
```
final_inpainted/
└── {frame}_{cam}.jpg       # RGB만
```

변경 후:
```
final_output/
├── rgb/
│   └── {frame}_{cam}.jpg               # 인페인팅 완료 RGB
│
├── depth/
│   └── {frame}_{cam}.png               # Composited dense depth
│                                        # uint16, mm 단위 (0~65535 → 0~65.535m)
│                                        # 0 = depth 없음
│
├── confidence/
│   └── {frame}_{cam}.png               # 픽셀별 복원 신뢰도
│                                        # uint8 (0~255)
│
├── method_log/
│   └── {frame}_{cam}.json              # 프레임별 사용된 depth 추정 방법 기록  ← NEW
│
└── point_cloud/
    └── accumulated_static.ply           # Step 1 정적 포인트 클라우드
                                         # (기존 step1_warped/에서 이동)
```

**하위 호환성:** `final_inpainted/` → `final_output/rgb/`로 심볼릭 링크 또는 복사 유지.

### 3.2 Confidence Map 정의 (세분화)

각 픽셀의 복원 출처에 따라 신뢰도를 부여한다.
**Step 2 내부의 4가지 방법을 구분**하여 depth loss 적용 가능 여부를 정밀하게 판단한다.

| 값 | 출처 | depth 신뢰도 | depth loss 사용 |
|----|------|-------------|-----------------|
| **255** | 원본 배경 + LiDAR sparse depth | cm급 정확 | O — 강한 supervision |
| **224** | Step 1 Temporal (LiDAR 기반 Z-buffer) | cm급 | O — 강한 supervision |
| **192** | Step 1 Temporal (pseudo depth 10m 기반) | 무의미 | **X — depth 합성에서 제외** |
| **160** | Step 2 Mono+LiDAR 정합 (Depth Anything + RANSAC) | ~10cm | O — 중간 supervision |
| **128** | Step 2 Mono only (스케일 없음) | 상대적만 | **X — depth loss 부적합** |
| **96** | Step 2 RANSAC 평면 추정 | 평면 가정 유효 시에만 | 조건부 — 평면 영역만 |
| **64** | Step 3 AI 생성 (ControlNet depth 가이드) | ControlNet 수준 | X — 참고용만 |
| **0** | 복원 실패 / cv2.inpaint fallback | 없음 | X |

**depth loss 적용 기준:**
- confidence >= 160 (0.63 normalized) → depth loss 활성화
- confidence < 160 → depth loss 비활성화 (RGB loss만 적용)

**생성 로직:**

```python
confidence = np.full((H, W), 255, dtype=np.uint8)   # 기본: 원본 배경

# Step 1 복원 영역
step1_filled = (orig_mask == 0) & (step1_result != 0)

if has_real_lidar_depth:
    # LiDAR 기반 Z-buffer → 신뢰도 높음
    confidence[step1_filled] = 224
else:
    # Pseudo depth(10m) 기반 → 기하학적으로 무의미
    confidence[step1_filled] = 192

# Step 2/3 복원 영역 — 사용된 방법에 따라 차등
step23_filled = (orig_mask == 0) & (~step1_filled)

# method_used: Step 2에서 기록한 프레임별 방법
if method_used == 'mono_lidar':
    confidence[step23_filled] = 160
elif method_used == 'mono_only':
    confidence[step23_filled] = 128
elif method_used == 'ransac_plane':
    confidence[step23_filled] = 96
else:  # 'simple_inpaint' or fallback
    confidence[step23_filled] = 0

# Step 3 AI 생성 영역 (Step 2 guide 위에 AI가 생성한 RGB)
ai_generated = step23_filled & step3_inpainted_mask
confidence[ai_generated] = min(confidence[ai_generated], 64)
```

### 3.3 Composited Depth Map 생성

여러 출처의 depth를 하나의 dense depth map으로 합성한다.

**합성 우선순위 (높은 순):**

```
1순위: LiDAR sparse depth (Stage 2 Preprocessing 출력)
       → 가장 정확. 존재하는 픽셀만.

2순위: Step 1 Z-buffer depth (LiDAR 기반만)
       → 실제 관측 기반 재투영. 복원 영역에 대한 실측 depth.
       ⚠️ pseudo depth(10m) 기반 Z-buffer는 제외

3순위: Step 2 depth guide (Depth Anything + LiDAR 정합)
       → 모노큘러 추정이지만 LiDAR로 스케일 보정됨.

4순위: Step 2 depth guide (Depth Anything only / RANSAC / inpaint)
       → 상대적 depth만. 스케일 불확실. 참고용으로만 포함.
```

**합성 수식:**

```python
depth_final = np.zeros((H, W), dtype=np.float32)

# 1순위: LiDAR
lidar_valid = lidar_depth > 0
depth_final[lidar_valid] = lidar_depth[lidar_valid]

# 2순위: Step 1 Z-buffer (LiDAR 기반만, pseudo depth 제외)
if has_real_lidar_depth:
    zbuf_valid = (step1_zbuffer > 0) & (~lidar_valid)
    depth_final[zbuf_valid] = step1_zbuffer[zbuf_valid]
# else: pseudo depth 기반 Z-buffer는 합성에서 제외

# 3순위: Step 2 depth guide (방법별 분기)
remaining = depth_final == 0
if method_used == 'mono_lidar':
    # metric-aligned depth → 합성에 포함
    depth_final[remaining] = step2_depth_guide[remaining]
elif method_used in ('mono_only', 'ransac_plane', 'simple_inpaint'):
    # 스케일 불확실 → 합성에는 포함하되 confidence로 구분
    depth_final[remaining] = step2_depth_guide[remaining]
```

### 3.4 Method Log (프레임별 추정 방법 기록)

Step 2에서 프레임마다 어떤 depth 추정 방법이 사용되었는지 기록한다.

```json
{
  "frame": "seq0_000000_FRONT",
  "step1": {
    "depth_source": "lidar",
    "filled_ratio": 0.73,
    "points_accumulated": 142857
  },
  "step2": {
    "method": "mono_lidar",
    "lidar_anchor_count": 1247,
    "ransac_inlier_ratio": 0.89,
    "scale": 1.023,
    "shift": -0.15
  },
  "step3": {
    "ai_filled_ratio": 0.08,
    "controlnet_strength": 0.8
  }
}
```

### 3.5 각 Step별 수정 사항

#### Step 1: Temporal Accumulation

**변경:**
- Z-buffer를 파일로 저장
- Pseudo depth 사용 여부를 플래그로 기록

```python
# 현재: 재투영 시 Z-buffer 계산 후 RGB만 저장
# 변경: Z-buffer도 함께 저장 + depth source 기록

# 출력 추가
step1_depth/
└── {frame}_{cam}.png    # uint16, mm 단위

# depth source 기록
step1_meta = {
    'depth_source': 'lidar' if lidar_exists else 'pseudo',
    'filled_ratio': float(step1_filled.sum()) / mask_area
}
```

**영향 범위:** `step1_temporal_accumulation.py`의 재투영 함수에서 depth 배열을 파일로 저장하는 코드 추가. 기존 로직 변경 없음.

#### Step 2: Geometric Guide

**변경:**
- 사용된 방법과 RANSAC 통계를 method_log로 기록
- `step2_hole_masks/` 출력을 Step 3에서 활용하도록 경로 연결

```python
# 현재: 방법 선택 후 결과만 저장
# 변경: 방법명 + 통계를 JSON으로 기록

step2_meta = {
    'method': method_name,             # 'mono_lidar' | 'mono_only' | 'ransac_plane' | 'simple_inpaint'
    'lidar_anchor_count': int(count),
    'ransac_inlier_ratio': float(ratio),
    'scale': float(scale),
    'shift': float(shift)
}
```

**영향 범위:** 기존 dense depth 생성 로직 변경 없음. 로깅 코드만 추가.

#### Step 3: AI Inpainting + Depth Composition

**변경:**
- Confidence map 생성 로직 추가
- Composited depth map 생성 로직 추가
- `step2_hole_masks/` 활용 (기존 자체 재계산 대신)

```python
# step3_final_inpainting.py에 추가할 함수

def compose_depth(lidar_depth, step1_zbuffer, step2_guide, orig_mask, step1_meta, step2_meta):
    """여러 출처의 depth를 우선순위에 따라 합성.
    pseudo depth 기반 Z-buffer는 자동 제외."""
    ...

def create_confidence_map(orig_mask, step1_filled_mask, step1_meta, step2_meta):
    """복원 출처별 신뢰도 맵 생성. Step 2 방법별 세분화."""
    ...

def create_method_log(frame_name, step1_meta, step2_meta, step3_stats):
    """프레임별 추정 방법 및 통계를 JSON으로 기록."""
    ...
```

#### Orchestrator: approach2_sequential.py

**변경:**
- `final_output/` 디렉토리 구조 생성 (rgb/, depth/, confidence/, method_log/, point_cloud/)
- 각 Step 출력물을 최종 구조로 조립
- `final_inpainted/` 하위 호환 유지 (rgb/ 심볼릭 링크 또는 복사)
- Step 간 메타데이터(step1_meta, step2_meta) 전달 경로 확보

---

## 4. Stage 4 연계 설계

### 4.1 prepare_metadata.py 수정

메타데이터에 depth 및 confidence 경로 추가:

```json
{
  "file_path": "final_output/rgb/seq0_000000_FRONT.jpg",
  "depth_path": "final_output/depth/seq0_000000_FRONT.png",
  "confidence_path": "final_output/confidence/seq0_000000_FRONT.png",
  "transform_matrix": [ ... ],
  "intrinsics": [ ... ],
  "width": 1920,
  "height": 1280,

  "velocity": { ... },
  "rolling_shutter": { ... }
}
```

### 4.2 data_loader.py 수정

```python
class ReconstructionDataset(Dataset):
    def __init__(self, ..., load_depth=False):
        self.load_depth = load_depth

    def __getitem__(self, idx):
        item = self.items[idx]
        result = {
            'image': load_image(item['file_path']),        # [3, H, W]
            'extrinsic': load_extrinsic(item),             # [4, 4]
            'intrinsic': load_intrinsic(item),             # [3, 3]
        }

        if self.load_depth and 'depth_path' in item:
            depth_mm = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)  # uint16
            result['depth'] = depth_mm.astype(np.float32) / 1000.0           # [H, W] meters

            conf = cv2.imread(item['confidence_path'], cv2.IMREAD_UNCHANGED) # uint8
            result['confidence'] = conf.astype(np.float32) / 255.0           # [H, W] 0~1

        return result
```

### 4.3 Rasterizer Depth 출력 (신규)

**Stage 4에서 depth loss를 계산하려면 rendered_depth가 필요하다.**
현재 3DGS/3DGUT rasterizer는 RGB만 출력한다.

```python
# 현재: RGB만 렌더링
rendered_image = rasterizer(
    means3D, opacity, scales, rotations, sh,
    viewmatrix, projmatrix
)

# 변경: depth도 함께 출력
rendered_image, rendered_depth = rasterizer(
    means3D, opacity, scales, rotations, sh,
    viewmatrix, projmatrix,
    render_depth=True   # Gaussian 중심의 depth를 alpha-weighted sum으로 계산
)
# rendered_depth: [1, H, W] in meters
```

**구현 방식:**
- NVIDIA diff-gaussian-rasterization이 depth 출력을 지원하는 경우: 플래그 활성화
- 미지원 시: Gaussian 중심 좌표의 카메라 z축 투영으로 근사

```python
# Fallback: Gaussian 중심 기반 depth 근사
def compute_gaussian_depth(means3D, viewmatrix):
    """각 Gaussian의 카메라 좌표계 depth 계산"""
    cam_coords = (viewmatrix[:3, :3] @ means3D.T + viewmatrix[:3, 3:4]).T
    return cam_coords[:, 2]  # z-axis = depth
```

### 4.4 Depth-supervised Loss 추가

```python
# losses.py에 추가

def depth_loss(rendered_depth, gt_depth, confidence, threshold=0.63):
    """Confidence-weighted depth supervision loss.

    Args:
        rendered_depth: [1, H, W] rasterizer 출력 depth (meters)
        gt_depth: [H, W] composited GT depth (meters)
        confidence: [H, W] normalized 0~1
        threshold: confidence 이 값 이상인 픽셀만 supervision 적용
                   0.63 = 160/255 (Mono+LiDAR 정합 이상만)
    """
    valid = (gt_depth > 0) & (confidence >= threshold)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=rendered_depth.device)

    depth_error = torch.abs(rendered_depth.squeeze(0)[valid] - gt_depth[valid])
    weights = confidence[valid]

    return (depth_error * weights).sum() / weights.sum()
```

**Training loop 수정:**

```python
# 기존
L_total = (1 - lambda_dssim) * L1_rgb + lambda_dssim * L_ssim

# 추가 (--use_depth 활성화 시)
if args.use_depth and 'depth' in batch and 'confidence' in batch:
    L_depth = depth_loss(rendered_depth, batch['depth'], batch['confidence'],
                         threshold=args.depth_confidence_threshold)
    L_total = L_total + args.alpha_depth * L_depth
```

**파라미터:**
- `--use_depth`: depth supervision 활성화 (기본값: False)
- `--alpha_depth`: depth loss 가중치 (기본값: 0.1, 튜닝 필요)
- `--depth_confidence_threshold`: confidence 최소 기준 (기본값: 0.63)

**하위 호환성:** `--use_depth` 기본값 False → 기존 동작 변경 없음.

### 4.5 초기 포인트 클라우드 개선

현재: `--initial_ply` 미제공 시 랜덤 50K점
변경: `final_output/point_cloud/accumulated_static.ply` 자동 탐색

```python
# 우선순위
ply_candidates = [
    args.initial_ply,                                    # 명시적 지정
    f"{data_root}/final_output/point_cloud/accumulated_static.ply",
    f"{data_root}/step1_warped/accumulated_static.ply",  # 하위 호환
]

initial_ply = next((p for p in ply_candidates if p and os.path.exists(p)), None)
```

---

## 5. 전체 데이터 흐름 (변경 후)

```
Stage 2 (Preprocessing)
────────────────────────
depth_maps/          (LiDAR sparse, uint16)
masks/               (동적 객체, uint8)
                │
                ▼
Stage 3 (Inpainting)
────────────────────────────────────────────────────────────

Step 1: Temporal Accumulation
  Input:  images/ + masks/ + poses/ + depth_maps/
  Output: step1_warped/{frame}_{cam}.png      (RGB)
          step1_depth/{frame}_{cam}.png       (Z-buffer depth)       ← NEW
          step1_meta/{frame}_{cam}.json       (depth_source 기록)    ← NEW
          step1_warped/accumulated_static.ply  (정적 포인트)
                │
                ▼
Step 2: Geometric Guide
  Input:  step1_warped/ + depth_maps/
  Output: step2_depth_guide/{frame}.png       (Dense depth)
          step2_hole_masks/{frame}.png        (구멍 마스크)
          step2_meta/{frame}.json             (방법 + RANSAC 통계)   ← NEW
                │
                ▼
Step 3: Final Inpainting + Depth Composition
  Input:  step1_warped/ + step2_depth_guide/ + step2_hole_masks/     ← 변경 (hole_masks 활용)
          + images/ + masks/
          + step1_depth/ + depth_maps/                                ← NEW
          + step1_meta/ + step2_meta/                                 ← NEW
  Output: step3_final_inpainted/{frame}.jpg   (RGB)
          step3_depth/{frame}.png             (Composited depth)     ← NEW
          step3_confidence/{frame}.png        (Confidence map)       ← NEW
                │
                ▼
Orchestrator: Final Output Assembly
  Output: final_output/
            ├── rgb/{frame}_{cam}.jpg
            ├── depth/{frame}_{cam}.png
            ├── confidence/{frame}_{cam}.png
            ├── method_log/{frame}_{cam}.json                        ← NEW
            └── point_cloud/accumulated_static.ply

                │
                ▼
Stage 4 (3D Reconstruction)
────────────────────────────────────────────────────────────

prepare_metadata.py
  Input:  final_output/rgb/ + final_output/depth/ + final_output/confidence/ + poses/
  Output: train_meta/train_pairs.json  (RGB + depth + confidence 경로 포함)

3DGS / 3DGUT Training
  Input:  train_pairs.json + final_output/
  Render: RGB + depth (rasterizer depth 출력 추가)                    ← NEW
  Loss:   L1_rgb + SSIM + α * confidence-weighted depth loss
          (confidence >= threshold 인 픽셀만 depth supervision)      ← NEW
  Init:   final_output/point_cloud/accumulated_static.ply
  Output: gaussians.ply
```

---

## 6. 파일 변경 목록

| 파일 | 변경 유형 | 변경 내용 |
|------|-----------|-----------|
| `Inpainting/step1_temporal_accumulation.py` | 수정 | Z-buffer depth 저장 + pseudo depth 감지 플래그(step1_meta) |
| `Inpainting/step2_geometric_guide.py` | 수정 | 사용 방법 + RANSAC 통계 로깅(step2_meta) |
| `Inpainting/step3_final_inpainting.py` | 수정 | Composited depth + confidence map(세분화) + method_log 생성 |
| `Inpainting/approach2_sequential.py` | 수정 | `final_output/` 구조 생성, Step간 meta 전달, 파일 조립 |
| `reconstruction/prepare_metadata.py` | 수정 | depth/confidence 경로를 JSON에 포함 |
| `reconstruction/data_loader.py` | 수정 | depth/confidence 로딩 기능 추가 |
| `reconstruction/losses.py` | 수정 | confidence-weighted depth loss + threshold 기반 필터링 |
| `reconstruction/approach1_3dgs.py` | 수정 | rasterizer depth 출력 + `--use_depth` 옵션 |
| `reconstruction/approach2_3dgut.py` | 수정 | rasterizer depth 출력 + `--use_depth` 옵션 |

---

## 7. 리스크 및 고려사항

### 7.1 Depth 정확도 (세분화)

| 출처 | 절대 정확도 | 한계 | depth loss 적용 |
|------|------------|------|-----------------|
| LiDAR sparse | ~cm | 희소 (이미지 픽셀의 ~1%) | O |
| Step 1 Z-buffer (LiDAR 기반) | ~cm | 복원 영역에만 존재 | O |
| Step 1 Z-buffer (pseudo 10m) | 무의미 | 기하학적 정보 없음 | **X — 합성 제외** |
| Step 2 Mono+LiDAR 정합 | ~10cm | RANSAC fitting 품질에 의존 | O |
| Step 2 Mono only | 상대적 | 절대 스케일 없음 | **X** |
| Step 2 RANSAC 평면 | 평면 영역에서 유효 | 비평면 영역 부정확 | 조건부 |
| Step 2 Simple inpaint | 없음 | cv2.inpaint fallback | **X** |

**대응:** confidence map 세분화 + threshold 기반 필터링으로 불확실한 depth가 학습에 미치는 영향을 차단.

### 7.2 Rasterizer Depth 출력

| 항목 | 상태 |
|------|------|
| NVIDIA diff-gaussian-rasterization depth 지원 | 확인 필요 (버전 의존) |
| Fallback (Gaussian 중심 z투영) | 구현 가능, 정확도 낮음 |
| gsplat 라이브러리 대안 | depth 출력 기본 지원 |

**리스크:** rasterizer가 depth를 네이티브로 출력하지 않으면, alpha-weighted depth 근사의 정확도가 supervision 효과를 제한할 수 있음.

### 7.3 저장 용량 증가

100 프레임 × 5 카메라 × 1920×1280 기준:

| 파일 | 포맷 | 프레임당 | 총 (500장) |
|------|------|----------|-----------|
| RGB | jpg (Q95) | ~500KB | ~250MB |
| Depth | uint16 png | ~2MB | ~1GB |
| Confidence | uint8 png | ~100KB | ~50MB |
| Method log | json | ~1KB | ~0.5MB |

총 추가 용량: **약 1GB** (depth가 대부분). 허용 가능한 수준.

### 7.4 하위 호환성

- `--use_depth` 플래그 기본값 = False → 기존 동작 변경 없음
- `final_inpainted/` 경로 유지 (심볼릭 링크)
- depth/confidence 파일 없어도 Stage 4 정상 동작
- `--depth_confidence_threshold` 로 depth supervision 범위 조절 가능

### 7.5 향후 확장

- **Depth refinement**: 3DGS 학습 중 rendered depth로 depth map 역보정
- **Uncertainty-aware densification**: confidence가 낮은 영역에서 Gaussian 밀도 제어
- **Multi-scale depth**: 해상도별 depth supervision

---

## 8. 구현 순서 (권장)

```
Phase 0: 진단 (구현 전 확인)
  ├── depth_maps/ 실제 존재 여부 확인 (Waymo 데이터셋별)
  ├── Step 1 pseudo depth(10m) 발생 비율 측정
  ├── Step 2 방법별 사용 비율 측정 (mono_lidar vs fallback)
  └── Rasterizer depth 출력 지원 여부 확인

Phase 1: Depth 전달 경로 확보
  ├── Step 1 Z-buffer 저장 + pseudo depth 감지 플래그
  ├── Step 2 방법 로깅 (step2_meta)
  ├── Step 3 composited depth 생성 (pseudo depth 제외 로직 포함)
  ├── Step 3 confidence map 생성 (8단계 세분화)
  └── Orchestrator final_output/ 구조 + method_log 조립

Phase 2: Stage 4 연계
  ├── prepare_metadata.py depth/confidence 경로 추가
  ├── data_loader.py depth/confidence 로딩
  ├── losses.py confidence-weighted depth loss (threshold 필터링)
  ├── Rasterizer depth 출력 확장 또는 fallback 구현
  └── 3DGS/3DGUT --use_depth, --alpha_depth, --depth_confidence_threshold 옵션

Phase 3: 검증
  ├── depth map 시각화 확인
  ├── confidence map 분포 확인 (8단계 분포가 의미 있는지)
  ├── method_log 기반 프레임별 품질 분석
  └── depth supervision 유무에 따른 reconstruction 품질 비교
```

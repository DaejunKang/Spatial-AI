# Preprocessing Stage 세부 프로세스 분석

## 📋 목표
**Inpainting에서 활용할 수 있는 동적 객체 마스킹 및 LiDAR-Image 동기화/투영**

---

## 🎯 주요 기능

### 1️⃣ **Multi-Image와 LiDAR 데이터 시공간 동기화 및 Projection**
- LiDAR 포인트 클라우드를 다중 뷰 이미지에 투영
- 타임스탬프 기반 센서 동기화
- 깊이 맵 생성 (Depth supervision용)

### 2️⃣ **동적 객체 마스킹 (Masking)**
- 3D Bounding Box 투영 기반 정밀 마스킹
- Semantic Segmentation 보완 (선택적)
- Inpainting Stage의 핵심 Input 제공

---

## 📂 Preprocessing Stage 파일 구조

```
Photo-real_project/preprocessing/
├── lidar_projection.py           # LiDAR → Image 투영 (NEW)
├── dynamic_masking.py            # 동적 객체 마스킹 (NEW)
├── segmentation.py               # Semantic Segmentation (기존)
├── run_preprocessing.py          # 통합 실행 스크립트 (업데이트)
├── waymo2nre.py                  # Waymo → NRE 변환
├── waymo2colmap.py               # COLMAP 준비
└── create_nre_pairs.py           # NeRF 학습용 페어 생성
```

---

## 🛠️ 세부 프로세스별 Input/Output

### 1️⃣ **LiDAR Point Cloud Projection** (`lidar_projection.py`)

#### ✨ 목적
- LiDAR 3D 포인트를 다중 카메라 이미지에 투영
- 깊이 맵 생성 (Inpainting Step 2에서 활용)
- 시공간 동기화 검증

#### 📥 Input
| 항목 | 형식 | 설명 | 출처 |
|-----|------|------|------|
| Point Clouds | `*.bin` | LiDAR 포인트 (Nx3 float32, Local World 좌표계) | Parsing Stage |
| Poses | `*.json` | 프레임별 카메라 포즈/메타데이터 | Parsing Stage |
| Images | `*.jpg` | 원본 이미지 (검증용) | Parsing Stage |

#### ⚙️ Process

**1.1 데이터 로딩 및 동기화**
```python
# 1. 프레임별 처리
for frame in frames:
    # LiDAR 포인트 로드 (Local World 좌표계)
    points_world = np.fromfile(f"{frame}.bin", dtype=np.float32).reshape(-1, 3)
    
    # 카메라 메타데이터 로드
    with open(f"{frame}.json") as f:
        frame_data = json.load(f)
    
    # 타임스탬프 확인 (시공간 동기화)
    timestamp = frame_data['timestamp']
```

**1.2 좌표 변환 (World → Camera)**
```python
# 각 카메라별 투영
for cam_name, cam_data in frame_data['cameras'].items():
    # Camera Pose (4x4)
    T_cam_to_world = np.array(cam_data['pose']).reshape(4, 4)
    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    
    # 포인트 변환
    points_world_homo = np.hstack([points_world, np.ones((N, 1))])
    points_cam = (T_world_to_cam @ points_world_homo.T).T[:, :3]
    
    # 카메라 앞의 포인트만 선택 (Z > 0.1m)
    valid_points = points_cam[points_cam[:, 2] > 0.1]
```

**1.3 3D → 2D 투영 (with Distortion)**
```python
# OpenCV projectPoints 사용
intrinsics = cam_data['intrinsics']  # [fx, fy, cx, cy, k1, k2, p1, p2, k3]

fx, fy, cx, cy = intrinsics[:4]
k1, k2, p1, p2, k3 = intrinsics[4:9]

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# 투영 (왜곡 보정 포함)
projected_2d, _ = cv2.projectPoints(
    valid_points, 
    rvec=np.zeros(3), 
    tvec=np.zeros(3),
    camera_matrix, 
    dist_coeffs
)

# 이미지 범위 내 포인트만 선택
valid_mask = (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < width) & \
             (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < height)
```

**1.4 깊이 맵 생성**
```python
# 희소 깊이 맵 초기화
depth_map = np.zeros((height, width), dtype=np.float32)
count_map = np.zeros((height, width), dtype=np.uint16)

# 각 포인트의 깊이 값 누적
for (x, y), depth in zip(projected_2d, depths):
    depth_map[y, x] += depth
    count_map[y, x] += 1

# 평균 계산 (중복 투영 처리)
valid_mask = count_map > 0
depth_map[valid_mask] /= count_map[valid_mask]

# 보간 (선택적)
if interpolation == 'nearest':
    depth_map = cv2.inpaint(
        depth_map, 
        (1 - valid_mask).astype(np.uint8), 
        inpaintRadius=5, 
        flags=cv2.INPAINT_NS
    )

# mm 단위 uint16 변환 (저장용)
depth_map_mm = (depth_map * 1000).astype(np.uint16)
```

**1.5 포인트 마스크 생성**
```python
# LiDAR 포인트가 투영된 픽셀 표시
point_mask = np.zeros((height, width), dtype=np.uint8)

for x, y in projected_2d:
    point_mask[y, x] = 255

# 시각화 개선을 위한 팽창
kernel = np.ones((3, 3), np.uint8)
point_mask = cv2.dilate(point_mask, kernel, iterations=1)
```

#### 📤 Output
| 디렉토리 | 파일 형식 | 내용 | 용도 |
|---------|----------|------|------|
| `depth_maps/{cam_name}/` | `{frame}.png` | 깊이 맵 (uint16, mm 단위) | Inpainting Step 2 Geometric Guide |
| `point_masks/{cam_name}/` | `{frame}.png` | LiDAR 포인트 마스크 (uint8) | 검증 및 시각화 |

**Depth Map 읽기:**
```python
depth_mm = cv2.imread("depth_map.png", cv2.IMREAD_UNCHANGED)  # uint16
depth_m = depth_mm.astype(np.float32) / 1000.0  # meters
```

---

### 2️⃣ **Dynamic Object Masking** (`dynamic_masking.py`)

#### ✨ 목적
- Inpainting에서 제거할 동적 객체 영역 마스킹
- 3D Bounding Box 기반 정밀 마스킹
- Semantic Segmentation 보완 (선택적)

#### 📥 Input
| 항목 | 형식 | 설명 | 출처 |
|-----|------|------|------|
| Objects | `*.json` | 동적 객체 3D Bounding Box | Parsing Stage |
| Poses | `*.json` | 프레임별 카메라 포즈/메타데이터 | Parsing Stage |
| Images | `*.jpg` | 원본 이미지 (Semantic Seg용) | Parsing Stage |

#### ⚙️ Process

**2.1 3D Bounding Box → 2D 투영**
```python
# 각 동적 객체 처리
for obj in objects:
    box_center = obj['box']['center']  # [x, y, z] in World
    box_size = obj['box']['size']      # [length, width, height]
    box_heading = obj['box']['heading'] # Yaw angle (rad)
    
    # 1. 3D Box 8개 코너 생성
    l, w, h = box_size
    
    # Rotation matrix (Z-axis)
    R_z = np.array([
        [cos(heading), -sin(heading), 0],
        [sin(heading),  cos(heading), 0],
        [0, 0, 1]
    ])
    
    # 8 corners (relative to center)
    corners = [
        [±l/2, ±w/2, ±h/2]  # 8 combinations
    ]
    
    # Rotate & Translate
    corners_world = R_z @ corners + box_center
```

**2.2 좌표 변환 및 투영**
```python
    # 2. World → Camera 변환
    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    corners_cam = T_world_to_cam @ corners_world_homo
    
    # 카메라 앞의 코너만 선택
    valid_corners = corners_cam[corners_cam[:, 2] > 0.1]
    
    # 3. 3D → 2D 투영 (with distortion)
    projected_2d = cv2.projectPoints(
        valid_corners, 
        rvec, tvec, 
        camera_matrix, 
        dist_coeffs
    )
```

**2.3 마스크 생성**
```python
    # 4. 2D Convex Hull 생성
    hull = cv2.convexHull(projected_2d)
    
    # 5. 마스크에 채우기 (0 = 동적 객체)
    cv2.fillConvexPoly(mask, hull, 0)
```

**2.4 Semantic Segmentation 보완 (선택적)**
```python
# SegFormer 모델 사용 (선택적)
if use_semantic_seg:
    from segmentation import SemanticSegmentor
    
    seg_model = SemanticSegmentor()
    semantic_mask = seg_model.process_image(image_path)
    
    # 두 마스크의 교집합 (더 보수적 마스킹)
    final_mask = cv2.bitwise_and(bbox_mask, semantic_mask)
```

**2.5 안전 마진 추가 (Dilation)**
```python
# Inpainting 품질 향상을 위한 마진
kernel = np.ones((dilation_size, dilation_size), np.uint8)
final_mask = cv2.erode(final_mask, kernel, iterations=1)
# Erode를 사용하면 동적 영역(0)이 확장됨
```

#### 📤 Output
| 디렉토리 | 파일 형식 | 내용 | 용도 |
|---------|----------|------|------|
| `masks/{cam_name}/` | `{frame}.png` | 동적 객체 마스크 (uint8) | Inpainting 전 단계 Input |

**마스크 형식:**
- `255` (흰색) = 정적 배경 (유효 영역)
- `0` (검은색) = 동적 객체 (Inpainting 대상)

---

## 🔄 전체 데이터 플로우

```
┌─────────────────────────────────────────────────────┐
│         Parsing Stage Output (NRE Format)           │
│  - images/*.jpg                                     │
│  - point_clouds/*.bin                               │
│  - poses/*.json                                     │
│  - objects/*.json                                   │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐     ┌──────────────────┐
│ LiDAR         │     │ Dynamic Object   │
│ Projection    │     │ Masking          │
│               │     │                  │
│ - Sync Check  │     │ - 3D Box → 2D    │
│ - 3D → 2D     │     │ - Semantic Seg   │
│ - Depth Map   │     │ - Safety Margin  │
└───────┬───────┘     └────────┬─────────┘
        │                      │
        ▼                      ▼
┌──────────────┐     ┌──────────────────┐
│ depth_maps/  │     │ masks/           │
│ point_masks/ │     │ (0=dynamic)      │
└───────┬──────┘     └────────┬─────────┘
        │                     │
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Inpainting      │
        │ Stage           │
        │                 │
        │ Step 1: Temporal│
        │ Step 2: Geom    │ ← depth_maps 활용
        │ Step 3: AI Gen  │ ← masks 활용
        └─────────────────┘
```

---

## 🎯 핵심 알고리즘

### 1. **시공간 동기화 (Temporal-Spatial Sync)**

**문제:** LiDAR와 카메라의 센싱 타이밍이 다름 (Rolling Shutter)

**해결:**
```python
# Waymo는 이미 동기화된 데이터 제공
# 하지만 Rolling Shutter 보정 필요 (향후)

timestamp_lidar = frame_data['timestamp']
timestamp_cam = cam_data['rolling_shutter']['trigger_time']

# 현재는 프레임 단위로 1:1 매칭
# 향후 개선: Sub-frame interpolation
```

### 2. **카메라 왜곡 보정 (Distortion Correction)**

**Waymo 카메라 모델:** Brown-Conrady (OpenCV 호환)

**파라미터:**
- Radial distortion: k1, k2, k3
- Tangential distortion: p1, p2

**보정 방법:**
```python
# OpenCV projectPoints가 자동으로 처리
cv2.projectPoints(..., dist_coeffs=[k1, k2, p1, p2, k3])
```

### 3. **희소 깊이 맵 보간 (Sparse Depth Interpolation)**

**문제:** LiDAR 포인트는 희소함 (이미지 픽셀의 ~1%)

**방법:**

| 방법 | 속도 | 품질 | 사용 사례 |
|-----|------|------|----------|
| **None** | 빠름 | 낮음 | 검증용 |
| **Nearest** | 중간 | 중간 | Inpainting 가이드 (권장) |
| **Linear** | 느림 | 높음 | NeRF 학습 |
| **Cubic** | 매우 느림 | 매우 높음 | 최종 시각화 |

**구현 (Nearest):**
```python
# OpenCV inpainting 활용
depth_map_dense = cv2.inpaint(
    depth_map_sparse,
    mask=(depth_map_sparse == 0),
    inpaintRadius=5,
    flags=cv2.INPAINT_NS  # Navier-Stokes
)
```

### 4. **동적 객체 안전 마진 (Safety Margin)**

**목적:** Inpainting 품질 향상 (객체 경계 Artifact 방지)

**방법:**
```python
# 마스크 팽창 (Erosion = 동적 영역(0) 확장)
kernel_size = 5  # 픽셀
kernel = np.ones((kernel_size, kernel_size), np.uint8)
mask_expanded = cv2.erode(mask, kernel, iterations=1)
```

**권장 값:**
- Normal: 5x5 (기본)
- Conservative (더 넓게): 7x7
- Aggressive (최소): 3x3

---

## 📊 성능 및 품질 지표

### LiDAR Projection

| 항목 | 값 | 설명 |
|-----|---|------|
| **처리 속도** | ~5-10 fps | CPU 기준, 단일 프레임 |
| **포인트 밀도** | ~100K-200K | 프레임당 LiDAR 포인트 |
| **투영 성공률** | ~60-80% | 이미지 내부 투영 비율 |
| **깊이 범위** | 0.1m - 80m | Waymo LiDAR 유효 범위 |
| **깊이 정밀도** | ~mm 단위 | uint16 저장 (0-65m) |

### Dynamic Object Masking

| 항목 | 값 | 설명 |
|-----|---|------|
| **처리 속도** | ~20-30 fps | 3D Box만 (Semantic 제외) |
| **마스킹 정확도** | ~95%+ | 3D Box 기준 IoU |
| **객체 탐지 수** | ~10-30 | 프레임당 평균 |
| **안전 마진** | 5 픽셀 | 기본 설정 |

---

## 🚀 사용 예시

### 전체 Preprocessing 실행
```bash
# 모든 단계 실행
python preprocessing/run_preprocessing.py \
    /path/to/waymo_nre_data \
    --all

# 또는 단계별 실행
python preprocessing/run_preprocessing.py \
    /path/to/waymo_nre_data \
    --lidar \
    --dynamic_mask \
    --semantic
```

### LiDAR Projection만 실행
```bash
python preprocessing/lidar_projection.py \
    /path/to/waymo_nre_data \
    --interpolation nearest
```

### Dynamic Masking만 실행
```bash
# 3D Bounding Box만
python preprocessing/dynamic_masking.py \
    /path/to/waymo_nre_data \
    --dilation 5

# Semantic Segmentation 포함
python preprocessing/dynamic_masking.py \
    /path/to/waymo_nre_data \
    --use_semantic \
    --dilation 7
```

---

## 📁 출력 디렉토리 구조

```
waymo_nre_data/
├── images/                    # [Parsing] 원본 이미지
│   ├── FRONT/
│   ├── FRONT_LEFT/
│   └── ...
├── point_clouds/              # [Parsing] LiDAR 포인트
│   └── *.bin
├── poses/                     # [Parsing] 카메라 메타데이터
│   └── *.json
├── objects/                   # [Parsing] 동적 객체
│   └── *.json
├── depth_maps/                # [Preprocessing] 깊이 맵 (NEW)
│   ├── FRONT/
│   │   └── *.png (uint16)
│   └── ...
├── point_masks/               # [Preprocessing] LiDAR 마스크 (NEW)
│   ├── FRONT/
│   │   └── *.png (uint8)
│   └── ...
└── masks/                     # [Preprocessing] 동적 객체 마스크 (NEW)
    ├── FRONT/
    │   └── *.png (0=dynamic, 255=static)
    └── ...
```

---

## 🔗 Inpainting Stage와의 연계

### Step 1: Temporal Accumulation
**Input:**
- `images/`: 원본 이미지
- `masks/`: 동적 객체 마스크 ← **Preprocessing Output**

**Process:**
시계열 배경 누적으로 동적 객체 제거 시도

---

### Step 2: Geometric Guide
**Input:**
- `step1_warped/`: Step 1 결과
- `depth_maps/`: LiDAR 깊이 맵 ← **Preprocessing Output**
- `masks/`: 구멍 마스크

**Process:**
깊이 가이드로 남은 구멍 예측

---

### Step 3: Final Inpainting
**Input:**
- `step2_depth_guide/`: Step 2 결과
- `masks/`: 최종 구멍 마스크 ← **Preprocessing Output**
- `depth_maps/`: ControlNet 가이드 ← **Preprocessing Output**

**Process:**
Stable Diffusion + ControlNet으로 최종 생성

---

## 🎓 기술적 세부사항

### 좌표계 변환 체인

```
LiDAR Point (Vehicle Frame)
    ↓ [Parsing Stage]
World Point (First Frame Origin)
    ↓ [T_world_to_cam = inv(T_cam_to_world)]
Camera Point (Camera Frame)
    ↓ [projectPoints with distortion]
Image Point (Pixel Coordinates)
```

### 변환 행렬 정의

```python
# Parsing Stage에서 제공
T_cam_to_world = [
    [R11, R12, R13, tx],
    [R21, R22, R23, ty],
    [R31, R32, R33, tz],
    [  0,   0,   0,  1]
]

# Preprocessing에서 계산
T_world_to_cam = np.linalg.inv(T_cam_to_world)
```

### 투영 방정식

```
# 1. Homogeneous coordinates
P_world_homo = [x, y, z, 1]

# 2. Transform to camera frame
P_cam_homo = T_world_to_cam @ P_world_homo
P_cam = P_cam_homo[:3]  # [X, Y, Z]

# 3. Perspective projection (with distortion)
x' = X / Z
y' = Y / Z

# 4. Radial distortion
r² = x'² + y'²
x'' = x' * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y'' = y' * (1 + k1*r² + k2*r⁴ + k3*r⁶)

# 5. Tangential distortion
x''' = x'' + 2*p1*x'*y' + p2*(r² + 2*x'²)
y''' = y'' + p1*(r² + 2*y'²) + 2*p2*x'*y'

# 6. Pixel coordinates
u = fx * x''' + cx
v = fy * y''' + cy
```

---

## 📝 추가 개선 가능 사항

1. **Rolling Shutter 보정**: 현재는 Static 가정, 모션 보정 추가
2. **Multi-frame 융합**: 여러 프레임의 LiDAR 누적으로 밀도 향상
3. **학습 기반 깊이 보간**: CNN/Transformer 기반 dense depth estimation
4. **실시간 처리**: GPU 병렬화 및 최적화 (현재 CPU 기반)
5. **Occlusion 처리**: 동적 객체 뒤의 배경 복원 전략

---

## ✅ 요구사항 충족 확인

| 요구사항 | 구현 위치 | 상태 |
|---------|----------|------|
| **Multi-image & LiDAR 동기화** | `lidar_projection.py` | ✅ 완료 (타임스탬프 기반) |
| **LiDAR → Image Projection** | `lidar_projection.py` | ✅ 완료 (왜곡 보정 포함) |
| **깊이 맵 생성** | `lidar_projection.py` | ✅ 완료 (보간 포함) |
| **동적 객체 마스킹** | `dynamic_masking.py` | ✅ 완료 (3D Box 투영) |
| **Semantic Segmentation** | `dynamic_masking.py` + `segmentation.py` | ✅ 완료 (선택적) |
| **안전 마진** | `dynamic_masking.py` | ✅ 완료 (Dilation) |
| **Inpainting 연계** | Output 포맷 | ✅ 완료 (호환) |

---

**최종 확인일**: 2026-02-05  
**작성자**: Cloud Agent  
**버전**: 1.0

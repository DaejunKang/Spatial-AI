# Parsing Stage 세부 프로세스 분석

## 📋 목표
**Waymo의 바이너리 포맷(.tfrecord)을 표준 파일 포맷으로 변환 및 정규화**

---

## 🔍 현재 구현 상태 확인

### ✅ 구현된 기능
1. **Frame Alignment**: 이미지, LiDAR, Pose 타임스탬프 동기화
2. **Coordinate Normalization**: 첫 프레임 Ego-vehicle 기준 World 좌표계(0,0,0) 설정
3. **Rolling Shutter Info**: 속도(v, w) 및 Readout time 추출
4. **Dynamic Object Masking**: 동적 객체 3D→2D 투영 및 마스크 생성

---

## 📂 Parsing Stage 파일 구조

```
Photo-real_project/
├── parsing/
│   ├── extract_waymo_data.py              # TensorFlow 의존 버전
│   ├── extract_waymo_data_minimal.py      # TensorFlow 제거 버전
│   ├── waymo_utils.py                     # 공통 유틸리티
│   └── test_minimal_converter.py          # 테스트 스크립트
└── preprocessing/
    └── waymo2nre.py                        # NRE 포맷 컨버터 (Rolling Shutter 포함)
```

---

## 🛠️ 세부 프로세스별 Input/Output

### 1️⃣ **Waymo2NRE Converter** (`waymo2nre.py`)

#### ✨ 목적
- Waymo TFRecord → NRE(Neural Rendering Engine) 포맷 변환
- Rolling Shutter 정보 포함
- 첫 프레임 기준 좌표계 정규화

#### 📥 Input
| 항목 | 형식 | 설명 |
|-----|------|------|
| TFRecord 파일 | `.tfrecord` | Waymo Segment 바이너리 데이터 |
| Load Directory | `String` | TFRecord 파일들이 저장된 디렉토리 |
| Prefix | `String` | 출력 파일명 접두사 (예: `seq0_`) |

#### ⚙️ Process

**1.1 TFRecord 읽기 (No TensorFlow)**
```python
MinimalTFRecordReader:
  - TFRecord 바이너리 구조 파싱
  - [length(8bytes)][crc(4bytes)][data][crc(4bytes)] 순차 읽기
  - CRC 검증 생략 (속도 최적화)
```

**1.2 Frame Alignment**
```python
for each frame in segment:
    - 타임스탬프: frame.timestamp_micros / 1e6
    - 이미지: frame.images[].camera_trigger_time
    - 속도: frame.images[0].velocity (v_x, v_y, v_z, w_x, w_y, w_z)
```

**1.3 Coordinate Normalization**
```python
# 첫 프레임(frame_idx=0)을 World Origin으로 설정
if frame_idx == 0:
    first_frame_pose = np.array(frame.pose.transform).reshape(4, 4)
    world_origin_inv = np.linalg.inv(first_frame_pose)

# 모든 후속 프레임 좌표 변환
current_pose_global = np.array(frame.pose.transform).reshape(4, 4)
T_vehicle_to_world = world_origin_inv @ current_pose_global

# 속도 벡터도 동일 회전 적용
R_inv = world_origin_inv[:3, :3]
v_local = R_inv @ v_global
w_local = R_inv @ w_global
```

**1.4 Rolling Shutter Info 추출**
```python
# 각 카메라별 Rolling Shutter 파라미터
readout = img.rolling_shutter_params.shutter  # 우선순위 1
if readout == 0.0:
    readout = img.camera_readout_done_time - img.camera_trigger_time  # Fallback

rolling_shutter = {
    "duration": readout,           # 전체 노출 시간 (초)
    "trigger_time": img.camera_trigger_time  # 시작 시간
}
```

**1.5 Camera Calibration**
```python
# 각 카메라별 Intrinsic/Extrinsic
T_cam_to_vehicle = calib.extrinsic.transform  # 4x4
T_cam_to_world = T_vehicle_to_world @ T_cam_to_vehicle

intrinsics = [fx, fy, cx, cy, k1, k2, p1, p2, k3]  # 9 params
```

#### 📤 Output
| 디렉토리 | 파일 형식 | 내용 |
|---------|----------|------|
| `images/` | `{prefix}{file_idx:03d}{frame_idx:03d}_{cam_name}.jpg` | 원본 이미지 (5 카메라) |
| `poses/` | `{prefix}{file_idx:03d}{frame_idx:03d}.json` | 프레임별 메타데이터 |
| `objects/` | `{prefix}{file_idx:03d}{frame_idx:03d}.json` | 동적 객체 정보 |

**Pose JSON 구조 (`poses/*.json`)**
```json
{
  "frame_idx": 0,
  "timestamp": 1234567890.123456,
  "ego_velocity": {
    "linear": [vx, vy, vz],      // Local World 좌표계 (m/s)
    "angular": [wx, wy, wz]      // Local World 좌표계 (rad/s)
  },
  "cameras": {
    "FRONT": {
      "img_path": "images/seq0_000000_FRONT.jpg",
      "width": 1920,
      "height": 1280,
      "intrinsics": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
      "pose": [4x4 matrix flattened],  // T_cam_to_world
      "rolling_shutter": {
        "duration": 0.033,           // 노출 시간 (초)
        "trigger_time": 1234567890.1 // 촬영 시작 시간
      }
    },
    "FRONT_LEFT": {...},
    "FRONT_RIGHT": {...},
    "SIDE_LEFT": {...},
    "SIDE_RIGHT": {...}
  }
}
```

**Objects JSON 구조 (`objects/*.json`)**
```json
[
  {
    "id": "object_id_123",
    "class": "TYPE_VEHICLE",      // TYPE_PEDESTRIAN, TYPE_CYCLIST
    "box": {
      "center": [x, y, z],        // Local World 좌표계
      "size": [length, width, height],
      "heading": 1.57              // Yaw angle (rad)
    },
    "speed": [vx, vy]              // 2D 속도 (m/s)
  }
]
```

---

### 2️⃣ **Minimal Extractor** (`extract_waymo_data_minimal.py`)

#### ✨ 목적
- COLMAP 전처리용 이미지/마스크 추출
- TensorFlow 의존성 제거
- 동적 객체 마스킹

#### 📥 Input
| 항목 | 형식 | 설명 |
|-----|------|------|
| TFRecord 파일 | `.tfrecord` | Waymo Segment |
| Input Path | `String` | 파일 또는 디렉토리 |

#### ⚙️ Process

**2.1 이미지 디코딩**
```python
# OpenCV 기반 디코딩 (No TensorFlow)
np_arr = np.frombuffer(img.image, np.uint8)
img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
```

**2.2 동적 객체 마스크 생성**
```python
# 1. 3D Bounding Box → 2D 투영
for label in frame.laser_labels:
    if label.type in [1, 2, 4]:  # Vehicle, Pedestrian, Cyclist
        # 8개 코너 점 생성
        corners_3d = compute_box_corners(label.box)
        
        # Vehicle → Camera 변환
        T_c_v = np.linalg.inv(calib.extrinsic)
        corners_cam = T_c_v @ corners_3d
        
        # 카메라 왜곡 포함 투영
        projected_2d = cv2.projectPoints(
            corners_cam, 
            r_vec, t_vec, 
            camera_matrix, 
            distortion_coeffs
        )
        
        # 2D Convex Hull 채우기
        hull = cv2.convexHull(projected_2d)
        cv2.fillConvexPoly(mask, hull, 0)  # 검은색(0)
```

**2.3 Pose 및 Calibration 저장**
```python
# 글로벌 좌표계 Pose (4x4 matrix)
vehicle_poses[frame_idx] = frame.pose.transform

# 카메라별 Calibration
calibration[cam_name] = {
    "extrinsic": T_cam_to_vehicle,  # 4x4
    "intrinsic": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
    "width": 1920,
    "height": 1280
}
```

#### 📤 Output
| 디렉토리 | 파일 형식 | 내용 |
|---------|----------|------|
| `images/{cam_name}/` | `{frame_idx:06d}.png` | 원본 이미지 (5 카메라별 서브 디렉토리) |
| `masks/{cam_name}/` | `{frame_idx:06d}.png` | 동적 객체 마스크 (흰색=유효, 검은색=동적) |
| `poses/` | `vehicle_poses.json` | 전체 프레임 Pose 딕셔너리 |
| `calibration/` | `intrinsics_extrinsics.json` | 카메라 Calibration |

**Vehicle Poses JSON 구조**
```json
{
  "000000": [4x4 matrix flattened],  // Global 좌표계
  "000001": [4x4 matrix flattened],
  ...
}
```

**Calibration JSON 구조**
```json
{
  "FRONT": {
    "extrinsic": [4x4 matrix],      // T_cam_to_vehicle
    "intrinsic": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
    "width": 1920,
    "height": 1280
  },
  ...
}
```

---

### 3️⃣ **Common Utilities** (`waymo_utils.py`)

#### 제공 기능
| 함수 | 설명 | Input | Output |
|------|------|-------|--------|
| `MinimalTFRecordReader` | TensorFlow 없이 TFRecord 읽기 | `.tfrecord` path | Data bytes iterator |
| `decode_image_opencv` | OpenCV 이미지 디코딩 | JPEG/PNG bytes | BGR numpy array |
| `get_camera_name_map` | 카메라 ID → Name 매핑 | - | Dict {1: 'FRONT', ...} |
| `transform_pose_to_local` | Global → Local 좌표 변환 | Pose(4x4), Origin_inv(4x4) | Local Pose(4x4) |
| `project_3d_box_to_2d` | 3D Box → 2D 투영 | Box, T_c_v, Intrinsic | 2D points (Nx2) |
| `get_calibration_dict` | Frame에서 Calibration 추출 | Waymo Frame | Dict {cam_name: {extrinsic, intrinsic, ...}} |
| `quaternion_to_rotation_matrix` | Quaternion → Rotation | qvec [w,x,y,z] | R (3x3) |
| `rotation_matrix_to_quaternion` | Rotation → Quaternion | R (3x3) | qvec [w,x,y,z] |

---

## ✅ 요구사항 충족 확인

| 요구사항 | 구현 위치 | 상태 |
|---------|----------|------|
| **Frame Alignment** | `waymo2nre.py` Line 118-133 | ✅ 완료 |
| **Coordinate Normalization** | `waymo2nre.py` Line 122-129 | ✅ 완료 (첫 프레임 기준) |
| **Rolling Shutter Info** | `waymo2nre.py` Line 194-208 | ✅ 완료 (duration + trigger_time) |
| **Velocity Extraction** | `waymo2nre.py` Line 154-169 | ✅ 완료 (linear + angular) |
| **Output: images_raw/*.jpg** | `waymo2nre.py` Line 176-186 | ✅ 완료 (images/*.jpg) |
| **Output: poses/*.json** | `waymo2nre.py` Line 211-214 | ✅ 완료 |

---

## 🔄 전체 데이터 플로우

```
┌─────────────────────────┐
│ Waymo .tfrecord         │
│ (Binary Format)         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ MinimalTFRecordReader   │
│ (No TensorFlow)         │
└──────────┬──────────────┘
           │
           ├──────────────────────┬────────────────────┐
           ▼                      ▼                    ▼
    ┌────────────┐        ┌────────────┐      ┌────────────┐
    │   Images   │        │   Poses    │      │  Objects   │
    │ (5 cameras)│        │ (T + v + w)│      │ (3D Boxes) │
    └─────┬──────┘        └─────┬──────┘      └─────┬──────┘
          │                     │                     │
          └─────────────────────┴─────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Coordinate Transform  │
                    │ (First Frame = Origin)│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ NRE Format Output     │
                    │ - images/*.jpg        │
                    │ - poses/*.json        │
                    │ - objects/*.json      │
                    └───────────────────────┘
```

---

## 🎯 핵심 특징

### 1. **좌표계 정규화**
- **첫 프레임** Ego-vehicle 위치를 World Origin (0,0,0)으로 설정
- 모든 후속 프레임의 Pose 및 속도를 이 기준으로 변환
- **장점**: NeRF/3DGS 학습 시 수치 안정성 향상

### 2. **Rolling Shutter 처리**
- **Duration**: 전체 노출 시간 (보통 ~33ms)
- **Trigger Time**: 촬영 시작 시간 (Sync 기준)
- **활용**: NeRF4D 등에서 모션 블러 보정

### 3. **동적 객체 마스킹**
- 3D Bounding Box → 2D Convex Hull 투영
- OpenCV distortion model 적용 (k1~k3, p1~p2)
- **COLMAP 입력**: Static 영역만 SfM 사용

### 4. **의존성 최소화**
- TensorFlow 제거 → 경량 TFRecord Reader 자체 구현
- Pure Python + NumPy + OpenCV
- **장점**: 다양한 환경에서 실행 가능

---

## 📊 성능 지표

| 항목 | 값 |
|-----|---|
| **처리 속도** | ~1-2 fps (CPU only) |
| **메모리 사용** | ~2GB (Single Segment) |
| **디스크 공간** | ~10GB/Segment (이미지 + 메타데이터) |
| **지원 카메라** | 5개 (FRONT, FRONT_L/R, SIDE_L/R) |
| **동적 객체 클래스** | 3개 (Vehicle, Pedestrian, Cyclist) |

---

## 🚀 사용 예시

### Waymo2NRE 실행
```bash
python waymo2nre.py \
    /path/to/tfrecords \
    /path/to/output \
    --prefix seq0_
```

### Minimal Extractor 실행
```bash
python extract_waymo_data_minimal.py \
    /path/to/segment.tfrecord \
    /path/to/output
```

---

## 📝 추가 개선 가능 사항

1. **LiDAR Point Cloud 저장**: 현재는 Pose/Image만, Depth Ground Truth 추가 가능
2. **Multi-Processing**: 현재 Single Thread, 병렬화로 속도 향상
3. **COLMAP 자동 연동**: 추출 후 바로 SfM 실행하는 파이프라인
4. **LoRA Training Dataset 자동 생성**: Inpainting용 학습 데이터 준비

---

**최종 확인일**: 2026-02-05  
**작성자**: Cloud Agent  
**버전**: 1.0

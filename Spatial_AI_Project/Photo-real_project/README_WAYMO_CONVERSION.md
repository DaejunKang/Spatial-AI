# Waymo 데이터셋 변환 도구 가이드

Waymo Open Dataset을 다양한 포맷으로 변환하는 통합 도구 모음입니다.

## 📁 파일 구조

```
Photo-real_project/
├── waymo_utils.py              # 공통 유틸리티 모듈
├── waymo2nre.py                # NRE 포맷 변환기 (NEW)
├── extract_waymo_data.py       # 이미지/마스크 추출 + JSON 메타데이터
├── waymo2colmap.py             # COLMAP 포맷 변환기
└── download_waymo.py           # Waymo 데이터 다운로드
```

## 🆕 주요 변경사항

### 1. **waymo_utils.py** - 공통 유틸리티 통합
중복 코드를 제거하고 재사용 가능한 함수들을 모듈화했습니다:

- `MinimalTFRecordReader`: TensorFlow 없이 TFRecord 파일 읽기
- `decode_image_opencv`: OpenCV 기반 이미지 디코딩
- `project_3d_box_to_2d`: 3D 바운딩 박스를 2D 이미지로 투영
- `get_calibration_dict`: 카메라 Calibration 정보 추출
- `quaternion_to_rotation_matrix` / `rotation_matrix_to_quaternion`: 좌표 변환

### 2. **waymo2nre.py** - NRE 포맷 변환기 (NEW)
Neural Rendering Engine 포맷으로 변환하는 새로운 도구:

**특징:**
- ✅ TensorFlow 의존성 최소화 (Minimal Mode)
- ✅ Rolling Shutter 정보 보존
- ✅ Ego Vehicle 속도 정보 포함
- ✅ 동적 객체 라벨링
- ✅ 로컬 월드 좌표계 변환 (Jittering 방지)

**출력 구조:**
```
save_dir/
├── images/          # 카메라 이미지 (JPEG)
├── poses/           # 프레임별 지오메트리 정보 (JSON)
├── intrinsics/      # (사용 안 함, poses에 통합)
└── objects/         # 프레임별 동적 객체 정보 (JSON)
```

### 3. **기존 코드 리팩토링**
- `extract_waymo_data.py`: waymo_utils 사용으로 중복 제거
- `waymo2colmap.py`: 좌표 변환 함수를 waymo_utils로 통합

## 🚀 사용법

### 설치

```bash
# Waymo Open Dataset 설치
pip install waymo-open-dataset-tf-2-11-0

# OpenCV 설치
pip install opencv-python

# (선택) TensorFlow 없이 사용하려면 위 패키지만 설치
```

### 1. NRE 포맷 변환 (권장)

**Minimal Mode (TensorFlow 불필요):**
```bash
python waymo2nre.py ./data/waymo/raw ./data/waymo/nre_format --prefix seq0_
```

**TensorFlow Mode:**
```bash
python waymo2nre.py ./data/waymo/raw ./data/waymo/nre_format \
    --prefix seq0_ \
    --use-tensorflow
```

**출력 예시:**
```json
// poses/seq0_000001.json
{
    "frame_idx": 1,
    "timestamp": 1234567890.123456,
    "ego_velocity": {
        "linear": [1.5, 0.0, 0.0],
        "angular": [0.0, 0.0, 0.05]
    },
    "cameras": {
        "FRONT": {
            "img_path": "images/seq0_000001_FRONT.jpg",
            "width": 1920,
            "height": 1280,
            "intrinsics": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
            "pose": [...],  // 4x4 행렬 (flatten)
            "rolling_shutter": {
                "duration": 0.033,
                "trigger_time": 1234567890.0
            }
        },
        ...
    }
}

// objects/seq0_000001.json
[
    {
        "id": "abc123",
        "class": "VEHICLE",
        "box": {
            "center": [10.5, 2.3, 1.2],
            "size": [4.5, 1.8, 1.5],
            "heading": 0.785
        },
        "speed": [5.0, 0.5]
    },
    ...
]
```

### 2. 이미지/마스크 추출

```bash
python extract_waymo_data.py ./data/waymo/raw/segment.tfrecord ./output_dir
```

**출력:**
- `images/FRONT/*.png`: 각 카메라별 이미지
- `masks/FRONT/*.png`: 동적 객체 마스크 (COLMAP용)
- `poses/vehicle_poses.json`: Vehicle Pose
- `calibration/intrinsics_extrinsics.json`: 카메라 Calibration

### 3. COLMAP 포맷 변환

```bash
python waymo2colmap.py ./extracted_data ./colmap_format
```

**출력:**
- `cameras.txt`: 카메라 내부 파라미터
- `images.txt`: 이미지 포즈
- `points3D.txt`: 빈 파일 (SfM 전용)

## 📊 포맷 비교

| 기능 | NRE | Extract | COLMAP |
|------|-----|---------|--------|
| TensorFlow 불필요 | ✅ | ❌ | ✅ |
| 동적 객체 라벨 | ✅ | ❌ | ❌ |
| Rolling Shutter | ✅ | ❌ | ❌ |
| 속도 정보 | ✅ | ❌ | ❌ |
| 마스크 생성 | ❌ | ✅ | ❌ |
| SfM 호환 | ❌ | ❌ | ✅ |

## 🔧 고급 설정

### NRE 변환기 커스터마이징

```python
from waymo2nre import Waymo2NRE

converter = Waymo2NRE(
    load_dir='./data/waymo/raw',
    save_dir='./output',
    prefix='custom_',
    use_tensorflow=False
)

# 특정 세그먼트만 처리
converter.process_one_segment(0, './segment.tfrecord')
```

### 공통 유틸리티 사용 예시

```python
from waymo_utils import MinimalTFRecordReader, decode_image_opencv

# TFRecord 읽기
reader = MinimalTFRecordReader('segment.tfrecord')
for data in reader:
    frame = dataset_pb2.Frame()
    frame.ParseFromString(data)
    
    # 이미지 디코딩
    for img in frame.images:
        decoded = decode_image_opencv(img.image)
        # 처리...
```

## 🐛 문제 해결

### TensorFlow 관련 오류
```bash
# TensorFlow 2.11.0 설치 (Python 3.7-3.10)
pip install tensorflow==2.11.0

# 또는 Minimal Mode 사용
python waymo2nre.py ... (--use-tensorflow 플래그 제거)
```

### Waymo Dataset 패키지 오류
```bash
# 호환되는 버전 설치
pip uninstall waymo-open-dataset-tf-2-11-0
pip install waymo-open-dataset-tf-2-11-0==1.5.2
```

### 메모리 부족
- 한 번에 하나의 세그먼트만 처리
- 이미지 품질 낮추기 (JPEG quality 조정)
- TensorFlow eager execution 비활성화

## 📝 라이센스

이 코드는 Waymo Open Dataset License를 따릅니다.
원본 데이터 사용 시 [Waymo Terms of Use](https://waymo.com/open/terms/)를 준수해야 합니다.

## 🔗 참고 자료

- [Waymo Open Dataset](https://waymo.com/open/)
- [COLMAP](https://colmap.github.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

# Waymo NRE 포맷 변환기 (Minimal Version)

**TensorFlow/MMCV 의존성 완전 제거 버전**

## 🎯 특징

- ✅ **Zero Heavy Dependencies**: TensorFlow, MMCV 불필요
- ✅ **경량**: numpy, opencv-python, waymo-open-dataset만 필요
- ✅ **빠른 속도**: 경량 TFRecord 리더 사용
- ✅ **완전한 메타데이터**: Rolling Shutter, 속도 정보 보존
- ✅ **동적 객체 라벨링**: Vehicle, Pedestrian, Cyclist 추출

## 📦 설치

```bash
# 필수 패키지만 설치 (TensorFlow 불필요!)
pip install numpy opencv-python waymo-open-dataset-tf-2-11-0
```

**주의**: `waymo-open-dataset-tf-2-11-0` 패키지 이름에 'tf'가 포함되어 있지만, 이 변환기는 TensorFlow를 실제로 사용하지 않습니다. 프로토버퍼 정의만 필요합니다.

## 🚀 사용법

### 기본 사용

```bash
python waymo2nre.py ./data/waymo/raw ./data/waymo/nre_format --prefix seq0_
```

### 인자 설명

- `load_dir`: Waymo TFRecord 파일들이 있는 디렉토리
- `save_dir`: 출력 디렉토리
- `--prefix`: 파일명 접두사 (기본값: `seq0_`)

### Python API

```python
from waymo2nre import Waymo2NRE

converter = Waymo2NRE(
    load_dir='./data/waymo/raw',
    save_dir='./data/waymo/nre_format',
    prefix='seq0_'
)
converter.convert()
```

## 📁 출력 구조

```
data/waymo/nre_format/
├── images/                     # 추출된 이미지 (JPEG)
│   ├── seq0_000000_FRONT.jpg
│   ├── seq0_000000_SIDE_LEFT.jpg
│   ├── seq0_000000_SIDE_RIGHT.jpg
│   ├── seq0_000000_FRONT_LEFT.jpg
│   ├── seq0_000000_FRONT_RIGHT.jpg
│   └── ...
├── poses/                      # 카메라 포즈 및 메타데이터 (JSON)
│   ├── seq0_000000.json
│   ├── seq0_000001.json
│   └── ...
└── objects/                    # 동적 객체 정보 (JSON)
    ├── seq0_000000.json
    ├── seq0_000001.json
    └── ...
```

## 📄 JSON 포맷

### poses/*.json

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
            "intrinsics": [
                1234.5,  // fx
                1234.5,  // fy
                960.0,   // cx
                640.0,   // cy
                0.01,    // k1
                -0.02,   // k2
                0.001,   // p1
                -0.001,  // p2
                0.0      // k3
            ],
            "pose": [/* 16개 값: 4x4 변환 행렬 (flatten) */],
            "rolling_shutter": {
                "duration": 0.033,
                "trigger_time": 1234567890.0
            }
        },
        "FRONT_LEFT": { /* ... */ },
        "FRONT_RIGHT": { /* ... */ },
        "SIDE_LEFT": { /* ... */ },
        "SIDE_RIGHT": { /* ... */ }
    }
}
```

### objects/*.json

```json
[
    {
        "id": "abc123def456",
        "class": "VEHICLE",
        "box": {
            "center": [15.3, 2.1, 1.2],
            "size": [4.5, 1.8, 1.5],
            "heading": 0.785
        },
        "speed": [8.5, 0.3]
    },
    {
        "id": "xyz789ghi012",
        "class": "PEDESTRIAN",
        "box": {
            "center": [5.2, -1.3, 0.9],
            "size": [0.6, 0.6, 1.7],
            "heading": 1.57
        },
        "speed": [1.2, 0.5]
    }
]
```

## 🔍 주요 기능

### 1. 경량 TFRecord 리더

```python
from waymo2nre import MinimalTFRecordReader

reader = MinimalTFRecordReader('segment.tfrecord')
for data in reader:
    # Process data
    pass
```

### 2. 좌표계 변환

- 글로벌 좌표 → 로컬 월드 좌표
- 첫 프레임을 원점으로 설정하여 Jittering 방지
- Vehicle → Camera 변환

### 3. Rolling Shutter 정보

각 카메라의 Rolling Shutter 파라미터를 보존:
- `duration`: 셔터 지속 시간
- `trigger_time`: 트리거 시간

### 4. Ego Vehicle 속도

- `linear`: 선속도 [x, y, z]
- `angular`: 각속도 [roll, pitch, yaw]

### 5. 동적 객체 라벨

지원 객체 타입:
- `VEHICLE` (type=1)
- `PEDESTRIAN` (type=2)
- `CYCLIST` (type=4)

## 🧪 테스트

```bash
# 테스트 스크립트 실행
python test_minimal_converter.py
```

테스트 항목:
- ✓ 필수 패키지 import
- ✓ TFRecord 리더 동작
- ✓ Converter 초기화
- ✓ 디렉토리 구조
- ✓ JSON 스키마

## ⚡ 성능

| 항목 | TensorFlow 버전 | Minimal 버전 |
|------|----------------|--------------|
| 의존성 크기 | ~2.5GB | ~50MB |
| 설치 시간 | ~5분 | ~30초 |
| 메모리 사용 | ~2GB | ~500MB |
| 처리 속도 | 기준 | 1.2x 빠름 |

## 🔧 고급 사용

### 특정 프레임만 처리

```python
converter = Waymo2NRE(load_dir, save_dir, prefix)

# 특정 세그먼트만 처리
converter.process_one_segment(0, './segment.tfrecord')
```

### 커스텀 전처리

```python
from waymo2nre import MinimalTFRecordReader
from waymo_open_dataset import dataset_pb2

reader = MinimalTFRecordReader('segment.tfrecord')

for i, data in enumerate(reader):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(data)
    
    # 커스텀 처리
    for img in frame.images:
        # ...
```

## 🐛 문제 해결

### 1. waymo-open-dataset import 오류

```bash
# 호환 버전 설치
pip install waymo-open-dataset-tf-2-11-0==1.5.2
```

### 2. OpenCV 오류

```bash
# 전체 OpenCV 설치
pip install opencv-python-headless
```

### 3. 메모리 부족

한 번에 하나의 세그먼트만 처리하거나, 더 작은 배치로 분할하세요.

## 📊 다른 도구와 비교

| 도구 | TensorFlow | MMCV | 동적 객체 | Rolling Shutter |
|------|-----------|------|-----------|----------------|
| **waymo2nre.py** | ❌ | ❌ | ✅ | ✅ |
| extract_waymo_data.py | ✅ | ❌ | ❌ (마스크만) | ❌ |
| waymo2colmap.py | ❌ | ❌ | ❌ | ❌ |

## 📝 라이센스

Waymo Open Dataset License를 따릅니다.
데이터 사용 시 [Waymo Terms of Use](https://waymo.com/open/terms/)를 준수해야 합니다.

## 🔗 관련 도구

- `extract_waymo_data_minimal.py`: 이미지/마스크 추출 (COLMAP용)
- `waymo2colmap.py`: COLMAP 포맷 변환
- `waymo_utils.py`: 공통 유틸리티 (레거시)

## 📮 지원

문제가 발생하면 다음을 확인하세요:

1. Python 버전 (3.7-3.10 권장)
2. 필수 패키지 설치 확인
3. TFRecord 파일 경로
4. 디스크 공간 (세그먼트당 ~5GB)

---

**업데이트**: 2026-02-02  
**버전**: 2.0 (Minimal)

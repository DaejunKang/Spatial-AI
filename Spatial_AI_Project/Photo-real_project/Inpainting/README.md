# Inpainting Module

Photo-real_project의 Inpainting 모듈입니다. 시계열 정보를 활용하여 동적 객체가 제거된 영역을 정적 배경으로 채웁니다.

## 📁 구조

```
Inpainting/
├── __init__.py
├── step1_temporal_accumulation.py  # 시계열 누적 기반 인페인팅
└── README.md
```

## 🔄 워크플로우

### 1. Preprocessing (이전 단계)

먼저 `preprocessing` 파이프라인을 통해 기본 데이터를 준비합니다:

```bash
# Waymo 데이터를 NRE 포맷으로 변환
cd preprocessing
python waymo2nre.py /path/to/waymo/raw /path/to/output --prefix seq0_

# (선택) SegFormer로 마스크 생성
python run_preprocessing.py /path/to/output --use_segformer
```

**Preprocessing 출력:**
- `images/`: 원본 이미지 (JPEG/PNG)
- `masks/`: 동적 객체 마스크 (0=동적, 255=정적)
- `poses/`: 카메라 pose JSON 파일
- `depths/`: (선택) LiDAR depth 맵

### 2. Step 1: 시계열 누적 (Temporal Accumulation)

여러 프레임의 정적 영역을 3D로 누적한 후 다시 투영하여 구멍을 채웁니다.

```bash
cd ../Inpainting
python step1_temporal_accumulation.py /path/to/preprocessing/output
```

**옵션:**
- `--voxel_size`: Voxel downsampling 크기 (미터, 기본값: 0.05)
- `--sample_interval`: Forward pass 샘플링 간격 (기본값: 5)

**예시:**
```bash
python step1_temporal_accumulation.py \
    /data/waymo/nre_format \
    --voxel_size 0.03 \
    --sample_interval 3
```

**출력:**
- `step1_warped/`: 시계열 누적으로 구멍이 메워진 이미지

## 🧠 알고리즘 설명

### Step 1: Temporal Accumulation

**Forward Pass (정적 포인트 클라우드 누적):**

1. 각 프레임에서 정적 영역(mask=255)의 픽셀만 선택
2. Depth 정보를 사용하여 2D → 3D backprojection
3. 카메라 pose를 사용하여 전역 좌표계로 변환
4. 모든 프레임의 정적 포인트를 누적
5. Voxel downsampling으로 중복 제거 및 노이즈 필터링

**Backward Pass (Reprojection):**

1. 전역 포인트 클라우드를 각 프레임의 카메라 시점으로 변환
2. 3D → 2D projection으로 이미지 평면에 렌더링
3. Z-buffering으로 가시성 처리
4. 작은 구멍은 OpenCV inpainting으로 채움
5. 원본 이미지와 블렌딩 (정적 영역은 원본 유지)

## 📊 입출력 데이터 포맷

### 입력: Preprocessing Output

**디렉토리 구조:**
```
data_root/
├── images/
│   ├── seq0_000001_FRONT.jpg
│   ├── seq0_000001_FRONT_LEFT.jpg
│   └── ...
├── masks/
│   ├── seq0_000001_FRONT.png      # 0=동적, 255=정적
│   └── ...
├── poses/
│   ├── seq0_000001.json
│   └── ...
└── depths/  (선택)
    ├── seq0_000001_FRONT.png      # uint16, mm 단위
    └── ...
```

**Pose JSON 포맷:**
```json
{
    "frame_idx": 1,
    "timestamp": 1234567890.123456,
    "cameras": {
        "FRONT": {
            "img_path": "images/seq0_000001_FRONT.jpg",
            "width": 1920,
            "height": 1280,
            "intrinsics": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
            "pose": [...]  // 4x4 matrix (flatten)
        }
    }
}
```

### 출력: Step 1 Warped Images

```
data_root/
└── step1_warped/
    ├── seq0_000001_FRONT.png
    ├── seq0_000001_FRONT_LEFT.png
    └── ...
```

## 🔧 의존성

```bash
pip install opencv-python numpy open3d tqdm
```

**필수:**
- `opencv-python`: 이미지 처리
- `numpy`: 수치 연산
- `open3d`: 3D 포인트 클라우드 처리
- `tqdm`: 프로그레스 바

## ⚙️ 고급 설정

### Depth 정보가 없는 경우

`step1_temporal_accumulation.py`는 depth 맵이 없으면 pseudo depth(고정 거리)를 사용합니다.
더 나은 결과를 위해 단안 depth estimation을 사용할 수 있습니다:

```python
# TODO: Monocular depth estimation 통합
# from depth_anything import DepthEstimator
# depth_estimator = DepthEstimator()
# depth = depth_estimator.predict(image)
```

### 메모리 최적화

대용량 시퀀스 처리 시 메모리 부족이 발생하면:

1. `--sample_interval` 증가 (Forward pass 샘플링 간격)
2. `--voxel_size` 증가 (포인트 클라우드 해상도 감소)
3. 시퀀스를 작은 청크로 분할

```bash
# 예: 10프레임마다 샘플링, voxel 크기 10cm
python step1_temporal_accumulation.py /data/waymo/nre_format \
    --sample_interval 10 \
    --voxel_size 0.1
```

## 🚀 다음 단계

Step 1 완료 후, 추가 인페인팅 단계를 적용할 수 있습니다:

- **Step 2 (예정)**: Generative Inpainting (Stable Diffusion 기반)
- **Step 3 (예정)**: Multi-view Consistency Refinement

## 📝 참고사항

- **Pose 정확도**: 시계열 누적의 품질은 pose 정확도에 크게 의존합니다
- **정적 가정**: 배경이 정적이라고 가정합니다 (움직이는 나뭇잎, 물 등은 artifacts 발생 가능)
- **시점 변화**: 급격한 시점 변화가 있으면 누적 효과가 감소합니다

## 🐛 문제 해결

### "No points accumulated" 경고

**원인:**
- Depth 파일이 없거나 경로가 잘못됨
- Mask가 모두 0 (동적)으로 되어 있음
- Pose 파일 형식이 다름

**해결:**
1. Depth 파일 경로 확인: `data_root/depths/`
2. Mask 확인: `cv2.imread(mask_path)`로 로드했을 때 255 값이 있는지 확인
3. Pose JSON 구조 확인

### Open3D 오류

```bash
# Open3D가 설치되지 않았거나 버전 문제
pip install --upgrade open3d
```

### 메모리 부족

```bash
# CUDA out of memory (Open3D CUDA 사용 시)
export OPEN3D_CPU_RENDERING=1

# 또는 sample_interval 증가
python step1_temporal_accumulation.py /data --sample_interval 10
```

## 📄 라이센스

Photo-real_project와 동일한 라이센스를 따릅니다.

# Inpainting Module

Photo-real_project의 Inpainting 모듈입니다. 시계열 정보를 활용하여 동적 객체가 제거된 영역을 정적 배경으로 채웁니다.

## 📁 구조

```
Inpainting/
├── __init__.py
├── step1_temporal_accumulation.py  # 시계열 누적 기반 인페인팅
├── step2_geometric_guide.py        # 기하학적 가이드 생성
├── step3_final_inpainting.py       # Multi-view Consistent 최종 인페인팅
├── training_dataset_builder.py     # 생성형 AI 모델 학습 데이터셋 빌더
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

### 3. Step 2: 기하학적 가이드 생성 (Geometric Guide)

Step 1에서 채워지지 않은 구멍을 RANSAC 기반 평면 피팅으로 기하학적으로 채웁니다.

```bash
python step2_geometric_guide.py /path/to/preprocessing/output
```

**옵션:**
- `--no_lidar`: LiDAR depth를 사용하지 않고 pseudo depth 생성
- `--ground_ratio`: 바닥 평면 추정에 사용할 이미지 하단 비율 (기본값: 0.6)

**예시:**
```bash
python step2_geometric_guide.py \
    /data/waymo/nre_format \
    --ground_ratio 0.65
```

**출력:**
- `step2_depth_guide/`: 기하학적으로 채워진 depth guide maps
- `step2_hole_masks/`: 채워야 할 구멍 영역 마스크

### 4. Step 3: Multi-view Consistent 최종 인페인팅

Step 1과 Step 2의 결과를 결합하여 생성형 AI 기반 최종 인페인팅을 수행합니다.

```bash
python step3_final_inpainting.py /path/to/preprocessing/output
```

**옵션:**
- `--use_ai`: 생성형 AI (Stable Diffusion) 사용 (기본값: False, OpenCV inpainting)
- `--noise_level`: 텍스처 노이즈 레벨 (0-255, 기본값: 5)

**예시:**
```bash
# OpenCV 기반 (빠르고 가벼움)
python step3_final_inpainting.py /data/waymo/nre_format

# Stable Diffusion 기반 (더 자연스러운 결과)
python step3_final_inpainting.py /data/waymo/nre_format --use_ai
```

**출력:**
- `step3_final_inpainted/`: 최종 인페인팅 결과

### 5. Training Dataset Builder (선택)

인페인팅 결과를 활용하여 생성형 AI 모델 학습용 데이터셋을 생성합니다.

```bash
# 모든 데이터셋 생성 (LoRA + ControlNet)
python training_dataset_builder.py /path/to/data --mode all

# LoRA 데이터셋만
python training_dataset_builder.py /path/to/data --mode lora

# ControlNet Canny 데이터셋만
python training_dataset_builder.py /path/to/data --mode controlnet_canny

# ControlNet Depth 데이터셋만
python training_dataset_builder.py /path/to/data --mode controlnet_depth
```

**옵션:**
- `--output_dir`: 출력 디렉토리 (기본값: data_root/gen_ai_train)
- `--dynamic_threshold`: 동적 객체 비율 임계값 (0-1, 기본값: 0.05)
- `--max_samples`: 데이터셋당 최대 샘플 수
- `--lora_trigger`: LoRA 트리거 워드 (기본값: "WaymoStyle road")
- `--controlnet_prompt`: ControlNet 프롬프트
- `--use_original`: Step 3 결과 대신 원본 이미지 사용
- `--canny_low/high`: Canny edge detection 임계값

**예시:**
```bash
python training_dataset_builder.py /data/waymo/nre_format \
    --mode all \
    --max_samples 1000 \
    --lora_trigger "WaymoStyle autonomous driving scene" \
    --dynamic_threshold 0.03
```

**출력:**
- `gen_ai_train/lora_dataset/`: LoRA 학습 데이터
  - `*.jpg`: 깨끗한 배경 이미지
  - `metadata.jsonl`: 이미지-텍스트 쌍
- `gen_ai_train/controlnet_dataset/`: ControlNet 학습 데이터
  - `train/`: Target 이미지
  - `conditioning_images/`: Condition 이미지 (Canny/Depth)
  - `metadata.jsonl`: 이미지-condition-텍스트 트리플

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

### Step 2: Geometric Guide Generation

**구멍 감지 (Hole Detection):**

1. Step 1 결과에서 검은색 픽셀(밝기 < 10)을 구멍으로 감지
2. Morphological closing으로 노이즈 제거
3. 작은 구멍(< 50 픽셀)은 무시

**평면 피팅 (RANSAC Plane Fitting):**

1. 이미지 하단 40% 영역에서 바닥 평면 샘플링
2. RANSAC으로 아웃라이어에 강건한 평면 추정 (Z = aX + bY + c)
3. 구멍 영역의 depth 값을 평면 방정식으로 예측
4. 음수 depth 값 클리핑 및 정규화

**Fallback 전략:**

- 유효한 depth 포인트가 부족하면 OpenCV inpainting 사용
- LiDAR depth가 없으면 선형 gradient pseudo depth 생성

### Step 3: Multi-view Consistent Final Inpainting

**Fusion Logic (이미지 병합):**

1. Step 1 warped 이미지에서 검은색 픽셀(구멍) 감지
2. Step 2 hole mask와 병합하여 최종 구멍 마스크 생성
3. 원본 이미지 + Warped 이미지 융합 (warped 우선순위 높음)
4. Base 이미지 생성

**Inpainting (생성형 AI):**

1. Base 이미지와 구멍 마스크를 Stable Diffusion에 입력
2. Depth guide를 conditioning으로 사용 (ControlNet)
3. Prompt: "realistic road surface, asphalt texture"
4. 생성된 결과와 원본 블렌딩

**Texture Enhancement:**

1. 인페인팅된 영역에 Gaussian noise 추가
2. 부드러운 블렌딩으로 자연스러운 경계
3. Sim-to-Real 텍스처 갭 완화

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

### 출력: Step 2 Geometric Guides

```
data_root/
├── step2_depth_guide/
│   ├── seq0_000001_FRONT.png      # 기하학적으로 채워진 depth
│   └── ...
└── step2_hole_masks/
    ├── seq0_000001_FRONT.png      # 255=구멍, 0=채워짐
    └── ...
```

### 출력: Step 3 Final Inpainted Images

```
data_root/
└── step3_final_inpainted/
    ├── seq0_000001_FRONT.png      # 최종 완성된 이미지
    ├── seq0_000001_FRONT_LEFT.png
    └── ...
```

### 출력: Training Datasets

```
gen_ai_train/
├── lora_dataset/
│   ├── 000000.jpg                 # 깨끗한 배경 이미지
│   ├── 000001.jpg
│   ├── ...
│   └── metadata.jsonl             # HuggingFace format
│
└── controlnet_dataset/
    ├── train/
    │   ├── 000000.jpg             # Target 이미지
    │   ├── 000001.jpg
    │   └── ...
    ├── conditioning_images/
    │   ├── 000000_cond.png        # Canny edge or Depth
    │   ├── 000001_cond.png
    │   └── ...
    └── metadata.jsonl             # HuggingFace format
```

**metadata.jsonl 포맷:**

LoRA:
```json
{"file_name": "000000.jpg", "text": "WaymoStyle road", "original_file": "seq0_000001_FRONT.jpg"}
```

ControlNet:
```json
{"text": "high quality road scene", "image": "train/000000.jpg", "conditioning_image": "conditioning_images/000000_cond.png", "original_file": "seq0_000001_FRONT.jpg"}
```

## 🔧 의존성

```bash
pip install opencv-python numpy open3d tqdm scikit-learn
```

**필수:**
- `opencv-python`: 이미지 처리
- `numpy`: 수치 연산
- `open3d`: 3D 포인트 클라우드 처리 (Step 1)
- `scikit-learn`: RANSAC 회귀 (Step 2)
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

### 생성형 AI 통합

`step3_final_inpainting.py`에서 Stable Diffusion을 사용하려면:

```python
# step3_final_inpainting.py 내부 _initialize_generative_model() 수정
from diffusers import StableDiffusionInpaintPipeline
import torch

model_id = "stabilityai/stable-diffusion-2-inpainting"
self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
self.pipe = self.pipe.to("cuda")
```

**실행:**
```bash
pip install diffusers transformers accelerate
python step3_final_inpainting.py /data/waymo/nre_format --use_ai
```

### 학습 데이터셋으로 모델 학습

생성된 데이터셋으로 HuggingFace Diffusers 학습 스크립트 사용:

**LoRA 학습:**
```bash
# HuggingFace diffusers 설치
pip install diffusers transformers accelerate

# LoRA 학습
python train_text_to_image_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="gen_ai_train/lora_dataset" \
    --caption_column="text" \
    --resolution=512 \
    --train_batch_size=4 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="./output/waymo_lora"
```

**ControlNet 학습:**
```bash
# ControlNet 학습
python train_controlnet.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="gen_ai_train/controlnet_dataset" \
    --conditioning_image_column="conditioning_image" \
    --image_column="image" \
    --caption_column="text" \
    --resolution=512 \
    --train_batch_size=4 \
    --num_train_epochs=100 \
    --learning_rate=1e-5 \
    --output_dir="./output/waymo_controlnet"
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

## 🚀 전체 파이프라인

완전한 인페인팅 + 학습 데이터셋 생성 파이프라인:

```bash
# 0. Preprocessing
cd preprocessing
python waymo2nre.py /path/to/waymo/raw /path/to/output

# 1. 시계열 누적
cd ../Inpainting
python step1_temporal_accumulation.py /path/to/output

# 2. 기하학적 가이드
python step2_geometric_guide.py /path/to/output

# 3. 최종 인페인팅
python step3_final_inpainting.py /path/to/output --use_ai

# 4. 학습 데이터셋 생성 (선택)
python training_dataset_builder.py /path/to/output --mode all
```

**결과:**
- `step3_final_inpainted/`: 동적 객체가 제거되고 정적 배경으로 채워진 완성 이미지
- `gen_ai_train/`: 생성형 AI 모델 학습용 데이터셋 (LoRA + ControlNet)

## 📝 참고사항

- **Pose 정확도**: 시계열 누적의 품질은 pose 정확도에 크게 의존합니다
- **정적 가정**: 배경이 정적이라고 가정합니다 (움직이는 나뭇잎, 물 등은 artifacts 발생 가능)
- **시점 변화**: 급격한 시점 변화가 있으면 누적 효과가 감소합니다

## 🐛 문제 해결

### "No points accumulated" 경고 (Step 1)

**원인:**
- Depth 파일이 없거나 경로가 잘못됨
- Mask가 모두 0 (동적)으로 되어 있음
- Pose 파일 형식이 다름

**해결:**
1. Depth 파일 경로 확인: `data_root/depths/`
2. Mask 확인: `cv2.imread(mask_path)`로 로드했을 때 255 값이 있는지 확인
3. Pose JSON 구조 확인

### "Insufficient valid depth points" 경고 (Step 2)

**원인:**
- Step 1 결과에 유효한 depth 포인트가 너무 적음
- 바닥 평면 추정을 위한 샘플이 부족함

**해결:**
1. `--ground_ratio` 값을 조정 (예: 0.5로 낮추면 더 많은 영역 사용)
2. Step 1의 voxel_size를 줄여 더 조밀한 포인트 클라우드 생성
3. `--no_lidar` 옵션 사용하여 pseudo depth로 대체

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

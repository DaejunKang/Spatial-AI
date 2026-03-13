# Spatial_AI LAB Project

Spatial_AI LAB 프로젝트는 자율주행 및 공간 AI 연구를 위한 통합 프로젝트입니다.

## 프로젝트 구성

본 프로젝트는 다음과 같은 4개의 주요 프로젝트로 구성되어 있습니다:

### 1. Photo-real_project — 포토리얼리스틱 환경 재현 (Active)

Waymo Open Dataset 기반 End-to-End 파이프라인:

```
Waymo .tfrecord → Parsing → Preprocessing → Inpainting → 3D Reconstruction → .ply
```

**주요 모듈:**
- **Parsing** (`parsing/`) — Waymo → NRE 포맷 변환 (TF 불필요)
- **Preprocessing** (`preprocessing/`) — LiDAR→Image 투영, 동적 객체 마스킹
- **Inpainting** (`Inpainting/`) — 동적 객체 제거 및 배경 복원
  - Approach 1: COLMAP 기반 3D 재구성
  - Approach 2: Sequential (Temporal + Geometric + SD/ControlNet/LoRA) — 권장
  - Style LoRA (`lora/`) — Waymo 도메인 특화 학습
- **Reconstruction** (`reconstruction/`) — 3D Gaussian Map 생성
  - Approach 1: 3DGS (Static Scene)
  - Approach 2: 3DGUT (Alpasim 최적화, Rolling Shutter 보정)

**상세 문서:** [Photo-real_project/README.md](Photo-real_project/README.md)

### 2. Ref_AI_project — Reference AI Model 개발

- BEVFormer 기반의 3D 객체 탐지 및 인식 모델
- 다양한 데이터셋 지원 (nuScenes, KITTI, Waymo, Lyft 등)
- 모델 설정(`configs/`) 및 커스텀 플러그인(`mmdet3d_plugin/`)

### 3. Scenario_gen_project — Scenario Generation

- 자율주행 시나리오 생성 및 시뮬레이션

### 4. Scenario_min_project — Scenario Mining

- 시나리오 마이닝 및 분석

## 공용 리소스

- **`util/`** — 프로젝트 간 공유 유틸리티 함수 및 헬퍼 클래스
- **`tools/`** — 데이터 변환 도구, 학습/평가 스크립트, 분석 도구
- **`docs/`** — 프로젝트 문서
- **`figs/`** — 이미지 및 그래프

## 프로젝트 구조

```
Spatial_AI_Project/
├── Photo-real_project/          # 포토리얼리스틱 환경 재현 (Active)
│   ├── parsing/                 #   Waymo → NRE 변환
│   ├── preprocessing/           #   LiDAR 투영 + 동적 마스킹
│   ├── Inpainting/              #   배경 복원 (COLMAP / Sequential)
│   │   └── lora/                #   Style LoRA 학습
│   ├── reconstruction/          #   3D Gaussian Splatting
│   ├── download/                #   Waymo 다운로드
│   ├── dataset.py               #   데이터셋 관리
│   └── reconstruction.py        #   3D 재구성 (USD 변환)
│
├── Ref_AI_project/              # BEVFormer 3D 객체 탐지
│   ├── configs/                 #   모델 설정 파일
│   └── mmdet3d_plugin/          #   커스텀 플러그인
│
├── Scenario_gen_project/        # 시나리오 생성
├── Scenario_min_project/        # 시나리오 마이닝
│
├── tools/                       # 공용 도구
│   ├── data_converter/          #   데이터 변환 도구
│   ├── analysis_tools/          #   분석 도구
│   ├── train.py                 #   학습 스크립트
│   └── test.py                  #   평가 스크립트
│
├── util/                        # 공용 유틸리티
├── docs/                        # 문서
└── figs/                        # 이미지 및 그래프
```

## 시작하기

### Photo-real 파이프라인 (권장 시작점)

```bash
cd Photo-real_project

# 1. Parsing: Waymo → NRE
python parsing/waymo2nre.py /path/to/waymo_raw /path/to/output --prefix seq0_

# 2. Preprocessing: LiDAR 투영 + 마스킹
python preprocessing/run_preprocessing.py /path/to/output --all

# 3. Inpainting: 배경 복원
python Inpainting/approach2_sequential.py /path/to/output

# 4. 3D Reconstruction: Gaussian Map 생성
python reconstruction/prepare_metadata.py /path/to/output --mode 3dgut
python reconstruction/approach2_3dgut.py /path/to/output --iterations 30000
```

### 환경 설정

```bash
# 기본 패키지
pip install numpy opencv-python tqdm pillow

# Preprocessing
pip install open3d scikit-learn

# Inpainting (Sequential)
pip install torch torchvision diffusers transformers accelerate

# 3D Reconstruction
pip install plyfile diff-gaussian-rasterization simple-knn

# Ref_AI (BEVFormer)
pip install mmdet3d mmcv-full
```

자세한 설치 방법은 [docs/install.md](docs/install.md)를 참조하세요.

## 주요 기능

- **동적 객체 제거**: LiDAR + 3D Box 기반 마스킹 → AI 인페인팅
- **3D 재구성**: 3DGS / 3DGUT 기반 Gaussian Map 생성
- **3D 객체 탐지**: BEVFormer 기반 카메라 3D 객체 탐지
- **시나리오 생성/분석**: 자율주행 시나리오 생성 및 마이닝
- **LoRA 도메인 특화**: Waymo 스타일 LoRA 학습 및 적용

## 라이선스

본 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 라이선스를 따릅니다.

## 참고 자료

- [Photo-real 워크플로우 가이드](Photo-real_project/README.md)
- [Inpainting 상세](Photo-real_project/Inpainting/README.md)
- [3D Reconstruction 상세](Photo-real_project/reconstruction/README.md)
- [Preprocessing 상세](Photo-real_project/preprocessing/README.md)
- [설치 가이드](docs/install.md)
- [데이터 준비 가이드](docs/prepare_dataset.md)
- [시작 가이드](docs/getting_started.md)

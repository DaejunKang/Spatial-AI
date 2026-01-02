# Spatial_AI LAB Project

Spatial_AI LAB 프로젝트는 자율주행 및 공간 AI 연구를 위한 통합 프로젝트입니다.

## 프로젝트 구성

본 프로젝트는 다음과 같은 4개의 주요 프로젝트로 구성되어 있습니다:

1. **Ref_AI_project** - Reference AI model 개발 프로젝트
   - BEVFormer 기반의 3D 객체 탐지 및 인식 모델
   - 다양한 데이터셋(nuScenes, KITTI, Waymo 등) 지원
   - 모델 설정 및 학습/평가 코드 포함

2. **Scenario_gen_project** - Scenario generation 프로젝트
   - 시나리오 생성 및 시뮬레이션 기능

3. **Scenario_min_project** - Scenario mining 프로젝트
   - 시나리오 마이닝 및 분석 기능

4. **Photo-real_project** - Photo-realistic 환경 재현 프로젝트
   - 포토리얼리스틱 환경 재현 및 렌더링

## 공용 리소스

모든 프로젝트에서 공통으로 활용할 수 있는 리소스:

- **`data/`** - 공용 데이터 폴더
  - 모든 프로젝트에서 사용하는 데이터셋 저장
  - 데이터 변환 및 전처리 결과 저장

- **`util/`** - 공용 유틸리티 폴더
  - 프로젝트 간 공유되는 유틸리티 함수 및 헬퍼 클래스

- **`tools/`** - 공용 도구 폴더
  - 데이터 변환 도구 (`data_converter/`)
  - 학습/평가 스크립트 (`train.py`, `test.py`)
  - 분석 도구 (`analysis_tools/`)
  - 기타 유틸리티 스크립트

## 프로젝트 구조

```
Spatial_AI_Project/
├── data/                    # 공용 데이터 폴더
├── util/                    # 공용 유틸리티 폴더
├── tools/                   # 공용 도구 폴더
│   ├── data_converter/      # 데이터 변환 도구
│   ├── analysis_tools/      # 분석 도구
│   ├── train.py             # 학습 스크립트
│   └── test.py              # 평가 스크립트
├── Ref_AI_project/          # Reference AI model 개발 프로젝트
│   ├── configs/             # 모델 설정 파일
│   └── mmdet3d_plugin/      # 커스텀 플러그인
├── Scenario_gen_project/    # Scenario generation 프로젝트
├── Scenario_min_project/    # Scenario mining 프로젝트
├── Photo-real_project/      # Photo-realistic 환경 재현 프로젝트
├── docs/                    # 문서
└── figs/                    # 이미지 및 그래프
```

## 시작하기

### 설치

자세한 설치 방법은 [docs/install.md](docs/install.md)를 참조하세요.

### 데이터 준비

데이터 준비 방법은 [docs/prepare_dataset.md](docs/prepare_dataset.md)를 참조하세요.

### 학습 및 평가

학습 및 평가 방법은 [docs/getting_started.md](docs/getting_started.md)를 참조하세요.

## 주요 기능

- **3D 객체 탐지 및 인식**: BEVFormer 기반의 카메라 기반 3D 객체 탐지
- **다양한 데이터셋 지원**: nuScenes, KITTI, Waymo, Lyft 등
- **시나리오 생성 및 분석**: 자율주행 시나리오 생성 및 마이닝
- **포토리얼리스틱 렌더링**: 현실적인 환경 재현

## 라이선스

본 프로젝트는 LICENSE 파일에 명시된 라이선스를 따릅니다.

## 참고 자료

- [설치 가이드](docs/install.md)
- [데이터 준비 가이드](docs/prepare_dataset.md)
- [시작 가이드](docs/getting_started.md)

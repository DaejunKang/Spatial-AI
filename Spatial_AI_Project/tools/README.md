# Tools 디렉토리

이 디렉토리는 자율주행 AI 개발에 필요한 실행 도구들을 포함합니다.

## util vs tools 차이점

- **util**: 재사용 가능한 유틸리티 클래스 및 함수 (시각화, 분석 등)
- **tools**: 실제 작업을 수행하는 실행 스크립트 (전처리, 평가, 배포 등)

## 디렉토리 구조

### data_tools/
데이터 처리 관련 도구

- **preprocess_data.py**: 데이터 전처리 도구
  - 포인트 클라우드 정규화
  - 데이터 필터링
  - 통계 계산

```bash
python tools/data_tools/preprocess_data.py \
    --input data.pkl \
    --output processed_data.pkl \
    --normalize standard \
    --filter-x -50 50 \
    --stats
```

### model_tools/
모델 관리 관련 도구

- **checkpoint_manager.py**: 체크포인트 관리 도구
  - 체크포인트 정보 추출
  - 체크포인트 비교
  - 체크포인트 병합
  - State dict 추출

```bash
# 체크포인트 정보 확인
python tools/model_tools/checkpoint_manager.py info checkpoint.pth

# 체크포인트 비교
python tools/model_tools/checkpoint_manager.py compare ckpt1.pth ckpt2.pth

# 체크포인트 병합
python tools/model_tools/checkpoint_manager.py merge \
    ckpt1.pth ckpt2.pth ckpt3.pth \
    --output merged.pth \
    --strategy average
```

### evaluation_tools/
모델 평가 관련 도구

- **evaluate_model.py**: 모델 평가 도구
  - Detection 메트릭 계산
  - 추론 시간 측정
  - 결과 비교

```bash
python tools/evaluation_tools/evaluate_model.py \
    --predictions predictions.json \
    --ground-truths ground_truths.json \
    --output results.json \
    --iou-threshold 0.5
```

### experiment_tools/
실험 관리 관련 도구

- **experiment_manager.py**: 실험 관리 도구
  - 실험 생성 및 관리
  - 실험 결과 추적
  - 실험 비교

```bash
# 새 실험 생성
python tools/experiment_tools/experiment_manager.py create \
    --name my_experiment \
    --config config.yaml \
    --description "Test experiment"

# 실험 목록
python tools/experiment_tools/experiment_manager.py list

# 실험 비교
python tools/experiment_tools/experiment_manager.py compare \
    --experiments exp1 exp2 exp3
```

### deployment_tools/
모델 배포 관련 도구

- **model_optimizer.py**: 모델 최적화 도구
  - 모델 양자화
  - ONNX 변환
  - 모델 프루닝

```bash
# ONNX 변환
python tools/deployment_tools/model_optimizer.py onnx \
    --model model.pth \
    --output model.onnx \
    --input-shape 1 3 224 224

# ONNX 최적화
python tools/deployment_tools/model_optimizer.py optimize \
    --onnx model.onnx \
    --output model_optimized.onnx
```

## 사용 예시

### 데이터 전처리 파이프라인
```bash
# 1. 데이터 전처리
python tools/data_tools/preprocess_data.py \
    --input raw_data.pkl \
    --output processed_data.pkl \
    --normalize standard \
    --filter-x -50 50 \
    --filter-y -50 50 \
    --stats

# 2. 통계 확인
cat processed_data.json
```

### 모델 평가 파이프라인
```bash
# 1. 모델 평가
python tools/evaluation_tools/evaluate_model.py \
    --predictions predictions.json \
    --ground-truths ground_truths.json \
    --output evaluation_results.json

# 2. 결과 확인
cat evaluation_results.json
```

### 실험 관리 워크플로우
```bash
# 1. 실험 생성
python tools/experiment_tools/experiment_manager.py create \
    --name baseline_experiment \
    --config configs/baseline.yaml

# 2. 학습 실행 (별도 스크립트)
python tools/train.py --config configs/baseline.yaml

# 3. 결과 저장
python tools/experiment_tools/experiment_manager.py save \
    --experiment baseline_experiment \
    --results results.json

# 4. 실험 비교
python tools/experiment_tools/experiment_manager.py compare \
    --experiments baseline_experiment improved_experiment
```

## 의존성

```bash
pip install torch numpy pyyaml onnx onnxruntime onnxsim
```

## 확장 가능성

각 도구는 독립적으로 사용할 수 있으며, 필요에 따라 확장할 수 있습니다:
- 새로운 데이터 처리 방법 추가
- 추가 평가 메트릭 구현
- 다른 모델 형식 지원 (TensorRT, OpenVINO 등)
- 실험 자동화 스크립트 추가

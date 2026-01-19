# 빠른 시작 가이드 (Quick Start)

## 1단계: 의존성 설치

```bash
cd Spatial_AI_Project/Photo-real_project
pip install -r requirements_extract.txt
```

## 2단계: 데이터 추출

### 방법 1: 직접 실행
```bash
python extract_waymo_data.py /path/to/waymo.tfrecord ./output
```

### 방법 2: 스크립트 사용
```bash
./run_extract_example.sh /path/to/waymo_raw ./output
```

### 방법 3: Python에서 직접 호출
```python
from extract_waymo_data import extract_universal

extract_universal(
    tfrecord_path="/path/to/segment_xxxx.tfrecord",
    output_dir="./output"
)
```

## 3단계: 결과 확인

```bash
# 출력 디렉토리 구조 확인
tree output/segment_xxxx

# 통계 확인
cat output/segment_xxxx/extraction_stats.json | python -m json.tool
```

## 주요 옵션

- **입력**: 단일 `.tfrecord` 파일 또는 디렉토리
- **출력**: 각 segment별로 별도 디렉토리 생성
- **자동 감지**: 이미지 자동 감지 (키 이름 무관)
- **마스크**: 빈 마스크 자동 생성 (COLMAP 호환)

## 문제 발생 시

1. **의존성 오류**: `pip install --upgrade tensorflow numpy opencv-python tqdm`
2. **메모리 부족**: 작은 배치로 나누어 처리
3. **이미지 미추출**: TFRecord 파일 형식 확인

자세한 내용은 `EXTRACT_USAGE.md` 참조

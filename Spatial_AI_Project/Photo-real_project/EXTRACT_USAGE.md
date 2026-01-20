# Waymo Data Extractor 사용 가이드

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements_extract.txt
```

또는 개별 설치:
```bash
pip install tensorflow numpy opencv-python tqdm
```

### 2. 데이터 추출 실행

#### 단일 파일 추출
```bash
python extract_waymo_data.py /path/to/segment_xxxx.tfrecord /path/to/output
```

#### 디렉토리 내 모든 파일 추출
```bash
python extract_waymo_data.py /path/to/waymo_raw /path/to/output
```

### 3. 출력 구조 확인

```
/path/to/output/
└── segment_xxxx/
    ├── images/
    │   ├── FRONT/
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   ├── FRONT_LEFT/
    │   ├── FRONT_RIGHT/
    │   ├── SIDE_LEFT/
    │   └── SIDE_RIGHT/
    ├── masks/
    │   └── (동일한 구조, 빈 마스크)
    └── extraction_stats.json
```

## 주요 특징

### 자동 이미지 감지
- JPEG Magic Header (FF D8) 검사
- PNG Magic Header (89PNG) 검사
- 최소 5KB 크기 필터링
- 키 이름에 의존하지 않음 (데이터 내용 기반 감지)

### 이중 파싱 전략
1. **SequenceExample 우선**: 비디오/시계열 데이터 처리
2. **Example 대체**: 스냅샷/프레임 데이터 처리

### 카메라 이름 추론
- 키 이름에서 자동 추론 (FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT)
- 인덱스 기반 매핑 (0=FRONT, 1=FRONT_LEFT, ...)
- 추론 실패 시 키 이름 사용 (특수문자 제거)

## 출력 통계

`extraction_stats.json` 파일에는 다음 정보가 포함됩니다:
- 각 시나리오별 프레임 수
- 타임스탬프 범위
- 총 이미지 수

## 문제 해결

### 이미지가 추출되지 않는 경우
1. 파일이 LiDAR 전용 데이터셋인지 확인
2. 이미지가 다른 압축 형식인지 확인
3. TFRecord 파일이 손상되지 않았는지 확인

### 메모리 부족 오류
- 대용량 파일의 경우 배치 처리 고려
- GPU 메모리 대신 CPU 사용 고려

## 예제 출력

```
🚀 Processing: segment_1234.tfrecord
Scanning Records: 100%|████████████| 200/200 [00:05<00:00, 38.2it/s]

==================================================
📊 Extraction Statistics Report
==================================================
✅ Total Images Saved: 1000
✅ Total Scenarios Found: 1
✅ Total Records Processed: 200

[Scenario Detail]
  - ID: segment_1234
    Frames: 200
    Timestamp Range: 0 ~ 199

💾 Statistics saved to: /path/to/output/segment_1234/extraction_stats.json
```

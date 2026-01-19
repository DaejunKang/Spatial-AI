#!/bin/bash
# Waymo Data Extractor 실행 예제 스크립트

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Waymo Data Extractor 실행 예제 ===${NC}"

# 기본 경로 설정 (사용자 환경에 맞게 수정)
WAYMO_RAW_DIR="${1:-/path/to/waymo_raw}"
OUTPUT_DIR="${2:-./waymo_extracted}"

echo -e "${YELLOW}입력 디렉토리: ${WAYMO_RAW_DIR}${NC}"
echo -e "${YELLOW}출력 디렉토리: ${OUTPUT_DIR}${NC}"

# 의존성 확인
echo -e "\n${GREEN}의존성 확인 중...${NC}"
python3 -c "import tensorflow as tf; import numpy as np; import cv2; import tqdm; print('✅ 모든 의존성 설치됨')" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  일부 의존성이 누락되었습니다. 설치 중...${NC}"
    pip install -r requirements_extract.txt
}

# 입력 경로 확인
if [ ! -d "$WAYMO_RAW_DIR" ] && [ ! -f "$WAYMO_RAW_DIR" ]; then
    echo -e "${YELLOW}⚠️  입력 경로가 존재하지 않습니다: ${WAYMO_RAW_DIR}${NC}"
    echo "사용법: $0 <입력_경로> [출력_경로]"
    echo "예제: $0 /path/to/waymo_raw ./output"
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 추출 실행
echo -e "\n${GREEN}데이터 추출 시작...${NC}"
python3 extract_waymo_data.py "$WAYMO_RAW_DIR" "$OUTPUT_DIR"

# 결과 확인
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 추출 완료!${NC}"
    echo -e "${YELLOW}출력 위치: ${OUTPUT_DIR}${NC}"
    
    # 통계 파일 확인
    STATS_FILE=$(find "$OUTPUT_DIR" -name "extraction_stats.json" | head -1)
    if [ -n "$STATS_FILE" ]; then
        echo -e "\n${GREEN}통계 파일: ${STATS_FILE}${NC}"
        echo "통계 내용:"
        python3 -c "import json; import sys; data=json.load(open('$STATS_FILE')); print(f\"시나리오 수: {len(data)}\"); [print(f\"  - {k}: {v['count']} frames\") for k,v in list(data.items())[:5]]"
    fi
else
    echo -e "\n${YELLOW}⚠️  추출 중 오류가 발생했습니다.${NC}"
    exit 1
fi

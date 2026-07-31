"""서버 연결 및 추론 관련 설정."""

# OpenAI 호환 서버 접속 정보
BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"

# 다중 GPU 복제본 엔드포인트 (데이터 병렬). 살아있는 것만 자동 사용.
# 복제본 기동 후(8002~8004) 여기 4개가 라운드로빈됨. 미기동 포트는 헬스체크로 자동 제외.
BASE_URLS = [
    "http://localhost:8001/v1",
    "http://localhost:8002/v1",
    "http://localhost:8003/v1",
    "http://localhost:8004/v1",
]

# 사용할 모델
MODEL = "nvidia/cosmos3-nano-reasoner"

# 추론 파라미터
TEMPERATURE = 0
MAX_TOKENS = 6144

# user 메시지 지시문 (시스템 프롬프트는 prompts.PROMPT)
USER_TEXT = "이 20초 클립을 v0.7.1 세그먼트 스키마로 태깅하라."

# 이벤트/윈도우 거동 분류 콜의 생성 상한(작은 JSON 출력이라 충분).
# 낮게 잡아 가끔 발생하는 <think> 폭주(→max_tokens까지 생성, ~86s) 꼬리를 차단.
CLASSIFY_MAX_TOKENS = 768

# --- 프레임 샘플링 ----------------------------------------------------------
# scene understanding 용: 전체 클립에서 시간축 균등 샘플링할 프레임 수
NUM_FRAMES = 16
# 프레임 긴 변 최대 픽셀 (개별 전송 시, 초과하면 비율 유지 축소. 0이면 원본)
FRAME_MAX_SIDE = 1280
# JPEG 인코딩 품질 (1~100)
JPEG_QUALITY = 85

# --- 윈도우(연속 청크) 파이프라인 --------------------------------------------
# 태깅 모드:
#   "montage"  : 단일 몽타주
#   "windowed" : 연속 윈도우 분할 후 통합
#   "events"   : egomotion 이벤트 중심 태깅(when=GT, what=모델) — 권장
TAG_MODE = "events"

# --- egomotion 이벤트 검출 임계값 ---
EVENT_DECEL_AX = -1.0     # ax(m/s^2) 감속 임계
EVENT_ACCEL_AX = 1.0      # ax 가속 임계
EVENT_STOP_SPEED = 1.5    # 정지 판정 속도(m/s)
EVENT_TURN_YAWRATE = 0.15 # yaw 편위 후보 검출 yaw rate(rad/s) — 이후 net heading으로 분류
EVENT_TURN_HEADING = 45.0 # 회전 확정 최소 net heading 변화(deg). 미만은 커브/차선변경 후보
EVENT_UTURN_HEADING = 150.0 # U턴 판정 net heading(deg)
EVENT_LC_HEADING_MAX = 22.0 # 차선변경: net heading |Δ| 상한(deg, 원래 진행방향 복귀)
EVENT_LC_LAT_MIN = 1.8    # 차선변경: 최소 측방 변위(m)
EVENT_LC_LAT_MAX = 6.0    # 차선변경: 최대 측방 변위(m, 이 이상은 회전/합류)
EVENT_MIN_SEC = 0.4       # 이벤트 최소 지속(초)
EVENT_MERGE_SEC = 0.6     # 같은 종류 이벤트 병합 간격(초)
EVENT_CTX_SEC = 1.5       # 이벤트 중심 ±컨텍스트(초). 3s창 → 서버 12f 캡에서 ~3.7fps(6s의 2배)
# obstacle(트랙) 기반 agent 이벤트 (egomotion 사각지대 보완)
OBST_LANE_HALF = 2.5      # ego 진행로 반폭(m, |center_y|)
OBST_AHEAD_M = 45.0       # 전방 관심 거리(m, center_x)
OBST_MIN_SEC = 1.0        # 트랙 in-corridor 최소 지속(초)
OBST_CUTIN_Y = 2.8        # cut-in 판정: 차로 밖(|y|>이값)에서 진입
EVENT_BASELINE_MIN_SEC = 1.5  # 무이벤트 baseline 세그먼트 최소 길이(초)
# 윈도우 길이(초)와 이동 간격(초). (WINDOW_SEC - WINDOW_STRIDE_SEC) = 겹침
WINDOW_SEC = 6.0
WINDOW_STRIDE_SEC = 5.0
# 윈도우 서브클립 mp4 기록 fps (서버가 자체 fps로 재샘플; 여유분 확보용)
SEND_FPS = 10
# 윈도우 서브클립 프레임 긴 변 최대(px, 0=원본)
WINDOW_MAX_SIDE = 1280
# 통합 시 같은 거동 세그먼트를 병합할 최대 시간 간격(초)
MERGE_GAP_SEC = 1.5

# 프레임 전송 방식:
#   "montage"    : 균등 샘플 N프레임을 시간순 그리드 1장으로 합쳐 전송(전 구간 커버, 이미지 1장)
#   "individual" : 프레임을 개별 이미지로 전송(서버 상한 MAX_IMAGES 장까지)
FRAME_MODE = "montage"
# 서버가 허용하는 프롬프트당 최대 이미지 수 (individual 모드 상한)
MAX_IMAGES = 5
# montage 그리드 열 수 (0이면 sqrt(N) 자동), 각 셀 긴 변 최대 픽셀
MONTAGE_COLS = 0
MONTAGE_CELL_MAX_SIDE = 640

# --- 데이터셋 ---------------------------------------------------------------
# PhysicalAI-AV-curated 데이터셋 루트
DATASET_ROOT = "/katech/datasets/PhysicalAI-AV-curated"

# 사용할 카메라 (camera_front_wide_120fov | camera_front_tele_30fov)
CAMERA = "camera_front_wide_120fov"

# 사용할 split 파일 (curation/ 하위)
SPLIT_FILE = "diverse_set-test.txt"

# 테스트에 사용할 클립 인덱스 (split 파일 내 순번, 0-based)
CLIP_INDEX = 0

# --- 출력 / 오버레이 --------------------------------------------------------
# 결과 저장 루트 (tags/<clip_id>.json, tags.jsonl, overlay/<clip_id>.mp4)
OUTPUT_DIR = "outputs"
# 오버레이 비디오 긴 변 최대 픽셀 (파일 크기 절감용 다운스케일, 0이면 원본)
OVERLAY_MAX_SIDE = 1280

# --- 비디오 소스 선택 --------------------------------------------------------
# "dataset": 위 DATASET_ROOT 에서 로컬 mp4 사용 (프레임 균등 추출 후 이미지 전송)
# "url"    : 아래 VIDEO_URL 원격 샘플 영상 사용 (임시 파일로 받아 동일 처리)
VIDEO_SOURCE = "dataset"

# VIDEO_SOURCE == "url" 일 때 사용할 원격 샘플 영상
VIDEO_URL = "https://download.samplelib.com/mp4/sample-5s.mp4"

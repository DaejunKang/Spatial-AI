"""Long-tail 이벤트 마이닝 택소노미 (gold 라벨링 + 파이프라인 검색의 단일 기준).

lane-keeping이 다수인 데이터셋에서 '의미 있는' 주행 상황을 다중 라벨로 분류/추출.
각 클립은 아래 카테고리 중 present인 것들의 집합으로 라벨된다(하나도 없으면 plain lane-keeping).
주 신호(source)는 검출 근거: egomotion / obj3d / map / vlm.

축: ego 기동 / 상호작용 / 맥락 / 정적환경 / 규칙.
2026-07-31 `legacy/new_tag.json`(v0.4 폐쇄어휘 50태그)의 정적환경(조명·기상·노면·정적장면·기하)
+ long-tail 이벤트(긴급차·동물·장애물·무단횡단·역주행·급기동 등)를 흡수. 각 흡수 태그의
visionary 실행 상태는 `STATUS`(runnable/vlm_only/sparse/gold). new_tag 원본의 GT rule 세부는
미채택(우리 데이터에 부재하는 nuScenes/Waymo GT 전제) — source/status 로 대체.
"""

TAXONOMY = [
    # (axis, key, label, source, hint)
    ("ego 기동", "lane_change_left",     "좌 차선변경",        "egomotion", "인접 좌차선으로 변경"),
    ("ego 기동", "lane_change_right",    "우 차선변경",        "egomotion", "인접 우차선으로 변경"),
    ("ego 기동", "turn_left",            "좌회전",             "egomotion", "교차로 좌회전(신호 보호)"),
    ("ego 기동", "turn_right",           "우회전",             "egomotion", "교차로/도로 우회전"),
    ("ego 기동", "u_turn",               "유턴",               "egomotion", "U턴"),
    ("ego 기동", "unprotected_left",     "비보호 좌회전",      "egomotion+obj3d", "대향차 대기 후 좌회전"),
    ("ego 기동", "stop",                 "정지",               "egomotion", "신호/정체로 완전 정지"),
    ("ego 기동", "decel_at_intersection","교차로 서행/감속",   "egomotion+map", "교차로 접근 감속(정지 아님)"),
    ("ego 기동", "creep",                "서행/기어가기",      "egomotion", "정체·혼잡 저속 주행"),
    ("ego 기동", "pull_over",            "갓길 정차",          "egomotion", "도로변으로 붙여 정차"),
    # ego 급기동·추월 (new_tag.json v0.4 흡수, 2026-07-31)
    ("ego 기동", "ego_hard_brake",       "급제동",             "egomotion", "종감속 3m/s²+ 0.5초+ 지속"),
    ("ego 기동", "ego_hard_steer",       "급조향",             "egomotion", "급격한 조향(yaw rate 큼)"),
    ("ego 기동", "ego_overtake",         "추월",               "egomotion+obj3d", "선행차 앞질러 전방 복귀"),

    ("상호작용", "cut_in",               "끼어들기(cut-in)",   "obj3d", "인접차로→ego차로 앞 완전 진입"),
    ("상호작용", "cut_in_attempt",       "끼어들기 시도",      "obj3d", "인접차가 ego차로 쪽으로 밀고 들어오나 완전 진입은 아님"),
    ("상호작용", "cut_out",              "빠짐(cut-out)",      "obj3d", "선행차가 ego차로에서 이탈"),
    ("상호작용", "lead_decel",           "선행차 감속",        "obj3d", "앞차가 감속(완만~급 포함, 넓은 범위)"),
    ("상호작용", "close_follow",         "근접 추종",          "obj3d", "짧은 차간거리 추종"),
    ("상호작용", "ped_crossing",         "보행자 횡단(+양보)", "obj3d", "보행자가 경로 횡단, ego 양보"),
    ("상호작용", "vru_roadside",         "도로변 보행자",      "obj3d", "보행자가 도로변에 있으나 경로 횡단은 아님(비횡단)"),
    ("상호작용", "cyclist_pm_near",      "자전거/PM 근접",     "obj3d", "자전거·킥보드 근접"),
    ("상호작용", "oncoming_encroach",    "대향차 침범",        "obj3d", "대향차가 중앙선 넘어 접근"),
    ("상호작용", "agent_yields_to_ego",  "상대차 양보(교차로)", "obj3d", "신호없는 교차로 등에서 횡단/대향차가 ego 보고 정지·양보, ego 통과"),
    ("상호작용", "ego_yields_to_agent",  "ego 양보(교차로)",   "obj3d+egomotion", "ego가 횡단/대향 차량·보행자에 양보해 정지·감속 후 통과"),
    # 긴꼬리 이벤트 (new_tag.json v0.4 흡수, 2026-07-31) — 큐레이션 타깃
    ("상호작용", "vehicle_cross_path",   "차량 경로 횡단",      "obj3d", "타차량이 ego 경로를 가로지름(교차로 등)"),
    ("상호작용", "wrong_way_vehicle",    "역주행 차량",         "obj3d", "대향/역방향 진행 차량"),
    ("상호작용", "stationary_vehicle_on_lane", "주행차로 정차차량", "obj3d", "주행 차로 내 정지 차량(장애)"),
    ("상호작용", "large_vehicle_proximity", "대형차 근접",      "obj3d", "트럭·버스 근접"),
    ("상호작용", "emergency_vehicle",    "긴급차량",           "obj3d+vlm", "구급/소방/경찰 등 긴급차량 출현"),
    ("상호작용", "animal_on_road",       "동물 출현",          "vlm+obj3d", "차도 위/인접 동물"),
    ("상호작용", "road_obstacle",        "차도 위 장애물",      "vlm+obj3d", "낙하물·적재물 등 경로상 장애물"),
    ("상호작용", "jaywalking",           "무단횡단",           "obj3d+vlm", "횡단보도 아닌 곳 보행자 횡단"),
    ("상호작용", "road_worker",          "도로 작업자",         "vlm", "차도/노변 작업자"),
    ("상호작용", "vulnerable_pedestrian", "취약 보행자",        "vlm", "휠체어·유아·노약자 등"),

    ("맥락",     "intersection_signalized",   "신호 교차로",   "vlm", "신호등 있는 교차로 통과"),
    ("맥락",     "intersection_unsignalized", "비신호 교차로", "vlm", "신호등 없는 교차로 통과(양보/우선권 협상)"),
    ("맥락",     "roundabout",           "회전교차로",         "map", "라운드어바웃"),
    ("맥락",     "merge_onramp",         "합류/램프",          "map", "본선 합류·진입로"),
    ("맥락",     "construction_cones",   "공사/콘",            "vlm", "공사구간·라바콘·차로차단"),
    ("맥락",     "toll_gate",            "요금소/차단기",      "vlm", "톨게이트·차단바"),
    # 도로환경(road class) — 클립 전체 성격. GT 없어 VLM/사람 판단.
    ("맥락",     "road_highway",         "고속/자동차전용",    "vlm", "고속도로·자동차전용도로"),
    ("맥락",     "road_urban_arterial",  "도심 간선",          "vlm", "도심 주간선/보조간선(차선표시 있는 큰길)"),
    ("맥락",     "road_backstreet",      "좁은 골목/이면도로", "vlm", "주택가 좁은 생활도로(차선표시 없음, 저속, 보행자·주차차량)"),
    ("맥락",     "road_rural",           "교외/시골길",        "vlm", "교외·시골 도로"),
    ("맥락",     "road_tunnel",          "터널",               "vlm", "터널 내부 주행"),
    ("맥락",     "road_bridge",          "교량",               "vlm", "교량 주행"),
    ("맥락",     "road_parking",         "주차장/구내",        "vlm", "주차장·건물 구내 저속 주행"),

    # 정적환경/조건 (new_tag.json v0.4 흡수, 2026-07-31) — 클립 성격의 present 태그(검색축).
    # 조명/기상/노면은 ODD dim(아래)에서 1급 검색 태그로 승격. 대부분 VLM 시각 판정.
    ("정적환경", "night",                "야간",               "vlm", "야간 주행(태양고도<-6°). 터널·지하 예외"),
    ("정적환경", "twilight",             "박명(여명/황혼)",     "vlm", "여명/황혼(태양고도 -6~+6°)"),
    ("정적환경", "day",                  "주간",               "vlm", "주간(태양고도>+6°)"),
    ("정적환경", "rain",                 "강우",               "vlm", "비 내림(강도 미구분)"),
    ("정적환경", "snow",                 "강설",               "vlm", "눈 내림(노면 적설과 독립)"),
    ("정적환경", "fog",                  "안개",               "vlm", "안개·연무 가시거리<1km"),
    ("정적환경", "clear_weather",        "맑음",               "vlm", "rain·snow·fog 모두 부재(파생)"),
    ("정적환경", "wet_road",             "노면 젖음",           "vlm", "노면 정반사·젖은 색조"),
    ("정적환경", "dry_road",             "노면 건조",           "vlm", "수막·적설·결빙 부재(파생)"),
    ("정적환경", "glare",                "역광/눈부심",         "vlm", "태양 직사·강반사로 노출 포화·플레어"),
    ("정적환경", "crosswalk_present",    "횡단보도 존재",        "vlm", "경로상 횡단보도(보행자 유무 무관)"),
    ("정적환경", "traffic_light_present", "신호등 존재",         "vlm", "경로상 차량용 신호등(상태 무관)"),
    ("정적환경", "sharp_curve",          "급곡선로",            "egomotion", "도로 선형 급곡선(R≤100m). 교차로 회전 제외"),
    ("정적환경", "undivided_road",       "비분리 도로",         "vlm", "양방향 물리적 분리물 부재(차선표시≠분리)"),
    ("정적환경", "congestion",           "정체/서행",           "egomotion", "평균<10km/h·정지출발≥2. 신호대기 단독 제외"),
    ("정적환경", "free_flow",            "원활 교통",           "egomotion", "정지출발 없음·평균≥30km/h"),
    ("정적환경", "crowd",                "보행자 밀집(군중)",    "obj3d", "전/측방 30m 내 보행자 8인+"),

    ("규칙",     "red_light_stop",       "적신호 정지",        "vlm+egomotion", "적색 신호로 정지"),
    ("규칙",     "signal_go",            "청신호 출발",        "vlm+egomotion", "녹색 전환 후 출발"),
]

KEYS = [t[1] for t in TAXONOMY]
BY_AXIS = {}
for axis, key, label, src, hint in TAXONOMY:
    BY_AXIS.setdefault(axis, []).append((key, label, src, hint))


# ODD 정적 환경 조건 — 클립당 차원별 '단일' 상태(라벨링 편의용).
# ⚠️ 검색·큐레이션용으로는 위 "정적환경" 축의 present 태그(night/rain/wet_road/...)가 정본(승격됨).
#    ODD 는 per-clip 단일값 표현으로만 잔존(gold 도구 미렌더). 조명/기상/노면 중복 주의.
# (dim_key, dim_label, [(opt_key, opt_label), ...])  첫 옵션은 unknown(미지정).
ODD = [
    ("road_env", "도로환경", [
        ("unknown", "미지정"), ("highway", "고속/자동차전용"), ("urban_arterial", "도심 간선"),
        ("residential", "주택가/이면도로"), ("rural", "교외/시골"), ("tunnel", "터널"),
        ("bridge", "교량"), ("parking_lot", "주차장/구내"), ("ramp_junction", "램프/분기"),
    ]),
    ("lighting", "조도", [
        ("unknown", "미지정"), ("day", "주간"), ("dawn_dusk", "여명/황혼"), ("night", "야간"),
    ]),
    ("weather", "날씨", [
        ("unknown", "미지정"), ("clear", "맑음"), ("overcast", "흐림"),
        ("rain", "비"), ("snow", "눈"), ("fog", "안개"),
    ]),
    ("surface", "노면", [
        ("unknown", "미지정"), ("dry", "건조"), ("wet", "젖음"), ("snow_ice", "눈/결빙"),
    ]),
]
ODD_DIMS = [d[0] for d in ODD]

# AUTO_GT = **순수 종방향 kinematic**만 — egomotion rule 단독으로 확정(순환/자명, 평가 제외).
# 근거(CLAUDE.md §2 두 층위): 종방향(stop/accel/decel/creep)은 egomotion 만으로 결정.
# 경로 기동(turn_left/turn_right/u_turn/lane_change)은 CAN yaw 만으론 under-determined
# (교차로 turn/커브/분기/u-turn 구분 불가) → CAN 은 전이 트리거일 뿐, 라벨은 map∪VLM 맥락 해소.
# 따라서 turn류는 AUTO_GT 에서 제외 = HUMAN_KEYS(gold 평가 대상)로 이동.
AUTO_GT = {"stop"}
# 나머지(경로기동·의미오버레이·상호작용·맥락)는 사람이 에피소드별로 라벨 = gold 평가 대상
HUMAN_KEYS = [k for k in KEYS if k not in AUTO_GT]

# new_tag.json v0.4(legacy/) 흡수 태그의 **visionary 실행 상태**(source는 TAXONOMY 튜플에).
# runnable=즉시 GT/규칙 도출 | vlm_only=GT부재·VLM 시각판정 | sparse=데이터 희소 | gold=사람 gold 필요.
# (new_tag 원본은 nuScenes/Waymo/ZOD GT rule 전제 — 우리 데이터엔 부재라 재도출 필요분을 표기.)
STATUS = {
    "night": "vlm_only", "twilight": "vlm_only", "day": "vlm_only",
    "rain": "vlm_only", "snow": "sparse", "fog": "sparse", "clear_weather": "vlm_only",
    "wet_road": "vlm_only", "dry_road": "vlm_only", "glare": "vlm_only",
    "crosswalk_present": "vlm_only", "traffic_light_present": "vlm_only",
    "sharp_curve": "runnable", "undivided_road": "vlm_only",
    "congestion": "runnable", "free_flow": "runnable", "crowd": "runnable",
    "vehicle_cross_path": "runnable", "wrong_way_vehicle": "sparse",
    "stationary_vehicle_on_lane": "runnable", "large_vehicle_proximity": "runnable",
    "emergency_vehicle": "gold", "animal_on_road": "sparse", "road_obstacle": "runnable",
    "jaywalking": "runnable", "road_worker": "vlm_only", "vulnerable_pedestrian": "vlm_only",
    "ego_hard_brake": "runnable", "ego_hard_steer": "gold", "ego_overtake": "runnable",
}


def auto_tags_from_arc(kinds):
    """egomotion arc kinds → ego 기동 태그.

    stop 은 순수 kinematic 확정(AUTO_GT). turn_left/right/u_turn 은 **CAN 트리거 후보**로
    방출(Phase A recall)하되 GT-확정 아님 — 방향/유형은 하류에서 map∪VLM 맥락으로 해소.
    """
    out = []
    ks = set(kinds)
    if "u_turn" in ks:
        out.append("u_turn")
    if "turn_left" in ks:
        out.append("turn_left")
    if "turn_right" in ks:
        out.append("turn_right")
    if "stop" in ks:
        out.append("stop")
    return out

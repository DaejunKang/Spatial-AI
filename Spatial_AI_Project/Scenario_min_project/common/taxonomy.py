"""Long-tail 이벤트 마이닝 택소노미 (gold 라벨링 + 파이프라인 검색의 단일 기준).

lane-keeping이 다수인 데이터셋에서 '의미 있는' 주행 상황을 다중 라벨로 분류/추출.
각 클립은 아래 카테고리 중 present인 것들의 집합으로 라벨된다(하나도 없으면 plain lane-keeping).
주 신호(source)는 검출 근거: egomotion / obj3d / map / vlm.
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

    ("규칙",     "red_light_stop",       "적신호 정지",        "vlm+egomotion", "적색 신호로 정지"),
    ("규칙",     "signal_go",            "청신호 출발",        "vlm+egomotion", "녹색 전환 후 출발"),
]

KEYS = [t[1] for t in TAXONOMY]
BY_AXIS = {}
for axis, key, label, src, hint in TAXONOMY:
    BY_AXIS.setdefault(axis, []).append((key, label, src, hint))


# ODD 정적 환경 조건 — 이벤트와 달리 클립당 차원별 '단일' 상태(대부분 VLM 시각 판단).
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

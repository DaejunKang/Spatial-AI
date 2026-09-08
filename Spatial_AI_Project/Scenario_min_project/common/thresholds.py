# -*- coding: utf-8 -*-
"""임계 상수 유일 허용 파일(pre-commit no-literal-thresholds 훅 대상).

threshold_set_id로 완전히 외부화하는 건 아직 미정(CLAUDE.md "설정으로 열어둘 것" 표,
`decisions/DV_DEFAULTS.md`) — 이 모듈은 우선 "같은 이름의 상수가 여러 파일에 흩어져
값이 갈리는 사고"(LEAD_IN 4중복·OBST_CUTIN_Y vs CUTIN_Y 값 충돌 실측)를 막는 최소
요건만 충족한다. 여기 없는 임계값이 새 파일에 등장하면 여기로 옮기고 import한다.

주의(2026-09-08): `task_episode/vlm_verify.py`·`task_episode/tag_v08.py`는 각자
`LEAD_IN = 3.0`을 여전히 로컬로 갖고 있다(이번 커밋 범위 밖 — 건드리지 않음). 이 모듈은
`task_episode/candidates.py`·`task_episode/map_lane.py`만 우선 이관한 상태다.
"""

LEAD_IN = 3.0        # 에피소드 onset 이전 접근 구간(초)
VALID_FRAC = 0.5     # map_lane 유효 프레임 비율 게이트

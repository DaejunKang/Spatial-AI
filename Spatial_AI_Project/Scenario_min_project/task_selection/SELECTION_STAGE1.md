# 1차 Clip 선별 랭킹 — 프로세스/로직

> **정본(2026-07-31 승격): 윈도우+video 반응성** — `run_selection.py` 기본 경로.
> `consolidate_episodes`로 ego 전이 윈도우 분할 → 윈도우별 ego 반응성 + **반응 윈도우 subclip을
> video로 VLM에 보내** 원인(외부 agent)을 **시간정렬** 확인. 몽타주 정지투영/오귀속을 해소.
> 여전히 **obj3d 없이 egomotion(CAN)+VLM video만** 사용(Stage1 전제 유지). 엔진=`folder_selection.rank_folder_windowed`.
> 실측(obj3d 대조): 몽타주 cut_in corroboration 6%(top50 0%) → 윈도우+video로 과검 대부분 제거·reactive 50/50.
> 아래 "몽타주 흥미도(Stage2)"는 **legacy**(`run_selection.py montage`)로 강등.

## 목적 & 제약
전체 주행 로그에서 **의미있는(추출가치) clip을 선별**하여 2차(episode 추출·태깅)로 넘김.
- 제약: raw 로그엔 **ego CAN + video만** 존재 (obj3d 3DOD·lane detection 없음/신뢰불가).
- (legacy) 선별은 **CAN arc_path(무료 1차) + VLM 경량 패스(2차)** 두 채널의 결합 랭킹.

```
전체 로그 → [Stage1 ego arc 점수(CAN, 전수)] + [Stage2 VLM interestingness(몽타주)] → 결합 랭킹 → 상위 K 선별 → 2차 태깅
```

## Stage 1 — ego arc 점수 (CAN only, 전수·무료)
egomotion(vx/ax/yaw)에서 net-heading 기반 검출(`events.detect_events`, taxo와 공유):
- 기동 플래그: `turn`(turn_left|right), `u_turn`, `lane_change`(L|R), `stop`
- **급감속** `harsh = max(0, -min(ax) - 3.0)` — 3 m/s² 초과 종방향 감속.
  ⭐ 핵심: cut-in·보행자·선행차 급정지 등 **agent 이벤트는 대개 ego 브레이크로 반영** → perception 없이도 "반응"을 포착.
- `n_decel` = decelerate 이벤트 수.

**점수**
```
score_ego = 2·turn + 3·u_turn + 2·lane_change + 1·stop + 1.5·min(harsh,3.0) + 0.4·n_decel
```
- 커버: ego 기동 + 반응성 agent 이벤트(급감속).
- 한계: **ego가 반응하지 않은** agent 이벤트(멀리 지난 보행자, 무반응 cut-in)는 CAN에 안 보임 → Stage2가 담당.

## Stage 2 — VLM interestingness (몽타주 이미지, 저비용)
- 입력: 클립 몽타주 12프레임(cell 360px) 단일 이미지 (video 대비 저비용).
- guided-JSON 출력: `{interesting: bool, score: 0~1, category, reason}`
  - category(coarse): turn_maneuver / intersection / stop_or_yield / cut_in / pedestrian_or_cyclist /
    lane_change / congestion / hazard_nearmiss / unusual_scene / routine_cruise
- 프롬프트: "notable/long-tail 상황(회전·교차로·양보·cut-in·보행자·차선변경·정체·위험·특이) vs 단순 순항"을
  calibrated score로. 순항=낮은 점수.
- 역할: **CAN이 못 보는 의미/agent 이벤트를 회수** + 거친 카테고리 부여.

## 결합 랭킹
```
en = score_ego / max(score_ego)          # 정규화
vs = vlm.score                            # 0~1
combined = max(en, vs) + 0.3·min(en, vs)  # 강한 채널 우선 + 합의 보너스
```
- `max(...)`: **어느 한 채널이라도 강하면 유의미** (기동만/agent만 각각 포착). 단순 평균이면 한 채널만 강한 clip이 저평가됨.
- `0.3·min`: 두 채널 합의 시 소폭 가산.
- 정렬 후 **상위 K 선별**(프로토타입 K=30/100). 최종 cutoff(top-K vs score 임계)는 선별-gold recall@K로 결정.

## 검증 (진행 중)
- **선별-gold**: 100 후보에 사람이 "유의미 y/n" 이진 라벨(예측 숨김·비편향).
- 지표: 결합 vs ego-only vs VLM-only의 top-K **precision/recall**, **recall@K 곡선**, **AUC**, 놓친 유의미·오선별 목록.
- 프로토타입 관찰: 상위 30 채널 기여 = ego우세 4 / VLM우세(ego 놓침) 15 / 둘다 11 → **두 채널 결합이 필수**임을 확인.

## 구현 파일 (프로토타입, job tmp)
- `select_prep.py`: 100 샘플 + 트랜스코드 + ego 점수 → `select/select_data.json`
- `run_select_vlm.py`: VLM 몽타주 interestingness → `select/select_vlm.json`
- `build_select_review.py`: 결합 랭킹 + 리뷰 HTML(`select/index.html`) + `selected30.json`
- `build_select_label.py`: 선별-gold 라벨링 도구(`select/label.html`)
- `eval_select.py`: gold 대비 정량화
- ego 검출 로직 근거: [[cutin-cutout-lane-relative]] (net-heading), taxo_detect/events.py

## 알려진 한계 / 후속 레버
- 정지 몽타주는 **시간적 이벤트(cut-in) 과소검출** 가능 → 후보 축소 후 video 재확인(2패스) 고려.
- raw 스케일 비용: Stage1이 순항 다수를 걸러 VLM 부하 축소. 순수 agent-무반응 이벤트만 VLM 전수/샘플 의존.
- 가중치(2/3/2/1, 0.3 보너스)·cutoff는 선별-gold 지표로 튜닝.

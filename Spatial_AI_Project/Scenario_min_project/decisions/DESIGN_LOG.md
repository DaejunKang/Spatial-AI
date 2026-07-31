# 설계 변경/개선 이력 (최신이 위)

---

## [2026-07-31] 저장소 구조: common / task_selection / task_episode 물리 분할
- 변경: flat 모듈을 2단계 task 기준으로 폴더 이관(`common`=공유 인프라, `task_selection`=Stage1, `task_episode`=Stage2). Task1 선별 알고리즘을 `selection.py`로 모듈화(구 job tmp 스크립트 통합).
- 이유: metadata labeling이 **① clip 선별 → ② episode 추출** 2단계이고, 공유 기반과 각 task 개별 알고리즘을 분리 관리하기 위함. 공유 인프라는 상위 프로젝트(Spatial-AI)로도 공유.
- 영향: import는 `run.sh` PYTHONPATH + venv `.pth`로 top-level 유지(전 모듈 21/21 OK). 스키마 json은 vocab 모듈과 함께 `common/`으로. 각 폴더 `README.md`=ownership 매니페스트.

## [2026-07-31] VLM self-consistency 투표 — 힌트 유지(B), 상호작용 합의는 불신
- 변경: Phase A의 VLM 5-vote는 GT 힌트를 계속 받되(`_prompt`의 "GT flagged"), **상호작용의 다채널 "합의"를 독립확인으로 신뢰하지 않음**.
- 이유: 힌트가 상호작용만 나열 → 맥락(도로/신호) 투표는 이미 독립(GT 부재라 VLM이 유일 원천). 상호작용 합의만 오염되나, Phase C에서 상호작용=`gt∩vlm`·맥락=VLM 다수결로 분리 처리하므로 상호작용 다수결 신뢰가 불필요.
- 영향: `candidates.py`(hint 유지), Phase C 설계(역할 분리).

## [2026-07-31] Track2 검색 입도 — 교차로 sig/unsig 유지, 도로유형만 병합
- 변경: `retrieve.MERGE`가 sig+unsig→intersection로 병합 중인 것을 **정정 대상**으로. 도로유형(urban/backstreet)만 병합 유지.
- 이유: 신호/비신호는 **cause와 결부**(비신호=양보 정차 vs 신호=신호 정지)라 하류 query 가치 있음 → 병합하면 안 됨. 병합 규칙 = (a)판단 애매 ∧ (b)query/cause 무가치일 때만.
- 영향: `retrieve.py`(MERGE에서 intersection 쌍 제거 예정), `taxonomy_merge_report.md`.

## [2026-07-31] cause 축 — 전이의 "왜", GT+맥락, Track2 1차 query 키
- 변경: cause를 전이(ego_action transition)에 결부된 "왜"로 정의. 모든 값(agent 포함)이 **GT 증거 + 맥락 추론**으로 해소. **Track2 검색 인덱스에 cause 축 추가 필요**(현재 v3 미포함 = 갭).
- 이유: agent조차 tracking만으론 인과 판정 불가(진행로 agent 존재 ≠ 전이 원인). 큐레이션은 주로 cause로 검색.
- 영향: `candidates.py`/`retrieve.py`(cause 산출·인덱스화 추가), `taxonomy.py`.

## [2026-07-31] ego_action — 두 층위 + 앵커 + 순방향 생성 (invariant 정정)
- 변경: "ego_action=rule, never model" invariant를 **정정**. 종방향(accel/decel/stop/creep)=egomotion rule 확정 / 경로(turn·lane_change·u_turn)=**CAN이 전이만 트리거 + 라벨은 맥락 해소(map∪VLM 앙상블)**. 앵커의 취지=전이 감지(상황 발생 지점→key_frame·세그먼트). 생성은 **순방향 SD→SA→MA**, MA는 마지막에 앵커+맥락으로 해소.
- 이유: CAN yaw만으론 교차로 turn / 커브 / 분기 / u-turn 구분 불가(under-determined). u_turn 제거·net-heading 노이즈가 이를 방증.
- 영향(드리프트 수정 대상): `tag_v08.V08_SCHEMA` MA-first→forward(MA-last) + ego_action 앵커화; `taxonomy.AUTO_GT`에서 turn류를 순수 rule에서 빼 "CAN트리거+맥락해소" 계층으로(stop/accel/decel/creep만 순수 kinematic).

## [2026-07-31] 세그먼테이션 = ego 전이 = critical scenario
- 변경: situation/세그먼트를 **ego_action 전이로만** 정의. ego 미반응 상황(non-reactive agent 이벤트)은 critical 아님 → 세그먼트 미생성. Track1·Track2 동일 세그먼트 백본 공유.
- 이유: critical의 정의를 ego 반응으로 확정(추후 문제 시 revisit). `key_frame_t`=rule(egomotion onset).
- 영향: `classify073.consolidate_episodes`(백본), 양 track.

## [2026-07-31] 산출물 두 제품(Track1/Track2) 병존
- 변경: **Track1 = v08형 SD/SA/MA 메타데이터**(라벨링) + **Track2 = v3 taxonomy 추출 retriever**(큐레이션). 공용 기반 공유.
- 이유: 라벨링과 큐레이션은 다른 산출물. v3 pivot이 SD/SA/MA를 드롭했었는데, 라벨링 제품(Track1)을 명시적으로 유지.
- 영향: `tag_v08`(Track1) vs `candidates/retrieve`(Track2), CLAUDE.md §2.

## [2026-07-31] v3 관점 전환 — 태거 = retriever (recall 우선)
- 변경: 태거를 분류기 아니라 **관대한 후보 생성기(retriever)** 로. Phase A(OR 앙상블·∩ 금지) → B(정규화·병합·랭킹) → C(precision 게이트·하류) → D(완전라벨 gold).
- 이유: 큐레이션에선 recall이 자산, **recall miss > FP**(miss는 영구 손실). 실측: OR합집합 recall 0.81(병합후 0.90) > 단일 VLM 0.67.
- 영향: `candidates.py`(Phase A), `retrieve.py`(Phase B). 이전 ∩ precision-우선(v2) 폐기.

## [이전] 2단계 퍼널 · map 재해석 · cut_in 희소 등
- Stage1(CAN+VLM 선별) / Stage2(3DOD+map episode). map `centerlines`=실제 경계선(재해석), 유효율 ~35%, is_intersection 죽음. cut_in 극희귀(gold 0/50, ~26/1966). obj3d vx/vy·occlusion 불신. 상세는 `PROJECT_DESIGN.md`·`SELECTION_STAGE1.md`.

# 설계 변경/개선 이력 (최신이 위)

> 각 엔트리는 소속 **Task**를 명시한다(컨벤션: `../docs/README.md`). Task = common | selection | episode(Track1|Track2) | multi.

---

## [2026-07-31] 저장소 정리: legacy/docs 분리, 활성/legacy 판별 가능화
> **Task**: multi
- 변경: 활성이 import 안 하는 구세대(`tagger`·`vocab`·`prompts`·`window/event_tagger`·`test/batch_readout`·`norm_embed`)를 `legacy/`로, 설계문서를 `docs/`로 분리. `build_client_pool`을 `common/client.py`로 추출. `legacy/docs/decisions`는 공통 폴더 + 파일별 Task 라벨.
- 이유: 데모용으로 관리 없이 작성돼 파일 트리로 용도 판별 불가했음. 타입별 공통 폴더 + Task 라벨로 중복 없이 정리.
- 영향: `.pth`/`run.sh` PYTHONPATH에 `legacy/` 추가(import 21+8 OK). 각 폴더 README.

## [2026-07-31] 저장소 구조: common / task_selection / task_episode 물리 분할
> **Task**: multi
- 변경: flat 모듈을 2단계 task 기준 폴더로 이관. Task1 선별을 `selection.py`로 모듈화.
- 이유: labeling이 ① clip 선별 → ② episode 추출 2단계. 공유 기반과 task별 알고리즘 분리.
- 영향: import는 PYTHONPATH+`.pth`로 top-level 유지. 스키마 json은 vocab 모듈과 동거.

## [2026-07-31] VLM self-consistency 투표 — 힌트 유지(B), 상호작용 합의는 불신
> **Task**: episode(Track2)
- 변경: Phase A VLM 5-vote는 GT 힌트 유지하되 상호작용 "합의"를 독립확인으로 신뢰 안 함.
- 이유: 힌트가 상호작용만 나열 → 맥락 투표는 이미 독립. 상호작용은 Phase C에서 `gt∩vlm`, 맥락은 다수결로 분리 처리.
- 영향: `candidates.py`, Phase C 설계.

## [2026-07-31] Track2 검색 입도 — 교차로 sig/unsig 유지, 도로유형만 병합
> **Task**: episode(Track2)
- 변경: `retrieve.MERGE`의 sig+unsig→intersection 병합을 정정 대상으로. 도로유형만 병합.
- 이유: 신호/비신호는 cause와 결부(비신호=양보 정차 vs 신호=신호 정지) → 병합 금지. 병합 규칙=(a)애매 ∧ (b)query/cause 무가치.
- 영향: `retrieve.py`, `docs/taxonomy_merge_report.md`.

## [2026-07-31] cause 축 — 전이의 "왜", GT+맥락, Track2 1차 query 키
> **Task**: multi (Track1+Track2)
- 변경: cause=전이의 "왜", 모든 값 GT+맥락 해소. **Track2 인덱스에 cause 축 추가 필요**(현재 갭).
- 이유: agent조차 tracking만으론 인과 판정 불가. 큐레이션은 주로 cause로 검색.
- 영향: `candidates.py`/`retrieve.py`, `taxonomy.py`.

## [2026-07-31] ego_action — 두 층위 + 앵커 + 순방향 생성 (invariant 정정)
> **Task**: multi (Track1+Track2)
- 변경: 종방향(accel/decel/stop/creep)=egomotion rule 확정 / 경로(turn·lane_change·u_turn)=CAN 전이 트리거 + 라벨은 맥락 해소(map∪VLM). 앵커=전이 감지(→key_frame·세그먼트). 생성은 순방향 SD→SA→MA, MA는 마지막 해소.
- 이유: CAN yaw만으론 교차로 turn/커브/분기/u-turn 구분 불가(under-determined).
- 영향(드리프트 수정 대상): `tag_v08.V08_SCHEMA` MA-first→forward + ego_action 앵커화; `taxonomy.AUTO_GT`에서 turn류를 맥락해소 계층으로.

## [2026-07-31] 세그먼테이션 = ego 전이 = critical scenario
> **Task**: multi
- 변경: 세그먼트를 ego_action 전이로만 정의. non-reactive는 critical 아님. 양 track 동일 백본.
- 이유: critical=ego 반응으로 확정(추후 revisit). `key_frame_t`=rule.
- 영향: `classify073.consolidate_episodes`.

## [2026-07-31] 산출물 두 제품(Track1/Track2) 병존
> **Task**: multi
- 변경: Track1=v08 SD/SA/MA 메타데이터(라벨링) + Track2=v3 taxonomy 추출(큐레이션). 공용 기반 공유.
- 이유: 라벨링과 큐레이션은 다른 산출물. v3가 드롭했던 SD/SA/MA(Track1)를 명시 유지.
- 영향: `tag_v08` vs `candidates/retrieve`.

## [2026-07-31] v3 관점 전환 — 태거 = retriever (recall 우선)
> **Task**: episode(Track2)
- 변경: 태거=관대한 후보 생성기. Phase A(OR 앙상블·∩ 금지)→B(정규화·병합·랭킹)→C(precision 하류)→D(완전 gold).
- 이유: 큐레이션은 recall이 자산, recall miss>FP. 실측 OR합집합 0.81(병합후 0.90)>VLM단독 0.67.
- 영향: `candidates.py`·`retrieve.py`. v2(∩ precision-우선) 폐기.

## [이전] 2단계 퍼널 · map 재해석 · cut_in 희소 등
> **Task**: multi
- Stage1(CAN+VLM 선별)/Stage2(3DOD+map). map centerlines=경계선(재해석·유효율 35%·is_intersection 죽음). cut_in 극희귀(0/50, ~26/1966). obj3d vx/vy·occlusion 불신. 상세 `docs/PROJECT_DESIGN.md`·`task_selection/SELECTION_STAGE1.md`.

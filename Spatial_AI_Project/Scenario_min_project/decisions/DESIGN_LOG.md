# 설계 변경/개선 이력 (최신이 위)

> 각 엔트리는 소속 **Task**를 명시한다(컨벤션: `../docs/README.md`). Task = common | selection | episode(Track1|Track2) | multi.

---

## [2026-07-31] 설계 문서 통합 — PROJECT_DESIGN 단일 정본
> **Task**: multi
- 변경: `docs/Concept_Design_v3` 제거(내용 90% PROJECT_DESIGN와 중복, 고유=수용기준만). `docs/PROJECT_DESIGN.md`를 **단일 정본**으로 재작성 — 개요(2단계·Track1/Track2)·데이터·아키텍처·관점전환·택소노미(5축60키)·Phase 상태표·가드레일·핵심발견·수용기준(흡수)·**현재상태&업무분담(WP1–8)**·코드맵. 참조 갱신(docs/README·CLAUDE.md·TEAM_REPORT·decisions/README).
- 이유: 두 문서 역할 동일(v3 설계)인데 stale(2026-07-23, Track1/Track2·windowed 승격·taxonomy 확장 미반영). 팀 업무 분담 목적이라 최신 상태+작업 패키지 필요.
- 영향: 설계 정본 1개로 단일화. 팀 공유/분담 기준.

## [2026-07-31] Stage1 기본 선별 = 윈도우+video 반응성으로 승격
> **Task**: selection
- 변경: `run_selection.py` 기본 경로를 **윈도우+video 반응성**(`folder_selection.rank_folder_windowed`)으로 전환. 몽타주 흥미도는 `montage` 서브커맨드(legacy)로 강등. `review.py`로 트랜스코드/HTML/샘플링 공용화(중복 제거). canonical 산출(index.html/select300.json)을 windowed 결과로 정본화(기존 winevent 재사용, 재컴퓨트 없이 `review` 재생성).
- 이유: obj3d 대조 검증에서 몽타주 cut_in corroboration 6%(top50 0%) vs 윈도우+video는 과검 대부분 제거·reactive 50/50. 시간정렬+video가 주변부 투영/오귀속을 구조적으로 해소. Stage1 전제(obj3d-free, CAN+VLM)는 유지(윈도우 엔진도 egomotion+VLM video만 사용).
- 영향: `run_selection.py`·`SELECTION_STAGE1.md`·README. 엔진은 `folder_selection`에 상주(Stage1 global + 폴더 공용). montage 흥미도 경로 보존(비교용).

## [2026-07-31] cut_in obj3d 검증 (Stage1 산출 사후 대조, 배선 아님)
> **Task**: episode (검증) / selection(대상)
- 변경: `task_episode/verify_cutin_obj3d.py` — 선별 산출(event_select/winevent_select)의 VLM cut_in 후보를 `taxo_detect.detect_taxonomy`(obj3d corridor)로 사후 대조. **task_selection엔 미배선**(Stage1은 obj3d-free 유지 확정).
- 결과(동일 300):
  - **몽타주 VLM cut_in**: 전체 31개 중 obj3d confirm 2 → **corroboration 6%**, top50 13개 중 **0/13(0%)**. → 사용자 지적("cut_in 대부분 오류") 정량 확증.
  - **윈도우+video**: cut_in 주장 자체가 31→3(전체)·13→3(top50)로 급감, confirm 1/3. 오검 대폭 감소.
  - **obj3d 기저율**: 전체 300 중 cut_in계열 7개(cut_in 2+attempt 5)=~2.3% → cut_in은 실제로 희소(gold 0/50·corridor 26/1966과 정합). 몽타주 10% 주장은 과검.
- 결론: 윈도우+video가 과검을 대부분 제거. 잔여 미세 cut_in/attempt 정밀 확정은 **obj3d를 권위로**(Stage2/task_episode에서), Stage1은 recall triage로 유지.

## [2026-07-31] 폴더별(조건별) 반응성 선별 알고리즘 (folder_selection)
> **Task**: selection
- 변경: `task_selection/folder_selection.py` 신규 — 데이터가 정적환경 조건 폴더(주간/야간·도심/골목)로 분리 저장된 경우, **폴더 안에서 ego motion에 영향 준 event(반응성)** 가 있는 clip 우선 선별.
- 설계: 패러다임 (B) egomotion+VLM 융합 유지. ego 반응성 점수(`react_ego_score`) = 급제동 2.5·min(harsh,4) ≫ 감속반복·정지출발 > 정지 > 차선변경 > 회전(0.6). VLM(`vlm_reactive`)은 조건 고정이라 맥락 판정 대신 **외부 agent/hazard 반응 여부** 확인(신호대기 routine 정지 배제, event_type enum). `combined=max(ego_norm,vlm)+0.3·min`, **정규화·랭킹은 폴더 단위**.
- 이유: 사용자 지시 — 조건별 폴더 전제에서 "ego 전이/반응이 있는 critical clip" 우선. Stage1(전역 흥미도 triage)과 **별개**(조건 고정 → VLM 역할이 맥락→반응확인으로 이동).
- 입력: `groups={조건:[clip_id]}` 또는 `groups_from_root(루트/<조건>/<clip>)`. 실행 `./run.sh task_selection/folder_selection.py <groups.json> [top_k]`.
- 검증: 합성 2폴더×4클립 end-to-end OK — 폴더별 독립 랭킹, 반응성 이벤트 분해(harsh_brake/decel_repeat/stop_go), ego 강하나 VLM 미확인 clip은 결합점수 하향(융합 의도대로).

## [2026-07-31] 정적환경 VLM 판정 추가 (흡수 태그 검출 배선)
> **Task**: episode(Track2)
- 변경: `vlm_verify._schema`/`_prompt`에 정적환경 필드 추가 — lighting(day/twilight/night)·weather(clear/rain/snow/fog)·road_surface(dry/wet) 단일택 + glare/crosswalk_present/traffic_light_present/undivided_road bool. 공용 매핑 `env_cats(v)`→taxonomy 키. `verify_clip`·`candidates._vlm_present` 양쪽에서 호출. `CTX`에 정적환경 키 추가(fuse에서 VLM 권위·GT 없이 유지).
- 이유: STATUS=vlm_only 정적환경 13태그(조명/기상/노면/glare/crosswalk/신호등/비분리)에 실제 판정 경로 부여. GT 부재 맥락이라 VLM이 유일 생성원(∩ 불필요, 다수결/단독).
- 검증: 스키마 유효·env_cats 매핑 OK, VLM 서버 1클립 end-to-end에서 twilight/clear_weather/dry_road/crosswalk_present/traffic_light_present 판정·매핑 확인.
- 미포함(후속): obj3d 기반 신규 태그는 obj3d GT 업데이트 확인 후. road_worker/vulnerable_pedestrian(이벤트, vlm)은 present enum 확장 별건.

## [2026-07-31] new_tag.json(v0.4 폐쇄어휘) 정적환경+long-tail 흡수
> **Task**: episode(Track2) + common
- 배경: 정적환경 태깅을 성급히 제거했다가(같은 날) 되돌림 — 최종 목적이 **search 기반 학습데이터 큐레이션**이라 정적환경 태그가 필요. `legacy/new_tag.json`(condition 25 + event 25, GT rule·cell_role·3값판정·backoff 갖춘 성숙 어휘)을 확인.
- 결정(옵션2): **new_tag 정본 채택 대신 taxonomy.py 구조 유지하며 흡수**. 검출기 배선 보존, GT rule 세부 미채택.
- 변경(`common/taxonomy.py`): 새 축 **정적환경**(조명 night/twilight/day·기상 rain/snow/fog/clear·노면 wet/dry·glare·crosswalk_present·traffic_light_present·undivided_road·crowd) + long-tail 이벤트(vehicle_cross_path·wrong_way·stationary_vehicle·large_vehicle·emergency_vehicle·animal·road_obstacle·jaywalking·road_worker·vulnerable_ped). 조명/기상/노면 9 승격. **KEYS 41→60**.
- **egomotion 기반 제외(사용자 결정)**: hard_brake/hard_steer/overtake/sharp_curve/congestion/free_flow 는 별도 태그로 추가 안 함 — egomotion primitives/기존 tag로 병합(congestion→`creep`, 급제동/급조향→events decelerate/turn·harsh_decel, overtake→lane_change). 추가 어휘로 미사용.
- **obj3d 기반 신규(사용자 결정)**: 어휘만 유지, 검출기는 obj3d 결과 확인 후 도입(deferred).
- `STATUS` dict: 흡수 태그별 visionary 실행상태(runnable/vlm_only/sparse/gold). ODD(조명/기상/노면 dim)는 검색 정본을 정적환경 축에 넘기고 per-clip 단일값 표현으로만 잔존.
- 영향: gold 도구 정적환경 축 노출(chips↑), 활성 import 9/9 OK, KEY 유일성 OK. 미구현 후속 = obj3d 신규 태그 검출기 + 정적환경 VLM 판정 + new_tag 3값판정/FDR 프레임 도입 여부.

## [2026-07-31] 드리프트 4건 코드 정합화 (설계 불변식 반영)
> **Task**: multi (Track1+Track2)
- 변경:
  1. `tag_v08.V08_SCHEMA` 순방향 재정렬 — required=`[scene_description, critical_components, chain_of_causation, cause, ego_action]`(MA-last). guided decoding이 관찰 선행 후 MA를 마지막 커밋. rec 출력도 동일 순서. ego_action은 `_ground_ego_action`(rule/arc 앵커) 유지.
  2. `taxonomy.AUTO_GT` = `{stop}`(순수 종방향 kinematic만). turn/u_turn류는 `HUMAN_KEYS`(맥락해소·평가대상)로 이동. `auto_tags_from_arc`는 Phase A recall 후보로 turn 계속 방출(주석 명시).
  3. `retrieve.MERGE`에서 `intersection_signalized/unsignalized` 제거 — sig/unsig 별도 유지. 도로유형(`road_urban_arterial/backstreet→road_surface`) 병합만 존치.
  4. Track2 인덱스에 **cause 축** 추가 — `candidates._cause_candidates`(카테고리→cause 사상, channels·vote_fraction 승계, 증거없음→other) → `retrieve._cause_axis`가 `index_clip` 에피소드에 `cause:[{cause, confidence, channels, from}]` 노출.
- 이유: CLAUDE.md §2 확정 불변식(순방향 SD→SA→MA / 두 층위 ego_action / cause=1차 query 키 / sig·unsig=cause 결부)과 코드 정합. [[ego_action 두 층위]]·[[cause 축]]·[[Track2 검색 입도]] 반영.
- 영향: `tag_v08.py`·`taxonomy.py`·`retrieve.py`·`candidates.py`. 스모크 테스트 4/4 통과, import ACTIVE 16/16. cause 단일 확정·상호작용 gt∩vlm은 Phase C 미구현(후속).

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

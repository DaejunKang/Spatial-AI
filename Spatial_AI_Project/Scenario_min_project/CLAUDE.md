# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

자율주행 20초 클립에 **scene description / analysis / meta-action 메타데이터를 tag**하고, long-tail 상황을 **추출**하는 파이프라인(KATECH VLA). VLM(NVIDIA Cosmos-Reason)은 **frozen** — 재학습하지 않고 GT(egomotion/obj3d/map)+규칙+VLM 조합 로직을 개선한다.

---

# 1. What the code does (현재 코드 실태)

## 2단계 구조 (최상위 개요)
metadata labeling = **① clip 선별 → ② 선별 clip에서 episode 추출·labeling** 의 2단계 퍼널. 두 단계를 **별도 task로 관리**한다.
- **Stage 1 = Task「선별」**: 전체 로그(**CAN+video만**, obj3d/map 없음) → `ego arc(egomotion) + VLM 흥미도(몽타주)` 결합 랭킹 → 상위 K. **recall 위주 triage**.
- **Stage 2 = Task「episode」**: 선별 clip(**3DOD + map** 사용) → egomotion **ego 전이**로 에피소드 분할 → 에피소드별 메타데이터(Track1 SD/SA/MA+cause / Track2 taxonomy 후보).
- 근거: obj3d/map 없는 raw 로그는 **싸게 넓게 선별**하고, 비싼 상세 태깅은 **선별 clip에만** 적용(비용·recall 퍼널).

## 폴더 구조 & ownership
```
common/          공유 인프라(상위 프로젝트 공유): config·paths·dataset·events·taxonomy·vocab073·client·overlay + schema
task_selection/  Task「선별」(Stage1): selection.py + SELECTION_STAGE1.md → gold_label/select/
task_episode/    Task「episode」(Stage2, 활성만):
                   base   classify073(consolidate)·taxo_detect·map_lane·vlm_verify
                   Track1 tag_v08          Track2 candidates·retrieve
legacy/          활성이 import 안 하는 구세대(v0.7.1 tagger·vocab·prompts·window/event_tagger·test/batch_readout, norm_embed)
docs/            설계·리포트(PROJECT_DESIGN[정본]·TEAM_REPORT·taxonomy_merge_report·STRUCTURE 등)
decisions/       설계 변경/개선 이력(DESIGN_LOG)
```
- import는 `run.sh` PYTHONPATH + venv `.pth`가 `common/task_selection/task_episode/legacy`를 top-level 경로로 올려 유지.
- **Task 라벨 컨벤션**: `legacy/docs/decisions`는 task별로 안 쪼개고 공통 폴더로 두되, **각 파일/엔트리 최상단에 `> **Task**: common|selection|episode(Track1|Track2)|multi` 라벨을 명시**(정본 `docs/README.md`).

## 명령어
정식 test/lint 프레임워크 없음. `test_readout.py`는 단위테스트가 아니라 단일클립 smoke run.
```bash
./run.sh <script.py> [args]        # venv 실행 + logs/latest.log 기록 (예: ./run.sh test_readout.py)
tail -f logs/latest.log            # 실시간 모니터
.venv/bin/python <script.py>       # 직접 실행
./run.sh batch_readout.py --limit 5 [--index i] [--overwrite] [--no-overlay]   # 배치(v0.7.1 경로) → outputs/tags/
.venv/bin/python -m pip install <pkg>     # venv pip 없으면 ensurepip 후 / 또는 ~/.local/bin/uv
for p in 8001 8002 8003 8004; do curl -s -o /dev/null -w "$p %{http_code}\n" localhost:$p/v1/models; done  # VLM 서버
python3 -m http.server 8080 --bind 127.0.0.1 --directory gold_label   # 라벨링/리뷰/데모 서빙(loopback→SSH터널)
docker container logs <container>  # 전체형(짧은 `docker logs` 아님)
```
임시 산출물은 `$CLAUDE_JOB_DIR/tmp`에 둔다(공용 `/tmp` 금지 — 병렬 job 충돌).

## 파이프라인 4세대가 공존 (혼동 주의)
저장소엔 서로 다른 스키마의 태깅 파이프라인이 층층이 쌓여 있다:
- **v0.7.1** `tagger.py`+`prompts.py`+`vocab.py`+`window_tagger.py`+`event_tagger.py` — 원래 세그먼트 메타데이터. **유일한 runnable 엔트리(`test_readout.py`, `batch_readout.py`)가 이걸 사용**(`config.TAG_MODE` 디스패치).
- **v0.7.3** `classify073.py`+`vocab073.py` — 2-pass 분류기. `consolidate_episodes()`(에피소드 병합)는 상위 파이프라인 공용.
- **v0.8** `tag_v08.py` — SD + critical_components 자유기술. `outputs_v08/tags/`(clip당 record JSON: window/ego_context/cause/scene_description/critical_components/chain_of_causation/search_tags).
- **v3(현재 방향)** `candidates.py`→`retrieve.py` (+ `taxo_detect`/`vlm_verify`/`map_lane`/`taxonomy`) — recall-우선 retriever. ad-hoc 스크립트로만 실행.

## 두 제품 (Track1 / Track2) — 공용 기반 공유
- **Track1 = v08형 세그먼트 메타데이터** (SD/SA/MA + cause + key_frame_t) — 자율주행 개발용 라벨링. authoritative 계보 = `tag_v08.py`.
- **Track2 = v3 추출 retriever** (taxonomy 카테고리 후보 검색) — 큐레이션. `candidates.py`→`retrieve.py`.
- 공용 기반: `events.py`(egomotion), `taxo_detect.py`(obj3d), `map_lane.py`(map), `vlm_verify.py`(VLM), `classify073.consolidate_episodes`(세그먼테이션).

## v3 검색 파이프라인 (Track2)
- **Phase A** `candidates.py` — 3채널 **OR 합집합**(ego-arc / obj3d-GT / VLM 5-vote temp>0), provenance·vote_fraction·multi 부착. ∩·게이트 미적용.
- **Phase B** `retrieve.py` — 정규화 + 병합(`taxonomy_merge_report.md`) + confidence 랭킹. 병합 원본은 `sub`에 보존.
- **Phase C/D** 미구현 — precision 게이트(∩·Tier-B·등급) + 완전라벨 gold.
- 실측 recall(gold 50): OR합집합 0.81(병합후 0.90), VLM단독 0.67, GT단독 0.17.

## 신호 채널
- `events.py` — egomotion 거동. **turn은 yaw 크기가 아니라 net-heading 변화량으로 분류**(u_turn 미검출 — CAN만으론 판별불가라 방향 L/R 통합). egomotion 로그는 비디오보다 길어 **카메라 timestamp 범위로 클립**(`load_egomotion_clip`).
- `taxo_detect.py` — obj3d 3DOD 상호작용. `detect_taxonomy()`.
- `map_lane.py` — map 차선. Branch A(lane-relative)/B(corridor fallback), `map_valid()` 게이트(유효율 ~35%).
- `vlm_verify.py` — VLM 검증/분류. `_prompt`/`_schema`(guided_json), `fuse()`(∩), `ground_signals()`.
- `taxonomy.py` — 단일 공유 택소노미. `AUTO_GT`/`HUMAN_KEYS`/`auto_tags_from_arc`.

## 데이터 규약 & 함정
- **두 데이터셋**: `config.DATASET_ROOT`=구 `PhysicalAI-AV-curated`(legacy `obstacle.offline`). **활성 데이터는 `paths.py`의 `visionary-nvidia`**(obj3d+map+egomotion). 신규 코드는 `paths.py`.
- **obj3d**: lidar frame `x=전방, y=좌우`, `boxes_3d`=11-DOF. **vx/vy 16%만 유효(위치미분 사용), occlusion NaN, 일부 clip `track_id` 컬럼 누락**(로더 방어).
- **map `centerlines`=실제 차선 경계선**(명칭 오류). ego차로=y=0 사이 인접 경계쌍. **`is_intersection` 전부 0(죽음)**, `timestamp_us`=0→frame_idx 선형매핑.
- **VLM 호출**: `guided_json`으로 enum 강제. 결정론 `TEMPERATURE=0`(config); **n-vote는 temp>0 명시**. 입력 `video_url`(subclip data URI, `SEND_FPS=10`이나 서버 ~12프레임 캡) 또는 `image_url`(몽타주).
- 설계 문서: `PROJECT_DESIGN.md`(정본)·`TEAM_REPORT.md`·`SELECTION_STAGE1.md`·`taxonomy_merge_report.md`.

---

# 2. Design constraints & rationale (인터뷰 확정)

## 제품 범위
- **두 제품 유지**: Track1(v08 SD/SA/MA 메타데이터) + Track2(v3 추출 retriever). CLAUDE.md는 둘 다 authoritative로 취급. *근거: 라벨링과 큐레이션은 다른 산출물이나 공용 기반을 공유.*

## Frozen tagger / anti-circularity (DO NOT VIOLATE)
- 태거 **fine-tuning 금지**(순환). **cosmos2(port 8000) 무접촉**(타 그룹). 우리 복제본 8001–8004(`config.BASE_URLS` 라운드로빈); **port 8001은 평가-frozen 인스턴스**이므로 8002–8004가 동일 frozen 가중치인지 유지 확인. NIM 컨테이너 파일 수정 시 blake3 체크섬으로 컨테이너 사망 → 서버는 운영 서버(재기동 승인 필요).
- 태거 출력이 태거 학습에 되먹임 금지. gold는 사람 라벨(태거 파생 아님), `AUTO_GT`는 평가 제외 — 이 anti-circularity 가드 유지.
- KV-cache·TTA·합성데이터·다운스트림 VLA 학습은 범위 밖.

## 세그먼테이션 = critical scenario = ego 전이
- **situation/세그먼트는 ego_action 전이로만 정의.** ego에 영향 없는 상황(non-reactive agent 이벤트)은 **critical scenario 아님** → 세그먼트 미생성. *근거: critical의 정의가 ego 반응. 추후 문제 시 revisit 가능.* Track1·Track2는 **동일 ego-전이 세그먼트 백본 공유**. `1 record = 1 transition = 1 behavior event`.
- **`key_frame_t`는 rule 기반**(egomotion 전이 onset, `ep["onset"]`). 모델 생성 금지.

## ego_action — 두 층위 + 앵커 + 순방향 생성
- **종방향(accel/decel/stop/creep) = egomotion rule 확정**(모델 불가).
- **경로(turn_L/R·lane_change_L/R·u_turn) = CAN이 전이만 트리거 + 라벨은 맥락 해소 = map ∪ VLM 앙상블.** *근거: CAN yaw만으론 교차로 turn / 커브 / 분기 / u-turn 구분 불가(under-determined).*
- **앵커의 취지 = 전이 감지**: ego_action 전이 지점이 상황 발생 지점 → rule/CAN이 이를 잡아 key_frame·세그먼트·종방향 확정.
- **생성은 순방향 SD→SA→MA**: 모델은 관찰 선행(scene_description → critical_components → …)으로 생성하고 **MA(ego_action)는 앵커+맥락으로 마지막에 해소**. behavior-first 금지.
- ✅ **드리프트 수정 완료(2026-07-31)**: `tag_v08.V08_SCHEMA`를 순방향(MA-last)으로 재정렬 — `[scene_description, critical_components, chain_of_causation, cause, ego_action]`. ego_action은 `_ground_ego_action`(rule/arc 앵커)로 최종 해소. `taxonomy.AUTO_GT={stop}`로 축소 — turn/u_turn류는 `HUMAN_KEYS`(맥락해소·평가대상)로 이동, `auto_tags_from_arc`는 Phase A recall 후보로만 turn 방출.

## cause 축
- `cause ∈ {agent, signal, road_geometry, other}` (boolean agent_present 아님).
- **cause = "그 ego 전이가 왜 일어났나"의 답** — 전이 결부(ego_action과 동일 앵커 구조).
- **모든 값이 GT 앵커 + 맥락 함께로 해소** — agent조차 tracking만으론 부족(진행로 agent 존재 ≠ 전이 원인). *근거: 인과는 GT 증거만으론 under-determined.*
- **cause = 큐레이션 1차 query 키** → **Track2 검색 인덱스에 cause 축 포함**. ✅ 추가 완료(2026-07-31): `candidates._cause_candidates`(카테고리→cause 사상, provenance·vf 승계, 증거없음→other) → `retrieve._cause_axis`가 index_clip 에피소드에 `cause:[{cause, confidence, channels, from}]` 축 노출. recall-first 후보(단일 확정은 Phase C).

## Track2 검색 입도 (패밀리별)
- **교차로 sig/unsig = 유지(병합 금지)**: cause와 결부(비신호=양보 정차 vs 신호=신호 정지)라 하류 query 가치 있음. ✅ 정정 완료(2026-07-31): `retrieve.MERGE`에서 intersection_* 제거, sig/unsig 별도 카테고리 유지(`road_urban_arterial/backstreet→road_surface`만 병합).
- **도로유형(urban/backstreet 등) = 병합 허용**: 판단 자체가 애매하고 query 가치 낮음.
- **병합 규칙**: (a) 판단이 애매하고 **(b) 하류 query/cause 가치가 없을 때만** 병합. 원본은 `sub` 보존.

## VLM 투표 (self-consistency n-vote)
- VLM 5-vote는 (Phase A) 후보 생성, (Phase B) vote_fraction→랭킹, (Phase C) **GT부재 맥락 카테고리의 precision=다수결**에 기여. **맥락(도로/신호/교차로)은 GT부재라 VLM이 유일 생성·정밀도 원천이며 독립적.**
- **결정(B)**: VLM 프롬프트에 **GT 힌트 유지**(상호작용 확인용). 단 **상호작용의 "multi 합의"는 독립확인으로 신뢰하지 않음** — Phase C에서 상호작용은 `gt∩vlm`으로, 맥락은 VLM 다수결로 분리 처리. *근거: 힌트는 상호작용만 나열하므로 맥락 투표는 이미 독립; 상호작용 합의만 오염되나 gt∩vlm이 별도로 처리하므로 다수결 신뢰 불필요.*

## 운영
- `docker container logs`(전체형). `python3`, venv(via `uv`), 터미널 세션마다 재활성화. 신규 데이터 접근은 `paths.py`(visionary) 사용.

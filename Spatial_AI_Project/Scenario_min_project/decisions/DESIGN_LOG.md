# 설계 변경/개선 이력 (최신이 위)

> 각 엔트리는 소속 **Task**를 명시한다(컨벤션: `../docs/README.md`). Task = common | selection | episode(Track1|Track2) | multi.

---

## [2026-08-31] same-anchor 재현성 실험 — 5-vote는 seed로 해소, tag_v08 잔차는 서빙층
> **Task**: multi

**배경**: 직전 [2026-08-31] 항목("재현성 불일치 조사")의 미결 최우선 항목 — "코드 감사만으로는
검증이 아니다. anchor를 고정하고 실제로 N회 반복 decode해 확인하라" — 을 실행한 결과다.
스크립트 `tmp/repro_experiment.py`(gitignore 대상, 재실행 가능), clip
`12c4c627-1311-42a2-a28c-307e55996e6f`(에피소드 1개, arc
`stop→accelerate→turn_right→decelerate`), 결과는 `tmp/repro_results/full_candidates.json`·
`full_tagv08.json`. 실행: `./run.sh tmp/repro_experiment.py --clip <id> --pipeline both --n 10`.

**방법**: anchor(subclip+프롬프트 텍스트 또는 GT 힌트)를 clip마다 **2회 독립 구성**해 md5/해시
비교로 먼저 고정을 확인한 뒤, 5개 조건 셀(현행/seed고정/포트고정/`n=5`단일요청/포트+seed)마다
같은 anchor로 10회 반복 decode. 두 실험 모두 **anchor는 2회 독립 구성 모두 완전 동일**
(subclip md5·프롬프트 해시·GT 힌트 해시 일치) — anchor 불일치가 원인일 가능성은 이번 clip에
한해 배제됨.

**결과 1 — candidates.py 5-vote(`temperature=0.7`), 10회 중 vote_fraction 패턴이 몇 종류
나왔는지**:

| 셀 | 조건 | distinct 패턴/10 |
|---|---|---|
| a_baseline | 현행 그대로(라운드로빈, seed 없음, `max_retries=2`) | **10/10** (한 번도 안 겹침) |
| b_seed | 라운드로빈 유지 + 투표별 `seed=1000+j` | 3/10 |
| c_pinned | 포트 8001 고정 + `max_retries=0`, seed 없음 | **10/10** |
| d_n5 | 단일 요청 `n=5`, seed 없음 | **10/10** |
| e_pinned_seed | 포트 8001 고정 + `max_retries=0` + `seed=1000+j` | **1/10 — 10회 전부 완전 동일** |

**결론 1**: "5-vote 비결정성"은 실재하며 anchor 문제가 아니다. **포트 고정만으로는(c) 전혀
개선되지 않고, 배치를 단일요청으로 묶어도(d) 개선되지 않는다 — 즉 서빙층 라우팅/배치 자체는
이 현상의 원인이 아니다.** 원인은 거의 전적으로 **seed 부재**(temp=0.7 샘플링에 RNG 고정이
없음)였다. seed만 줘도(b) 10→3으로 급감하고, **포트 고정과 seed를 같이 주면(e) 완전
결정론**이 된다. `candidates.py:138-140`에 `seed=BASE_SEED+j`를 추가하는 것이 검증된
해결책이다(미결 5번 "L1 seed 전달"이 이 실측으로 확정됨).

**결과 2 — tag_v08.py 2-pass(`_structure`, `temperature=0`, guided_json), 10회 반복 시
필드별 안정성**:

- `cause`: **4개 셀 × 10회 = 40/40 전부 `agent`로 완전 안정.**
- `ego_action`: 대부분 `ego_turn_right`이나 **셀·조건과 무관하게 10회 중 0~2회씩 다른 값으로
  튐**(`ego_lane_change_right`/`ego_lane_keep`/`ego_follow`) — `a_baseline` 1회,
  `c_pinned` 1회, `e_pinned_seed` 2회, `b_seed`만 0회(표본 10 규모라 우연일 수 있음).
  포트를 고정하고 seed까지 줘도(e) 사라지지 않았다.
- `scene_description`/`chain_of_causation` 길이는 매 회 제각각(263~400자).
- 부수: pass1(`_reason`, 자유서술) 자체도 완전히 결정적이진 않음 — 같은 anchor로 2회 독립
  호출 시 한 번은 길이가 1808자 vs 1274자로 크게 갈렸고(파일럿 n=1), 본 실행(전체 n=10 세트)
  에서는 2회 모두 1274자로 일치 — **재현 여부 자체가 비결정적**이라는 뜻.

**결론 2**: `temperature=0`(greedy)에서도 이 서빙 스택은 bitwise 재현을 보장하지 않는다.
`cause`처럼 결정 여유가 큰 필드는 흔들리지 않지만, `ego_action`처럼 후보 간 확률차가 근소한
필드는 anchor·포트·seed를 전부 고정해도 흔들린다. seed는 `temperature=0`에서 이론상
샘플링에 관여하지 않으므로(=argmax) 이 잔차는 **seed로 해소되는 종류가 아니다** — 서빙층
(프리픽스 캐시·chunked prefill·cudagraph 배치)의 부동소수 합산 순서 변동이라는 §8.3
`:725`의 원 서술이 이 지점에서 그대로 실측 확인된 것으로 판단한다. 다만 배치 변동을
0으로 만든 조건(요청을 단독으로 서버에 보내 동시 트래픽 0)까지는 이번 실험에서 통제하지
않았으므로, 이 잔차가 "서빙층 자체의 하한"인지 "동시 트래픽에 의한 배치 변동"인지는
추가 실험 대상으로 남긴다.

**갱신되는 미결 항목**: 직전 [2026-08-31] 항목의 "L1 seed 전달"은 본 실측으로 **효과
확정**(코드 반영은 별도 커밋 대상, 이번 세션은 관찰만). "L2는 제거 불가"는 유지하되,
tag_v08 쪽은 seed로 해소되지 않는 잔차가 실측됐다는 점을 구체화함.

**영향**: 코드 변경 없음(`tmp/repro_experiment.py` 신설은 실험 스크립트, `tmp/`는 gitignore
대상이라 저장소 추적 파일에는 영향 없음). `candidates.py:138-140`의 seed 추가가 다음
커밋의 구체적 변경 대상.

---

## [2026-08-31] 재현성 불일치 조사 — 원인은 decode 단이 아니라 집계/순회 층
> **Task**: multi

**배경**: "decode 단의 재현성 불일치를 알고 있어서 free scene description이 아닌 vocab 지정
(닫힌 어휘 점수화)으로 풀려 했다"는 전제로 시작한 조사. 그 전제의 유일한 근거는
`docs/design/pipeline_design_guide_v0.4.md:725`의 "기존에 확인된 5-vote 비결정성" 한 줄이다.
코드 전수 확인 + 서빙 컨테이너 실측으로 이 전제를 검증했다.

**관측**:

0. **방법론 한계 — "같은 입력으로 비교했는가"를 이 조사도 직접 검증하지 못했다.**
   이 조사는 plan mode에서 3개 탐색 서브에이전트를 배경 실행해 코드·문서를 감사하는 방식으로
   진행됐다 — 특정 clip에 대해 실제로 두 번 decode를 돌려 anchor(프롬프트 텍스트·서브클립)를
   로그로 남기고 diff한 **통제 실험이 아니다.** 사용자가 "정확히 같은 입력으로 비교한 게
   맞는지" 확인을 요청해 아래 1번을 재검증했고, 그 결과 **원인으로 지목한 버그가 실제 decode
   비교 경로에는 닿지 않는다는 것이 드러났다** — 최초 서술(아래)은 과잉 일반화였다.
   근본 문제: `pipeline_design_guide_v0.4.md:725`의 "기존에 확인된 5-vote 비결정성"도 언제
   누가 무엇을 비교해 확인했는지 기록이 없어(6번 참조) 이 조사가 그 원 관측을 재현·반증할
   방법이 없다. **아래 1은 "무엇이 비결정적인가"에 대한 정확한 관측이지, "5-vote decode
   비교가 이것 때문에 무효였다"는 검증은 아니다.**

1. **확정(실측)이나 활성 decode 경로에는 닿지 않음 — L3 집계/순회 층은 VLM 없이도
   산출물을 바꾸지만, 범위는 dead code·legacy에 한정된다.**
   `common/events.py:273` `for k in set(tid.tolist())` — parquet의 `track_id`가 문자열이고
   `run.sh`에 `PYTHONHASHSEED` 고정이 없어, 동일 입력을 3개 프로세스에서 실행하니 순회 순서가
   매번 달랐다(`80 ['120','156','97',...]` / `['142','21','77',...]` / `['29','21','30',...]`).
   이 순서가 `:307-313`의 안정정렬+dedup에서 "어느 트랙이 남는지"를 결정하고(`min_dist`가
   `:288`에서 `round(...,1)`로 양자화되어 동점이 잦음), 결과가 **`common/events.py:251`
   `detect_obstacle_events()`를 거쳐 `classify073.tag_clip_v073`·`legacy/event_tagger.py:130`
   으로만 흐른다.**
   재확인 결과 `tag_clip_v073`을 호출하는 활성 코드가 **없다**(`tag_v08.py`/`candidates.py`/
   `vlm_verify.py`/`folder_selection.py`는 `classify073`에서 `consolidate_episodes`·
   `_causal_agent` 등만 가져다 쓰고 `tag_clip_v073`은 아무도 호출하지 않음). 실제 decode
   anchor를 만드는 활성 경로는 별개 함수를 쓴다: `tag_v08.py:226`의
   `detect_obj3d_events()`(`events.py:187-248`)는 `tracks.items()`(**dict**, 삽입순 —
   해시 랜덤화 영향 없음)를 쓰고, `taxo_detect._load_tracks`(`:66`)는 `range(len(...))`로
   파일 순서 그대로 순회한다. `candidates.py`/`vlm_verify.py`가 프롬프트에 넣는 GT 힌트도
   `vlm_verify.py:105` `sorted(cands)`로 정렬 후 삽입된다. 서브클립 영상도 md5 3회 동일(4번).
   **즉 활성 파이프라인(tag_v08/candidates)의 anchor(GT 힌트 텍스트 + 영상)는 이번에 재확인한
   범위에서 실행 간 동일하다** — 이 버그로는 그 경로의 decode 비교 불일치를 설명할 수 없다.
   집계/순회 층에 남아 있는 활성 경로 결함은 **anchor 내용이 아니라 산출물 순서/선별**에
   국한된다: `folder_selection.py:277-285`(무이벤트 클립 `clip_score=0.0` 대량 동점 +
   `as_completed` 수집 순서 → **어느 clip이 top-K에 뽑히는지**가 실행마다 달라짐 — 이건
   "같은 clip의 decode 비교"가 아니라 "애초에 어느 clip을 도느냐"의 문제), `:247`(어느 3개
   윈도우를 VLM에 보낼지가 동점으로 갈림 — 이 역시 anchor 선택 단계, decode 자체 아님),
   `candidates.py:125,131,143-151` set 순회 → `retrieve.py:67,75,81` **출력 배열 순서**(값 아님),
   `sort_keys=True` repo 전체 0건(§8.3 `:731` 해시 비교가 키 순서만으로 실패),
   `folder_selection.py:279-283` 표본 상대 정규화, `review.py:26,35`(seed=42 고정이나 pool이
   `os.listdir` 산물이라 클립 1개 증감에 표본 300개 재배치 — `:33` 주석은 풀 불변일 때만 참).

2. **유력·미계측 — L2 서빙 층.** `cosmos_dj`(8001) 컨테이너 env 실측:
   `NIM_ENABLE_KV_CACHE_REUSE=1`(프리픽스 캐시 ON), `NIM_ENABLE_CHUNKED_PREFILL=1`,
   `NIM_MAX_NUM_SEQS=256`, `cudagraph_mode=FULL_AND_PIECEWISE`, seed 관련 env 없음.
   §8.3 `:725`가 원인 후보로 지목한 경로가 전부 활성 — `temperature=0`이어도 bitwise 재현이
   보장되지 않는다. 클라이언트도 부하 변동을 만든다: `ThreadPoolExecutor(8/12)`
   (`folder_selection.py:274,300`), 5표를 서로 다른 복제본으로 분산(`candidates.py:139`),
   `common/client.py:16-23` 헬스체크(2s) 통과분만 풀 구성 → `len(pool)` 1~4 가변,
   openai SDK 기본 `max_retries=2` 무기록 재시도.

3. **반증 해소 — 복제본 가중치 동일성.** `trunk_실측_20260814.md:97`의 "8001·8002-4가 다른
   이미지 → 가중치 동일성 미검증" 지적을 재확인: `cosmos_dj_snap:latest`(f8dcbf66)의 Parent가
   `nvcr.io/nim/nvidia/cosmos3-reasoner:latest`(19add3ed)이고, 4개 컨테이너 env 해시 동일,
   `/v1/models`의 `created` 4개 모두 `1788134979`. snap은 베이스 이미지 위에 NIM 캐시 레이어를
   commit한 자식 — 별개 가중치 아님.

4. **용의선상 제외 — 프레임 추출.** `sample_montage`/`write_subclip` 각 3회 실행 md5 동일
   (5,727,505B/181frames). OpenCV 시크·mp4v 인코딩은 이 빌드에서 재현적.

5. **설계상 의도 — L1 5-vote.** `candidates.py:26-27` `VOTE_N=5`,`VOTE_TEMP=0.7` — 저장소 유일의
   temp>0 경로. 호출 14곳 전부 `seed`·`top_p`·`top_k`·`n`·`logprobs` 미전달이고, `client.py`가
   파라미터를 캡슐화하지 않는 순수 팩토리라 공통 주입 지점 자체가 없다. → "불일치"가 아니라
   **재현 불가**. seed 전달로 대부분 해소 가능(미착수).

6. **근거 추적 불가.** §8.3 `:725`의 "기존에 확인된 5-vote 비결정성"은 DESIGN_LOG·logs/(25개)·
   docs/ 어디에도 원 관측(날짜·수치·해시 비교)이 없다. `tag_vocab_v0.4.json`의
   `versioning.reproducibility_params` 5군을 기록하는 코드도 없다(`disclosure.py`의 4필드가
   유일한 메타이며 디코딩 축 없음). `CLAUDE.md:88`이 정본으로 지목한
   `실험계획_추론경로_260824.md`는 main 워킹트리에 없다(실체는 워크트리의
   `docs/experiments/experiment_design_v0.1.md`).

**판단**: vocab 지정(닫힌 어휘 점수화) 전환을 **재현성 근거로 추진하지 않는다.** 1번 재검증
결과 활성 파이프라인의 anchor는 이미 결정적이므로, 그 경로에서 진짜 decode 비결정성이
관측된다면 원인은 "anchor가 달랐다"가 아니라 2번(서빙층: 프리픽스 캐시·chunked prefill·
cudagraph·배치 변동) 또는 5번(5-vote는 설계상 temp=0.7+seed 없음)일 가능성이 높다. 워크트리
프로토타입 `task_episode/seq_score.py`(`.claude/worktrees/selection-dist/`)도 `logprobs=True,
top_logprobs=K`로 동일한 L2 서빙 경로를 탄다 — vocab 지정이 서빙층 비결정성을 우회하지
못한다. 점수화는 후보 간 logprob 근소차에 argmax가 걸리는데, 실측된 충돌 필드 9개
(`tag_vocab_v0.4.json` `scoring_method.collision_note`, 2026-08-28: 19개 중 9개 — 좌/우 쌍 +
`road_geometry`/`road_condition`,`stop_sign`/`stop_line` 등 접두어 공유 계열)가 정확히 그
지점이다. 전환은 증상을 "텍스트가 눈에 띄게 다름"에서 "점수 1e-3 차이로 라벨이 조용히
뒤집힘"으로 바꿀 뿐이다. vocab 지정의 정당한 근거는 재현성이 아니라 어휘 곱 폭발 억제·판정
일관성(지침서 §1.4, `:81`)이다.

**부수 발견(수정은 별도)**: `common/dataset.py:269,274`가 원본 30fps 프레임을 전부 쓰면서
컨테이너 fps 헤더만 `SEND_FPS=10`으로 선언 — 실측 6.0s 창이 181프레임/선언 18.1초(3배 팽창).
서버는 `NIM_MEDIA_IO_KWARGS={"video":{"fps":4.0}}`로 **선언 타임라인 기준** 재샘플하므로,
CLAUDE.md의 "서버 ~12프레임 캡" 서술은 캡이 아니라 4fps 재샘플이며 §4 전환 규칙("캡이 상한")의
전제가 부정확하다.

**미결(다음 조치 대상, 이번 세션에서 착수 안 함)**:
- **통제된 same-anchor 재현성 실험 (최우선, 0번의 대안)** — 이번 조사는 코드 감사이지 실측
  실험이 아니다. 실제 검증은: 특정 clip·윈도우 1개를 고정 → anchor(프롬프트 텍스트 전문·
  서브클립 md5·GT 힌트)를 로그로 저장 → `tag_v08._structure`/`candidates._vlm_present`를
  N회 반복 호출 → anchor 로그가 N회 모두 동일한지 먼저 확인하고, 동일함에도 decode 출력이
  갈리면 그때 L2(서빙층)로 원인을 좁힌다. 이 실험 없이는 "5-vote 비결정성"도 이번 조사의
  1번 정정도 최종 확정이 아니다
- L3 결정화: `PYTHONHASHSEED=0` 고정, `events.py:273`류 set 순회 `sorted()`화, 정렬 2차 키 추가,
  산출 JSON `sort_keys=True`
- L1 seed 전달: `common/client.py` 공통 호출 래퍼 신설 후 `candidates.py` n-vote에 `seed` 부여
- L2는 제거 불가 — 배치/캐시 민감도를 계측해 허용 오차로 문서화(§8.3 `:732`)하고 평가 실행
  규약을 단일 엔드포인트+`workers=1`로 고정하는 방안 검토
- L4 매니페스트: `common/manifest.py` 신설, `tag_vocab_v0.4.json`의 재현성 5군 스탬프
- CLAUDE.md의 "서버 ~12프레임 캡" 서술 정정

**영향**: 코드 변경 없음(조사·기록 전용). `common/events.py`·`task_episode/candidates.py`·
`task_selection/folder_selection.py`·`common/client.py`·`common/dataset.py`가 향후 수정 대상.

---

## [2026-08-28] 설계변수 기본값 코드 구현 1차 — 곡률보정 Stage2 배선, GT배치·공개필드 배선
> **Task**: multi

- 변경:
  1. **B1 결정**: `common/gt_placement.py` 신설 — GT 주입(DV-8) 미분류 필드 기본값을 `gt_before`로 확정(기존 `gt_free` 안이 아님). 이유: `gt_free`(미주입)는 grounding 실패 위험을 조용히 방치하지만 `gt_before`(관측 사실 주입)는 최악의 경우도 과잉 주입일 뿐 정보 손실이 아님 — 안전한 쪽 강등.
  2. **공개의무 배선**: `common/disclosure.py` 신설, `tag_v08.py`/`candidates.py`/`vlm_verify.py`에 `flags`로 스탬프, `retrieve.py`에 pass-through. `s2_conditioning` 실측값(`rule_injected`)이 스키마 선언 기본값(`minimal`)과 불일치함을 그대로 기록 — 코드 동작은 안 바꾸고 사실만 정직하게 남김(§3 GT 힌트 전환 규칙 준수).
  3. **곡률보정(A2) Stage2 배선**: `map_lane.road_curvature_over`/`default_curvature_fn` 신설, `events.detect_events(curvature_fn=...)` opt-in 파라미터 추가(기본 None — 미지정 호출은 전부 기존 동작 그대로, map import 없음), `tag_v08.py`에 기본 배선.
  4. `decisions/DV_DEFAULTS.md` 신설 — 설계변수 전체의 현재 기본값·실행상태 스냅샷(살아있는 레퍼런스, 결정 이력 아님).

- 근거(곡률보정 안전성 실측): 데모 150 clip 중 map_valid=True 37개 전체에 on/off 대조 실행(VLM 불필요, `detect_events`+`consolidate_episodes`만). **kind 분류가 바뀐 clip 1개(2.7%), 에러 0건.** 그 1건(`d2f9f633-...`)은 raw heading 49°(문턱 45° 바로 위) · lat 7.4m(차선변경 대역 밖) · map frac 0.9(고신뢰) 조건에서 turn_left가 완전히 소거됨 — 설계 의도(완만한 대곡선의 회전 오검출 방지)와 정확히 일치하는 방향. 경계값+map_valid 동시 조건 실측 케이스는 이 1건뿐이라 폭넓은 검증은 아니지만, 낮은 blast radius(37개 중 1개)와 방향 일치성을 근거로 배선 확정.

- **자체 회귀 발견 및 정정**: 이번 작업 중 `common/vocab073.py`가 모듈 최상단에서 `meta_tagging_*_v0.7.3.json` 3개를 무조건 로드하는데, `tag_v08.py`가 `from vocab073 import EGO_ACTIONS, OBJECT_TYPES, RELATIONS`로 **직접**(non-lazy) import한다는 걸 이전 v0.7.3 스키마 삭제 결정(2026-08-28 앞선 엔트리, 본 로그 미기재분) 당시 놓쳤다. 그때는 `classify073.tag_clip_v073()`(비활성 함수)만 확인하고 `tag_v08.py`의 별도 top-level 의존을 못 봤음 — Track1 authoritative 경로 import 자체가 깨져 있었다. `git checkout HEAD --`로 4개 파일 전부 복구(main+워크트리), import 정상화 재확인.

- 영향: `common/gt_placement.py`·`common/disclosure.py`·`task_episode/map_lane.py`·`common/events.py`·`task_episode/tag_v08.py`·`task_episode/candidates.py`·`task_episode/vlm_verify.py`·`task_episode/retrieve.py`. Stage1(`folder_selection.vlm_curve_check`/`verify_ambiguous_curves`)은 구현만 하고 랭킹 경로엔 미배선(VLM 비용 검토 후 별도 진행).

---
> **Task**: episode

# DESIGN_LOG — 2026-08-24 설계 지침서 v0.4 도입

기존 항목 아래에 이어 붙인다. 각 엔트리는 **무엇을 / 왜 / 무엇이 바뀌는가**로 구성한다.

---

## D-2026-08-24-01 · 설계 지침서 v0.4 및 어휘 v0.4 도입

**변경**: clean-slate 설계서(v0.1)와 구 계열(meta_vocab v0.1)을 병합한 v0.4를 목표 설계 정본으로 채택. 어휘는 `tag_vocab_v0.4.json` 단일 원천.

**근거**: 두 계보를 대조한 결과 각각이 상대의 공백을 메웠다. clean-slate 쪽이 원칙 수준 결함 4건을 잡았고(가시성≠존재, 단계 병렬 구조, Action-first 배제 범위 명시, 중간 산출물 영속화), 구 계열 쪽이 소비자 계층 6건을 보유했다(파생 뷰, 선별 접속, 규칙 산출성과 신뢰성 분리, 표지 계층화, 판정 범위 2단, gold 복수 정답).

**영향**: 코드는 아직 v0.4를 구현하지 않는다. CLAUDE.md를 3층 구조(현재 코드 실태 / 목표 설계 / 불변 규칙 + 전환 규칙)로 재작성하여 두 층의 혼동을 막는다.

---

## D-2026-08-24-02 · 앵커 정의 정정 — 경로 유형 라벨은 CAN 확정값이 아님

**변경**: "자차 거동은 전량 CAN anchored" → **"전이 시점과 종방향은 CAN 규칙 확정, 경로 유형(좌/우회전·차선변경·U턴) 라벨은 지도 ∪ VLM으로 해소"**.

**근거**: CAN 요레이트만으로는 교차로 회전 / 도로 곡률 / 분기 / U턴이 구분되지 않는다(under-determined). 이는 저장소에 이미 기록된 제약이었으나, 어휘 재설계 과정에서 축을 "순수하게" 만들려는 단순화 압력으로 누락되었고 clean-slate 재작성이 복원 기회를 없앴다.

**영향**:
- 경로 유형 라벨은 규칙 확정값이 아니므로 **평가 대상**이며 기계 채점 기준(AUTO_GT)으로 사용 금지
- 앵커 비오염 원칙의 적용 범위는 **전이 시점 확정**까지. 경로 라벨의 맥락 해소는 예외가 아니라 이후 단계
- 지침서 §2.3, 어휘 `ego_action.anchor_policy`, CLAUDE.md §3에 반영

**되돌린 결정**: `vocab_lint.py`의 `RETIRED`에 `"전량 CAN"` 등록

---

## D-2026-08-24-03 · 미래 정보 차단 원칙 폐기

**변경**: "`key_frame_t` 이후 관측을 인과 귀속 근거로 사용 금지" → **폐기**. 판정에는 시간창 전 구간을 제한 없이 사용한다.

**근거**: 본 산출물은 라벨이지 예측기가 아니다. 태깅 시점에 이미 전 구간이 확보되어 있으므로, 사후 정보를 막으면 인과 판정 정확도만 낮아지고 얻는 것이 없다. 원래의 우려(사후 정보 기반 인과 라벨로 예측 모델을 학습하면 추론 시점에 없는 정보를 전제하는 패턴을 배움)는 **소비 단계 주의사항**이지 파이프라인 제약이 아니다.

**영향**:
- 인과 라벨의 판정 근거 범위를 export 문서에 표기 — 예측 모델 학습 용도 시 시점 제한 재라벨 필요 가능
- 위반 금지 원칙 2번을 "판정 근거 시점 표기"로 교체
- 이에 딸려 설계했던 영상 블록 분할(인과용/전체용 프리픽스)도 철회

**되돌린 결정**: `RETIRED`에 `"미래 정보 차단"` 등록

---

## D-2026-08-24-04 · 목적 우선순위 원칙 신설 — 라벨 품질이 측정 편의에 우선

**변경**: 상위 원칙 신설(지침서 §1.3.1). 측정 목적 제약이 라벨 품질을 낮추면 **측정 쪽을 조정**한다(표본 한정 이중 실행, 조건 분리 보고). 전역 금지는 순환성·재현성처럼 어기면 결과가 무효가 되는 항목에 한정.

**근거**: 미래 정보 차단과 같은 계열의 제약이 문서 전반에 흩어져 있었다. 개별 수정만으로는 같은 유형이 다시 유입된다.

**영향 — 함께 완화된 항목 2건**
- **S2 입력의 최소 단서**: 금지 → **기본값**. 규칙 판정 주입이 라벨 정확도를 높이면 채택하고 전파율은 표본 한정 독립 실행으로 측정(실험 C)
- **프레임·뷰 순환 배제**: 기본 샘플링은 후보 비의존 유지, **후보 기반 추가 프레임은 플래그 기록 후 허용**. 뷰는 축소만 금지하고 확대는 허용(축소는 누락을 만들고 확대는 만들지 않음)

**유지된 항목**: 물리 게이트의 시간 순서(인과율 자체), 반응 없는 원인의 에피소드 미생성(Action-first 구조 결정)

---

## D-2026-08-24-05 · S1 물리 정보의 S2 주입 — 3-way 확정

**변경**: 주입 클래스를 `gt_only` / `gt_before` / `gt_free` **3종으로 확정**. 영상 뒤 주입(`gt_after`)은 기본값에서 제외.

**근거**: `gt_after`의 도입 근거였던 "모델이 규칙과 다른 값을 낼 여지를 남겨 모순 탐지"는 **주입 없이도 성립**한다 — VLM 독립 실행 후 규칙 판정과 사후 대조하면 동일 신호를 얻고 토큰 비용이 낮다. 규칙 맥락이 필수인 경우는 객체 참조 성립뿐이며 `gt_before`가 담당.

**영향**:
- 필드 단위 배정표 확정(지침서 §3.1 DV-8) — 판정 기준 3개(정보 없이 답할 수 있는가 / 복사만 하는가 / 편향시키는가)
- `gt_before` 작성 규칙: **관측 사실만**. "감속 중"은 관측, "위험"·"끼어드는 중"은 판정이므로 제외
- 금지가 아니라 기본값 선택 — 실험 C에서 라벨 정확도 이득 확인 시 재상정

**되돌린 결정**: `RETIRED`에 `"gt_before → gt_after"`, `"GT-after 블록"` 등록

---

## D-2026-08-24-06 · 전이 검출 4중 필터 구조 확정

**변경**: 앵커 검출에서 사건이 아닌 변화를 거르는 **규칙 구조**를 고정(임계값은 별도). 순서: 크기 → 지속 → 곡률 보정 → 병합.

**근거**: 도로 곡률 추종 조향·차로 내 흔들림·노면 요철·가다서다가 걸러지지 않으면 에피소드가 폭증하여 모집단 자체가 무의미해진다. 임계값을 하이퍼파라미터로 미루면서 **구조까지 함께 미룬 상태**였다.

**설계 결정 3**:
- 횡방향은 순간 요레이트가 아니라 **누적 방위 변화량** — 완만한 대곡선과 급한 소회전이 순간값으로 구분되지 않음
- 곡률 보정은 자차 방위 변화에서 **도로 기하가 설명하는 몫을 뺀 잔차**로 판정. 보정 출처(`map`/`estimated`/`none`)를 기록
- 병합은 마지막 — 앞 필터 통과분끼리만 병합해야 미미한 변동이 되살아나지 않음

**영향**: `transition_filters_passed`·`curvature_correction_source`·`merged_from` 기록 의무. 필터 통과 직전 탈락 사례를 별도 로그로 축적하여 임계 조정 근거로 사용

---

## D-2026-08-24-07 · 규칙 불확실성 5신호 정의 (참조 유실 복구)

**변경**: `rule_uncertainty` 5신호에 산출 방법 정의. v1 가동 범위는 **지도 유효성 + 기하 여유 2개**로 한정.

**근거**: 지침서 2곳이 참조하는데 어휘에서 `null`이었다. 구 계열(v0.8.1)에서 정의했던 것이 clean-slate 어휘 교체 시 유실되어, **참조는 살아남고 정의만 사라진** 상태였다.

**영향**: 5신호 동시 튜닝은 근거 없는 상수를 5개 늘린다. 나머지 3신호(검출 품질·소스 상충·시계열 흔들림)는 값만 기록해 분포를 축적한 뒤 임계를 도출

---

## D-2026-08-24-08 · 물리 게이트 6검사 판정 로직 정의

**변경**: 항목명뿐이던 B 게이트에 입력·판정 로직·위반 처리 명시.

**핵심 2**:
- **반응 지연은 단일 상수 금지** — 신호 대기와 끼어들기 대응의 지연 분포가 다르므로, 하나의 범위로 묶으면 특정 원인 유형이 체계적으로 탈락한다. 원인 유형별 gold 실측에서 도출
- **위반 시 라벨 삭제 금지** — 위반 항목을 기록해 오류 유형 분류·재라벨 우선순위에 사용. 삭제하면 임계가 잘못돼도 통계에서 사라져 알 수 없다

**영향**: 산출은 `오답 확정` / `미판정` 두 값뿐. **통과율을 정확도로 보고 금지**(위반 금지 원칙 5)

---

## D-2026-08-24-09 · 추론 실행 방식 — 3층 구성 (잠정)

**변경**: 필드 처리 경로를 3층으로 구성. GT 필드는 호출 없음 / 의미 필드는 **생성 1회 + 결과 JSON 재프리필** / 판별이 갈리는 소수 필드만 후보 시퀀스 채점 보정.

**근거**: 필드별 개별 질의(약 35회)와 전체 생성(디코드 300~400토큰) 사이의 균형점. 값이 채워진 JSON을 다시 프리필하면 모든 필드 값 자리가 존재하므로 **한 번의 프리필로 전 필드 분포**를 얻는다.

**주의**:
- 재프리필 확신도는 모델이 스스로 쓴 값에 조건화된 값이다. "정답일 확률"인지 "자기 답에 대한 확신"인지는 gold 대비 calibration으로만 구분
- **점수화 기본값은 시퀀스 방식**. 한 위치 분포만 읽는 방식(first-token)은 후보 첫 토큰이 같으면 판별 불가이며, 실측상 충돌 38개 값의 대부분이 **좌/우 쌍**이라 최상위 실패 모드에서 판별력을 잃는다
- **잠정 구성** — 실험 A(arm ①~⑤)로 확정

---

## D-2026-08-24-10 · 어휘 정합성 검사 도구 도입

**변경**: `vocab_lint.py` 추가. 검사 9종 — 폐기 표현 잔존 / 개념 중복 정의 / enum 명명 혼용 / auto_checks 참조 / 파생 뷰 참조 / first-token 충돌 / 지침서 폐기 표현 / **지침서→어휘 참조 무결성** / gt_only↔rule_derivable 교차.

**근거**: 이번 세션에서 부분 수정 후 잔존물이 남는 사고가 반복되었다(window 정의 3곳, gt_after 2곳, enum 명명 2건). 사람 검토로는 놓친다.

**도입 즉시 검출된 실제 불일치**:
- `window`와 `episode_structure.window_semantics`의 개념 중복 → 최상위 `window` 단일 정본으로 통합
- 지침서 `prev_transition`·`budget_truncated` ↔ 어휘 `prev_ego_transition`·`budget_truncation` → 어휘 쪽으로 통일
- 품질 등급이 `"rule_verified — 설명"` 형태의 결합 문자열 → 키-설명 분리(기계 참조 가능하게)

**운용 규칙**: 어휘·지침서 변경 커밋에서 실행. **결정이 뒤집힐 때마다 `RETIRED` 사전에 항목 추가** — 이 목록이 본 로그의 "되돌린 결정"과 짝을 이룬다

---

## 미결 (다음 결정 대상)

| 항목 | 결정 주체 | 선행 조건 |
|---|---|---|
| 어휘 파편화 정리 (현행 4벌 → 단일) | 설계 회의 | v0.4 도입의 전제 |
| 파생 뷰 원칙 위반 리팩터 범위 | 설계 회의 | 뷰 이름이 검출 로직에 박힌 범위 산정 |
| 임계값 전체 (전이 검출·반응 지연·불확실성·후보 범위) | 실측 후 선언 | gold·분포 실측 |
| 추론 경로 확정 | 실험 A | 서빙 재기동 |
| first-token 적용 가능 필드 | 실험 B | 토크나이저만으로 즉시 가능 |
| GT 주입 배치 재검토 | 실험 C | §2.2 상충 판정 완료(D-04로 해소) |
| 신뢰도의 경로 간 비교 가능성 | 검증 | 실험 A 결과 |


## [2026-08-03] Task「선별」실행파일 배포 패키징 (외부 기관 서버, 코드 비노출)
> **Task**: selection
- 변경: `deploy/` 신규 — ① `app/runtime.py` 외부 설정(JSON/env) → **모듈 네임스페이스 주입**(paths·config·client·selection·folder_selection)으로 원본 소스 수정 없이 데이터경로·VLM엔드포인트 교체 ② `app/cli.py` `rank`/`review`/`selftest`/`print-config` ③ `shield.py` 빌드 스테이징에서 프롬프트·카테고리 상수를 난독 blob으로 AST 치환 ④ `build.sh` 스테이징→shield→Nuitka onefile→검증(strings)→SHA256SUMS ⑤ `Dockerfile` 멀티스테이지(최종 이미지에 소스 부재) ⑥ `app/guard.py` 선택적 유효기간·호스트잠금.
- 이유: 타 기관 서버에 올려 구동하되 알고리즘(점수식·임계값·프롬프트·융합 로직) 노출 방지. 파이썬 배포는 소스/바이트코드가 그대로 노출되므로 네이티브 컴파일이 최소 요건.
- 부수 변경: `task_episode/classify073.py` 의 `vocab073` import를 **지연 로드**(`_v()`)로 전환. import 시점에 `schema/*.json` 3개를 읽던 것을 제거 — Stage1 선별 경로는 `consolidate_episodes` 만 쓰므로, 배포물에 **Stage2 taxonomy 자산이 아예 들어가지 않게** 됨(동작 불변).
- 보호 범위(정직): 소스·바이트코드 부재·docstring 제거·프롬프트 평문 미노출은 확보. 그러나 (a) **VLM 서버를 상대가 호스팅하면 요청 로그로 프롬프트 노출**(구조적) (b) 네이티브 리버싱은 여전히 가능 (c) 입출력 대량 관측으로 랭킹 함수 근사 모방 가능 (d) guard는 계약 이행 보조일 뿐 보안 경계 아님. → 컨테이너+유효기간 빌드+NDA 조합 권고, 프롬프트가 결정적 IP면 추론 엔드포인트를 우리가 운영.
- 산출물 노출 축소: `ranking.json` 에서 점수 기여항(`terms`=내부 가중치)은 기본 제외(`output.include_score_terms` 로만 노출).
- 검증: 소스모드/바이너리 모두 실데이터(1966 clip) selftest·rank·리뷰HTML 통과. VLM 서버 8001–8004 전부 down이라 **VLM 채널은 미검증**(구조적 경로만 확인).

## [2026-08-03] 타 기관 큐레이션 인터페이스 정의서 — 질의=세그먼트 / 반환=클립
> **Task**: multi
- 맥락: 두 작업이 별개다. **(A)** 우리 labeling 파이프라인을 타 기관 서버에 올려 **우리가** 실행(→ 실행파일 wrapping·IP 보호). **(B)** 확보된 metadata로 **타 기관이** 검색엔진 기반 큐레이션 알고리즘을 개발(→ 이 정의서). 상대는 라벨링도, 우리 실행파일도 다루지 않는다.
- 변경: `deploy/schema/` 를 작업별로 분리 — `selection/`(A: 실행파일 산출물 스키마, dist 동봉) · `curation/`(B: `INTERFACE_SPEC.md` + 세그먼트 교환용 사본 + `clip_index` 스키마) · `SHARE.md`(공유등급·검증기록).
- 설계 결정: **질의 단위=세그먼트(의미 축 보유) / 반환 단위=클립**. 세그먼트가 걸리면 소속 클립이 함께 결과에 포함되고, 세그먼트는 근거·구간 위치로 동반. 한 클립 다중 매치는 dedup 후 근거 배열. 이를 위해 `clip_index`(clip_id·duration_s·n_segments·video) 상위 엔티티를 신설 — 세그먼트 0건 클립도 유지.
- 범위: **Track1(세그먼트 메타태깅)만.** Track2 카테고리 후보(카테고리·confidence·provenance)는 recall-우선 후보라 임계값 책임이 상대로 넘어가므로 이번 정의서에서 제외, 카테고리 검색을 열 때 별도 버전.
- 교환용 사본 = 내부 정본에서 `x_generation`(앵커/생성순서)·`x_constrained_decoding`(guided_json 강제) 제거 + `x_record_unit`·`x_field_source` 추가. `properties`·`x_fill_rules`·`x_role_derivation` 은 해석에 필요하므로 유지.
- 비공유: `meta_tagging_opt1_model_output_v0.7.3.json`(무엇을 모델에 생성시키는지 = 설계 IP), `meta_tagging_vocab_v0.7.3.json`(라벨 공간 설계 — 상대가 라벨링을 안 하므로 불필요), 프롬프트, 가중치. 값 의미 사전·검증 절차·참조 구현도 제외(상대 개발 영역).
- 기준: **상호운용에 필요한가(→준다) vs 그 답에 도달한 방법인가(→안 준다)**.
- 검증(실데이터): `outputs_v073/tags/` 실제 산출물 10파일·세그먼트 14건으로 교환본 draft-07 검증 위반 0, `clip_index` 조립·조인 무결성(고아 0, n_segments 일치, 0건 클립 유지) OK, 세그먼트 질의→클립 롤업 동작 확인(cause=agent 5세그→3클립 등).
- **검증 중 발견·수정**: 시간 필드의 `maximum: 20` 제약이 실측 클립 길이(20.01~20.08s)를 탈락시킴 → **내부 정본(`common/schema/meta_tagging_seg_opt1_schema_v0.7.3.json`)과 교환본 양쪽에서 상한 제거**. 기존 태깅 산출물의 `warnings` 에 이미 찍히던 건.

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

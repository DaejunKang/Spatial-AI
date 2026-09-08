# Claude Code 에이전트 체제 설계

에이전트 정의 파일 7개는 `agents/` 폴더에 있다. 저장소의 `.claude/agents/`에 그대로 복사하면 된다.

---

## 1. 설계 원칙

**사람 체제의 분리 원칙을 에이전트에 그대로 적용한다.** 라벨을 만드는 에이전트가 채점하지 않고, 채점 에이전트가 파이프라인을 고치지 않는다. 한 에이전트가 두 역할을 겸하는 순간 순환 평가가 코드 안에서 조용히 일어난다.

| 축 | 에이전트 | 하는 일 | 하지 않는 일 |
|---|---|---|---|
| **생산** | rule-engineer · vlm-engineer | 파이프라인 S0~S3 구현 | 채점, 게이트 통과를 위한 임계 조정 |
| **검증** | verifier · dataset-curator | 준거·채점·평가셋 | 파이프라인 코드 수정 |
| **공용** | schema-keeper · experiment-runner · design-scribe | 정본 관리·실험 실행·기록 | 판정 |

**에이전트마다 "금지" 절이 있다.** 이것이 역할 경계의 실체다. description은 언제 부르는지, 금지는 무엇을 하면 안 되는지를 정한다.

---

## 2. 에이전트 7종

| 에이전트 | 한 줄 | 호출 시점 | 산출물 |
|---|---|---|---|
| **schema-keeper** | 어휘·지침서 단일 원천 | 필드 추가/변경, 버전 갱신, 린트 | 어휘 JSON, lint 통과, RETIRED |
| **rule-engineer** | 규칙 계층 S0+S1 | 에피소드 분할, 위치·거동·계측·후보·불확실성 | 에피소드 구간 + subjects[] + GT-before |
| **vlm-engineer** | S2·S3 VLM 호출 | 프롬프트 조립, 점수화, 생성, 매니페스트 | 라벨 + 중간 산출물 + 매니페스트 |
| **verifier** | 물리 게이트 + 채점 | 술어 설계·전수 검사·역검증, 실험 채점, 리포트 | 명세표 + 위반 분포 + 자동 리포트 |
| **dataset-curator** | 평가셋·gold | 스모크셋 변환, gold 도구, IAA | 스모크셋 + gold + IAA |
| **experiment-runner** | 실험 실행 | arm 구성, 브랜치, 원시 결과 | 카드 + 매니페스트 + 결과 파일 |
| **design-scribe** | 결정 기록 | 결정·번복·외부 반영 시 | DESIGN_LOG + conflict_register |

**통합 근거**: anchor+physics는 둘 다 규칙 계층이고 VLM 미호출이라 경계가 같다. gate+scorer는 둘 다 검증 측이고 파이프라인을 수정하지 않는다는 금지가 같다. 부하가 확인되면 다시 나눈다.

---

## 3. 인수인계 — 누가 무엇을 누구에게

```
schema-keeper ──── 어휘 로더 계약 ──────────▶ rule / vlm
rule-engineer ──── 구간 · 후보 · 계측 · GT-before ─▶ vlm-engineer
vlm-engineer ───── 라벨 · 중간 산출물 ─────▶ verifier (게이트 전수 검사)
                                            └──▶ experiment-runner (arm 실행)
experiment-runner ─ 원시 결과 + 매니페스트 ─▶ verifier (채점)
dataset-curator ── 스모크셋 · gold ─────────▶ verifier (채점 · 게이트 역검증)
verifier ───────── 리포트 ──────────────────▶ (사람 판정)
design-scribe ──── DESIGN_LOG · RETIRED ────▶ schema-keeper (린트 갱신)
```

**인수 형식은 파일이다.** 에이전트 간 전달은 구두 요약이 아니라 표준 경로의 파일로만 한다. 전달 파일이 없으면 다음 단계가 시작되지 않는다.

| 인수물 | 경로 (제안) | 생산자 → 소비자 |
|---|---|---|
| 에피소드 구간 + subjects + 계측 | `outputs/episodes/<clip>.json` | rule → vlm |
| 라벨 + 중간 산출물 | `outputs/labels/<clip>/{s2,s3,final}.json` | vlm → verifier, runner |
| 실험 카드 | `experiments/cards/<id>.md` | runner → verifier (사전 등록) |
| 원시 결과 | `experiments/results/<id>/` | runner → verifier |
| 매니페스트 | `experiments/results/<id>/manifest.json` | runner → verifier |
| 스모크셋 | `data/smoke/<version>/` | curator → 전원 |
| gold | `data/gold/<version>/` (접근 제한) | curator → verifier |
| 리포트 | `reports/<id>.md` (자동 생성) | verifier → 사람 |

---

## 4. 메인 세션의 위임 규칙 (CLAUDE.md 추가분)

메인 세션은 오케스트레이터다. 직접 구현하지 않고 아래 규칙으로 위임한다.

```
요청에 다음이 포함되면 → 해당 에이전트
  어휘·스키마·필드·값 집합·버전·린트                    → schema-keeper
  전이·앵커·에피소드 분할·3DOD·지도·위치·계측·후보        → rule-engineer
  프롬프트·VLM·점수화·생성·서빙·매니페스트                → vlm-engineer
  게이트·술어·역검증·채점·지표·전파율·calibration·리포트  → verifier
  스모크셋·gold·라벨링 도구·IAA·변환                     → dataset-curator
  실험·arm·브랜치·재현                                  → experiment-runner
  결정·기록·번복·외부 문서 반영                          → design-scribe

한 요청이 두 축(생산+검증)에 걸치면 → 분리해서 순서대로 위임. 한 에이전트에 몰지 않는다.
```

**위임하지 않고 메인이 직접 하는 것**: 우선순위 판단, 에이전트 간 충돌 중재, 사람에게 결정 요청.

---

## 5. 강제되는 규칙 — 어느 에이전트도 어기지 못함

1. **앵커 비오염** — key_frame_t·종방향 거동은 CAN 규칙. 모델 유래 값 진입 금지 (rule-engineer 외 수정 불가)
2. **자가 채점 금지** — 생산 에이전트의 산출물은 verifier만 채점. 생산 에이전트가 낸 "정확도"는 승격 근거가 아님
3. **사전 등록 없는 채점 거절** — verifier는 등록 기준·시각이 없는 요청을 거절
4. **무기록 fallback 금지** — 대체 경로 사용은 플래그. rule-engineer·vlm-engineer 공통
5. **1 브랜치 1 변경축** — experiment-runner가 강제. 다변경 카드는 반려
6. **필요조건 ≠ 정답** — verifier의 게이트 산출은 두 값뿐. 통과율을 정확도로 쓰지 않음
7. **8000 무접촉** — vlm-engineer·experiment-runner 공통

---

## 6. 운용 방식

**자가 점검은 무제한, 승격만 검증.** 생산 에이전트는 스모크셋으로 얼마든지 자기 점검을 한다 — 대기 없음. 단 잣대는 dataset-curator가 만든 공용 셋이어야 하고, 정본 병합은 verifier의 채점을 거친다.

**막히면 blocker 파일.** 에이전트가 진행 불가 상태를 만나면 `reports/BLOCKER_<date>_<agent>.md`를 남기고 멈춘다. 우회 구현으로 넘어가지 않는다.

**첫 실행 순서** (서빙 복구 전에도 가능한 것부터)
1. schema-keeper — 어휘 로더 계약, 린트 CI 연결
2. experiment-runner — 실험 B (토큰 충돌, 서빙 불요)
3. dataset-curator — 스모크셋 표본 선정, 변환 스크립트
4. verifier — 술어 명세표 3종 (문서 선행)
5. rule-engineer — 서빙 불요, 규칙 계층
6. vlm-engineer — 서빙 복구 후

---

## 7. 모델 배정 (권장)

| 성격 | 에이전트 | 이유 |
|---|---|---|
| 설계·추론 비중 높음 | vlm-engineer · verifier · design-scribe | 트레이드오프 판단, 반례 탐색, 충돌 감지 |
| 정형 작업 비중 높음 | schema-keeper · experiment-runner · dataset-curator | 규약 준수, 반복 실행 |
| 중간 | rule-engineer | 규칙 구현 + 함정 대응 |

정형 작업 에이전트는 가벼운 모델로도 충분하다. 비용보다 **역할 경계 준수**가 중요하므로, 모델 선택보다 금지 절이 지켜지는지를 먼저 확인한다.

---

## 8. pre-commit 훅 — 금지 절을 코드로 강제

### 무엇인가

git은 커밋을 기록하기 **직전에** `.git/hooks/pre-commit` 스크립트를 실행한다. 스크립트가 0이 아닌 값으로 끝나면 커밋이 **기록되지 않는다.** 즉 "커밋할 수 있는 것"의 조건을 코드로 정하는 장치다.

에이전트 정의의 금지 절은 프롬프트일 뿐이라 어겨도 아무 일이 없다. 훅은 그중 **구조적으로 검사 가능한 것**을 실제로 막는다. 의도는 검사하지 못하지만 — "게이트를 통과시키려 임계를 조정했다"는 판별 불가 — 그 결과 나타나는 형태(임계 상수가 파이프라인 파일에 리터럴로 등장)는 잡는다.

### 훅 6종 (`hooks/`)

| 훅 | 막는 것 | 근거 |
|---|---|---|
| `check_vocab_sync` | 어휘·지침서 변경 시 `vocab_lint` 오류 | 두 파일 불일치 = 병합 거부 |
| `check_no_literal_thresholds` | 임계 상수를 `thresholds` 모듈 밖에서 숫자로 대입 | LEAD_IN 4중복, 2.8 vs 3.5 값 충돌 실측 |
| `check_import_separation` | 검증 코드가 파이프라인 import / 파이프라인이 검증 import | 채점기의 태거 종속, 게이트 맞춤 경로 차단 |
| `check_vocab_duplication` | 어휘 enum 값이 schema 밖 .py에 6개 이상 리터럴 | 어휘 복제 → 정본 이탈 |
| `check_gold_isolation` | 검증 측 밖에서 gold 경로 참조 | gold 학습 사용 금지 (불변식) |
| `check_experiment_card` | 결과 커밋 시 카드 부재·사전 등록 시각 없음·변경 축 2개 이상 | 사전 등록, 1 브랜치 1 변경축 |

### 설치

```bash
pip install pre-commit
# .pre-commit-config.yaml 은 이미 저장소 루트에 있음(hooks/ 안이 아님) — 별도 복사 불필요
pre-commit install                                         # .git/hooks/pre-commit 생성
pre-commit run --all-files                                 # 기존 파일 전체 1회 점검
```

경로는 `hooks/hook_utils.py` 상단에 모여 있다. 저장소 배치가 다르면 그 파일만 고친다.
(2026-09-08 정정: 이 절이 원래 가리키던 `hooks/_common.py`·`cp hooks/.pre-commit-config.yaml .`
경로는 실제 배치와 달라 오류였음 — 위가 실제 배치 기준 수정본.)

### 한계 — 정직하게

- `git commit --no-verify`로 우회할 수 있다. 훅은 **실수를 막는 장치**이고 고의를 막는 장치가 아니다. 고의는 리뷰와 로그가 잡는다
- 어휘 복제 검사는 문자열 매칭 휴리스틱이라 오탐·미탐이 있다. 정당한 경우 `# vocab-literal-ok` 주석으로 명시하게 하여, 우회 사실 자체를 코드에 남기도록 했다
- 훅은 커밋 시점에만 작동한다. 커밋하지 않고 실행만 하는 실험 코드는 잡지 못한다 — 그래서 실험 결과 커밋 시 카드를 요구하는 훅을 둔 것이다

### 검증

위반 사례 5종으로 전부 차단 확인, 수정 후 전부 통과 확인. 오탐 1건(`_common.py` 자체의 gold 경로 문자열) 수정 완료.
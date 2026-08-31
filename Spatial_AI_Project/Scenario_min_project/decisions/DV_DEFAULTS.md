# 설계변수(DV) 현재 기본값 — 살아있는 레퍼런스

> **Task**: multi

지침서(`docs/design/pipeline_design_guide_v0.4.md`) §3 설계공간 + 어휘(`common/schema/tag_vocab_v0.4.json`)
`design_variables`/`field_execution_policy`/`sampling_policy`/`transition_detection`/`rule_uncertainty` +
`common/config.py` 실제 상수를 대조한 스냅샷. **"실행" 열이 핵심** — 같은 "기본값"이라도 실제 코드가
그 값으로 동작 중인지, 스키마에만 선언되고 코드는 없는지가 다르다.

이 파일은 결정 이력이 아니라 **현재 상태 조회용**이다. 값이 바뀌면 이 파일과 DESIGN_LOG.md를 같은
커밋에서 갱신한다(§9 동기화 규칙과 동일 원칙).

---

## A. S0 앵커 — 전이검출·곡률보정

| # | 변수(키워드) | 위치 | 현재 기본값 | 가능한 값 | 실행 |
|---|---|---|---|---|---|
| A1 | 전이검출 임계 | `config.py` `EVENT_TURN_HEADING`(45)·`EVENT_LC_HEADING_MAX`(22)·`EVENT_TURN_YAWRATE`(0.15) 등 | 위 상수값 | 실측 후 재선언(`threshold_set_id`) | 코드 구동 중 |
| A2 | 곡률보정 소스 | `transition_detection.filters[2].source` | `map_lane.default_curvature_fn`이 map_valid 자동 판별(2026-08-28 배선) | `map`(Stage2) / `vlm_confirmed`(Stage1) / `estimated`(예비) / `none` | 코드 구동 중 — `tag_v08.py`에 배선(아래 실측 참조) |
| A3 | 곡률보정 애매구간 마진 | `folder_selection.CURVE_AMBIGUOUS_MARGIN` | 10.0° | 임의 조정 | 코드 구동 중, 단 `rank_folder_windowed` 등 실제 랭킹 경로엔 미배선 |
| A4 | rule_uncertainty 가동 범위 | `rule_uncertainty.v1_scope` | `map_validity`+`geometric_margin` 2개만 | 5신호 전체 | map_validity만 코드 구동(`map_lane.map_valid`), 나머지 스키마만 |

## B. S1→S2 주입 (DV-8)

| # | 변수 | 위치 | 현재 기본값 | 가능한 값 | 실행 |
|---|---|---|---|---|---|
| **B1** | **GT 배치 미분류 필드 기본값** | `common/gt_placement.py` `DEFAULT_CLASS` | **`gt_before`**(2026-08-28 결정 — 아래 근거) | `gt_only`/`gt_free`로 되돌리기 가능 | **코드 구동 중** |
| B2 | 물리정보 가공 수준(DV-1) | `design_variables.DV-1` | **미정**(실험 대상, 강제 선언 안 함) | L0(원측정값) / L1(파생 판단) | 미구현 |
| B3 | S2 조건화 공개값 | `flags.s2_conditioning` (스키마) / `common/disclosure.py` (실측) | 스키마 기본값=`minimal`, **실측값=`rule_injected`**(불일치 — 아래 참조) | — | 코드 구동 중(disclosure 스탬프) |

**B1 근거**: 미분류 필드를 `gt_free`(미주입)로 두면 grounding 실패 위험을 조용히 방치한다.
`gt_before`(관측 사실 주입)는 최악의 경우도 "과잉 주입"일 뿐 정보 손실은 아니다 — 안전한 쪽으로 강등.

**B3 불일치 기록**: `flags.s2_conditioning`의 스키마 선언 기본값은 `minimal`(§1.3.1 "단계 독립 기본값")이지만,
`tag_v08.py`/`candidates.py`/`vlm_verify.py`는 실제로 GT 카테고리 힌트를 항상 주입 중이라 실측 사실은
`rule_injected`다. CLAUDE.md §3 "VLM 프롬프트의 GT 힌트" 전환 규칙("실험 없이 기존 경로 안 바꿈")에 따라
**코드 동작은 바꾸지 않고, 사실을 정직하게 기록**하는 쪽을 택함(`common/disclosure.py`).

## C. S2 입력 구성

| # | 변수 | 위치 | 현재 기본값 | 가능한 값 | 실행 |
|---|---|---|---|---|---|
| C1 | 시간창·프레임률(DV-2) | `design_variables.DV-2` | F1 고정 | F2(거동 적응) | 코드 구동 중(고정 오프셋) |
| C2 | 뷰 선택(DV-3) | `design_variables.DV-3` | V1 고정 뷰셋 | V2(GT 기반 적응) | 코드 구동 중(고정) |
| C3 | 판정 근거 시점 | `window.causal_judgment_scope` | `full_window` | `pre_anchor`(대조실험용) | 코드 구동 중, disclosure로 기록 |
| C4 | 후보 프레임 샘플링 | `sampling_policy.additional` | `false` | `true` | 코드 구동 중(추가 로직 자체 없음), disclosure로 기록 |
| C5 | 뷰 확대 | `sampling_policy.view` | `false` | `true` | 미구현, disclosure로 기록 |

## D. 실행 구조

| # | 변수 | 위치 | 현재 기본값 | 가능한 값 | 실행 |
|---|---|---|---|---|---|
| D1 | 중간산출물 외부화(DV-7) | `design_variables.DV-7` | P1 2-pass | P2 단일 pass | 파일별 혼재(historical) |
| D2 | 추론 실행층(3층구성) | `field_execution_policy.execution_layers` | GT무호출/재프리필/보정채점 | 실험 A arm ①~⑤ | 미구현 |
| D3 | 조건화 강도 | 지침서 §7.1 | strong | weak | 미구현(스위치 없음) |

## E. 점수화 (실측 완료)

| # | 변수 | 위치 | 현재 기본값 | 실행 |
|---|---|---|---|---|
| E1 | 점수화 기본방식 | `scoring_method.default` | `sequence` | 실측 완료(`scoring_probe.py`), 추론엔진 자체는 미구현 |
| E2 | 필드별 배정 | `scoring_method.field_assignment` | 19개 중 10=first_token/9=sequence | 실측 데이터 반영 완료 |

## F. 곡률보정 구현 배선 상태 (2026-08-28)

| # | 항목 | 위치 | 상태 |
|---|---|---|---|
| F1 | `events.detect_events(curvature_fn=...)` | `common/events.py` | 구현+테스트+**`tag_v08.py`에 기본 배선 완료**(map_valid clip만 자동 적용) |
| F2 | `map_lane.road_curvature_over()` / `default_curvature_fn()` | `task_episode/map_lane.py` | 구현 완료. 실측 데이터에 "경계값+map_valid 동시" 케이스가 없어 품질은 목업으로만 검증 |
| F3 | `folder_selection.verify_ambiguous_curves()` | `task_selection/folder_selection.py` | 구현 완료. 랭킹 경로엔 미배선(VLM 비용 검토 필요 — 별도 승인 후 배선) |

---

## 변경 방법

이 표의 "#" 코드나 위치 경로를 지정하면 그 항목을 변경한다.

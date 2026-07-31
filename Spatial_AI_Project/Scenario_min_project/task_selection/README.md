# task_selection — Task「선별」 (Stage 1)

전체 로그에서 **의미있는 clip을 recall 위주로 선별**. raw 로그엔 CAN+video만 있으므로 obj3d/map 미사용.

`ego arc 점수(egomotion) + VLM 흥미도(몽타주) → 결합 랭킹 → 상위 K`

> **(B) 점진 이관**: 파일은 repo 루트 flat 유지. 이 폴더는 소속·역할 매니페스트.

## 개별 알고리즘 (이 task 전용)
| 파일 | 역할 |
|---|---|
| `selection.py` | **선별 알고리즘 모듈** — `ego_score`(급감속=agent반응 대리신호 포함) · `vlm_interest`(몽타주 guided_json: interesting/score/category/reason) · `combined_score`(max+0.3·min) · `rank_clips`. (구 job tmp의 select_prep/run_select_vlm/build_select_review 통합) |
| `SELECTION_STAGE1.md` | 알고리즘 명세·한계·후속 레버 |

## 공용 기반 의존 (→ `common/`)
`events`(ego arc) · `dataset`(몽타주) · `config`(MODEL/TEMPERATURE) · `paths` · `tagger`(client pool)

## 산출물
`gold_label/select/` — `select_data.json`(ego 점수)·`select_vlm.json`(VLM 흥미도)·`selected30.json`(선별)·선별/데모 리뷰 HTML·데모 mp4.

## 핵심 원칙 (상위 CLAUDE.md §2)
- **두 채널 결합 필수**(실측: VLM단독으로 못 잡는 것 다수 — 상위30 중 VLM우세 15). ego=reactive, VLM=non-reactive 상보.
- VLM 흥미도의 **신뢰 신호 = notable/routine 거친 랭킹**뿐. 세부 category는 미검증 참고값.
- 선별은 **triage**(recall) — 확정 라벨은 Stage 2(task_episode).

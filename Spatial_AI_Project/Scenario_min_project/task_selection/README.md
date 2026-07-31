# task_selection — Task「선별」 (Stage 1)

전체 로그에서 **의미있는 clip을 recall 위주로 선별**. raw 로그엔 CAN+video만 있으므로 obj3d/map 미사용.

`ego arc 점수(egomotion) + VLM 흥미도(몽타주) → 결합 랭킹 → 상위 K`

> **(B) 점진 이관**: 파일은 repo 루트 flat 유지. 이 폴더는 소속·역할 매니페스트.

## 개별 알고리즘 (이 task 전용)
| 파일 | 역할 |
|---|---|
| `selection.py` | **선별 알고리즘 모듈** — `ego_score`(급감속=agent반응 대리신호 포함) · `vlm_interest`(몽타주 guided_json: interesting/score/category/reason) · `combined_score`(max+0.3·min) · `rank_clips`. (구 job tmp의 select_prep/run_select_vlm/build_select_review 통합) |
| `run_selection.py` | Stage1 정식 진입점. **기본=윈도우+video 반응성(정본)** — 전체 clip에서 gold 제외 후 무작위 N 랭킹 → 상위 K + `<video>` 리뷰 HTML. `review`=재랭킹 없이 HTML, `montage`=legacy 흥미도 |
| `folder_selection.py` | **폴더별(조건별) 선별 + windowed 반응성 엔진**(공유). `rank_folder_windowed`(ego 전이 윈도우별 반응성 × VLM video 확인, 시간정렬) = Stage1·폴더 공용 엔진. `verify`=폴더무시 검증. clip-level montage는 legacy |
| `review.py` | 공용 리뷰 유틸 — 표본 샘플링·H.264 트랜스코드·`<video>` 리뷰 HTML |
| `SELECTION_STAGE1.md` | 알고리즘 명세·한계·후속 레버 |

## 공용 기반 의존 (→ `common/`)
`events`(ego arc) · `dataset`(몽타주) · `config`(MODEL/TEMPERATURE) · `paths` · `client`(pool)

## 산출물
`gold_label/select/` — `select{N}.json`(랭킹)·`selected{K}.json`(선별)·`folder_selection.json`(폴더별)·`index.html`(리뷰)·`vids/`(트랜스코드, git 제외).

## 두 선별 방식
- **전역 선별**(`run_selection`): 정적환경 미분류 전체 로그 → ego+VLM 흥미도로 롱테일 triage.
- **폴더별 선별**(`folder_selection`): 조건(주간/야간·도심/골목)이 이미 폴더로 분리된 경우 → 폴더 안에서 **ego motion에 영향 준 event(반응성)** 우선. 조건 고정이라 VLM은 맥락 판정 대신 **반응 여부 확인**.

## 핵심 원칙 (상위 CLAUDE.md §2)
- **두 채널 결합 필수**(실측: VLM단독으로 못 잡는 것 다수 — 상위30 중 VLM우세 15). ego=reactive, VLM=non-reactive 상보.
- VLM 흥미도의 **신뢰 신호 = notable/routine 거친 랭킹**뿐. 세부 category는 미검증 참고값.
- 선별은 **triage**(recall) — 확정 라벨은 Stage 2(task_episode).

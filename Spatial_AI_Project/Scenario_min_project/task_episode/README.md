# task_episode — Task「episode」 (Stage 2)

선별된 clip에서 **episode 추출·메타데이터 labeling**. 여기서만 **3DOD + map** 사용.

`egomotion ego 전이로 에피소드 분할 → 에피소드별 메타데이터`

두 제품이 동일 세그먼트 백본 위에서:
- **Track1** — SD/SA/MA + cause + key_frame (자율주행 개발용 라벨링)
- **Track2** — taxonomy 후보 검색(큐레이션 retriever)

> **(B) 점진 이관**: 파일은 repo 루트 flat 유지. 이 폴더는 소속·역할 매니페스트.

## 개별 알고리즘 (이 task 전용)
| 파일 | 역할 |
|---|---|
| `classify073.py` | 에피소드 consolidation(`consolidate_episodes`) = 세그먼트 백본. + v0.7.3 2-pass |
| `taxo_detect.py` | obj3d 3DOD 상호작용 검출(`detect_taxonomy`) |
| `map_lane.py` | map 차선(경계선 재해석)·Branch A/B·`map_valid` 게이트 |
| `vlm_verify.py` | VLM 검증/분류·`fuse`(∩)·`ground_signals` |
| `candidates.py` | **Track2 Phase A** — OR 앙상블 후보생성(ego/gt/VLM 5-vote) |
| `retrieve.py` | **Track2 Phase B** — 정규화·병합·confidence 랭킹 |
| `tag_v08.py` | **Track1** — SD/critical_components 메타데이터(v0.8) |
| `event_tagger.py`·`window_tagger.py` | legacy 태깅 경로 |
| `norm_embed.py` | 임베딩 정규화(자유기술→어휘) |

## 공용 기반 의존 (→ `common/`)
`events`(전이) · `paths`(obj3d/map) · `dataset`(subclip) · `taxonomy` · `config` · `tagger`

## 산출물
`gold.json`(episode gold) · `gold_label/`(라벨링 도구·리뷰) · `outputs*/`(태그)

## 핵심 원칙 (상위 CLAUDE.md §2)
- **세그먼트 = ego 전이 = critical**. non-reactive 제외. `key_frame_t`=rule.
- **ego_action 두 층위**: 종방향=rule 확정 / 경로=CAN트리거+맥락(map∪VLM). **앵커+순방향 SD→SA→MA 생성**.
- **cause** = 전이의 "왜", GT+맥락, **Track2 검색 인덱스에 필요**(현재 미포함=갭).
- **Track2 병합**: 교차로 sig/unsig 유지(cause 결부) / 도로유형만 병합.
- ⚠️ 드리프트(수정 대상): `tag_v08` MA-first schema, `AUTO_GT`의 turn류, `retrieve.MERGE`의 sig/unsig.

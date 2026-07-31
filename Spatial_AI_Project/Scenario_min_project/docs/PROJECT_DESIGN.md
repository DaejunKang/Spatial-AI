# VLA Long-tail 메타데이터·추출 파이프라인 — 설계본 (단일 정본)

> KATECH VLA. 자율주행 20초 클립에 메타데이터를 **tag**하고 long-tail 상황을 **추출**.
> VLM(NVIDIA Cosmos-Reason)은 **frozen** — 재학습 없이 GT(egomotion/obj3d/map)+규칙+VLM 조합을 개선.
> **이 문서가 유일 설계본**(구 `Concept_Design_v3` 흡수·폐기). 코드 실태·불변식은 `../CLAUDE.md`, 이력은 `../decisions/DESIGN_LOG.md`, 팀 공유는 `TEAM_REPORT.md`. 최신 갱신 2026-07-31.

---

## 1. 개요 — 무엇을 만드나

metadata labeling = **① clip 선별 → ② 선별 clip에서 episode 추출·labeling** 의 2단계 퍼널.
싼 신호(CAN+video)로 **넓게 선별**하고, 비싼 상세 태깅(3DOD+map)은 **선별 clip에만** 적용(비용·recall 퍼널).

**두 제품(공용 기반 공유):**
- **Track1** = v08형 **세그먼트 메타데이터**(SD/SA/MA + cause + key_frame) — 자율주행 개발용 라벨링. 계보 `tag_v08.py`.
- **Track2** = v3 **추출 retriever**(taxonomy 카테고리 후보 검색) — 큐레이션. `candidates.py`→`retrieve.py`.
- 공용 기반: `events`(egomotion) · `taxo_detect`(obj3d) · `map_lane`(map) · `vlm_verify`(VLM) · `classify073.consolidate_episodes`(세그먼테이션) · `taxonomy`(단일 어휘).

---

## 2. 데이터 & 제약

- **데이터**: visionary-nvidia(`paths.py`). egomotion(CAN, GT) · obj3d(3DOD; 위치신뢰, **vx/vy 16%만 유효**, occlusion NaN, 일부 clip track_id 누락) · map(pseudo; `centerlines`=실제 경계선, **`is_intersection` 죽음**, 유효율 ~35%).
- **제약**: raw 대규모 로그엔 **CAN+video만** → **Stage1 선별은 obj3d 없이**(CAN+VLM), Stage2 라벨링만 3DOD+map.
- **VLM**: `guided_json` enum 강제. 결정론 `TEMPERATURE=0`, n-vote만 temp>0. 입력 `video_url`(subclip) 또는 `image_url`(몽타주). 복제본 8001–8004 라운드로빈.

---

## 3. 아키텍처

```
전체 로그(CAN+video)
  │  STAGE 1 「선별」(task_selection) — obj3d 없이
  │    정본: ego 전이 윈도우별 반응성 × VLM(video, 같은 윈도우) 시간정렬 → 상위 K
  │    (legacy: 클립레벨 몽타주 흥미도)
  ▼
선별 clip → STAGE 2 「episode」(task_episode) — 3DOD+map
  │    consolidate_episodes(ego 전이) 분할
  │    ├ Track1  tag_v08 : SD→critical→cause→MA(앵커 해소)  세그먼트 메타데이터
  │    └ Track2  Phase A 후보(OR앙상블) → B 정규화·병합·랭킹 → C precision → D gold
  ▼
에피소드별 메타데이터 / 검색·추출 후보(provenance·confidence)
```

- **세그먼테이션 = ego 전이 = critical**: `1 record = 1 transition = 1 behavior event`. ego 무영향(non-reactive) 상황은 세그먼트 미생성. `key_frame_t`=rule(전이 onset).

---

## 4. 관점 전환 — 태거 = retriever (Track2 핵심)

태거를 **분류기가 아니라 retriever(관대한 후보 생성기)** 로. 큐레이션에선 **recall이 자산**, precision은 하류 human 확인으로 보완. **recall miss > FP**(FP는 걸러내면 되지만 miss는 영원히 못 씀).

태거 오류 3종 → 처리(재학습 아님):
| 오류 | 예 | 처리 |
|---|---|---|
| 슬롯 오류 | traffic_light를 object_type에 | 검색계층 **정규화·별칭 흡수**(Phase B) |
| 입도 오류 | backstreet↔urban 구분 불가 | **카테고리 병합**(Phase B, 단 sig/unsig은 cause 결부라 병합 금지) |
| 누락(recall miss) | 후보에 없음 | **OR 앙상블 확대**(Phase A, 진짜 병목) |

---

## 5. 택소노미 (`common/taxonomy.py`)

**5축, KEYS 60**: ego기동 · 상호작용 · 맥락 · **정적환경** · 규칙.
- `legacy/new_tag.json`(v0.4 폐쇄어휘) 흡수(2026-07-31): 정적환경(조명/기상/노면/glare/crosswalk/신호등/비분리) + long-tail 이벤트(긴급차·동물·장애물·무단횡단·역주행·취약보행자).
- `STATUS`: 태그별 visionary 실행상태(runnable/vlm_only/sparse/gold). egomotion 기반(hard_brake 등)은 기존 tag로 병합. obj3d 기반 신규는 어휘만(검출 deferred).
- `AUTO_GT={stop}`(순수 kinematic만·평가 제외). turn/u_turn/lane_change는 **CAN 트리거+맥락(map∪VLM) 해소** → HUMAN_KEYS(평가 대상).

---

## 6. 신호 채널 & Phase 상세 (상태 표기)

**신호 채널**: `events`(egomotion 거동, net-heading turn) · `taxo_detect.detect_taxonomy`(obj3d 상호작용·cut_in corridor) · `map_lane`(차선/분기, `map_valid` 게이트) · `vlm_verify`(검증·정적환경 판정·`fuse` ∩).

**Track2 Phase (실행 단위):**
| Phase | 내용 | 상태 |
|---|---|---|
| **A** 후보생성 | 3채널 OR 합집합(ego/obj3d/VLM 5-vote temp>0), provenance·vote_fraction·multi. ∩·게이트 금지 | ✅ `candidates.py` (+cause 축) |
| **B** 정규화·병합·랭킹 | 별칭·슬롯 흡수, 도로유형 병합(sig/unsig 유지), confidence 랭킹, `sub` 보존 | ✅ `retrieve.py` |
| **C** precision 게이트 | GT신뢰→`gt∩vlm` / GT부재 맥락→다수결 / 부분→Tier-B human. 학습셋 등급(train_ready\|needs_review\|low_conf) | ⛔ 미구현 |
| **D** gold & 측정 | 완전라벨 gold ≥60(present/absent, sparse 금지). recall=1차 지표, precision은 C 승격 후 | ⛔ 미구현(현 sparse gold 50) |
| **E** 어휘 통일 | taxonomy ↔ v0.7.3 vocab 매핑 | 🟡 부분(vocab073 공유) |
| **F** map 시간축 | lane-index 카테고리에 다프레임 경계선 누적 | ⛔ 선택 |

**Stage1 선별(task_selection)**: ✅ windowed+video 반응성 정본화(`run_selection.py`, 엔진 `folder_selection.rank_folder_windowed`). 폴더별 선별(`folder_selection`)·리뷰 유틸(`review.py`) 포함.

**Track1(tag_v08)**: 🟡 스키마 순방향(MA-last) 재정렬 완료. ego_action rule/arc 앵커(`_ground_ego_action`) — map∪VLM 완전 해소는 후속.

---

## 7. 가드레일 / 범위 밖 (DO NOT VIOLATE)

- 태거 **fine-tuning 금지**(순환). 태거 출력이 태거 학습에 되먹임 금지. gold=사람 라벨, `AUTO_GT`는 평가 제외.
- **cosmos2(port 8000) 무접촉**(타 그룹). 우리 복제본 8001–8004, **8001=평가-frozen**. NIM 컨테이너 파일 수정 시 blake3로 컨테이너 사망 → **운영 서버(재기동 승인 필요)**.
- KV-cache·TTA·합성데이터·다운스트림 VLA 학습·비용최적화 = 범위 밖(별도 트랙).
- 운영: `python3`, venv(uv), `docker container logs`(전체형), 신규 데이터는 `paths.py`.

---

## 8. 핵심 발견 (실측, 설계 근거)

1. 선별엔 **ego+VLM 상보 필수**(ego=reactive / VLM=non-reactive·맥락).
2. **cut_in 극희귀**(gold 0/50, obj3d ~2.3%/300, corridor 26/1966) — 희소 자체가 결과.
3. **몽타주 정지투영 = 주변부 이벤트를 ego에 오귀속**(cut_in corroboration 몽타주 6%·top50 0%) → **윈도우+video 시간정렬로 과검 제거**(obj3d 대조 확인).
4. **map 유효 ~35%**(경계선 재해석) → corridor fallback.
5. **VLM은 맥락 볼륨 강·미세분류 약** → Phase A(다중투표)+B(병합).
6. **gold sparse → precision 왜곡** → Phase D(완전라벨) 필요.

---

## 9. 수용 기준 (구 Concept_Design 흡수)

- **A**: OR 앙상블 후보, ∩·precision 게이트 미적용(recall 우선), provenance/confidence 부착.
- **B**: 정규화·병합으로 슬롯/입도 오류 흡수, confidence-ranked 결과, 원본 `sub` 보존.
- **C**: precision 게이트를 검색 이후로 분리(∩·Tier-B human), 학습셋 등급 부착.
- **D**: 완전라벨 gold + recall 1차 리포트, precision은 승격 후. FP는 `{실제FP|gold누락|검출기오류}` 분류.
- 공통: 태거 미학습, cosmos2/8000 무접촉, python3·docker container logs.

---

## 10. 현재 상태 & 업무 분담 (팀 분담용)

**완료(2026-07-31 기준)**: 저장소 재정리(common/task_selection/task_episode/legacy/docs/decisions) · 택소노미 5축60키(new_tag 흡수·정적환경) · 드리프트 4건 수정(tag_v08 순방향·AUTO_GT·sig/unsig·cause 축) · 정적환경 VLM 판정 · **Stage1 windowed+video 승격** · folder_selection · obj3d cut_in 검증 도구 · 신호 검출/검증 모듈 · sparse gold 50 · 데모.

**작업 패키지(분담 후보):**
| # | 패키지 | 소속 | 모듈 | 선행 |
|---|---|---|---|---|
| WP1 | **Stage1 정량검증** — 300에 critical y/n gold → windowed recall@K, montage 대비 | selection | `run_selection`·`review` | gold 라벨 |
| WP2 | **완전라벨 gold ≥60**(present/absent, sparse 금지) = Phase D | episode | `gold_tool.py`·gold.json | — |
| WP3 | **Phase C precision 게이트**(gt∩vlm·다수결·Tier-B·등급) | episode(Track2) | `vlm_verify.fuse`·신규 | Phase A/B |
| WP4 | **obj3d 검출기 배선**(GT 업데이트 후: cut_in/신규 obj3d 태그) | episode | `taxo_detect`·`map_lane` | obj3d GT 업데이트 |
| WP5 | **정적환경 VLM 스케일 판정 + gold** | episode(Track2) | `vlm_verify.env_cats`·`candidates` | — |
| WP6 | **Track1 tag_v08 완성**(ego_action map∪VLM 해소, 순방향 생성 검증) | episode(Track1) | `tag_v08` | — |
| WP7 | **실제 조건 폴더 적용**(folder_selection groups_from_root) | selection | `folder_selection` | 조건 폴더 데이터 |
| WP8 | **map 유효율 개선**(Phase F, 다프레임 누적) | episode | `map_lane` | 선택 |

- 각 WP는 **Track/Task 경계로 독립**(공용 기반은 `common/`에서 공유) → 병렬 분담 가능. 변경은 `decisions/DESIGN_LOG.md`에 Task 라벨로 기록.

---

## 11. 코드/산출물 맵

- **common**: `config paths dataset events taxonomy vocab073 client overlay` + `schema/`
- **task_selection**: `selection run_selection folder_selection review` + `SELECTION_STAGE1.md`
- **task_episode**: base(`classify073 taxo_detect map_lane vlm_verify`) / Track1(`tag_v08`) / Track2(`candidates retrieve`) / 검증(`verify_cutin_obj3d`) / `gold_tool`
- **산출물**: `gold.json` · `gold_label/`(라벨링·리뷰·선별) · `outputs*/`
- **문서**: 본 문서(정본) · `TEAM_REPORT.md` · `SELECTION_STAGE1.md` · `taxonomy_merge_report.md` · `../CLAUDE.md` · `../decisions/DESIGN_LOG.md`

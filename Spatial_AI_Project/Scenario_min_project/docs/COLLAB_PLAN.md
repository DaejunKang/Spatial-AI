# VLA Long-tail 파이프라인 — 설계 · 협업 구도

> KATECH VLA. 자율주행 20초 클립에 scene/analysis/meta-action 메타데이터를 **tag**하고 long-tail 상황을 **추출**. VLM(NVIDIA Cosmos-Reason)은 **frozen** — 재학습 없이 GT(egomotion·obj3d·map)+규칙+VLM 조합을 개선.
> 단일 설계 정본 · 2단계 퍼널 · 두 제품(Track1·Track2) · 갱신 2026-07-31
> (claude.ai 공유 Artifact와 동일 내용의 텍스트본. 설계 상세는 `PROJECT_DESIGN.md`.)

---

## 1. 개요 — 무엇을 만드나

metadata labeling = **① clip 선별 → ② 선별 clip에서 episode 추출·labeling** 의 2단계 퍼널. 싼 신호(CAN+video)로 **넓게 선별**하고, 비싼 상세 태깅(3DOD+map)은 **선별 clip에만** 적용(비용·recall 퍼널).

- **Track 1 — 세그먼트 메타데이터**: v08형 SD/SA/MA + cause + key_frame. 자율주행 개발용 라벨링. 계보 `tag_v08.py`. 생성은 관찰 선행 순방향(SD→critical→cause→MA), MA는 rule/arc 앵커로 해소.
- **Track 2 — 추출 retriever**: taxonomy 카테고리 후보 검색. 큐레이션용. `candidates.py`→`retrieve.py`. 태거=관대한 후보 생성기(recall 우선), precision은 하류에서 회복.
- 공용 기반: `events`(egomotion) · `taxo_detect`(obj3d) · `map_lane`(map) · `vlm_verify`(VLM) · `classify073.consolidate_episodes`(세그먼테이션) · `taxonomy`(단일 어휘).

## 2. 아키텍처

```
전체 로그(CAN+video)
  │  STAGE 1 「선별」(task_selection) — obj3d 없이
  │    정본: ego 전이 윈도우별 반응성 × VLM(video, 같은 윈도우) 시간정렬 → 상위 K
  │    (legacy: 클립레벨 몽타주 흥미도)
  ▼
선별 clip → STAGE 2 「episode」(task_episode) — 3DOD+map
  │    consolidate_episodes(ego 전이) 분할. 세그먼트 = ego 전이 = critical (1 record = 1 transition)
  │    ├ Track1  tag_v08 : SD→critical→cause→MA(앵커)  세그먼트 메타데이터
  │    └ Track2  Phase A(OR앙상블) → B(정규화·병합·랭킹) → C(precision) → D(gold)
  ▼
에피소드별 메타데이터 / 검색·추출 후보(provenance·confidence)
```

## 3. 현재 상태

**완료(2026-07-31)**: 저장소 재정리 · 택소노미 5축60키(new_tag 흡수·정적환경) · 드리프트 4건 수정 · 정적환경 VLM 판정 · **Stage1 windowed+video 승격** · folder_selection(조건별) · obj3d cut_in 검증 도구 · sparse gold 50 · 데모.

| Track2 Phase | 내용 | 상태 |
|---|---|---|
| A 후보생성 | 3채널 OR 합집합(ego/obj3d/VLM 5-vote)·provenance·cause 축. ∩·게이트 금지 | ✅ DONE |
| B 정규화·병합·랭킹 | 별칭·슬롯 흡수·도로유형 병합(sig/unsig 유지)·confidence 랭킹 | ✅ DONE |
| C precision 게이트 | gt∩vlm·다수결·Tier-B human·학습셋 등급 | ⛔ TODO |
| D gold & 측정 | 완전라벨 gold ≥60(present/absent)·recall 1차 지표 | ⛔ TODO |
| E 어휘 통일 | taxonomy ↔ v0.7.3 vocab 매핑 | 🟡 WIP |
| Track1 tag_v08 | 순방향 스키마 완료 · ego_action map∪VLM 완전 해소 후속 | 🟡 WIP |

## 4. 협업 구도 — 업무 분담

편성 = **리드 1(나) + 엔지니어 2 + gold 라벨러 1–2(on-demand)**. 엔지니어는 **Track 경계로 1인 1트랙**(코드 충돌 없음), 리드는 공용 기반·통합·GT 조율, 라벨러는 gold 공급.

### 팀 편성
| 역할 | 담당 | 모듈 | WP |
|---|---|---|---|
| **리드 (나)** | 공용 기반·통합·GT 조율·가드레일 | `common/` | WP4 · WP8 · 통합/리뷰 |
| **엔지니어 A** | Track2 추출·큐레이션(precision) | `candidates·retrieve·vlm_verify` | WP3 · WP5 (+WP2 gold 설계·검증) |
| **엔지니어 B** | Track1 라벨링 + 선별 | `tag_v08·run_selection·folder_selection` | WP6 · WP1 · WP7 |
| **gold 라벨러 1–2** | gold 공급(on-demand, 전체 병목) | `gold_label` 도구 | WP2 → WP1 |

### 작업 패키지 (WP)
| # | 패키지 | 담당 | 모듈 | 선행 |
|---|---|---|---|---|
| WP1 | Stage1 정량검증 — 300 critical y/n gold → windowed recall@K, montage 대비 | 엔지니어 B | `run_selection·review` | gold 라벨 |
| WP2 | 완전라벨 gold ≥60 (present/absent, sparse 금지) = Phase D 기반 | gold 라벨러 | `gold_tool·gold.json` | 없음(즉시) |
| WP3 | Phase C precision 게이트 (gt∩vlm·다수결·Tier-B·등급) | 엔지니어 A | `vlm_verify.fuse`+신규 | Phase A/B(완료)·WP2 병행 |
| WP4 | obj3d 검출기 배선 (GT 업데이트 후 cut_in·신규 obj3d 태그) | 리드 | `taxo_detect·map_lane` | obj3d GT 업데이트 |
| WP5 | 정적환경 VLM 스케일 판정 + gold | 엔지니어 A | `vlm_verify.env_cats·candidates` | 없음(즉시) |
| WP6 | Track1 tag_v08 완성 (ego_action map∪VLM 해소·순방향 검증) | 엔지니어 B | `tag_v08` | 없음(즉시) |
| WP7 | 실제 조건 폴더 적용 (folder_selection groups_from_root) | 엔지니어 B | `folder_selection` | 조건 폴더 데이터 |
| WP8 | map 유효율 개선 (Phase F, 다프레임 누적·선택) | 리드 | `map_lane` | 없음(선택) |

### 지금 시작 순서 (선행 무관 = 즉시 착수)
| 담당 | 지금 바로 | 다음(선행 해소 후) |
|---|---|---|
| gold 라벨러 | **WP2** 완전 gold ≥60 — **최우선** | WP1 critical y/n 300 |
| 엔지니어 A | **WP5** 정적환경 VLM (gold 쌓이는 동안) | WP3 Phase C (gold 후) |
| 엔지니어 B | **WP6** tag_v08 완성 | WP1 정량검증(gold 후) · WP7(폴더 후) |
| 리드(나) | common 리뷰·통합·**obj3d GT 일정 조율** | WP4(GT 후) · WP8 선택 |

### 이 편성의 핵심 포인트
- **gold가 최대 병목** — WP1·WP2·WP3가 모두 사람 gold 의존. 라벨러 **조기 투입**이 전체 속도 결정 → 라벨러 1–2명을 **WP2부터 즉시**.
- **2인이라 GT 인프라(WP4·WP8)는 리드가 흡수** — 엔지니어 A/B는 각자 Track에만 집중(컨텍스트 스위칭 최소).
- **병렬성**: A=Track2 / B=Track1+선별 → 코드 경로 분리(candidates·retrieve vs tag_v08, 선별은 task_selection). 충돌 지점은 `common/`뿐.
- **공용 기반 변경 규약**: `common/` 수정은 3인 전체 영향 → `decisions/DESIGN_LOG.md`에 Task 라벨로 기록·공지(리드 승인).
- **외부 의존**: obj3d GT 업데이트(타 그룹)가 WP4·WP3(gt∩vlm) 선행 → 일정 확인.

## 5. 가드레일 — 절대 위반 금지
- 태거 **fine-tuning 금지**(순환). 태거 출력이 태거 학습에 되먹임 금지. gold=사람 라벨, `AUTO_GT`는 평가 제외.
- **cosmos2(port 8000) 무접촉**(타 그룹). 복제본 8001–8004, **8001=평가-frozen**. NIM 컨테이너 파일 수정 시 blake3로 컨테이너 사망 → **운영 서버(재기동 승인 필요)**.
- KV-cache·TTA·합성데이터·다운스트림 VLA 학습·비용최적화 = **범위 밖**(별도 트랙).

## 6. 코드/문서 맵
- **common**: `config paths dataset events taxonomy vocab073 client overlay` + `schema/`
- **task_selection**: `selection run_selection folder_selection review` + `SELECTION_STAGE1.md`
- **task_episode**: `classify073 taxo_detect map_lane vlm_verify tag_v08 candidates retrieve verify_cutin_obj3d gold_tool`
- **문서**: 설계 정본 `docs/PROJECT_DESIGN.md` · 본 협업본 `docs/COLLAB_PLAN.md` · 코드실태·불변식 `CLAUDE.md` · 이력 `decisions/DESIGN_LOG.md` · 팀 리포트 `docs/TEAM_REPORT.md`

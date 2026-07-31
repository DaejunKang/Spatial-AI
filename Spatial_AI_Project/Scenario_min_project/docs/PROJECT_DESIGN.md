# VLA Long-tail 추출 파이프라인 — 개념 설계 (v3, 단일 설계본)

> KATECH. 자율주행 로그에서 long-tail 상황을 자동 tag·추출.
> **이 문서가 최신·유일 설계본**입니다. (이전 ∩ precision-우선 버전은 폐기 — v3는 recall-우선 retriever 아키텍처로 대체.)
> 실행 명세는 `Concept_Design_v3`, 팀 공유는 `TEAM_REPORT.md`. as of 2026-07-23.

---

## 0. 관점 전환 (핵심)
태거를 **분류기(정확 라벨 생성기)가 아니라 retriever(관대한 후보 생성기)** 로 사용.
- 학습용 큐레이션에선 **recall이 자산**, precision은 하류 human 확인으로 보완.
- **recall miss(후보에 아예 없음) > FP** — FP는 걸러내면 되지만 miss는 영원히 못 씀.
- 태거는 **frozen**(재학습 금지·순환). GT+규칙+VLM 조합을 개선.

태거 오류 3종 → 처리(재학습 아님):
| 오류 | 예 | 처리 |
|---|---|---|
| 슬롯 오류 | traffic_light를 object_type에 | 검색계층 **정규화·별칭 흡수** (Phase B) |
| 입도 오류 | backstreet↔urban 구분 불가 | **카테고리 병합** (Phase B) |
| 누락(recall miss) | 후보에 없음 | **OR 앙상블 확대** (Phase A, 진짜 병목) |

---

## 1. 목적 & 데이터 제약
- **목적**: lane-keeping이 다수인 로그에서 의미있는/희귀 상황(회전·정지·보행자·끼어들기·비보호좌회전·정체·악천후 등)을 다중라벨로 **검색·추출** → 다운스트림 학습데이터.
- **데이터**: visionary-nvidia. egomotion(CAN, GT), obj3d(3DOD, 위치신뢰/vx·vy 불신/occlusion NaN), map(pseudo, `centerlines`=실제 경계선·`is_intersection` 사용불가).
- **제약**: raw 대규모 로그엔 **CAN+video만** → 1차 선별은 CAN+VLM, 2차 라벨링만 3DOD+map.

---

## 2. 아키텍처 — 2단계 퍼널 × 검색 파이프라인
```
전체 로그(CAN+video)
  │  STAGE 1 선별: ego arc(CAN) + VLM 흥미도(몽타주) → 결합랭킹 → 상위 K
  ▼
선별 clip → STAGE 2 episode 라벨링 (3DOD+map):
     egomotion 분할
     ─ Phase A  recall-우선 후보생성 (OR 앙상블)
     ─ Phase B  정규화·병합·랭킹
     ─ Phase C  precision 게이트 (검색 이후·하류)
     ─ Phase D  gold & retrieval 측정
  ▼
에피소드별 후보 태그 (provenance·confidence·등급) → 검색·추출
```

---

## 3. 택소노미 (`taxonomy.py`)
4축 + ODD. AUTO_GT{turn/stop/u_turn}=egomotion 자동(평가 제외), 나머지=평가 대상.
- ego기동 / 상호작용 / 맥락(교차로·도로유형) / 규칙(신호). 어휘는 **v0.7.3 KPI vocab과 통일**(Phase E).

---

## 4. Phase 상세 (실행 단위)

### Phase A — recall-우선 후보 생성 (OR 앙상블) · HIGH
- **다채널 OR 합집합**(어느 채널이든 잡으면 후보):
  1. **VLM self-consistency n-vote** (n=5, temp>0) — **1표라도 나오면 후보**
  2. **GT 검출기** (`taxo_detect`, obj3d)
  3. **ego-arc** (`events`, egomotion)
- 각 후보에 **provenance**(어느 채널) + **vote_fraction/vote_var**(VLM confidence) 부착. 다채널 동시검출=고신뢰.
- ⚠️ **∩(확인전용)·precision 게이트 금지** — Phase C로. 목표는 recall 최대화.
- 근거: ego(reactive)·VLM(non-reactive) 상보성 (선별 상위30: VLM우세15/ego우세4/둘다11).

### Phase B — 정규화·랭킹·병합 · HIGH
- **정규화 계층**: 슬롯 오류·별칭 흡수(태거 재실행 없이 검색시점 매핑).
- **병합**: 혼동 리포트(카테고리쌍 투표 불일치)로 태거가 못 구분하는 쌍 병합(backstreet+urban→urban 등). → `taxonomy_merge_report.md`.
- **confidence 랭킹**: vote_var·다채널 합의로 순위. 검색결과=confidence-ranked.

### Phase C — precision 게이트 (검색 이후·하류) · HIGH
- **역할 매트릭스**(GT 가용성): GT 신뢰 → `gt∩vlm` 확인 / GT 부재 → self-consistency 다수결 / 부분 → C2.
- **Tier-B human 확인**: 저신뢰·부분 후보를 사람이 확인해 승격(버리지 않음).
- **학습셋 등급**: `train_ready | needs_review | low_conf`.

### Phase D — gold & retrieval 측정 · HIGH
- **완전라벨 gold** ≥60 clip (present/absent 명시, **sparse 금지**).
- **지표**: 카테고리별 **recall = 1차 지표**. precision은 C 승격 후. FP는 `{실제FP|gold누락|검출기오류}` 분류.
- AUTO_GT(turn/stop/u_turn) 평가 제외(순환) 유지.

### Phase E — 어휘 통일 · MED (병렬)
- 추출 taxonomy ↔ v0.7.3 vocab(object_type·relation·ego_action·occlusion·cause) 매핑·단일 공유. `occlusion_emerging`(dart-out) 타깃 추가.

### Phase F — map 시간축 집계 · LOW (선택)
- lane-index 카테고리(lane_change·cut_in)에만 다프레임 경계선 누적으로 map 유효율↑. 그 외 corridor fallback 유지.

---

## 5. 가드레일 / 범위 밖
- 태거 fine-tuning 금지(순환). KV-cache·TTA·비용최적화·합성데이터·다운스트림 VLA 학습은 **범위 밖**(별도 트랙).
- **cosmos2(8000) 무접촉**(타 그룹). 복제본 8001–8004. python3, docker container logs.

---

## 6. 핵심 발견 (실측, 설계 근거)
1. 선별엔 **ego+VLM 둘 다 필수**.
2. **cut_in 극희귀**(gold 0/50, ~26/1966) — 희소 자체가 결과.
3. **map 활용 ~35%**(경계선 재해석, 근접경계 누락) → Phase F 대상.
4. **VLM은 맥락 볼륨 강·미세분류 약**(construction 환각 등) → Phase A(다중투표)+B(병합).
5. **gold sparse → precision 왜곡** → Phase D(완전라벨).

---

## 7. 현재 상태 & 다음
- **완료**: 택소노미, egomotion/obj3d/map/VLM 검출·검증 모듈, sparse gold 50, 선별 프로토타입, 데모.
- **다음(v3)**: **Phase A(OR 앙상블 · VLM 5-vote)** 착수 → B(정규화·병합) → C(precision 하류) → D(완전 gold).

## 8. 코드/산출물
`events.py taxo_detect.py vlm_verify.py map_lane.py taxonomy.py classify073.py` · `gold.json` gold_label/ · 실행명세 `Concept_Design_v3` · 팀리포트 `TEAM_REPORT.md`

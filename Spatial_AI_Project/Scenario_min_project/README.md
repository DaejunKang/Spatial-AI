# KATECH VLA — 자율주행 에피소드 태깅 & Long-tail 추출

자율주행 클립에서 **자차 거동 사건(에피소드) 메타데이터를 태깅**하고 long-tail 상황을 **추출**하는 파이프라인.
VLM(NVIDIA Cosmos-Reason)은 **frozen** — 재학습하지 않고 GT(egomotion/obj3d/map)+규칙+VLM 조합 로직을 개선한다.

## Architecture — 2단계 퍼널

- **Stage 1 선별**: 전체 로그(CAN+video만, obj3d/map 없음) → ego arc + VLM 흥미도(몽타주) 결합 랭킹 → 상위 K. recall 위주 triage
- **Stage 2 에피소드**: 선별 clip(3DOD + map 사용) → egomotion 전이로 분할 → 에피소드별 메타데이터

싸게 넓게 선별하고, 비싼 상세 태깅은 선별분에만 적용하는 구조다.

## 두 제품

| 제품 | 내용 | 계보 |
|---|---|---|
| **Track1** | v08형 세그먼트 메타데이터(scene description/critical components + cause + ego_action) — 라벨링 | `task_episode/tag_v08.py` |
| **Track2** | recall-우선 retriever(3채널 OR 합집합: ego-arc/obj3d-GT/VLM n-vote) — 큐레이션 검색 | `task_episode/candidates.py` → `retrieve.py` |

실측 recall(gold 50clip): OR 합집합 0.81(병합 후 0.90), VLM 단독 0.67, GT 단독 0.17.

## 폴더 구조

```
common/          공유 인프라: config·paths·dataset·events·taxonomy·client·overlay + schema
task_selection/  Stage1: selection.py + folder_selection.py → gold_label/select/
task_episode/    Stage2: base(classify073·taxo_detect·map_lane·vlm_verify)
                        Track1 tag_v08          Track2 candidates·retrieve
deploy/          Stage1 배포 패키징
legacy/          활성이 import 안 하는 구세대(v0.7.1 tagger·window/event_tagger 등)
docs/            설계 정본(PROJECT_DESIGN.md)·목표설계(design/pipeline_design_guide_v0.4.md)
decisions/       설계 변경 이력(DESIGN_LOG.md), 설계변수 기본값 스냅샷(DV_DEFAULTS.md)
```

## 실행

```bash
./run.sh <script.py> [args]        # venv 실행 + logs/latest.log
```

정식 test/lint는 없다. VLM 서버는 NVIDIA NIM(cosmos3-nano-reasoner) 복제본 4개를 라운드로빈으로 사용한다.

## 현재 상태

문서는 3층 구조로 관리한다 — **현재 코드 실태 / 목표 설계(v0.4, 미구현) / 불변 규칙**. 자세한 내용과
운영 지침은 [`CLAUDE.md`](CLAUDE.md) 참조.

최근 작업: VLM 디코딩 재현성 조사 — same-anchor 통제 실험으로 원인을 실측 확정했다(anchor를 완전히
고정한 뒤 반복 decode). Track2의 5-vote 비결정성은 seed 부여로 10/10→1/10(사실상 완전 재현)까지
해소됨을 검증해 반영했고, Track1의 `temperature=0` 잔차는 서빙층(프리픽스 캐시·배치 구성) 기인으로
확인해 별도 과제로 남겨뒀다. 근거와 수치는 [`decisions/DESIGN_LOG.md`](decisions/DESIGN_LOG.md) 최신
항목 참조.

## 불변 규칙 요약

- **Frozen tagger**: fine-tuning 금지(순환성 차단). 평가용 복제본(port 8001)은 별도 taggers와 무접촉
- **앵커 비오염**: 전이 시점(`key_frame_t`)과 종방향 거동은 CAN 규칙 산출 — 모델 출력이 중간 경유로도 진입 금지
- **재현성**: seed 고정만으로는 불충분(배치 크기·프리픽스 캐시 히트에 따라 결과가 흔들림) — 매니페스트로 기록

전체 규칙은 [`CLAUDE.md`](CLAUDE.md) §3 참조.

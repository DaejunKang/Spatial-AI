# Taxonomy Merge Report (Phase B) — 2026-07-24

혼동 리포트(gold 50 · Phase A OR 후보 대비)로 태거가 못 구분하는 쌍을 병합. 검색용 통합, 세부는 Phase C 복원.

## 혼동 근거 (gold=행 → 후보가 낸 것)
- road_urban_arterial → {urban 33, **backstreet 11**, rural 5}
- road_backstreet     → {**urban 5**, backstreet 7}
- intersection_signalized   → {sig 22, **unsig 6**}
- intersection_unsignalized → {unsig 10, **sig 1**}

## 적용 병합 (retrieve.MERGE)
| 원본 | → 대표 | 근거 |
|---|---|---|
| road_urban_arterial, road_backstreet | **road_surface** | VLM이 생활도로 유형 구분 못함(양방향 혼동) |
| intersection_signalized, intersection_unsignalized | **intersection** | VLM 신호상태 혼동(~21%) |

## 효과
- OR recall 0.81 → **0.90**. road_surface 0.98, intersection 1.00.
- 세부(신호상태·도로유형)는 tag.sub에 보존 → Phase C(precision·human)에서 복원.

## 미병합 (소수지지·근거부족)
road_rural/highway/tunnel/bridge/parking — 지지도 작아 유지. 추가 데이터 후 재검토.

# decisions — 설계 변경/개선 이력

설계·아키텍처가 바뀔 때마다 **무엇을·왜** 바꿨는지 여기 기록한다. 코드(diff)는 git이, **의도·근거**는 이 폴더가 담당한다.

## 사용법
- 설계 변경이 생기면 `DESIGN_LOG.md` 최상단에 새 엔트리 추가(최신이 위).
- 큰 결정은 별도 파일 `ADR-NNNN-<제목>.md`로 남겨도 됨.
- 엔트리 형식 (**Task 라벨 필수** — 컨벤션 `../docs/README.md`):
  ```
  ## [YYYY-MM-DD] 제목
  > **Task**: common | selection | episode(Track1|Track2) | multi
  - 변경: 무엇을 바꿨나
  - 이유: 왜 (근거·실측·트레이드오프)
  - 영향: 파일/후속 작업·드리프트
  ```
- 확정된 불변식은 `CLAUDE.md §2`(design constraints)에 요약, 이 폴더엔 **이력·이유 전문**.

## 참고 문서
`../CLAUDE.md`(코드설명+제약) · `../PROJECT_DESIGN.md`(개념 v3) · `../Concept_Design_v3`(실행명세) · `../taxonomy_merge_report.md`

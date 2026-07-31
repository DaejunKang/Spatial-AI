# docs — 설계·리포트 (공통 폴더)

## Task 라벨 컨벤션 (전 폴더 공통)
`legacy/` · `docs/` · `decisions/` 는 task별로 쪼개지 않고 **공통 폴더**로 둔다. 대신 **각 파일/엔트리 최상단에 소속 task를 명시**한다:

```
> **Task**: common | selection | episode(Track1|Track2) | multi
```

- 새 리포트·설계문서·이력 엔트리·legacy 정리 시 **이 라벨을 반드시** 붙인다.
- 이렇게 하면 폴더를 task 하위에 중복 생성하지 않고, 파일 내부 라벨로 소속을 판별한다.

## 문서 목록
| 문서 | Task | 내용 |
|---|---|---|
| `PROJECT_DESIGN.md` | multi | **설계 정본** — 개요·아키텍처·Phase·현재상태·업무분담 (구 Concept_Design_v3 흡수) |
| `TEAM_REPORT.md` | multi | 팀 공유 리포트 |
| `taxonomy_merge_report.md` | episode(Track2) | 병합 근거·정정 |
| `STRUCTURE.md` | common | 데이터셋 구조 |
| `meta_tag_format.md` | episode(Track1) | v0.8 SD/SA/MA 태그 포맷 |
| `Claude_task_v0.7.3` | episode(Track1) | v0.7.3 태스크 명세(legacy 참고) |
| `meta_tagging_*_v0.7.3*` | episode | v0.7.3 vocab/sample(참고) |

(선별 알고리즘 명세 `SELECTION_STAGE1.md`는 `task_selection/`에 위치.)

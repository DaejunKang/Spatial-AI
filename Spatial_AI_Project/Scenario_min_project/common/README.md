# common — 공용 기반 (상위 프로젝트로 공유)

두 task(선별·episode) 및 상위 프로젝트가 함께 쓰는 인프라. 특정 단계에 종속되지 않는 데이터 접근·egomotion·VLM client·공유 어휘.

> **(B) 점진 이관**: 실제 파일은 아직 repo 루트에 flat로 있음(임포트 무결성 유지). 이 폴더는 **소속·역할 매니페스트**. 물리 이동 시 `common`을 PYTHONPATH에 올려 `import events` 등을 유지.

## 소속 파일 (repo 루트)
| 파일 | 역할 |
|---|---|
| `config.py` | 서버 접속(BASE_URLS 8001–8004)·추론 파라미터(MODEL·TEMPERATURE·SEND_FPS 등) |
| `paths.py` | **visionary-nvidia** 데이터셋 경로(video/egomotion/obj3d/map). 신규 코드의 데이터 접근 표준 |
| `dataset.py` | 프레임/몽타주 샘플링·subclip 생성·data URI (`sample_montage`·`write_subclip`·`to_data_uri`) |
| `events.py` | **egomotion primitives** — `load_egomotion_clip`(카메라 범위 클립)·`detect_events`(net-heading 거동). 양 단계 공용 |
| `taxonomy.py` | 단일 공유 택소노미(4축+ODD)·`AUTO_GT`/`HUMAN_KEYS`/`auto_tags_from_arc` |
| `tagger.py` | VLM client pool(`build_client_pool`, 8001–4 헬스체크 라운드로빈) |
| `vocab.py`·`vocab073.py` | KPI 어휘(v0.7.1/v0.7.3) 로더·스키마·검증 |
| `overlay.py` | 세그먼트 태그 영상 burn-in(정성 확인 util) |

## 규약(요약, 상세는 상위 CLAUDE.md §2)
- 이 계층은 태거 frozen·anti-circularity에 중립적 인프라. cosmos2(8000) 무접촉.
- 데이터 함정: obj3d x=전방/y=좌우·vx·vy 불신, map centerlines=경계선·is_intersection 죽음. egomotion 로그는 비디오보다 김.

# PhysicalAI-AV-curated 데이터셋 & 태깅 파이프라인

루트: `/katech/datasets/PhysicalAI-AV-curated`
> 요청 경로는 소문자 `physicalAI-AV-curated` 였으나 실제 디렉토리는 대문자 `PhysicalAI-AV-curated` 하나뿐(소문자 경로 없음).

## 규모
- **전체 1,966 클립** (metadata 인덱스 = mp4 개수 = 1,966, 카메라별). 클립당 20초, 30fps, 1920×1080.
- split(`curation/*.txt`, 한 줄=clip_id): **각 파일 마지막 줄에 개행이 없어 `wc -l`은 1 적게 셈** — 실제 개수는 아래.

| split | 실제 클립 수 |
| --- | --- |
| diverse_set-test / val | 150 / 150 |
| diverse_set-train | 700 |
| original_set-test / val | 150 / 150 |
| original_set-train | 700 |

## 디렉토리
```
PhysicalAI-AV-curated/
├── camera/{camera_front_wide_120fov, camera_front_tele_30fov}/   # 각 1,966 클립
│     <clip_id>.<cam>.mp4                    # 영상 (20s, 30fps, 605프레임)
│     <clip_id>.<cam>.timestamps.parquet     # 프레임 타임스탬프 (µs↔frame_index)
│     <clip_id>.<cam>.blurred_boxes.parquet  # 프라이버시 블러 2D 박스
├── curation/            # split 목록 (위 표)
├── labels/
│   ├── egomotion/          # <clip_id>.egomotion.parquet         (자차 상태, 원본 로그 전체)
│   ├── egomotion.offline/  # <clip_id>.egomotion.offline.parquet (클립 정합 궤적)
│   └── obstacle.offline/   # <clip_id>.obstacle.offline.parquet  (3D 장애물 트랙)
├── metadata/clip_index_curated.parquet   # 전체 클립 인덱스 (1,966행)
└── reasoning/ood_reasoning.parquet        # OOD reasoning (15행, train만)
```
모든 산출물은 `clip_id`(UUID) 기준 join. 예) `0c5239c7-a3d6-467a-9f32-bebb4a1bb436`.

## Parquet 스키마 (pyarrow 로 실측 확인)
- **metadata/clip_index_curated** (1,966행): `clip_id, split(train/test/val), chunk, clip_is_valid`.
- **egomotion/** (자차 상태, 고빈도 ~84Hz): `timestamp(µs), qx/qy/qz/qw, x/y/z, vx/vy/vz, ax/ay/az, curvature`.
- **egomotion.offline/** (클립 정합, ~10Hz, 202행): `timestamp, qx..qw, x/y/z` (속도·가속도 없음).
- **obstacle.offline/** (프레임·객체별 3D 박스): `timestamp_us, track_id, center_x/y/z, size_x/y/z, orientation_x/y/z/w, label_class, reference_frame(rig), reference_frame_timestamp_us`.
  - `label_class`: automobile, heavy_truck, trailer, person, rider, protruding_object …
- **camera .timestamps** (605행): `timestamp(µs), frame_index(0-based)`.
- **camera .blurred_boxes**: `frame_index, x1,y1,x2,y2` (픽셀).

## ⚠ 시간 정렬 주의 (중요)
- `egomotion`(online)은 **비디오(20s)보다 긴 원본 주행 로그 전체**(66~140s). 비디오는 그 맨 앞 ~20초.
  → 반드시 카메라 timestamps 범위 `[t_min, t_max]`로 **필터한 뒤** `t초 = (ts − t_min)/1e6` 로 매핑.
- `egomotion.offline`·`obstacle.offline`은 **이미 클립 정합(0~20s)**, 카메라와 동일 클럭 → 필터 불필요.
- 검증: online(필터)↔offline 궤적 평균 1.1m 일치, 카메라 30fps 균일 → 매핑 정확.

## 태깅 파이프라인 (이벤트 중심 하이브리드, meta_tagging v0.7.1)
"언제(when)"는 라벨 데이터(GT)로, "무엇(what)"은 VLM으로 분리.
- **egomotion 이벤트** (`events.detect_events`): 감속·가속·정지·좌/우회전 → 정확한 시각·프레임.
- **obstacle 이벤트** (`events.detect_obstacle_events`): ego 진행로 내 트랙 → 정속 선행차·cut-in·횡단 등(egomotion 사각지대 보완, GT 클래스·위치 제공).
- **모델 분류**: 각 이벤트 ±`EVENT_CTX_SEC`(1.5s→3s창) 서브클립을 `video_url`로 전송해 subject/거동 분류.
  서버 프레임 캡이 ~12프레임 고정이라 3s창 = ~3.7fps(6s창의 2배). 시간은 이벤트가 정의.
- 무이벤트 구간은 `ego_lane_keep` baseline으로 채워 전 구간 커버. 출력 세그먼트에 `t_start/t_end`(초) + `frame_start/frame_end`(원본 30fps, 0부터) 병기.
- 모듈: `events.py`, `event_tagger.py`, `window_tagger.py`, `tagger.py`(`TAG_MODE=events`), `overlay.py`, `vocab.py`.

## 실행 & 확장
```bash
./run.sh test_readout.py                              # 단일 클립 (config.CLIP_INDEX)
./run.sh batch_readout.py --limit 5                   # 앞 5개 + 오버레이
./run.sh batch_readout.py --split diverse_set-test.txt --limit 0 --no-overlay   # 테스트셋 150 전체
./run.sh batch_readout.py --split all --limit 0 --no-overlay --workers 4        # ★ 전체 1,966 확장
```
- `--split all` = metadata의 전체 1,966 클립. 다른 split은 `curation/` 파일명 지정.
- `--start/--limit`로 구간 분할, `--overwrite`로 재처리, `--workers`로 동시 처리.
- 산출물: `outputs/tags/<clip_id>.json` · `outputs/tags.jsonl` · `outputs/overlay/<clip_id>.mp4`.
- **확장 준비 확인**: 전체 1,966 클립 모두 camera/timestamps/egomotion/obstacle 라벨 완비(누락 0).

## 환경
- venv(`.venv`)에 `openai, pyarrow, opencv-python-headless, numpy, jsonschema` 설치됨.
- 추론 서버: NIM 컨테이너 `cosmos_dj` (`nvidia/cosmos3-reasoner`, vLLM 0.14.1, 아키텍처 qwen3_vl), `localhost:8001`.
  - 비디오 프레임: 프로세서 `fps`(현재 2)로 결정, 프레임당 ~1,025토큰·총 ~12프레임 상한. 밀도↑ 하려면 컨테이너 `/opt/nim/workspace/processor_config.json`의 `fps` 상향 후 재기동 필요.

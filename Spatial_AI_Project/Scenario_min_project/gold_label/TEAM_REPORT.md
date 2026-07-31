# VLA Long-tail 추출 파이프라인 — 팀 공유 리포트

as of 2026-07-23 · 담당 djkang · 자율주행 로그 long-tail 상황 자동 tag·추출

---

## 0. 한눈에 (현황)
- **2단계 퍼널**: ① clip 선별(CAN+VLM) → ② episode 메타데이터 labeling(3DOD+map+VLM).
- **최신 설계 방향(v3)**: 태거를 **분류기 아닌 retriever(관대한 후보 생성기)** 로. recall 우선(넓게 건짐), precision은 하류 human 확인으로 회복.
- 태거는 **frozen**(재학습 안 함). GT(egomotion/obj3d/map)+규칙+VLM 조합을 개선.

---

## 1. 모델 & 서빙

| 항목 | 값 |
|---|---|
| 모델 | **NVIDIA Cosmos-Reason nano** (`nvidia/cosmos3-nano-reasoner`, Qwen3-VL 계열 reasoning VLM) |
| 서빙 | vLLM (NIM), OpenAI-compatible API. **복제본 4개 8001–8004** 라운드로빈 |
| ⚠️ 제약 | **cosmos2(port 8000) 무접촉** — 타 그룹 사용 중. 서버 재기동은 승인 필요(운영 서버) |
| 디코딩 | `TEMPERATURE=0`, **guided_json**(`response_format` json_schema)로 enum 강제 — 어휘 고정 |
| 입력 | (2차) `video_url` = subclip data URI (`SEND_FPS=10`, 서버가 ~12프레임 샘플) / (1차) `image_url` = 몽타주 12프레임 |
| 토큰 | reason `MAX_TOKENS=6144`, 검증 512~2048, 분류 768 |

---

## 2. tag JSON 구성 (3종)

### 2-A. 2차 파이프라인 출력 (clip당, `outputs_v08/tags/<clip>.json`)
```
{
  "clip_id": "...", "ok": true, "mode": "v08",
  "records": [ {                         # 에피소드(세그먼트)별
    "segment_id": 1,
    "window": [t0, t1],                  # 에피소드 시간창(초)
    "key_frame_t": 0.0,                  # onset
    "ego_context": {                     # egomotion GT
       "ego_action": "ego_stop",         # enum(EGO_ACTIONS)
       "arc": ["stop","accelerate"],     # 거동 시퀀스
       "arc_rule": "ego_stop" },
    "cause": "agent",                    # enum: agent|signal|road_geometry|other
    "scene_description": "Nighttime, rainy ... stopped at red light behind a white van ...",  # VLM 자유서술
    "ego_intent": null,
    "critical_components": [ {           # 핵심 객체(장면 영향 요소)
       "description": "...", "why_critical": "...",
       "object_type": "vehicle", "relation": null,   # GT(obj3d) 근거
       "source": "gt|model|gt_injected",
       "ref": {"object_type":"vehicle","role":"preceding_vehicle","distance_m":11.4,"vru_state":null},
       "tags": ["object_type:vehicle"] } ],
    "chain_of_causation": "...",         # VLM 인과 서술
    "consistency": [{"gt":{...}, "covered_by_model": true}],  # GT↔모델 정합
    "search_tags": ["arc:accelerate","cause:agent","ego_action:ego_stop","object_type:vehicle"]  # 검색용 flat 태그
  } ]
}
```
- **핵심**: `window`(언제) = egomotion GT, `ego_action/arc`(ego 거동) = GT, `object_type/relation`(agent) = obj3d GT, `scene_description/chain_of_causation`(서술) = VLM, `search_tags` = 큐레이션 검색 인덱스.

### 2-B. gold (사람 정답 라벨, `gold.json`)
```
{ "<clip_id>": {
    "odd": {"lighting":"night"},                  # 정적환경(참고)
    "episodes": [{ "win":[0.5,8.5], "auto":["turn_right"],   # auto=egomotion GT(사람 라벨 X)
                   "arc":["decelerate","turn_right"],
                   "human":["intersection_signalized","road_urban_arterial"], "note":"" }],
    "manual": [{ "win":[0,4], "human":[], "manual":true }] } }   # 기동 밖 수동 구간
```

### 2-C. taxonomy 융합 출력 (검색/추출용)
- 에피소드별 **taxonomy 카테고리 집합** + source(GT/obj3d/map/VLM) + confidence. (v3에선 provenance·vote_fraction 부착)

---

## 3. Prompt (실제 사용문)

### 3-A. 2차 — 자유 서술(reason)
> "{GT hint}\nAnalyze this driving segment. Describe in English, freely: (1) scene description (road, environment, ego intent), (2) each critical component that influences the ego's behavior and why it is critical, (3) the chain of causation for the ego's behavior. Use the GT arc/objects as grounding but verify and describe from the video." + `video_url`

### 3-B. 2차 — 구조화(structure, guided_json = V08_SCHEMA)
> "Structure the above analysis into JSON ... For ego_action, classify the ego's actual maneuver from its enum (a stop-then-turn is a turn/unprotected turn, not just a stop)."
> **V08_SCHEMA**: `ego_action`(enum), `cause`(enum), `scene_description`(≤400), `critical_components`[{description,why_critical,object_type(enum|null),relation(enum|null)}](maxItems 6), `chain_of_causation`

### 3-C. VLM 검증/분류 (`vlm_verify._prompt`, guided_json)
> "You are verifying driving events in this video segment. Ego arc = {arc}. GT detectors flagged: {cands}. STRICT RULE: include an event in `present` ONLY if clearly visible; when uncertain, exclude. ... Also classify: road_env(one), intersection(signalized/unsignalized/none), extras(roundabout/merge/construction/toll only if clearly present), red_light_stop/signal_go."
> 출력: `{present[], road_env, intersection, extras[], red_light_stop, signal_go}`

### 3-D. 1차 선별 — interestingness (몽타주, guided_json)
> "Frames from a 20s driving clip. Decide if NOTABLE/long-tail (turn, intersection, stop/yield, cut-in, pedestrian, lane change, congestion, hazard) vs ROUTINE cruising. Return: interesting(bool), score(0-1), category, reason. Plain cruising = low score."

---

## 4. 필요 공유사항 (리스트업)

**접속·환경**
- [ ] VLM 서버: `http://localhost:8001~8004/v1` (원격은 SSH 터널). **cosmos2/8000 절대 무접촉.**
- [ ] 실행 파이썬: `/home/daejun/vla-tagging/.venv`. 서버 상태: `docker logs`.
- [ ] 데이터셋: `/katech/datasets/visionary-nvidia` (obj3d 3DOD + map + egomotion). raw 로그엔 CAN+video만 있음.

**코드·산출물**
- [ ] 파이프라인: `events.py`(egomotion) · `taxo_detect.py`(obj3d GT검출) · `vlm_verify.py`(VLM 검증·융합) · `map_lane.py`(map 경계선·분기) · `taxonomy.py`(공유 어휘) · `classify073.py`
- [ ] gold/평가: `gold.json`, `gold_label/`(라벨링 도구·리뷰·데모, `python3 -m http.server 8080`로 서빙 중)
- [ ] 문서: `PROJECT_DESIGN.md`(개념설계), `Concept_Design_v3`(최신 실행명세), `SELECTION_STAGE1.md`, 본 리포트

**어휘·규격 (합의 필요)**
- [ ] `taxonomy.py` = 추출/검색 단일 기준(4축: ego기동·상호작용·맥락·규칙 + ODD).
- [ ] **v0.7.3 KPI vocab**(object_type·relation·ego_action·occlusion·cause)과 추출 taxonomy **매핑 통일** 필요(Phase E). `occlusion_emerging`(dart-out) 타깃 추가 논의.
- [ ] search_tags 규격(`key:value` flat) — 검색·큐레이션 인터페이스.

**현재 이슈 (공유·논의)**
- [ ] **cut_in 극희귀**: 데이터셋에 ~1.3%(gold 0/50). 타깃 수집/합성 여부 논의.
- [ ] **map 활용 ~35%**: 근접 경계선 누락 → 나머지 corridor fallback. (`is_intersection` 필드는 사용불가)
- [ ] **VLM 미세분류 한계**: 도로유형·신호상태·construction 혼동/환각. → 택소노미 병합 + 다중투표(v3).
- [ ] **gold sparse → precision 측정 왜곡**: 완전라벨 세트(≥60) 필요(Phase D).

**다음 단계 (v3 로드맵)**
- [ ] Phase A: recall-우선 **OR 앙상블**(VLM 5-vote + GT검출기 + ego-arc 합집합, provenance/confidence).
- [ ] Phase B: 정규화·혼동병합·confidence 랭킹.
- [ ] Phase C: precision 게이트(∩·역할매트릭스·Tier-B human) — 검색 이후로 분리.
- [ ] Phase D: 완전라벨 gold + recall 1차지표.

---

## 5. 핵심 요약 (팀 메시지)
1. 태거는 **retriever** — 넓게 건지고(recall), 하류에서 human이 정밀화(precision).
2. 태그는 **에피소드 단위 시간동기** — "언제·무엇"이 함께. 언제=egomotion GT, 무엇=obj3d GT + VLM 서술/검증.
3. **어휘 통일(taxonomy ↔ v0.7.3 vocab)** 이 추출→KPI 평가 연결의 관건.

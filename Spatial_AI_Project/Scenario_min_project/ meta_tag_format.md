# 동적 거동 세그먼트 태깅 스키마 v0.7.1

어휘 라벨 공간(`meta_tagging_vocab_v0.7.1.json`)은 동일. 태깅 **단위와 출력 구조**를 세그먼트 기준으로 재정의한 버전.

## 태깅 단위

- **Clip** = 20초 로깅 파일 (하나의 입력)
- **Segment** = Clip 내 5~10초 구간. 태깅의 기본 단위
- **레코드 1건 = 세그먼트 1건 = 동적 거동 이벤트 1건**
- 각 세그먼트는 다음 중 **하나**를 태깅:
  - `subject="ego"` — ego 자체의 동적 거동 (⑤ ego_action)
  - `subject="agent"` — ego에 영향을 주는 agent의 동적 거동 (① 종류 + ②/③ 거동)
- **객체 나열이 아님.** 정적 씬의 모든 객체를 뽑는 것이 아니라, ego 거동 또는 ego영향 거동이 발생하는 구간만 세그먼트로 잘라 태깅한다.
- 한 Clip에서 여러 세그먼트가 나올 수 있고, 시간적으로 겹칠 수 있다(예: agent cut-in 구간과 그에 반응한 ego 정지 구간).

## 필드 정의

| 필드 | 타입 | 조건 | 설명 |
| --- | --- | --- | --- |
| `clip_id` | string | 필수 | 20초 클립 식별자 |
| `segment_id` | int | 필수 | 클립 내 세그먼트 번호 |
| `t_start` | number(0–20) | 필수 | 세그먼트 시작(초) |
| `t_end` | number(0–20) | 필수 | 세그먼트 종료(초). 길이 5~10s |
| `subject` | `ego` \| `agent` | 필수 | 거동 주체 |
| `ego_action` | enum\|null | subject=ego | ⑤ 자차 거동 11종 |
| `object_type` | enum\|null | subject=agent | ① 종류 8종 |
| `role` | enum\|null | subject=agent | `preceding_vehicle`/`adjust`/`crossing` |
| `longitudinal_action` | enum\|null | role=preceding_vehicle / stationary_parked | ② 종방향 7종 |
| `relation` | enum\|null | role=adjust/crossing | ③ 관계 13종 (무손실 단일) |
| `vru_detail` | enum\|null | object_type=pedestrian | VRU 세부 7종 |
| `difficulty` | int(1–10)\|null | 선택 | 난이도 루브릭 |

## 필드 채움 규칙 (subject 판별)

- **ego 세그먼트**: `ego_action` 채움. `object_type·role·longitudinal_action·relation·vru_detail` = null
- **agent 세그먼트**: `object_type`+`role` 채움. `ego_action` = null
  - `role=preceding_vehicle` → `longitudinal_action` 사용, `relation` = null
  - `role=adjust`/`crossing` → `relation` 사용, `longitudinal_action` = null
  - `object_type=pedestrian` → `vru_detail` 추가 가능

## 확정된 설계 결정

1. **관계 무손실 단일** — 객체당 ego 영향 관계 1개만 태깅하는 원칙이므로 `relation` 단일로 정보 손실 없음. ego 무영향 거동(후행·추월·밀착)은 취소 어휘로 배제.
2. **`role`에서 `{category}` 제거** — 클래스는 `object_type`이 전담, `role`은 상호작용 유형 3종.
3. **세그먼트 길이 5~10s, 0~20s 클립 내** — JSON Schema로는 t_start/t_end 범위만 강제, 길이 5~10s는 코드 검증(아래).

## 서버 테스트 형식 (`cosmos_dj`, localhost:8001)

```python
import json, re, jsonschema
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
VOCAB  = json.load(open("meta_tagging_vocab_v0.7.1.json"))
SCHEMA = json.load(open("meta_tagging_segment_schema_v0.7.1.json"))

SYS = (
    "You tag dynamic driving behaviors in a 20s clip. Use ONLY codes from VOCABULARY.\n"
    "Segment the clip into 5-10s windows; each window captures ONE dynamic behavior "
    "event: either an ego maneuver (subject=ego) or an ego-affecting agent maneuver "
    "(subject=agent). This is NOT per-object listing.\n"
    "Think in <think></think>, then emit a JSON array of segment records conforming "
    "to SEGMENT_SCHEMA inside <answer></answer>.\n\n"
    f"VOCABULARY:\n{json.dumps(VOCAB, ensure_ascii=False)}\n\n"
    f"SEGMENT_SCHEMA:\n{json.dumps(SCHEMA, ensure_ascii=False)}"
)

resp = client.chat.completions.create(
    model="nvidia/cosmos3-nano-reasoner",
    temperature=0, max_tokens=4096,
    extra_body={"allow_logprobs": True},
    messages=[
        {"role":"system","content":SYS},
        {"role":"user","content":[
            {"type":"text","text":"이 20초 클립을 v0.7.1 세그먼트 스키마로 태깅하라."},
            {"type":"video_url","video_url":{"url":"file:///path/to/clip_20s.mp4"}},
        ]},
    ],
)
out = resp.choices[0].message.content

# 검증: enum·필수필드 + 세그먼트 길이 5~10s
ans = re.search(r"<answer>(.*?)</answer>", out, re.S).group(1)
for rec in json.loads(ans):
    jsonschema.validate(rec, SCHEMA)
    assert 5 <= rec["t_end"] - rec["t_start"] <= 10, "segment length out of range"
```

## 출력 예시 (Clip KATECH-0472, 20s)

```json
{"clip_id":"KATECH-0472","segment_id":1,"t_start":2.0,"t_end":9.0,"subject":"agent","ego_action":null,"object_type":"motorcycle","role":"adjust","longitudinal_action":null,"relation":"sudden_cut_in","vru_detail":null,"difficulty":8}
{"clip_id":"KATECH-0472","segment_id":2,"t_start":6.0,"t_end":14.0,"subject":"ego","ego_action":"ego_reactive_stop","object_type":null,"role":null,"longitudinal_action":null,"relation":null,"vru_detail":null,"difficulty":8}
```

세그먼트 1은 오토바이의 급끼어들기 거동(2–9s), 세그먼트 2는 그에 반응한 ego 정지 거동(6–14s). 시간 겹침은 인과(agent 거동 → ego 반응)를 그대로 반영하며, EPR/Causal-F1 분석 시 `subject=agent → subject=ego` 세그먼트 쌍의 시간 중첩이 인과 후보가 된다.

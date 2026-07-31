"""meta_tagging v0.7.1 어휘 로더 · segment 스키마 · 프롬프트 · 검증.

- 어휘 라벨 공간: meta_tagging_vocab_v0.7.1.json (축별 허용 코드)
- 태깅 단위: Segment (20s 클립 내 5~10s 구간, 레코드 1건 = 동적 거동 이벤트 1건)
- 세그먼트 레코드 스키마: build_segment_schema() 로 어휘에서 생성 → json 파일로도 저장
"""

import json
from pathlib import Path

# 축 name_en
AX_OBJECT_TYPE = "object_type"
AX_LON_ACTION = "longitudinal_action"
AX_RELATION = "relation"
AX_EGO_ACTION = "ego_action"
AX_VRU = "vru_detail"

# role 은 어휘의 per-label role 힌트와 별개인 상호작용 유형 3종 (문서 v0.7.1)
ROLE_ENUM = ["preceding_vehicle", "adjust", "crossing"]
SUBJECT_ENUM = ["ego", "agent"]

VOCAB_FILENAME = "meta_tagging_vocab_v0.7.1.json"
SCHEMA_FILENAME = "meta_tagging_segment_schema_v0.7.1.json"


def _find(filename: str) -> Path | None:
    here = Path(__file__).parent
    for p in sorted(here.iterdir()):
        if p.is_file() and p.name.strip() == filename:
            return p
    return None


def load_vocab(path: Path | None = None) -> dict:
    path = path or _find(VOCAB_FILENAME)
    if path is None:
        raise FileNotFoundError(f"{VOCAB_FILENAME} 를 찾을 수 없습니다.")
    return json.loads(path.read_text(encoding="utf-8"))


def _axis(vocab: dict, name_en: str) -> dict:
    for ax in vocab["axes"]:
        if ax["name_en"] == name_en:
            return ax
    raise KeyError(f"축 {name_en} 없음")


def codes(vocab: dict, name_en: str) -> list[str]:
    return [lb["code"] for lb in _axis(vocab, name_en)["labels"]]


# --- segment JSON Schema ----------------------------------------------------
# 클립 실측 길이(~20.17s)를 수용하기 위한 시간 상한. 문서상 20s 이나 데이터는 20.2s.
CLIP_MAX_SEC = 21


def build_segment_schema(vocab: dict) -> dict:
    """세그먼트 레코드 1건의 JSON Schema(draft-07)를 어휘에서 생성.

    frame_start/frame_end(원본 fps 프레임 인덱스, 0부터) 와 windows(기여 윈도우 목록)를
    포함. 시간(t_start/t_end)은 윈도우가 정의하며 초 단위로 병기된다.
    """
    lo, hi = vocab["difficulty_rubric"]["scale"]

    def enum_or_null(name_en):
        return {"type": ["string", "null"], "enum": codes(vocab, name_en) + [None]}

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "meta_tagging_segment_v0.7.1",
        "type": "object",
        "additionalProperties": False,
        "required": ["clip_id", "segment_id", "t_start", "t_end", "subject"],
        "properties": {
            "clip_id": {"type": "string"},
            "segment_id": {"type": "integer", "minimum": 1},
            "t_start": {"type": "number", "minimum": 0, "maximum": CLIP_MAX_SEC},
            "t_end": {"type": "number", "minimum": 0, "maximum": CLIP_MAX_SEC},
            "frame_start": {"type": ["integer", "null"], "minimum": 0},
            "frame_end": {"type": ["integer", "null"], "minimum": 0},
            "windows": {"type": "array", "items": {"type": "integer"}},
            "event_kind": {"type": ["string", "null"]},
            "subject": {"type": "string", "enum": SUBJECT_ENUM},
            "ego_action": enum_or_null(AX_EGO_ACTION),
            "object_type": enum_or_null(AX_OBJECT_TYPE),
            "role": {"type": ["string", "null"], "enum": ROLE_ENUM + [None]},
            "longitudinal_action": enum_or_null(AX_LON_ACTION),
            "relation": enum_or_null(AX_RELATION),
            "vru_detail": enum_or_null(AX_VRU),
            "difficulty": {"type": ["integer", "null"], "minimum": lo, "maximum": hi},
        },
    }


def load_or_build_schema(vocab: dict, write: bool = True) -> dict:
    """스키마 파일이 있으면 로드, 없으면 어휘에서 생성 후 파일로 저장."""
    p = _find(SCHEMA_FILENAME)
    if p is not None:
        return json.loads(p.read_text(encoding="utf-8"))
    schema = build_segment_schema(vocab)
    if write:
        out = Path(__file__).parent / SCHEMA_FILENAME
        out.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return schema


# --- 프롬프트 ---------------------------------------------------------------
DEFAULT_FRAMES_DESC = (
    "The input is a 20-second driving clip, given as frames sampled uniformly "
    "over the whole clip in temporal order."
)
MONTAGE_FRAMES_DESC = (
    "The input is a 20-second driving clip, given as ONE grid montage of frames "
    "sampled uniformly over the whole clip. Tiles run left-to-right, top-to-bottom "
    "in time order; each tile is labeled with its frame number and timestamp in "
    "seconds. Use those timestamps to set t_start / t_end."
)


def build_prompt(
    vocab: dict, schema: dict, frames_desc: str = MONTAGE_FRAMES_DESC
) -> str:
    """segment 기반 동적 거동 태깅 시스템 프롬프트."""
    return f"""You tag dynamic driving behaviors in a 20s clip. Use ONLY codes from VOCABULARY.

{frames_desc}

Segment the clip into 5-10s windows; each window (segment) captures ONE dynamic
behavior EVENT — either an ego maneuver (subject=ego) or an ego-affecting agent
maneuver (subject=agent). This is NOT per-object listing: only cut a segment where
a dynamic ego behavior or an ego-affecting behavior actually occurs. A clip may
yield zero, one, or several segments, and segments may overlap in time (e.g. an
agent cut-in and the ego stop that reacts to it).

Field rules:
- subject=ego  -> fill ego_action; object_type/role/longitudinal_action/relation/vru_detail = null
- subject=agent -> fill object_type + role; ego_action = null
    - role=preceding_vehicle -> use longitudinal_action; relation = null
    - role=adjust or crossing -> use relation; longitudinal_action = null
    - object_type=pedestrian -> vru_detail may be added
- t_start/t_end in seconds within [0,20]; segment length 5-10s.
- difficulty: integer 1-10 (1-2 easy ... 9-10 safety-critical) or null.

Think in <think></think> using at most 5 short sentences — do NOT enumerate every
option or deliberate at length. Then immediately output ONLY a JSON array of segment
records inside <answer></answer>. Each record MUST conform to SEGMENT_SCHEMA. If no
dynamic behavior is present, output <answer>[]</answer>. Always close </think> and
emit <answer> before you run out of space.

VOCABULARY:
{json.dumps(vocab, ensure_ascii=False)}

SEGMENT_SCHEMA:
{json.dumps(schema, ensure_ascii=False)}
"""


# --- 검증 -------------------------------------------------------------------
def validate_segments(
    records: list, vocab: dict, schema: dict, enforce_length: bool = False
) -> tuple[list, list[str]]:
    """세그먼트 레코드 목록을 검증.

    JSON Schema(enum·필수·타입) + subject 별 필드 채움 규칙.
    enforce_length=True 일 때만 문서의 5~10s 길이 규칙을 경고로 남긴다(기본 완화).
    반환: (레코드 목록, 경고 목록). 위반은 경고로만 남긴다.
    """
    import jsonschema

    warns: list[str] = []
    if not isinstance(records, list):
        return [], ["<answer> 가 JSON 배열이 아님"]

    validator = jsonschema.Draft7Validator(schema)
    for i, rec in enumerate(records):
        tag = f"seg[{i}]"
        if not isinstance(rec, dict):
            warns.append(f"{tag} 레코드가 객체가 아님")
            continue
        for err in sorted(validator.iter_errors(rec), key=lambda e: list(e.path)):
            loc = ".".join(str(x) for x in err.path) or "(root)"
            warns.append(f"{tag}.{loc} 스키마 위반: {err.message}")

        # 길이 5~10s (frame 국소화에서는 보통 그보다 짧으므로 기본 비강제)
        if enforce_length:
            ts, te = rec.get("t_start"), rec.get("t_end")
            if isinstance(ts, (int, float)) and isinstance(te, (int, float)):
                dur = te - ts
                if not (5 <= dur <= 10):
                    warns.append(f"{tag} 길이 {dur:.1f}s (5~10s 범위 밖)")

        # subject 별 필드 채움 규칙
        subj = rec.get("subject")
        ego_only = ["object_type", "role", "longitudinal_action", "relation", "vru_detail"]
        if subj == "ego":
            if not rec.get("ego_action"):
                warns.append(f"{tag} subject=ego 인데 ego_action 비어있음")
            for f in ego_only:
                if rec.get(f) is not None:
                    warns.append(f"{tag} subject=ego 인데 {f} 채워짐(null 이어야 함)")
        elif subj == "agent":
            if rec.get("ego_action") is not None:
                warns.append(f"{tag} subject=agent 인데 ego_action 채워짐")
            if not rec.get("object_type"):
                warns.append(f"{tag} subject=agent 인데 object_type 비어있음")
            role = rec.get("role")
            if role not in ROLE_ENUM:
                warns.append(f"{tag} subject=agent 인데 role 누락/미허용: {role!r}")
            elif role == "preceding_vehicle":
                if not rec.get("longitudinal_action"):
                    warns.append(f"{tag} role=preceding_vehicle 인데 longitudinal_action 비어있음")
                if rec.get("relation") is not None:
                    warns.append(f"{tag} role=preceding_vehicle 인데 relation 채워짐")
            elif role in ("adjust", "crossing"):
                if not rec.get("relation"):
                    warns.append(f"{tag} role={role} 인데 relation 비어있음")
                if rec.get("longitudinal_action") is not None:
                    warns.append(f"{tag} role={role} 인데 longitudinal_action 채워짐")
            if rec.get("vru_detail") is not None and rec.get("object_type") != "pedestrian":
                warns.append(f"{tag} vru_detail 은 pedestrian 에만 유효")

    return records, warns

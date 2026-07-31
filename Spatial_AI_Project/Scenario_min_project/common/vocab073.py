"""meta_tagging v0.7.3 로더 · 앵커 매핑 · role 파생 · normalization · 검증.

schema/ 의 3파일 사용:
  meta_tagging_vocab_v0.7.3.json       (어휘 + cause/signal_state/occlusion/role_derivation/normalization/freetext_reject)
  meta_tagging_opt1_model_output_v0.7.3.json  (모델 <answer> guided_json 스키마)
  meta_tagging_seg_opt1_schema_v0.7.3.json    (최종 저장 레코드 스키마)
"""

import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent / "schema"

VOCAB = json.loads((SCHEMA_DIR / "meta_tagging_vocab_v0.7.3.json").read_text(encoding="utf-8"))
MODEL_OUTPUT_SCHEMA = json.loads((SCHEMA_DIR / "meta_tagging_opt1_model_output_v0.7.3.json").read_text(encoding="utf-8"))
SEG_SCHEMA = json.loads((SCHEMA_DIR / "meta_tagging_seg_opt1_schema_v0.7.3.json").read_text(encoding="utf-8"))

# guided_json 용 (draft-07 서브셋: type/props/required/additionalProperties)
GUIDED = {k: MODEL_OUTPUT_SCHEMA[k] for k in
          ("type", "additionalProperties", "required", "properties")}

ROLE_DERIVATION = VOCAB["role_derivation"]
NORMALIZATION = VOCAB["normalization"]
FREETEXT_REJECT = set(VOCAB["freetext_reject"])
CAUSE_ENUM = VOCAB["cause"]["enum"]
SIGNAL_ENUM = VOCAB["signal_state"]["enum"]
OCCLUSION_ENUM = VOCAB["occlusion"]["enum"]


def _axis_codes(name_en):
    for ax in VOCAB["axes"]:
        if ax["name_en"] == name_en:
            return [l["code"] for l in ax["labels"]]
    return []


OBJECT_TYPES = _axis_codes("object_type")
LON_ACTIONS = _axis_codes("longitudinal_action")
RELATIONS = _axis_codes("relation_wrt_ego")
EGO_ACTIONS = _axis_codes("ego_action")


# --- ego_action 앵커 매핑 (egomotion 이벤트 → ego_action, RULE) --------------
# reactive 단어 제거(ego_stop/ego_evade). ego_follow ∪ ego_lane_keep 병합
# (추종/순항/감속 구별 불가 → 모두 ego_lane_keep). 정지만 ego_stop, 회전만 turn.
def anchor_ego_action(event_kind: str, has_lead: bool = False) -> str:
    if event_kind == "stop":
        return "ego_stop"
    if event_kind == "turn_left":
        return "ego_turn_left"
    if event_kind == "turn_right":
        return "ego_turn_right"
    if event_kind == "evade":
        return "ego_evade"
    # decelerate / accelerate / baseline / 기타 → 병합된 lane_keep
    return "ego_lane_keep"


def derive_role(longitudinal_action, relation):
    """behavior → role (role_derivation). 모델은 role 미생성."""
    if relation and relation in ROLE_DERIVATION:
        return ROLE_DERIVATION[relation]
    if longitudinal_action and longitudinal_action in ROLE_DERIVATION:
        return ROLE_DERIVATION[longitudinal_action]
    return None


def normalize_field(field: str, value):
    """normalization 맵으로 동의어/오타 교정. (교정된 {field:value} dict 반환 or None)."""
    if value in NORMALIZATION:
        return NORMALIZATION[value]  # 예: {"object_type":"vehicle"} 또는 {"cause":"signal"}
    return None


def assemble_record(anchor: dict, model_out: dict) -> tuple[dict, list]:
    """앵커 + 모델출력 → 최종 seg 레코드. normalization·fill_rules·role 파생 적용.

    anchor: clip_id, segment_id, key_frame_t, t_start, t_end, ego_action, difficulty?
    반환: (record, notes[])
    """
    notes = []
    m = dict(model_out)

    # 1) normalization: 모델이 낸 값이 정규화 맵에 있으면 교정 (필드 이동 포함)
    cause = m.get("cause")
    for f in ("object_type", "longitudinal_action", "relation"):
        v = m.get(f)
        norm = normalize_field(f, v) if v else None
        if norm:
            for nf, nv in norm.items():
                if nf == f:
                    m[f] = nv
                else:  # 필드 이동 (예: traffic_light → cause=signal)
                    if nf == "cause":
                        m["cause"] = nv; cause = nv; m[f] = None
                    else:
                        m[nf] = nv; m[f] = None
            notes.append(f"normalize {v!r}→{norm}")

    cause = m.get("cause")
    lon = m.get("longitudinal_action")
    rel = m.get("relation")

    # 2) relation ↔ longitudinal_action 상호배타
    if lon and rel:
        # 선행차 종방향이면 lon 유지, 아니면 relation 유지
        if lon in LON_ACTIONS and rel not in RELATIONS:
            m["relation"] = None; rel = None
        else:
            m["longitudinal_action"] = None; lon = None
        notes.append("mutual-exclusive lon↔rel 조정")

    # 3) cause 별 fill rules
    if cause == "agent":
        m["signal_state"] = None
    elif cause == "signal":
        # 신호는 '유무'만 (색 판정 정확도 낮아 signal_state 폐기 → null)
        for f in ("object_type", "vru_attr", "occlusion",
                  "longitudinal_action", "relation", "vru_state", "signal_state"):
            m[f] = None
    else:  # road_geometry | other
        for f in ("object_type", "vru_attr", "occlusion", "longitudinal_action",
                  "relation", "vru_state", "signal_state"):
            m[f] = None

    # 4) role 파생
    role = derive_role(m.get("longitudinal_action"), m.get("relation")) if cause == "agent" else None

    rec = {
        "clip_id": anchor["clip_id"],
        "segment_id": anchor["segment_id"],
        "key_frame_t": round(anchor["key_frame_t"], 2),
        "t_start": round(anchor["t_start"], 2),
        "t_end": round(anchor["t_end"], 2),
        "ego_action": anchor["ego_action"],
        "cause": cause,
        "object_type": m.get("object_type"),
        "vru_attr": m.get("vru_attr"),
        "occlusion": m.get("occlusion"),
        "role": role,
        "longitudinal_action": m.get("longitudinal_action"),
        "relation": m.get("relation"),
        "vru_state": m.get("vru_state"),
        "signal_state": m.get("signal_state"),
        "difficulty": anchor.get("difficulty"),
    }
    return rec, notes


def validate_record(rec: dict) -> list[str]:
    """seg 스키마(draft-07) 검증 → 경고 목록."""
    import jsonschema
    v = jsonschema.Draft7Validator(SEG_SCHEMA)
    return [f"seg[{rec.get('segment_id')}].{'.'.join(map(str,e.path)) or '(root)'}: {e.message}"
            for e in v.iter_errors(rec)]

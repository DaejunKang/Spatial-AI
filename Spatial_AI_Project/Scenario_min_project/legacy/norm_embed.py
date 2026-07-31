"""임베딩 기반 정규화 — 자유기술(한국어/영어) → 정규 어휘 코드.

키워드 매칭 대신 다국어 문장임베딩 코사인 유사도로 매핑(재현율↑).
정규 어휘 설명은 vocab073(label_ko + note)에서 자동 구성.
"""

import numpy as np

from vocab073 import VOCAB

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None
_refs = {}  # axis -> (codes[], matrix[NxD])

# 영문 레퍼런스 (모델이 영어로 서술 → 영어로 매칭). 과매칭 억제 위해 임계 높임.
_OBJ_DESC = {
    "vehicle": "a car, passenger vehicle, sedan, SUV or van",
    "large_vehicle": "a truck, bus, trailer or large heavy construction vehicle",
    "motorcycle": "a motorcycle or motorbike",
    "bicycle_micromobility": "a bicycle, cyclist, scooter or micromobility rider",
    "pedestrian": "a pedestrian, a person walking on foot",
    "animal": "an animal such as a dog or deer",
    "emergency_vehicle": "an emergency vehicle: ambulance, police car or fire truck",
    "other": "a traffic cone, bollard, debris or fallen object on the road",
}
_REL_DESC = {
    "cross_ego_path": "an object crossing the ego vehicle's path",
    "oncoming_cross": "an oncoming object crossing into the ego path",
    "merge": "a vehicle merging into the ego lane",
    "brake_check": "the lead vehicle suddenly brake-checking",
    "block_lane": "a stopped or parked object blocking the lane",
    "door_open": "a parked car opening its door",
    "board_alight": "passengers boarding or alighting a stopped vehicle",
    "cut_in": "a vehicle cutting in front of the ego vehicle",
    "cut_out": "the lead vehicle cutting out of the lane",
    "sudden_cut_in": "a vehicle aggressively and suddenly cutting in",
    "alongside_parallel": "a vehicle driving alongside in parallel",
    "oncoming_narrow": "oncoming traffic on a narrow road requiring yielding",
}
_CAUSE_DESC = {
    "agent": "caused by a road user (vehicle, pedestrian, cyclist)",
    "signal": "caused by a traffic light or signal",
    "road_geometry": "caused by road curvature, an intersection or road geometry",
    "other": "no specific external cause; self-initiated cruising",
}


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _axis_desc(name_en):
    for ax in VOCAB["axes"]:
        if ax["name_en"] == name_en:
            return {l["code"]: f"{l.get('label_ko','')} {l.get('note') or ''}"
                    for l in ax["labels"] if l.get("gt") is not False or True}
    return {}


_AXIS_SRC = {
    "object_type": lambda: _OBJ_DESC,
    "relation": lambda: _REL_DESC,
    "cause": lambda: _CAUSE_DESC,
}


def _ref(axis):
    if axis not in _refs:
        desc = _AXIS_SRC[axis]()
        codes = list(desc.keys())
        mat = _get_model().encode([desc[c] for c in codes], normalize_embeddings=True)
        _refs[axis] = (codes, np.asarray(mat))
    return _refs[axis]


def normalize(text: str, axis: str, thr: float = 0.42):
    """text 를 axis 정규어휘 중 최근접 코드로. 임계 미만이면 None."""
    if not text or not text.strip():
        return None, 0.0
    codes, mat = _ref(axis)
    e = _get_model().encode(text, normalize_embeddings=True)
    sims = mat @ e
    i = int(np.argmax(sims))
    return (codes[i], float(sims[i])) if sims[i] >= thr else (None, float(sims[i]))


def component_tags(text: str) -> list:
    """자유기술 → [object_type:*, relation:*] (임베딩)."""
    tags = []
    ot, s1 = normalize(text, "object_type", 0.50)   # 임계↑: 과매칭(환경묘사→객체) 억제
    if not ot:
        return tags   # 객체가 아니면(신호·도로·환경 서술) relation 부여 안 함
    tags.append(f"object_type:{ot}")
    rel, s2 = normalize(text, "relation", 0.52)      # agent 일 때만 relation
    if rel:
        tags.append(f"relation:{rel}")
    return tags


def infer_cause(texts: list, thr: float = 0.40) -> str | None:
    """여러 기술을 합쳐 cause 추정(보조용)."""
    joined = " ".join(t for t in texts if t)
    c, s = normalize(joined, "cause", thr)
    return c

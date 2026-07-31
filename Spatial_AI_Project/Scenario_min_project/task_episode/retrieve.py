# -*- coding: utf-8 -*-
"""Phase B — 정규화·병합·confidence 랭킹 (검색 인덱스화).

Phase A 후보(OR 합집합)를 검색 가능한 형태로:
  - 정규화(별칭·슬롯 흡수) — guided-decode라 슬롯오류 적음, 별칭만.
  - 병합: 태거가 못 구분하는 쌍(혼동 리포트 근거)을 검색용으로 통합.
    · road_urban_arterial + road_backstreet → road_surface (도심/이면 생활도로)
    · intersection_signalized + intersection_unsignalized → intersection (신호상태 구분은 Phase C에서 복원)
  - confidence: 채널·vote·multi로 후보 순위.
정밀 구분(sig/unsig, urban/backstreet)은 Phase C(precision·human)에서 회복.
"""

ALIAS = {}   # 슬롯/별칭 (guided-decode라 현재 비어있음. 필요시 추가)

MERGE = {
    "road_urban_arterial": "road_surface",
    "road_backstreet": "road_surface",
    "intersection_signalized": "intersection",
    "intersection_unsignalized": "intersection",
}
# 병합 대표 → 원본 세부(Phase C 복원용)
MERGE_SUB = {
    "road_surface": ["road_urban_arterial", "road_backstreet"],
    "intersection": ["intersection_signalized", "intersection_unsignalized"],
}


def normalize(cat):
    cat = ALIAS.get(cat, cat)
    return MERGE.get(cat, cat)


def confidence(info):
    """후보 confidence [0,1]: 채널 가중 + vote + multi 보너스."""
    ch = set(info.get("channels", []))
    vf = info.get("vote_fraction", 0) or 0.0
    s = 0.0
    if "ego" in ch:
        s += 0.45          # egomotion GT (거의 확정)
    if "gt" in ch:
        s += 0.40          # obj3d GT
    if "vlm" in ch:
        s += 0.50 * vf     # VLM 투표 비례
    if len(ch) >= 2:
        s += 0.25          # 다채널 합의
    return round(min(s, 1.0), 2)


def index_clip(cand_clip):
    """Phase A clip 결과 → 정규화·병합·랭킹된 검색 인덱스.
    반환: {clip_id, episodes:[{win, tags:[{cat, confidence, channels, sub?}...정렬]}]}
    """
    out = {"clip_id": cand_clip.get("clip_id"), "episodes": []}
    for ep in cand_clip.get("episodes", []):
        merged = {}   # norm_cat → {channels, vote_fraction(max), subs}
        for cat, info in ep["candidates"].items():
            nc = normalize(cat)
            m = merged.setdefault(nc, {"channels": set(), "vote_fraction": 0.0, "subs": set()})
            m["channels"] |= set(info.get("channels", []))
            m["vote_fraction"] = max(m["vote_fraction"], info.get("vote_fraction", 0) or 0.0)
            if nc != cat:
                m["subs"].add(cat)
        tags = []
        for nc, m in merged.items():
            info = {"channels": sorted(m["channels"]), "vote_fraction": round(m["vote_fraction"], 2)}
            t = {"cat": nc, "confidence": confidence(info), "channels": info["channels"]}
            if m["subs"]:
                t["sub"] = sorted(m["subs"])       # Phase C 세부복원 후보
            tags.append(t)
        tags.sort(key=lambda t: -t["confidence"])
        out["episodes"].append({"win": ep["win"], "arc": ep.get("arc", []),
                                "ego_action": ep.get("ego_action"), "tags": tags})
    return out

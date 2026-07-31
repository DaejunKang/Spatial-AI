# -*- coding: utf-8 -*-
"""Phase B — 정규화·병합·confidence 랭킹 (검색 인덱스화).

Phase A 후보(OR 합집합)를 검색 가능한 형태로:
  - 정규화(별칭·슬롯 흡수) — guided-decode라 슬롯오류 적음, 별칭만.
  - 병합 규칙: (a) 판단이 애매하고 ∧ (b) 하류 query/cause 가치가 없을 때만 통합.
    · road_urban_arterial + road_backstreet → road_surface (도로유형: 애매+query가치 낮음 → 병합 OK)
    · 교차로 sig/unsig 는 **병합 금지** — cause 와 결부(비신호=양보 정차 vs 신호=신호 정지)라
      하류 query 가치가 있음. 별도 카테고리로 유지.
  - confidence: 채널·vote·multi로 후보 순위.
정밀 구분(urban/backstreet)만 Phase C(precision·human)에서 회복.
"""

ALIAS = {}   # 슬롯/별칭 (guided-decode라 현재 비어있음. 필요시 추가)

MERGE = {
    "road_urban_arterial": "road_surface",
    "road_backstreet": "road_surface",
    # 교차로 sig/unsig 는 cause 결부 → 병합하지 않음(하류 query 가치).
}
# 병합 대표 → 원본 세부(Phase C 복원용)
MERGE_SUB = {
    "road_surface": ["road_urban_arterial", "road_backstreet"],
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


def _cause_axis(ep):
    """cause 후보 → 랭킹된 cause 축. cause 는 큐레이션 1차 query 키(별도 축으로 노출).
    반환: [{cause, confidence, channels, from}...정렬]  (없으면 [])."""
    causes = []
    for cz, ci in ep.get("cause_candidates", {}).items():
        info = {"channels": ci.get("channels", []), "vote_fraction": ci.get("vote_fraction", 0.0)}
        causes.append({"cause": cz, "confidence": confidence(info),
                       "channels": info["channels"], "from": ci.get("from", [])})
    causes.sort(key=lambda c: -c["confidence"])
    return causes


def index_clip(cand_clip):
    """Phase A clip 결과 → 정규화·병합·랭킹된 검색 인덱스.
    반환: {clip_id, episodes:[{win, tags:[{cat, confidence, channels, sub?}...], cause:[...]}]}
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
                                "ego_action": ep.get("ego_action"), "tags": tags,
                                "cause": _cause_axis(ep)})   # cause 축(전이의 "왜")
    return out

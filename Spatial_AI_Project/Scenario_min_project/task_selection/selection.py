# -*- coding: utf-8 -*-
"""Task「선별」(Stage 1) — 의미있는 clip 선별 알고리즘 (모듈화).

raw 로그엔 CAN+video만 있음(obj3d/map 없음). 두 채널 결합 랭킹:
  ego arc 점수(egomotion, 무료·전수) + VLM 흥미도(몽타주, 저비용) → 결합 → 상위 K.
recall 위주 triage. 이전 job tmp 스크립트(select_prep/run_select_vlm/build_select_review)를 통합.

공용 기반 사용: events(egomotion) · dataset(몽타주) · config(MODEL/TEMPERATURE) · paths.
"""
import json

import events
from config import MODEL, TEMPERATURE
from dataset import jpeg_to_data_uri, sample_montage

# VLM 흥미도 guided_json
_CATS = ["turn_maneuver", "intersection", "stop_or_yield", "cut_in", "pedestrian_or_cyclist",
         "lane_change", "congestion", "hazard_nearmiss", "unusual_scene", "routine_cruise"]
_SCHEMA = {"type": "object", "additionalProperties": False,
           "required": ["interesting", "score", "category", "reason"],
           "properties": {"interesting": {"type": "boolean"}, "score": {"type": "number"},
                          "category": {"type": "string", "enum": _CATS},
                          "reason": {"type": "string", "maxLength": 160}}}
_PROMPT = (
    "These are frames (time order) from a 20-second driving clip (front camera). "
    "Decide if this clip is a NOTABLE / long-tail driving situation worth extracting for a dataset "
    "(turn, intersection negotiation, stop/yield, cut-in, pedestrian/cyclist, lane change, congestion, "
    "hazard/near-miss, unusual scene), versus ROUTINE straight lane-keeping/cruising. "
    "Return JSON: interesting (true if notable), score 0-1 (how notable/rare), category (best coarse "
    "label; routine_cruise if not notable), reason (short). Be calibrated: plain cruising = low score.")


def ego_score(clip_id):
    """egomotion arc 점수 (전수·무료). 급감속=agent 반응 대리신호 포함."""
    ev = events.detect_events(clip_id)
    eg = events.load_egomotion_clip(clip_id)
    if not ev.get("ok"):
        return {"score": 0.0, "kinds": [], "harsh_decel": 0.0}
    kinds = [e["kind"] for e in ev["events"]]
    ks = set(kinds)
    turn = int(bool({"turn_left", "turn_right"} & ks))
    lc = int(bool({"lane_change_left", "lane_change_right"} & ks))
    stop = int("stop" in ks)
    min_ax = float(min(eg["ax"])) if eg["ok"] and len(eg["ax"]) else 0.0
    harsh = max(0.0, -min_ax - 3.0)                 # 3m/s^2 초과 급감속
    ndecel = sum(1 for k in kinds if k == "decelerate")
    score = 2 * turn + 2 * lc + 1 * stop + 1.5 * min(harsh, 3.0) + 0.4 * ndecel
    return {"score": round(score, 2), "kinds": sorted(ks), "harsh_decel": round(-min_ax, 1)}


def vlm_interest(client, clip_id):
    """VLM 몽타주 흥미도 → {interesting, score, category, reason} (실패 시 None)."""
    import paths as P
    try:
        uri = jpeg_to_data_uri(sample_montage(str(P.video_path(clip_id)), num_frames=12, cell_max_side=360))
        r = client.chat.completions.create(
            model=MODEL, temperature=TEMPERATURE, max_tokens=512,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": uri}}]}],
            extra_body={"guided_json": _SCHEMA})
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None


def combined_score(ego_norm, vlm_score):
    """결합 점수 = 강한 채널 우선 + 합의 보너스. 한 채널이라도 강하면 유의미."""
    return max(ego_norm, vlm_score) + 0.3 * min(ego_norm, vlm_score)


def rank_clips(clip_ids, client_pool, top_k=None):
    """clip 목록 → 결합점수 내림차순 랭킹. 각 원소에 ego/vlm/combined·provenance.

    반환: [{clip_id, ego, vlm, combined}]  (top_k 지정 시 상위 K).
    """
    ego = {c: ego_score(c) for c in clip_ids}
    max_ego = max((ego[c]["score"] for c in clip_ids), default=1.0) or 1.0
    rows = []
    for i, cid in enumerate(clip_ids):
        v = vlm_interest(client_pool[i % len(client_pool)], cid) or {}
        en = ego[cid]["score"] / max_ego
        vs = float(v.get("score", 0) or 0)
        rows.append({"clip_id": cid, "ego": ego[cid], "vlm": v,
                     "combined": round(combined_score(en, vs), 3)})
    rows.sort(key=lambda r: -r["combined"])
    return rows[:top_k] if top_k else rows

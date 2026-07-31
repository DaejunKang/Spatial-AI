# -*- coding: utf-8 -*-
"""VLM 검증/분류 계층 — 넓은 GT 후보를 VLM으로 좁히고, VLM전용 맥락을 산출.

에피소드 구간별로 subclip을 VLM에 보내 guided-decode:
  (1) present 상호작용/기동세부: GT 후보를 힌트로 주되, 실제 보이는 것만 확정(FP 제거 + 놓침 보완)
  (2) 도로환경(road_env) 단일 분류
  (3) 신호/비신호 교차로
  (4) 기타 맥락(roundabout/merge/construction/toll) + red_light/signal_go
클립 단위 taxonomy present 집합을 반환. gold 대비 재평가용.
"""
import json
import tempfile
from pathlib import Path

from config import MODEL, SEND_FPS, TEMPERATURE, WINDOW_MAX_SIDE
from dataset import to_data_uri, video_meta, write_subclip
import events
import classify073 as C
import taxo_detect as D

LEAD_IN = 3.0

# VLM이 검증할 '의미적' 상호작용만. ego 운동학(close_follow/creep/decel)은 GT 전담 → 제외.
_INTER = {
    "cut_in": "an adjacent-lane vehicle moves FULLY INTO ego's lane just ahead (completes the merge)",
    "cut_in_attempt": "an adjacent vehicle edges toward/into ego's lane but does NOT fully complete",
    "cut_out": "a vehicle ahead in ego's lane LEAVES the lane sideways",
    "lead_decel": "the vehicle ahead decelerates (any noticeable slow-down, gentle to hard)",
    "ped_crossing": "a pedestrian crosses ego's path",
    "cyclist_pm_near": "a cyclist or PM(kickboard) rider close to ego's path",
    "oncoming_encroach": "an oncoming vehicle crosses the centerline toward ego's lane",
    "agent_yields_to_ego": "a cross/oncoming vehicle STOPS to yield while EGO proceeds (unsignalized)",
    "ego_yields_to_agent": "EGO stops/slows to yield to a crossing/oncoming vehicle or pedestrian",
    "lane_change_left": "ego changes to the left lane",
    "lane_change_right": "ego changes to the right lane",
}
_ROAD = ["highway", "urban_arterial", "backstreet", "rural", "tunnel", "bridge", "parking", "unknown"]
_ROAD_DESC = ("highway=freeway/expressway (no cross traffic, high speed); "
              "urban_arterial=city road WITH painted lane markings and multiple lanes; "
              "backstreet=NARROW residential street with NO/faint lane markings, parked cars "
              "(if painted lane lines are clearly present, it is urban_arterial NOT backstreet); "
              "rural=countryside road with fields/trees, few buildings; tunnel; bridge; "
              "parking=INSIDE a parking lot with parking stalls/bays (NOT a through-road; a narrow "
              "street with moving traffic is backstreet, NOT parking); unknown when unclear")
_INTX = ["signalized", "unsignalized", "none"]
_EXTRA = ["roundabout", "merge_onramp", "construction_cones", "toll_gate"]
# road_env VLM값 → taxonomy 키
_ROAD_MAP = {"highway": "road_highway", "urban_arterial": "road_urban_arterial",
             "backstreet": "road_backstreet", "rural": "road_rural", "tunnel": "road_tunnel",
             "bridge": "road_bridge", "parking": "road_parking", "unknown": None}


def _schema():
    return {
        "type": "object", "additionalProperties": False,
        "required": ["present", "road_env", "intersection", "extras", "red_light_stop", "signal_go"],
        "properties": {
            "present": {"type": "array", "items": {"type": "string", "enum": list(_INTER)}},
            "road_env": {"type": "string", "enum": _ROAD},
            "intersection": {"type": "string", "enum": _INTX},
            "extras": {"type": "array", "items": {"type": "string", "enum": _EXTRA}},
            "red_light_stop": {"type": "boolean"},
            "signal_go": {"type": "boolean"},
        }}


def _prompt(cands, arc):
    deflines = "\n".join(f"- {k}: {v}" for k, v in _INTER.items())
    hint = ", ".join(sorted(cands)) if cands else "(none)"
    return (
        "You are verifying driving events in this video segment. Ego kinematics (from GT): "
        f"arc = {'+'.join(arc) if arc else 'straight'}.\n"
        f"GT motion detectors flagged these POSSIBLE events (may be false positives): {hint}.\n"
        "STRICT RULE: include an event in `present` ONLY if it is CLEARLY and unambiguously visible "
        "in the video. When uncertain, DO NOT include it (default to excluding). Reject flagged events "
        "that you cannot clearly confirm.\n\n"
        f"Event definitions:\n{deflines}\n\n"
        "Also classify the static context (be conservative):\n"
        f"- road_env (exactly one): {_ROAD_DESC}\n"
        "- intersection: signalized (traffic lights visible) / unsignalized (a junction with NO lights) / none\n"
        f"- extras (leave EMPTY unless clearly present): roundabout=a TRUE circular junction with a "
        "central island (NOT a curve or a normal turn); merge_onramp=highway merge/on-ramp; "
        "construction_cones=ONLY if you clearly see orange traffic cones, barriers, or road workers "
        "(normal curbs, poles, parked cars, guardrails are NOT construction — if unsure, leave empty); "
        "toll_gate=toll booth/barrier.\n"
        "- red_light_stop: ego stopped for a RED light; signal_go: ego started on GREEN.\n"
        "Return JSON only.")


# 융합용 카테고리 집합
SEM = set(_INTER)                    # 의미 상호작용 — GT후보를 VLM이 '확인만'
CTX = {"road_highway", "road_urban_arterial", "road_backstreet", "road_rural", "road_tunnel",
       "road_bridge", "road_parking", "intersection_signalized", "intersection_unsignalized",
       "roundabout", "merge_onramp", "construction_cones", "toll_gate", "red_light_stop", "signal_go"}
KIN = {"close_follow", "creep", "vru_roadside"}   # GT 전담(운동학 + obj3d 존재)


def fuse(gt_cats, vlm_cats):
    """확인전용(∩) 융합.
      - 의미 상호작용: VLM은 GT 후보를 확인만(추가 금지) → gt ∩ vlm
      - 맥락: VLM 권위
      - ego 운동학(close_follow/creep): GT 전담
      - decel_at_intersection: GT 감속 ∧ VLM 교차로 확인
    """
    gt, vlm = set(gt_cats), set(vlm_cats)
    out = (vlm & SEM & gt) | (vlm & CTX) | (gt & KIN)
    if "decel_at_intersection" in gt and (vlm & {"intersection_signalized", "intersection_unsignalized"}):
        out.add("decel_at_intersection")
    return out


def ground_signals(cats, clip_id, w0, w1):
    """red_light_stop/signal_go를 egomotion으로 그라운딩(모순·환각 제거).
      - red_light_stop: 창 안에서 ego가 실제 정지(저속)해야 유지.
      - signal_go: 정지 상태에서 가속 출발(뒤가 앞보다 빠름)해야 유지.
    """
    cats = set(cats)
    if not ({"red_light_stop", "signal_go"} & cats):
        return cats
    eg = events.load_egomotion_clip(clip_id)
    if not eg["ok"]:
        return cats
    import numpy as np
    m = (eg["t"] >= w0) & (eg["t"] <= w1)
    if m.sum() < 2:
        return cats
    sp = eg["speed"][m]
    has_stop = float(sp.min()) < 1.5
    n = len(sp)
    rising = float(sp[-max(1, n // 3):].mean()) > float(sp[:max(1, n // 3)].mean()) + 1.0
    if "red_light_stop" in cats and not has_stop:
        cats.discard("red_light_stop")
    if "signal_go" in cats and not (has_stop and rising):
        cats.discard("signal_go")
    return cats


def verify_clip(client, path, clip_id):
    """클립의 VLM 검증/분류 → {clip_id, ok, cats:set, episodes:[...]}."""
    try:
        dur = video_meta(path)["duration_s"]
    except Exception as e:
        return {"clip_id": clip_id, "ok": False, "error": str(e), "cats": set()}
    ev = events.detect_events(clip_id)
    eps = C.consolidate_episodes(ev["events"]) if ev.get("ok") else []
    gt = D.detect_taxonomy(clip_id)
    # GT 후보를 에피소드 창별로 배분
    def cands_in(w0, w1):
        return set(d["cat"] for d in gt["detail"] if not (d["t1"] < w0 or d["t0"] > w1))

    windows = []
    if eps:
        for ep in eps:
            windows.append((max(0.0, ep["onset"] - LEAD_IN), min(dur, ep["t1"] + 1.0), ep["kinds"]))
    else:
        windows.append((0.0, dur, []))

    cats = set()
    ep_out = []
    tmp = Path(tempfile.mkdtemp(prefix="vlmv_"))
    try:
        for wi, (w0, w1, arc) in enumerate(windows):
            cands = cands_in(w0, w1)
            sub = tmp / f"w{wi}.mp4"
            write_subclip(path, w0, w1, sub, WINDOW_MAX_SIDE, SEND_FPS)
            uri = to_data_uri(sub); sub.unlink(missing_ok=True)
            v = None
            for mt in (1024, 2048):
                r = client.chat.completions.create(
                    model=MODEL, temperature=TEMPERATURE, max_tokens=mt,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": _prompt(cands, arc)},
                        {"type": "video_url", "video_url": {"url": uri}}]}],
                    extra_body={"guided_json": _schema()})
                try:
                    v = json.loads(r.choices[0].message.content); break
                except Exception:
                    if r.choices[0].finish_reason != "length":
                        break
            if not v:
                ep_out.append({"win": [round(w0, 1), round(w1, 1)], "error": "vlm_parse"}); continue
            wc = set(v.get("present", []))
            # 맥락 → taxonomy 키
            rk = _ROAD_MAP.get(v.get("road_env"))
            if rk: wc.add(rk)
            ix = v.get("intersection")
            if ix == "signalized": wc.add("intersection_signalized")
            elif ix == "unsignalized": wc.add("intersection_unsignalized")
            for e in v.get("extras", []): wc.add(e)
            if v.get("red_light_stop"): wc.add("red_light_stop")
            if v.get("signal_go"): wc.add("signal_go")
            wc = ground_signals(wc, clip_id, w0, w1)   # 신호 egomotion 그라운딩
            cats |= wc
            ep_out.append({"win": [round(w0, 1), round(w1, 1)], "cats": sorted(wc)})
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return {"clip_id": clip_id, "ok": True, "cats": cats, "episodes": ep_out}

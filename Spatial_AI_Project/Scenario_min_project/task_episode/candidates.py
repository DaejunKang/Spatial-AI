# -*- coding: utf-8 -*-
"""Phase A — recall-우선 후보 생성 (OR 앙상블).

에피소드 단위로 3채널 합집합을 후보로:
  (ego)  ego-arc 기동      — events.detect_events → auto_tags_from_arc
  (gt)   obj3d GT 검출기   — taxo_detect.detect_taxonomy detail (창 오버랩)
  (vlm)  VLM self-consistency n-vote (temp>0) — 1표라도 나오면 후보

각 후보에 provenance(어느 채널)·vote_fraction(VLM)·multi(다채널 동시)를 부착.
⚠️ ∩(확인전용)·precision 게이트 미적용 — recall 최대화. precision은 Phase C.
"""
import json
import tempfile
from collections import Counter
from pathlib import Path

from config import MODEL
from dataset import to_data_uri, video_meta, write_subclip
import events
import taxonomy
import classify073 as C
import taxo_detect as D
import vlm_verify as VV

VOTE_N = 5
VOTE_TEMP = 0.7
LEAD_IN = 3.0

# cause 축(전이의 "왜") 후보 매핑 — 각 present 카테고리를 cause 로 사상.
# recall-first: 단일 확정이 아니라 후보 집합(provenance 부착). 정밀 단일해소는 Phase C.
# 근거(CLAUDE.md §2): cause = ego 전이의 답, 큐레이션 1차 query 키 → Track2 인덱스 필수.
_CAUSE_OF = {
    # agent 상호작용(진행로 상대) — 단, GT+맥락 필요(존재≠원인)이나 recall 후보로 부착
    "cut_in": "agent", "cut_in_attempt": "agent", "cut_out": "agent",
    "lead_decel": "agent", "close_follow": "agent", "ped_crossing": "agent",
    "vru_roadside": "agent", "cyclist_pm_near": "agent", "oncoming_encroach": "agent",
    "agent_yields_to_ego": "agent", "ego_yields_to_agent": "agent",
    # signal — 신호등/신호 상태
    "red_light_stop": "signal", "signal_go": "signal", "intersection_signalized": "signal",
    # road_geometry — 도로 형상(교차로 협상/합류/회전/분기)
    "intersection_unsignalized": "road_geometry", "roundabout": "road_geometry",
    "merge_onramp": "road_geometry", "turn_left": "road_geometry",
    "turn_right": "road_geometry", "u_turn": "road_geometry",
}


def _cause_candidates(cand):
    """present 카테고리 dict → cause 후보 {cause:{channels, vote_fraction, from}}.
    provenance(channels)·vf 는 기여 카테고리에서 승계. 증거 없으면 other."""
    cc = {}
    for cat, e in cand.items():
        cz = _CAUSE_OF.get(cat)
        if not cz:
            continue
        m = cc.setdefault(cz, {"channels": set(), "vote_fraction": 0.0, "from": []})
        m["channels"] |= set(e.get("channels", []))
        m["vote_fraction"] = max(m["vote_fraction"], e.get("vote_fraction", 0.0) or 0.0)
        m["from"].append(cat)
    if not cc:
        cc["other"] = {"channels": set(), "vote_fraction": 0.0, "from": []}
    # set → sorted list (JSON 직렬화)
    return {cz: {"channels": sorted(m["channels"]),
                 "vote_fraction": round(m["vote_fraction"], 2),
                 "from": sorted(m["from"])} for cz, m in cc.items()}


def _vlm_present(client, uri, cands, arc):
    """VLM 1회 호출 → present 상호작용 ∪ 맥락 카테고리 집합."""
    r = client.chat.completions.create(
        model=MODEL, temperature=VOTE_TEMP, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": VV._prompt(cands, arc)},
            {"type": "video_url", "video_url": {"url": uri}}]}],
        extra_body={"guided_json": VV._schema()})
    try:
        v = json.loads(r.choices[0].message.content)
    except Exception:
        return set()
    out = set(v.get("present", []))
    rk = VV._ROAD_MAP.get(v.get("road_env"))
    if rk:
        out.add(rk)
    ix = v.get("intersection")
    if ix == "signalized":
        out.add("intersection_signalized")
    elif ix == "unsignalized":
        out.add("intersection_unsignalized")
    for e in v.get("extras", []):
        out.add(e)
    if v.get("red_light_stop"):
        out.add("red_light_stop")
    if v.get("signal_go"):
        out.add("signal_go")
    out |= VV.env_cats(v)                              # 정적환경(조명/기상/노면/glare/crosswalk/신호등/비분리)
    return out


def generate_candidates(client_pool, path, clip_id, n_vote=VOTE_N):
    """clip → 에피소드별 후보 dict.
    반환: {clip_id, ok, dur, episodes:[{win, candidates:{cat:{channels:[...], vote_fraction, multi}}}]}
    """
    try:
        dur = video_meta(path)["duration_s"]
    except Exception as e:
        return {"clip_id": clip_id, "ok": False, "error": str(e)}
    ev = events.detect_events(clip_id)
    if not ev.get("ok"):
        return {"clip_id": clip_id, "ok": False, "error": ev.get("reason")}
    eps = C.consolidate_episodes(ev["events"])
    gt = D.detect_taxonomy(clip_id)                       # obj3d GT detail

    def gt_in(w0, w1):
        return set(d["cat"] for d in gt["detail"] if not (d["t1"] < w0 or d["t0"] > w1))

    if not eps:                                           # 순수 주행 → 전체창 1개
        eps = [{"onset": 0.0, "t1": dur, "kinds": [], "ego_action": "ego_lane_keep"}]

    out_eps = []
    tmp = Path(tempfile.mkdtemp(prefix="candA_"))
    try:
        for i, ep in enumerate(eps):
            w0 = max(0.0, ep["onset"] - LEAD_IN); w1 = min(dur, ep["t1"] + 1.0)
            # --- ego 채널: arc 기동 ---
            ego_cats = set(taxonomy.auto_tags_from_arc(ep["kinds"]))
            for k in ep["kinds"]:
                if k in ("lane_change_left", "lane_change_right"):
                    ego_cats.add(k)
            # --- gt 채널: obj3d 상호작용 ---
            gt_cats = gt_in(w0, w1)
            gt_only = gt_cats - ego_cats
            # --- vlm 채널: n-vote ---
            sub = tmp / f"e{i}.mp4"
            write_subclip(path, w0, w1, sub, 1280, 10)
            uri = to_data_uri(sub); sub.unlink(missing_ok=True)
            votes = Counter()
            hint = gt_cats                                 # GT 후보를 힌트로(확정 아님)
            for j in range(n_vote):
                cl = client_pool[j % len(client_pool)]
                votes.update(_vlm_present(cl, uri, hint, ep["kinds"]))
            # --- 합집합 병합 ---
            cand = {}
            for c in ego_cats:
                cand.setdefault(c, {"channels": []})["channels"].append("ego")
            for c in gt_only:
                cand.setdefault(c, {"channels": []})["channels"].append("gt")
            for c, cnt in votes.items():
                e = cand.setdefault(c, {"channels": []})
                e["channels"].append("vlm"); e["vote_fraction"] = round(cnt / n_vote, 2)
            for c, e in cand.items():
                e["multi"] = len(e["channels"]) >= 2       # 다채널 동시 = 고신뢰
            cause_cands = _cause_candidates(cand)          # cause 축(전이의 "왜") 후보
            out_eps.append({"win": [round(w0, 1), round(w1, 1)], "arc": ep["kinds"],
                            "ego_action": ep["ego_action"], "candidates": cand,
                            "cause_candidates": cause_cands})
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return {"clip_id": clip_id, "ok": True, "dur": dur, "episodes": out_eps}

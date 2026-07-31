"""v0.8 프로토타입 — Scene Description + Critical Components (자유기술 + 사후 정규화).

패러다임 전환: 이산 maneuver enum 강제 대신,
  1) GT(egomotion arc + obstacle in-path)로 '무엇이 critical한가' 근거 제공
  2) 모델은 자유 기술(scene_description / critical_components / chain_of_causation) — 강점 활용
  3) 사후에 유사 토큰을 정규 어휘로 매핑(tags[]) — 고정어휘 강제 생성 아님
윈도우는 lead-in 포함(접근 구간부터)해 maneuver 아크 전체를 담는다.
"""

import re
import tempfile
from pathlib import Path

from config import MODEL, SEND_FPS, TEMPERATURE, WINDOW_MAX_SIDE

V08_MAX_TOKENS = 4096  # 자유기술(scene/critical/coc)은 길어 넉넉히(다객체 구조화 truncate 방지)
from dataset import to_data_uri, video_meta, write_subclip
from events import detect_events, detect_obj3d_events
from classify073 import consolidate_episodes, _causal_agent

LEAD_IN = 3.0  # 에피소드 onset 이전 접근 구간(초) 포함

_ARC = {"stop": "stop", "decelerate": "decelerate", "accelerate": "accelerate",
        "turn_left": "turn left", "turn_right": "turn right", "evade": "evasive maneuver"}

# 사후 정규화: 자유기술 → 정규 어휘 (유사 토큰 매핑)
_OBJ_KW = {"pedestrian": ["보행자", "행인", "사람", "pedestrian", "person"],
           "vehicle": ["차량", "승용", "car", "vehicle", "sedan", "택시"],
           "large_vehicle": ["트럭", "버스", "대형", "truck", "bus", "폐기물"],
           "motorcycle": ["오토바이", "이륜", "motorcycle"],
           "bicycle_micromobility": ["자전거", "킥보드", "bicycle", "rider", "라이더"]}
_REL_KW = {"cut_in": ["끼어", "cut in", "cut-in", "진입", "cut_in"],
           "cross_ego_path": ["횡단", "가로질", "cross", "경로.*가로"],
           "block_lane": ["막", "차단", "block", "정차"],
           "oncoming_cross": ["대향", "마주", "oncoming"],
           "alongside_parallel": ["병렬", "나란", "parallel", "alongside"]}
_SIGNAL_KW = ["신호등", "traffic light", "적신호", "red light", "신호"]


def _norm_tags(text: str):
    """자유기술 텍스트 → 정규 어휘 태그 목록(유사 토큰 매핑)."""
    t = text.lower()
    tags = []
    for code, kws in _OBJ_KW.items():
        if any(re.search(k.lower(), t) for k in kws):
            tags.append(f"object_type:{code}"); break
    for code, kws in _REL_KW.items():
        if any(re.search(k.lower(), t) for k in kws):
            tags.append(f"relation:{code}"); break
    if any(k.lower() in t for k in _SIGNAL_KW):
        tags.append("cause:signal")
    return tags


def _gt_ref(o):
    return {"object_type": o["object_type"], "role": o.get("role"),
            "relation": o.get("relation"), "distance_m": o.get("min_dist"),
            "vru_state": o.get("vru_detail")}


def _gt_tags(o):
    t = [f"object_type:{o['object_type']}"]
    if o.get("relation"):
        t.append(f"relation:{o['relation']}")
    return t


def _gt_syn(o):
    """GT 근거로 합성한 서술(모델 미서술 시 주입용)."""
    return (f"{o['object_type']} in ego path ({o.get('role')}, "
            f"{o.get('relation') or ''} {o.get('min_dist')}m ahead)")


def _covers(comp_text, obj_type):
    kws = _OBJ_KW.get(obj_type, [])
    t = comp_text.lower()
    return any(re.search(k.lower(), t) for k in kws)


def _gt_hint(ep, agents):
    arc = " → ".join(_ARC.get(k, k) for k in ep["kinds"])
    if agents:
        lst = "; ".join(f"{i+1}) {o['object_type']} ({o.get('role')}, "
                        f"{o.get('relation') or ''} {o.get('min_dist')}m ahead)"
                        for i, o in enumerate(agents))
        a = (f" GT-detected in-path objects (you MUST address each; do not omit): {lst}.")
    else:
        a = " No GT in-path objects."
    return f"GT: ego maneuver arc = [{arc}].{a}"


from vocab073 import EGO_ACTIONS, OBJECT_TYPES, RELATIONS
_CAUSE_ENUM = ["agent", "signal", "road_geometry", "other"]
_VRU_ENUM = ["crossing", "about_to_cross", "walking_along", "stationary", None]

# maxLength/maxItems 로 출력 길이 강제 제한 (nano 모델 장황→truncate 방지)
# ego_action·cause 를 모델 이해(scene/coc)에서 추출 — rule 대신 의미 기반(arc/GT 는 grounding).
V08_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["ego_action", "cause", "scene_description", "critical_components", "chain_of_causation"],
    "properties": {
        "ego_action": {"type": "string", "enum": EGO_ACTIONS},
        "cause": {"type": "string", "enum": _CAUSE_ENUM},
        "scene_description": {"type": "string", "maxLength": 400},
        "ego_intent": {"type": "string", "maxLength": 150},
        "critical_components": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["description", "why_critical"],
            "properties": {"description": {"type": "string", "maxLength": 160},
                           "why_critical": {"type": "string", "maxLength": 160},
                           # 모델이 enum 에서 직접 분류(guided decoding). 객체 아니면 null.
                           "object_type": {"type": ["string", "null"], "enum": OBJECT_TYPES + [None]},
                           "relation": {"type": ["string", "null"], "enum": RELATIONS + [None]}}}},
        "chain_of_causation": {"type": "string", "maxLength": 400}}}


def _reason(client, uri, hint):
    r = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=V08_MAX_TOKENS,
        messages=[{"role": "user", "content": [
            {"type": "text", "text":
             f"{hint}\nAnalyze this driving segment. Describe in English, freely: "
             f"(1) scene description (road, environment, ego intent), "
             f"(2) each critical component that influences the ego's behavior and why it is critical, "
             f"(3) the chain of causation for the ego's behavior. Use the GT arc/objects as grounding "
             f"but verify and describe from the video."},
            {"type": "video_url", "video_url": {"url": uri}}]}])
    return (r.choices[0].message.content or "").strip()


def _structure(client, uri, think, hint=""):
    import json
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Analyze this segment."},
                                     {"type": "video_url", "video_url": {"url": uri}}]},
        {"role": "assistant", "content": think},
        {"role": "user", "content": "Structure the above analysis into JSON (keep free-text in English; "
         "do not over-summarize). For ego_action, classify the ego's actual maneuver from its enum based on "
         f"your analysis (a stop-then-turn is a turn/unprotected turn, not just a stop). {hint}"}]
    for mt in (V08_MAX_TOKENS, 6144):   # truncate 시 토큰↑ 재시도
        r = client.chat.completions.create(
            model=MODEL, temperature=TEMPERATURE, max_tokens=mt,
            messages=msgs, extra_body={"guided_json": V08_SCHEMA})
        ch = r.choices[0]
        try:
            return json.loads(ch.message.content)
        except Exception:
            if ch.finish_reason != "length":
                return None   # 길이초과가 아니면 재시도 무의미
    return None


def _search_tags(rec) -> list:
    """큐레이션 검색용 flat 태그. role/dist 제외, relation 은 대표 1개(GT 우선).

    arc 유지(에피소드 거동 시퀀스), ego_action·cause·object_type·vru 포함.
    """
    t = set()
    ec = rec["ego_context"]
    t.add(f"ego_action:{ec['ego_action']}")
    for k in ec.get("arc", []):          # arc 유지
        t.add(f"arc:{k}")
    t.add(f"cause:{rec['cause']}")

    objs, vrus = set(), set()
    gt_rel, model_rel = None, None
    for c in rec.get("critical_components", []):
        ref = c.get("ref")
        if ref:                          # GT(obj3d) 근거 우선
            if ref.get("object_type"):
                objs.add(ref["object_type"])
            if ref.get("relation") and gt_rel is None:
                gt_rel = ref["relation"]
            if ref.get("vru_state"):
                vrus.add(ref["vru_state"])
        for tg in c.get("tags", []):
            if tg.startswith("object_type:"):
                objs.add(tg.split(":", 1)[1])
            elif tg.startswith("relation:") and model_rel is None:
                model_rel = tg.split(":", 1)[1]
        if c.get("source") == "gt_injected":
            t.add("flag:model_missed_gt")

    for o in objs:
        t.add(f"object_type:{o}")
    for v in vrus:
        t.add(f"vru:{v}")
    rel = gt_rel or model_rel            # 대표 relation 1개, GT 우선
    if rel:
        t.add(f"relation:{rel}")
    return sorted(t)


_LEFT = {"ego_turn_left", "ego_unprotected_left", "ego_u_turn"}


def _ground_ego_action(model_ea, arc):
    """[수정1] arc(egomotion 방향)로 ego_action 강제. 모델은 방향 내 의미변형만 선택.
    arc가 회전 방향의 진실 → 좌/우/무회전 일관성 강제(unprotected_left 남발 방지)."""
    has_l = "turn_left" in arc
    has_r = "turn_right" in arc
    if has_l and not has_r:
        return model_ea if model_ea in _LEFT else "ego_turn_left"
    if has_r and not has_l:
        return "ego_turn_right"                       # 우회전은 unprotected 아님
    if has_l and has_r:
        return model_ea if model_ea in (_LEFT | {"ego_turn_right"}) else "ego_turn_left"
    # arc에 회전 없음 → 회전 코드 무효
    if model_ea in (_LEFT | {"ego_turn_right"}):
        return "ego_stop" if "stop" in arc else "ego_lane_keep"
    return model_ea                                   # 비회전 의미라벨(stop/lane_keep/follow/yield/evade) 유지


def tag_clip_v08(client, path, clip_id: str) -> dict:
    result = {"clip_id": clip_id, "ok": False, "mode": "v08"}
    try:
        meta = video_meta(path); dur = meta["duration_s"]
        det = detect_events(clip_id)
        if not det["ok"]:
            result["error"] = det.get("reason"); return result
        obst = detect_obj3d_events(clip_id, dur)
        episodes = consolidate_episodes(det["events"])
        recs = []
        tmp = Path(tempfile.mkdtemp(prefix="v08_"))
        for i, ep in enumerate(episodes, 1):
            w0 = max(0.0, ep["onset"] - LEAD_IN)   # lead-in 포함 (접근 구간)
            w1 = min(dur, ep["t1"] + 1.0)
            sub = tmp / f"e{i}.mp4"
            write_subclip(path, w0, w1, sub, WINDOW_MAX_SIDE, SEND_FPS)
            uri = to_data_uri(sub); sub.unlink(missing_ok=True)
            _, overlapping = _causal_agent(obst, ep)
            in_path = [o for o in overlapping
                       if o.get("role") in ("crossing", "preceding_vehicle")
                       and o.get("min_dist", 999) < 30]
            hint = _gt_hint(ep, in_path)              # ② GT 객체 열거+커버리지 지시
            think = _reason(client, uri, hint)
            v = _structure(client, uri, think, hint) or {}
            comps = v.get("critical_components", []) or []
            # [수정2] object_type/relation 은 GT(obj3d)만 채택. 모델 enum 은 매칭용으로만 보관
            # (모델 과채움 door_open 차단). 매칭 안 된 모델 컴포넌트는 서술만(enum null).
            for c in comps:
                c["source"] = "model"; c["ref"] = None
                c["_mot"] = c.get("object_type")
                c["object_type"] = None; c["relation"] = None; c["tags"] = []

            consistency = []
            for o in in_path:
                hit = next((c for c in comps if c.get("_mot") == o["object_type"]), None)
                if hit:                              # 모델이 언급 + GT 근거 부여
                    hit["source"] = "gt"; hit["ref"] = _gt_ref(o)
                    hit["object_type"] = o["object_type"]; hit["relation"] = o.get("relation")
                    hit["tags"] = _gt_tags(o)
                    consistency.append({"gt": _gt_ref(o), "covered_by_model": True})
                else:                                # 모델 누락 → GT 주입(floor)
                    comps.append({"source": "gt_injected", "ref": _gt_ref(o),
                                  "object_type": o["object_type"], "relation": o.get("relation"),
                                  "description": _gt_syn(o),
                                  "why_critical": "GT in-path object (missed by model free-text; GT-injected)",
                                  "tags": _gt_tags(o), "grounded": False})
                    consistency.append({"gt": _gt_ref(o), "covered_by_model": False})

            # ego_action: 모델 추출(의미) → arc(egomotion)로 방향·회전유무 강제 그라운딩
            ego_action = _ground_ego_action(v.get("ego_action") or ep["ego_action"], ep["kinds"])
            # cause: GT agent 우선(신뢰) → 없으면 모델 추출 cause → signal 은 grounding 확인
            if in_path:
                cause = "agent"
            else:
                cause = v.get("cause") or "other"
                if cause == "signal":
                    from classify073 import _signal_present, _frame_uri
                    if not _signal_present(client, _frame_uri(path, ep["onset"])):
                        cause = "other"

            rec = {
                "segment_id": i, "window": [round(w0, 2), round(w1, 2)],
                "key_frame_t": round(ep["onset"], 2),
                "ego_context": {"ego_action": ego_action, "arc": ep["kinds"],
                                "arc_rule": ep["ego_action"]},  # arc-rule 라벨도 참고 보관
                "cause": cause,
                "scene_description": v.get("scene_description"),
                "ego_intent": v.get("ego_intent"),
                "critical_components": comps,
                "chain_of_causation": v.get("chain_of_causation"),
                "consistency": consistency, "think": think}
            rec["search_tags"] = _search_tags(rec)
            recs.append(rec)
        result["records"] = recs
        result["ok"] = True
    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"; result["trace"] = traceback.format_exc()[-400:]
    return result

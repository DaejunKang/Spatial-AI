"""v0.7.3 2-pass 분류기 + 조립.

Pass1: 비제약 추론(<think>) — 원인/객체/가림/거동/신호를 단계적으로.
Pass2: guided_json 으로 model_output_v0.7.3 스키마 enum 강제 추출.
이후 vocab073.assemble_record 로 앵커+모델출력 병합(role 파생·normalization·fill_rules).
"""

import base64
import tempfile
from pathlib import Path

from config import (
    CLASSIFY_MAX_TOKENS,
    EVENT_CTX_SEC,
    MODEL,
    SEND_FPS,
    TEMPERATURE,
    WINDOW_MAX_SIDE,
)
from dataset import to_data_uri, video_meta, write_subclip

# 에피소드 병합/인과 게이트 파라미터
_EP_GAP = 2.0          # 인접 egomotion 이벤트 병합 간격(초)
_ONSET_AFTER = 1.0     # agent가 onset 이후 이만큼 넘어 등장하면 비인과
_ONSET_BEFORE = 1.5    # agent가 onset 이전 이만큼 전에 종료면 비인과
_D_CROSS = 15.0        # crossing 인과 최대 거리(m)
_D_PREC = 30.0         # preceding 인과 최대 거리(m)
from events import CLASS_MAP, detect_events, detect_obstacle_events
from vocab073 import (
    GUIDED,
    OCCLUSION_ENUM,
    anchor_ego_action,
    assemble_record,
    derive_role,
    validate_record,
)

_AXES = (
    "cause=agent|signal|road_geometry|other. "
    "object_type(도로사용자만)=vehicle/large_vehicle/motorcycle/bicycle_micromobility/pedestrian/animal/emergency_vehicle/other. "
    "occlusion(cause=agent)=visible/partial/full_inferred/emerging/entering. "
    "longitudinal_action(선행차 종방향)=accelerate/decelerate/hard_brake/coming_to_stop/maintain_speed/reverse/stationary_parked "
    "XOR relation(횡·관계)=cross_ego_path/oncoming_cross/merge/yield_to_ego/brake_check/block_lane/door_open/board_alight/cut_in/cut_out/sudden_cut_in/alongside_parallel/oncoming_narrow. "
    "vru_attr/vru_state(보행자). signal_state(cause=signal)=red/yellow/green/red_arrow/green_arrow/flashing."
)

_HINT = {
    "stop": "egomotion GT: EGO가 정지함.",
    "decelerate": "egomotion GT: EGO가 감속함.",
    "accelerate": "egomotion GT: EGO가 가속함.",
    "turn_left": "egomotion GT: EGO가 좌회전함.",
    "turn_right": "egomotion GT: EGO가 우회전함.",
    "evade": "egomotion GT: EGO가 급회피함.",
}


_CAUSE_RULES = (
    "cause 판정: (1) 신호등/표지가 ego 전이(정지·출발)를 지배 → cause=signal, signal_state 기입, 나머지 null. "
    "(2) 도로사용자(선행차 감속/정지, 보행자·차량 횡단, cut-in 등)가 원인 → cause=agent, object_type+occlusion+(longitudinal_action XOR relation) 기입. "
    "(3) 곡선/과속방지턱 등 도로기하만으로 트리거 → cause=road_geometry, 나머지 null. "
    "(4) 위 어느 것도 아닐 때만 cause=other. "
    "신호와 선행차가 동시면 신호가 지배적이면 signal."
)


def _think(client, uri, hint):
    r = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=CLASSIFY_MAX_TOKENS,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"{hint} 이 구간에서 ego 거동의 전이 '원인'과 관련 "
             f"객체/신호/가림/거동을 단계적으로 추론하라. {_CAUSE_RULES} "
             f"축: {_AXES} 마지막 줄에 '결론: cause=<>, signal_state=<>, object_type=<>, "
             f"occlusion=<>, longitudinal_action=<>, relation=<>, vru_state=<>' 형식으로 요약(해당없으면 null)."},
            {"type": "video_url", "video_url": {"url": uri}}]}])
    return (r.choices[0].message.content or "").strip()


def _extract(client, uri, think):
    r = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=CLASSIFY_MAX_TOKENS,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "이 구간을 분석하라."},
                {"type": "video_url", "video_url": {"url": uri}}]},
            {"role": "assistant", "content": think},
            {"role": "user", "content": f"위 분석의 '결론'을 v0.7.3 model_output 레코드(JSON)로만 출력하라. "
             f"추론에서 신호가 원인이면 cause=signal, 도로사용자가 원인이면 cause=agent 로 정확히 반영하라. {_CAUSE_RULES}"}],
        extra_body={"guided_json": GUIDED})
    import json
    try:
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None


def _dominant(kinds: list) -> tuple:
    """에피소드 지배 거동 (우선순위 stop>evade>turn>lane_keep). (ego_action, kind)."""
    if "stop" in kinds:
        return "ego_stop", "stop"
    if "evade" in kinds:
        return "ego_evade", "evade"
    if "turn_left" in kinds:
        return "ego_turn_left", "turn_left"
    if "turn_right" in kinds:
        return "ego_turn_right", "turn_right"
    return "ego_lane_keep", kinds[0] if kinds else "decelerate"


def consolidate_episodes(events: list, gap: float = _EP_GAP) -> list:
    """[원칙1] 인접/겹치는 egomotion 이벤트를 하나의 거동 에피소드로 병합."""
    evs = sorted(events, key=lambda e: e["t0"])
    eps = []
    for e in evs:
        if eps and e["t0"] <= eps[-1]["t1"] + gap:
            eps[-1]["t1"] = max(eps[-1]["t1"], e["t1"])
            eps[-1]["kinds"].append(e["kind"])
        else:
            eps.append({"t0": e["t0"], "t1": e["t1"], "kinds": [e["kind"]]})
    for ep in eps:
        ep["onset"] = ep["t0"]
        ep["ego_action"], ep["dom_kind"] = _dominant(ep["kinds"])
    return eps


def _causal_agent(obst, ep):
    """[원칙2] 인과 게이트 통과한 agent. present-at-onset + in-path + 근접.
    반환: (primary or None, overlapping[])."""
    onset, t1 = ep["t0"], ep["t1"]
    overlapping = [o for o in obst if not (o["t1"] < onset - 1e-6 or o["t0"] > t1 + 1e-6)]
    cands = []
    for o in overlapping:
        if o["t0"] > onset + _ONSET_AFTER:      # onset 이후 등장 → 비인과
            continue
        if o["t1"] < onset - _ONSET_BEFORE:      # onset 전 종료 → 비인과
            continue
        role, d = o.get("role"), o.get("min_dist", 999)
        if role == "crossing" and d < _D_CROSS:
            cands.append((0, d, o))
        elif role == "preceding_vehicle" and d < _D_PREC:
            cands.append((1, d, o))
        # adjust(cut_in 등)은 ego 자발 기동의 원인으로 보지 않음 → 제외
    if not cands:
        return None, overlapping
    cands.sort(key=lambda x: (x[0], x[1]))
    return cands[0][2], overlapping


def _frame_uri(path, t: float, max_side: int = 960) -> str:
    """시각 t의 단일 프레임 JPEG data URI (grounding 용)."""
    import cv2
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, fr = cap.read(); cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w); fr = cv2.resize(fr, (int(w * s), int(h * s)))
    return "data:image/jpeg;base64," + base64.b64encode(
        cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def _signal_present(client, img_uri) -> bool:
    """[원칙3-B] grounding으로 실제 신호등 검출 여부."""
    if img_uri is None:
        return False
    r = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=120,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "이 장면에 교통 신호등(traffic light)이 실제로 있으면 "
             "그 바운딩박스를 <box>x1,y1,x2,y2</box> 로, 없으면 정확히 'none' 만 출력."},
            {"type": "image_url", "image_url": {"url": img_uri}}]}])
    txt = (r.choices[0].message.content or "").lower()
    return "<box>" in txt and "none" not in txt[:6]


def tag_clip_v073(client, path, clip_id: str) -> dict:
    """v0.7.3: 에피소드 병합 + 인과게이트 + 신호 grounding 2-pass 태깅."""
    result = {"clip_id": clip_id, "ok": False, "mode": "events_v073"}
    try:
        meta = video_meta(path)
        result["video"] = meta
        dur = meta["duration_s"]
        det = detect_events(clip_id)
        if not det["ok"]:
            result["error"] = det.get("reason"); return result
        obst = detect_obstacle_events(clip_id, dur)
        episodes = consolidate_episodes(det["events"])
        result["events"] = det["events"]
        result["episodes"] = episodes
        result["obstacle_events"] = obst

        records, review = [], []
        sid = 0
        consumed = [False] * len(obst)
        tmp = Path(tempfile.mkdtemp(prefix="v073_"))

        # 1) 에피소드 = 세그먼트 단위. onset 앵커 창.
        for ep in episodes:
            sid += 1
            onset = ep["onset"]
            w0, w1 = max(0.0, onset - EVENT_CTX_SEC), min(dur, onset + EVENT_CTX_SEC)
            sub = tmp / f"e{sid}.mp4"
            write_subclip(path, w0, w1, sub, WINDOW_MAX_SIDE, SEND_FPS)
            uri = to_data_uri(sub); sub.unlink(missing_ok=True)
            think = _think(client, uri, _HINT.get(ep["dom_kind"], ""))
            mo = _extract(client, uri, think) or {"cause": "other"}
            src = f"episode:{'+'.join(ep['kinds'])}"

            prim, overlapping = _causal_agent(obst, ep)
            # 에피소드에 겹친 agent 는 전부 흡수(중복 방지)
            for i, o in enumerate(obst):
                if o in overlapping:
                    consumed[i] = True

            if prim is not None:  # [원칙2] 인과 agent → cause=agent (GT 우선)
                m2 = dict(mo); m2["cause"] = "agent"; m2["object_type"] = prim["object_type"]
                if prim.get("role") == "preceding_vehicle":
                    m2["longitudinal_action"] = ("coming_to_stop" if ep["dom_kind"] == "stop"
                                                 else "maintain_speed")
                    m2["relation"] = None
                else:
                    m2["relation"] = prim.get("relation") or "cross_ego_path"
                    m2["longitudinal_action"] = None
                m2["occlusion"] = m2.get("occlusion") or "visible"
                mo = m2; src += f"+GT:{prim['object_type']}"
            else:
                # [원칙3-B] agent 없음: 정지면 신호등 grounding 검증
                if ep["ego_action"] == "ego_stop":
                    if _signal_present(client, _frame_uri(path, onset)):
                        mo = {**mo, "cause": "signal"}; src += "+signalGND"
                    else:
                        mo = {**mo, "cause": "other"}  # 불빛 오탐 방지
                # 비정지(자발 가/감속·회전)는 모델 cause 유지(road_geometry/other)

            anchor = {"clip_id": clip_id, "segment_id": sid, "key_frame_t": round(onset, 2),
                      "t_start": ep["t0"], "t_end": ep["t1"],
                      "ego_action": ep["ego_action"], "difficulty": None}
            rec, notes = assemble_record(anchor, mo)
            records.append(rec)
            review.append({"segment_id": sid, "source": src,
                           "window": [round(w0, 2), round(w1, 2)],
                           "t_start": ep["t0"], "t_end": ep["t1"], "key_frame_t": round(onset, 2),
                           "ego_action": ep["ego_action"], "think": think,
                           "model_out": mo, "record": rec, "notes": notes,
                           "warns": validate_record(rec)})

        # 2) 에피소드 미연관 GT agent 중 인과성 있는 것만 독립 세그먼트(순항 추종 등)
        for i, o in enumerate(obst):
            if consumed[i]:
                continue
            role, d = o.get("role"), o.get("min_dist", 999)
            if not ((role == "crossing" and d < _D_CROSS) or (role == "preceding_vehicle" and d < _D_PREC)):
                continue  # 원거리·비인과 agent 는 버림(원칙3)
            sid += 1
            mo = {"cause": "agent", "object_type": o["object_type"], "vru_attr": None,
                  "occlusion": "visible", "longitudinal_action": o.get("longitudinal_action"),
                  "relation": o.get("relation"), "vru_state": o.get("vru_detail"), "signal_state": None}
            anchor = {"clip_id": clip_id, "segment_id": sid,
                      "key_frame_t": round((o["t0"] + o["t1"]) / 2, 2),
                      "t_start": o["t0"], "t_end": o["t1"], "ego_action": "ego_lane_keep",
                      "difficulty": None}
            rec, notes = assemble_record(anchor, mo)
            records.append(rec)
            review.append({"segment_id": sid, "source": o["kind"] + "(standalone)",
                           "window": None, "t_start": o["t0"], "t_end": o["t1"],
                           "key_frame_t": anchor["key_frame_t"], "ego_action": "ego_lane_keep",
                           "think": "(obstacle GT 독립·순항 중 agent)", "model_out": mo,
                           "record": rec, "notes": notes, "warns": validate_record(rec)})

        result["segments"] = records
        result["review"] = review
        result["warnings"] = [w for rv in review for w in rv["warns"]]
        result["ok"] = True
    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-500:]
    return result

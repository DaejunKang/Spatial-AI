"""egomotion 이벤트 중심 태깅 (when=GT, what=모델).

egomotion 으로 동적 이벤트(감속/가속/정지/회전)의 정확한 시각을 잡고,
각 이벤트 중심 ±컨텍스트 구간을 모델에 보여 거동을 분류한다. 이벤트가 없는
구간은 baseline(ego_lane_keep) 으로 결정론적으로 채워 전체 시간을 커버한다.
윈도우 경계에서 거동이 끊기는 문제를 이벤트-중심 창으로 해소한다.
"""

import tempfile
from pathlib import Path

from config import (
    CLASSIFY_MAX_TOKENS,
    EVENT_BASELINE_MIN_SEC,
    EVENT_CTX_SEC,
    MODEL,
    SEND_FPS,
    TEMPERATURE,
    WINDOW_MAX_SIDE,
)
from dataset import to_data_uri, video_meta, write_subclip
from events import detect_events, detect_obstacle_events
from prompts import SCHEMA, VOCAB
from tagger import extract_answer
from window_tagger import _WINDOW_SYS, _behavior_key, _normalize, merge_segments
from vocab import validate_segments

_HINT = {
    "decelerate": "egomotion GT: the EGO is SLOWING DOWN here. The ego stays in its lane while slowing, so ego_action is usually ego_follow (following a slowing preceding vehicle) or ego_lane_keep. Identify the agent causing the slowdown (preceding vehicle decelerating/stopping, a crossing pedestrian/vehicle, etc.).",
    "accelerate": "egomotion GT: the EGO is GAINING SPEED here while keeping its lane. ego_action is usually ego_lane_keep (speed change is not a separate maneuver). Tag a preceding vehicle if relevant.",
    "stop": "egomotion GT: the EGO is STOPPED / near-stationary here. Identify what it is stopped for (preceding vehicle, red light, pedestrian, etc.) and tag that agent.",
    "turn_left": "egomotion GT: the EGO is TURNING LEFT here.",
    "turn_right": "egomotion GT: the EGO is TURNING RIGHT here.",
}

# 이벤트 종류 → egomotion GT 로 확정 가능한 ego 거동(항상 부여)
_EGO_DETERMINISTIC = {
    "stop": "ego_reactive_stop",
    "turn_left": "ego_turn_left",
    "turn_right": "ego_turn_right",
}


def _classify(client, sub_uri: str, hint: str) -> list:
    resp = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=CLASSIFY_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _WINDOW_SYS},
            {"role": "user", "content": [
                {"type": "text", "text": hint + " Classify the dynamic behavior(s) in this short clip."},
                {"type": "video_url", "video_url": {"url": sub_uri}},
            ]},
        ],
    )
    ch = resp.choices[0]
    return extract_answer(ch.message.content or "") or []


def _gaps(dur: float, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[0,dur] 에서 spans 로 덮이지 않은 구간."""
    spans = sorted(spans)
    gaps, cur = [], 0.0
    for a, b in spans:
        if a > cur + 1e-6:
            gaps.append((cur, min(a, dur)))
        cur = max(cur, b)
    if cur < dur - 1e-6:
        gaps.append((cur, dur))
    return gaps


def tag_clip_events(client, path, clip_id: str) -> dict:
    """이벤트 중심 태깅. tagger.tag_clip 과 호환되는 dict 반환."""
    result: dict = {"clip_id": clip_id, "ok": False, "mode": "events"}
    try:
        meta = video_meta(path)
        result["video"] = meta
        dur = meta["duration_s"]
        src_fps = meta["fps"] or 30.0

        det = detect_events(clip_id)
        if not det["ok"]:
            # egomotion 없으면 윈도우 방식으로 폴백
            from window_tagger import tag_clip_windowed
            r = tag_clip_windowed(client, path, clip_id)
            r["mode"] = "events->windowed_fallback"
            r["fallback_reason"] = det.get("reason")
            return r
        events = det["events"]
        result["events"] = events

        all_segs: list = []
        run_dumps = []
        tmpdir = Path(tempfile.mkdtemp(prefix="vla_ev_"))
        for i, ev in enumerate(events):
            c = ev["center"]
            w0, w1 = max(0.0, c - EVENT_CTX_SEC), min(dur, c + EVENT_CTX_SEC)
            sub = tmpdir / f"ev{i}.mp4"
            write_subclip(path, w0, w1, sub, WINDOW_MAX_SIDE, SEND_FPS)
            behaviors = _classify(client, to_data_uri(sub), _HINT.get(ev["kind"], ""))
            sub.unlink(missing_ok=True)

            det_ego = _EGO_DETERMINISTIC.get(ev["kind"])
            kept = []
            for b in behaviors:
                if not isinstance(b, dict):
                    continue
                b = _normalize(b)
                # stop/turn 처럼 GT 로 ego 거동이 확정되는 이벤트에선 모델의 ego 추정은 버리고
                # agent 거동만 취한다(ego 는 아래서 결정론적으로 부여).
                if det_ego and b.get("subject") == "ego":
                    continue
                s = dict(b)
                s.update({"clip_id": clip_id, "t_start": ev["t0"], "t_end": ev["t1"],
                          "windows": [i], "event_kind": ev["kind"]})
                kept.append(s)
            if det_ego:
                kept.append({"clip_id": clip_id, "subject": "ego",
                             "ego_action": det_ego,
                             "object_type": None, "role": None,
                             "longitudinal_action": None, "relation": None,
                             "vru_detail": None, "difficulty": None,
                             "t_start": ev["t0"], "t_end": ev["t1"],
                             "windows": [i], "event_kind": ev["kind"]})
            all_segs.extend(kept)
            run_dumps.append({"idx": i, "kind": ev["kind"], "t0": ev["t0"],
                              "t1": ev["t1"], "n_behaviors": len(behaviors)})

        # obstacle 트랙 기반 agent 이벤트 융합 (egomotion 사각지대: 정속 선행차·비반응 agent 등)
        obst = detect_obstacle_events(clip_id, dur)
        result["obstacle_events"] = obst
        for j, o in enumerate(obst):
            all_segs.append({"clip_id": clip_id, "subject": "agent",
                             "ego_action": None, "object_type": o["object_type"],
                             "role": o["role"], "longitudinal_action": o["longitudinal_action"],
                             "relation": o["relation"], "vru_detail": o["vru_detail"],
                             "difficulty": None, "t_start": o["t0"], "t_end": o["t1"],
                             "windows": [], "event_kind": o["kind"]})

        # 무이벤트 구간 baseline (주행 중 → ego_lane_keep)
        spans = [(e["t0"], e["t1"]) for e in events]
        for a, b in _gaps(dur, spans):
            if b - a >= EVENT_BASELINE_MIN_SEC:
                all_segs.append({"clip_id": clip_id, "subject": "ego",
                                 "ego_action": "ego_lane_keep", "object_type": None,
                                 "role": None, "longitudinal_action": None,
                                 "relation": None, "vru_detail": None,
                                 "difficulty": None, "t_start": round(a, 2),
                                 "t_end": round(b, 2), "windows": [],
                                 "event_kind": None})

        merged = merge_segments(all_segs, src_fps)
        _, warns = validate_segments(merged, VOCAB, SCHEMA)
        result["segments"] = merged
        result["raw_segments"] = all_segs
        result["event_runs"] = run_dumps
        result["warnings"] = warns
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result

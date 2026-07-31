"""윈도우(연속 청크) 분할 태깅 + 통합 (meta_tagging v0.7.1 segment).

20초 클립을 ~6초 윈도우로 잘라 각 윈도우 서브클립을 video_url 로 태깅하면,
윈도우당 프레임 밀도가 전체 클립을 한 번에 보는 것보다 높아 모션 이해가 낫다.
각 윈도우 세그먼트의 상대 시간을 전역 시간/프레임으로 변환한 뒤, 같은 거동을
윈도우 경계 너머로 병합해 클립 단위 최종 세그먼트로 통합한다.
"""

import tempfile
from pathlib import Path

from config import (
    CLASSIFY_MAX_TOKENS,
    MERGE_GAP_SEC,
    MODEL,
    SEND_FPS,
    TEMPERATURE,
    WINDOW_MAX_SIDE,
    WINDOW_SEC,
    WINDOW_STRIDE_SEC,
)
from dataset import to_data_uri, video_meta, write_subclip
from prompts import SCHEMA, VOCAB
from tagger import extract_answer
from vocab import ROLE_ENUM, SUBJECT_ENUM, codes, validate_segments

# 시간은 윈도우가 정의하므로 모델에는 '거동 분류'만 요청한다(모델은 시간맹).
_AX = ("object_type", "longitudinal_action", "relation", "ego_action", "vru_detail")
_EGO_CODES = set(codes(VOCAB, "ego_action"))
_OBJ_CODES = set(codes(VOCAB, "object_type"))
# 모델이 자주 쓰는 동의어 → 어휘 코드 (traffic_light 등 어휘에 없는 것은 그대로 두어 경고로 노출)
_OBJ_SYN = {"car": "vehicle", "van": "vehicle", "suv": "vehicle", "minivan": "vehicle",
            "truck": "large_vehicle", "bus": "large_vehicle", "heavy_truck": "large_vehicle",
            "trailer": "large_vehicle", "person": "pedestrian", "people": "pedestrian",
            "cyclist": "bicycle_micromobility", "bicyclist": "bicycle_micromobility",
            "bike": "bicycle_micromobility", "rider": "bicycle_micromobility"}
_VOCAB_TXT = (
    f"subject: {SUBJECT_ENUM}\n"
    f"role (EXACTLY one of these 3, never invent/append/misspell): {ROLE_ENUM}\n"
    + "\n".join(f"{ax}: {codes(VOCAB, ax)}" for ax in _AX)
)


def _window_bounds(dur: float) -> list[tuple[float, float]]:
    """[start, end] 윈도우 목록. 마지막은 클립 끝까지."""
    wins = []
    start = 0.0
    while start < dur - 1e-6:
        end = min(start + WINDOW_SEC, dur)
        wins.append((round(start, 3), round(end, 3)))
        if end >= dur - 1e-6:
            break
        start += WINDOW_STRIDE_SEC
    return wins


_WINDOW_SYS = f"""You classify dynamic driving behaviors in a SHORT video window (a few seconds cut from a 20s clip).
Watch how the scene CHANGES across frames to read motion. Do NOT report timestamps —
the window's time span is already known. Just list the dynamic behavior(s) present that
involve the ego vehicle:
- subject=ego (an ego maneuver) OR subject=agent (an ego-affecting agent maneuver). NOT static object listing.
Use ONLY these exact codes (copy them verbatim; do not invent, translate, append, or misspell):
{_VOCAB_TXT}
Field rules:
- subject=ego -> set ego_action ONLY; leave object_type/role/longitudinal_action/relation/vru_detail = null.
- subject=agent -> set object_type + role (role is EXACTLY one of {ROLE_ENUM}); ego_action = null.
    role=preceding_vehicle -> set longitudinal_action, relation=null.
    role=adjust or crossing -> set relation, longitudinal_action=null.
    object_type=pedestrian -> may add vru_detail.
- Do NOT emit more than one agent behavior of the same role per window. Merge duplicates.
Think briefly in <think> (<=4 sentences). Then output ONLY inside <answer> a JSON array of behaviors:
[{{"subject":"ego|agent","ego_action":<code|null>,"object_type":<code|null>,"role":<code|null>,"longitudinal_action":<code|null>,"relation":<code|null>,"vru_detail":<code|null>,"difficulty":<1-10|null>}}]
If nothing dynamic/ego-relevant happens in this window, output <answer>[]</answer>."""


def tag_window(client, sub_uri: str) -> tuple[list, str, dict]:
    """윈도우 서브클립 하나를 태깅해 거동 목록 반환(시간 필드 없음)."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _WINDOW_SYS},
            {"role": "user", "content": [
                {"type": "text", "text": "이 윈도우의 동적 거동을 분류하라."},
                {"type": "video_url", "video_url": {"url": sub_uri}},
            ]},
        ],
    )
    ch = resp.choices[0]
    segs = extract_answer(ch.message.content or "") or []
    return segs, ch.finish_reason, (resp.usage.model_dump() if resp.usage else {})


def _normalize(b: dict) -> dict:
    """subject/role 필드 채움 규칙(결정론적)에 맞게 비허용 필드를 null 로 정리."""
    b = dict(b)
    if b.get("subject") == "ego":
        for f in ("object_type", "role", "longitudinal_action", "relation", "vru_detail"):
            b[f] = None
        # ego 가감속은 어휘상 별도 액션이 없음(차로 유지 중 속도변화) → 미허용 코드는 lane_keep 로 보정
        if b.get("ego_action") not in _EGO_CODES:
            b["ego_action"] = "ego_lane_keep"
    elif b.get("subject") == "agent":
        b["ego_action"] = None
        ot = b.get("object_type")
        if ot not in _OBJ_CODES:              # 동의어 보정(미지 코드는 그대로 → 경고 노출)
            b["object_type"] = _OBJ_SYN.get(ot, ot)
        if b.get("role") == "preceding_vehicle":
            b["relation"] = None
        elif b.get("role") in ("adjust", "crossing"):
            b["longitudinal_action"] = None
        if b.get("object_type") != "pedestrian":
            b["vru_detail"] = None
    return b


def _behavior_key(s: dict):
    if s.get("subject") == "ego":
        return ("ego", s.get("ego_action"))
    return ("agent", s.get("object_type"), s.get("role"),
            s.get("relation"), s.get("longitudinal_action"), s.get("vru_detail"))


def merge_segments(segs: list, src_fps: float) -> list:
    """전역 시간으로 정렬 후 같은 거동을 인접/겹침 기준으로 병합."""
    segs = [s for s in segs if isinstance(s.get("t_start"), (int, float))
            and isinstance(s.get("t_end"), (int, float))]
    segs.sort(key=lambda s: (str(_behavior_key(s)), s["t_start"]))

    merged: list = []
    for s in segs:
        k = _behavior_key(s)
        if merged and _behavior_key(merged[-1]) == k and \
                s["t_start"] <= merged[-1]["t_end"] + MERGE_GAP_SEC:
            m = merged[-1]
            m["t_end"] = max(m["t_end"], s["t_end"])
            m["windows"] = sorted(set(m.get("windows", []) + s.get("windows", [])))
            if isinstance(s.get("difficulty"), int):
                m["difficulty"] = max(m.get("difficulty") or 0, s["difficulty"])
        else:
            merged.append(dict(s))

    # 시간순 재정렬 + 전역 프레임/segment_id 부여
    merged.sort(key=lambda s: s["t_start"])
    for i, m in enumerate(merged, 1):
        m["segment_id"] = i
        m["t_start"] = round(m["t_start"], 2)
        m["t_end"] = round(m["t_end"], 2)
        m["frame_start"] = int(round(m["t_start"] * src_fps))
        m["frame_end"] = int(round(m["t_end"] * src_fps))
    return merged


def tag_clip_windowed(client, path, clip_id: str) -> dict:
    """클립을 윈도우로 분할 태깅 후 통합. tagger.tag_clip 과 호환되는 dict 반환."""
    result: dict = {"clip_id": clip_id, "ok": False, "mode": "windowed"}
    try:
        meta = video_meta(path)
        result["video"] = meta
        dur = meta["duration_s"]
        src_fps = meta["fps"] or 30.0
        wins = _window_bounds(dur)
        result["windows"] = [{"idx": i, "t0": a, "t1": b} for i, (a, b) in enumerate(wins)]

        all_segs: list = []
        win_dumps = []
        tmpdir = Path(tempfile.mkdtemp(prefix="vla_win_"))
        for i, (t0, t1) in enumerate(wins):
            sub = tmpdir / f"win{i}.mp4"
            write_subclip(path, t0, t1, sub, WINDOW_MAX_SIDE, SEND_FPS)
            uri = to_data_uri(sub)
            behaviors, finish, usage = tag_window(client, uri)
            seen = set()
            for b in behaviors:
                if not isinstance(b, dict):
                    continue
                b = _normalize(b)
                key = _behavior_key(b)
                if key in seen:  # 윈도우 내 중복 제거
                    continue
                seen.add(key)
                # 시간은 모델이 아니라 윈도우가 정의
                s = dict(b)
                s["clip_id"] = clip_id
                s["t_start"] = t0
                s["t_end"] = t1
                s["windows"] = [i]
                all_segs.append(s)
            win_dumps.append({"idx": i, "t0": t0, "t1": t1, "finish": finish,
                              "n_raw": len(behaviors), "usage": usage})
            sub.unlink(missing_ok=True)

        merged = merge_segments(all_segs, src_fps)
        _, warns = validate_segments(merged, VOCAB, SCHEMA)
        result["segments"] = merged
        result["raw_segments"] = all_segs
        result["window_runs"] = win_dumps
        result["warnings"] = warns
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result

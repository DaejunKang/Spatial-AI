"""세그먼트 태그를 영상 위에 burn-in 하는 정성 확인용 오버레이 렌더러.

태그는 시간(t_start/t_end)만 있고 공간 좌표는 없으므로, 재생 시점 t 에 활성인
세그먼트를 좌상단 패널에 표시하고 하단에 전체 세그먼트 타임라인 + 플레이헤드를 그린다.
- ego 세그먼트 = 청록, agent 세그먼트 = 주황
- 활성 세그먼트는 강조, 비활성은 흐리게
"""

from pathlib import Path

# BGR
COL_EGO = (200, 200, 60)      # 청록
COL_AGENT = (40, 150, 240)    # 주황
COL_BG = (0, 0, 0)
COL_TXT = (255, 255, 255)
COL_DIM = (120, 120, 120)


def _diff_color(d):
    """난이도 색 (초록→빨강)."""
    if not isinstance(d, int):
        return COL_DIM
    if d <= 2:
        return (80, 200, 80)
    if d <= 4:
        return (120, 200, 160)
    if d <= 6:
        return (60, 200, 230)
    if d <= 8:
        return (40, 140, 240)
    return (50, 50, 235)


def _is_v073(seg):
    return "cause" in seg


def _seg_color(seg):
    if _is_v073(seg):
        # cause 색: signal=노랑, agent=주황, road_geometry/other=회청
        c = seg.get("cause")
        if c == "signal":
            return (60, 200, 230)
        if c == "agent":
            return COL_AGENT
        return COL_EGO
    return COL_EGO if seg.get("subject") == "ego" else COL_AGENT


def _seg_label(seg) -> str:
    sid = seg.get("segment_id")
    if seg.get("demo_label"):          # 데모: arc_path + 자연어 설명 커스텀 라벨
        return f"#{sid} {seg['demo_label']}"
    if _is_v073(seg):
        # v0.7.3: ego_action + cause + (원인별 상세)
        core = f"#{sid} {seg.get('ego_action')} ← {seg.get('cause')}"
        if seg.get("cause") == "signal" and seg.get("signal_state"):
            core += f":{seg['signal_state']}"
        elif seg.get("cause") == "agent":
            bits = [str(seg.get("object_type"))]
            if seg.get("role"):
                bits.append(f"role={seg['role']}")
            if seg.get("longitudinal_action"):
                bits.append(seg["longitudinal_action"])
            if seg.get("relation"):
                bits.append(seg["relation"])
            if seg.get("occlusion") and seg["occlusion"] != "visible":
                bits.append(f"occ={seg['occlusion']}")
            if seg.get("vru_state"):
                bits.append(f"vru={seg['vru_state']}")
            core += "  [" + " ".join(bits) + "]"
        return core
    if seg.get("subject") == "ego":
        core = f"EGO · {seg.get('ego_action')}"
    else:
        bits = [f"{seg.get('object_type')}", f"role={seg.get('role')}"]
        if seg.get("longitudinal_action"):
            bits.append(seg["longitudinal_action"])
        if seg.get("relation"):
            bits.append(seg["relation"])
        if seg.get("vru_detail"):
            bits.append(f"vru={seg['vru_detail']}")
        core = "AGENT · " + " ".join(bits)
    d = seg.get("difficulty")
    dtag = f"  (d{d})" if d is not None else ""
    return f"#{sid} {core}{dtag}"


def _put(img, text, org, color=COL_TXT, scale=0.6, th=1, bg=True):
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, tht), base = cv2.getTextSize(text, font, scale, th)
    x, y = org
    if bg:
        cv2.rectangle(img, (x - 3, y - tht - 3), (x + tw + 3, y + base + 2), COL_BG, -1)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
    return tht + base + 6


def _blend_rect(img, x0, y0, x1, y1, alpha=0.45, color=COL_BG):
    import cv2

    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return
    overlay = sub.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


def _resize_side(frame, max_side):
    import cv2

    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame
    s = max_side / max(h, w)
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def render_overlay(
    clip_path: Path | str,
    segments: list,
    out_path: Path | str,
    clip_id: str = "",
    max_side: int = 1280,
) -> Path:
    """clip 영상 위에 세그먼트 태그를 burn-in 한 mp4 를 out_path 에 쓴다."""
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps else 0.0

    # 첫 프레임으로 출력 해상도 결정
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("첫 프레임 읽기 실패")
    first = _resize_side(first, max_side)
    H, W = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter 열기 실패(mp4v 코덱 확인)")

    tl_h = max(40, 12 * max(1, len(segments)) + 22)  # 타임라인 높이

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _resize_side(frame, max_side)
        t = fidx / fps if fps else 0.0
        _draw_hud(frame, segments, t, dur, clip_id, tl_h)
        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    return out_path


def _draw_hud(frame, segments, t, dur, clip_id, tl_h):
    import cv2

    H, W = frame.shape[:2]

    # 상단: clip_id + 시간
    _blend_rect(frame, 0, 0, W, 30)
    _put(frame, f"{clip_id[:18]}  t={t:4.1f}/{dur:.1f}s", (8, 21), bg=False)

    # 좌상단: 활성 세그먼트 패널
    active = [s for s in segments if _is_active(s, t)]
    if active:
        lines = [_seg_label(s) for s in active]
        panel_h = 24 + 22 * len(lines)
        _blend_rect(frame, 0, 32, min(W, 760), 32 + panel_h)
        y = 54
        _put(frame, f"ACTIVE ({len(active)})", (8, y), COL_TXT, 0.55, 1, bg=False)
        y += 24
        for s in active:
            _put(frame, _seg_label(s), (8, y), _seg_color(s), 0.55, 1, bg=False)
            y += 22

    # 하단: 전체 타임라인
    _draw_timeline(frame, segments, t, dur, tl_h)


def _is_active(seg, t) -> bool:
    ts, te = seg.get("t_start"), seg.get("t_end")
    if not isinstance(ts, (int, float)) or not isinstance(te, (int, float)):
        return False
    return ts <= t <= te


def _draw_timeline(frame, segments, t, dur, tl_h):
    import cv2

    H, W = frame.shape[:2]
    y0 = H - tl_h
    _blend_rect(frame, 0, y0, W, H, alpha=0.55)
    pad = 10
    x0, x1 = pad, W - pad
    span = max(dur, 1e-6)

    def xt(tt):
        return int(x0 + (x1 - x0) * min(max(tt, 0), span) / span)

    # 각 세그먼트 막대
    row_y = y0 + 16
    for s in segments:
        ts, te = s.get("t_start"), s.get("t_end")
        if not isinstance(ts, (int, float)) or not isinstance(te, (int, float)):
            continue
        col = _seg_color(s)
        active = _is_active(s, t)
        thick = -1 if active else 1
        cv2.rectangle(frame, (xt(ts), row_y - 5), (xt(te), row_y + 5), col, thick)
        _put(frame, f"#{s.get('segment_id')}", (xt(ts) + 2, row_y - 7),
             col, 0.4, 1, bg=False)
        row_y += 12

    # 플레이헤드
    px = xt(t)
    cv2.line(frame, (px, y0 + 4), (px, H - 4), (255, 255, 255), 1)

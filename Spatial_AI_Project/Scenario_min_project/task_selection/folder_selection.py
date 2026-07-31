# -*- coding: utf-8 -*-
"""폴더별(정적환경 조건별) 선별 — 폴더 안에서 'ego motion에 영향을 준 event' 우선.

전제: 데이터가 조건 폴더로 분리 저장됨(주간/야간, 도심로/골목 등). 폴더가 정적환경을
고정하므로, 폴더 안에서는 **ego가 외부 event에 반응해 거동을 바꾼 clip**을 위로 올린다.
= Stage1(run_selection)과 별개 알고리즘: (조건 고정 → VLM 맥락 판정 불요, 대신 '반응 여부' 확인).

패러다임 (B) egomotion + VLM 융합, **반응성 이벤트 가중**.

**승격(2026-07-31): 윈도우 + video 시간정렬 방식이 정본**
  - `consolidate_episodes`로 ego 전이 윈도우 분할(추출 파이프라인과 동일 백본).
  - 윈도우별 ego 반응성(급제동≫정지출발/감속반복>정지>차선변경>회전) 계산.
  - **반응 윈도우의 subclip을 video로 VLM에 보내** 그 거동의 원인이 외부 agent인지 확인
    (시간정렬 → 신호대기 감속 오귀속·주변부 cut_in 투영 차단). 몽타주 정지투영 문제 해소.
  - 윈도우 점수 = ego_react × VLM확인(같은 윈도우). clip 점수 = 최강 반응 윈도우(+다중 소폭 가산).
  - clip-level montage(`rank_folder`/`vlm_reactive`)는 값싼 legacy 경로로 보존.

입력: groups = {조건: [clip_id...]}  또는  groups_from_root("루트/<조건>/<clip_id>/...").
API(정본): rank_folder_windowed · select_folders(→windowed) · verify.
실행: ./run.sh task_selection/folder_selection.py verify [N K seed]   # 폴더 무시 event 검증
      ./run.sh task_selection/folder_selection.py <groups.json> [top_k]
"""
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import classify073 as C
import events
import paths as P
from config import MODEL, SEND_FPS, TEMPERATURE, WINDOW_MAX_SIDE
from dataset import jpeg_to_data_uri, sample_montage, to_data_uri, video_meta, write_subclip
from selection import combined_score   # max(a,b)+0.3·min — legacy clip-level 융합


# --- ego 반응성 점수 (egomotion, 전수·무료) ---------------------------------
def react_ego_score(clip_id):
    """ego motion 반응성 점수. 급제동·정지출발·감속반복에 큰 가중(반응 신호).

    반환 {score, kinds, harsh_decel, terms}. terms=기여 항목(설명·디버깅).
    """
    ev = events.detect_events(clip_id)
    eg = events.load_egomotion_clip(clip_id)
    if not ev.get("ok"):
        return {"score": 0.0, "kinds": [], "harsh_decel": 0.0, "terms": {}}
    kinds = [e["kind"] for e in ev["events"]]
    ks = set(kinds)
    min_ax = float(min(eg["ax"])) if eg.get("ok") and len(eg["ax"]) else 0.0
    harsh = max(0.0, -min_ax - 3.0)                       # 3 m/s² 초과 급감속 = 반응 대리
    n_decel = sum(1 for k in kinds if k == "decelerate")
    stop = int("stop" in ks)
    stop_go = int("stop" in ks and "accelerate" in ks)     # 정지-출발(양보/협상 hesitation)
    turn = int(bool({"turn_left", "turn_right"} & ks))
    lc = int(bool({"lane_change_left", "lane_change_right"} & ks))
    terms = {
        "harsh_brake": 2.5 * min(harsh, 4.0),              # 반응성 최강 신호
        "decel_repeat": 1.0 * min(n_decel, 4),             # 감속 반복(hesitation)
        "stop_go": 0.8 * stop_go,                          # 정지-출발(협상/양보)
        "stop": 1.0 * stop,                                # 정지
        "lane_change": 1.0 * lc,                           # 차선변경(회피 가능)
        "turn": 0.6 * turn,                                # 회전(기동, 중간)
    }
    return {"score": round(sum(terms.values()), 2), "kinds": sorted(ks),
            "harsh_decel": round(-min_ax, 1), "terms": {k: round(v, 2) for k, v in terms.items() if v}}


# --- VLM 반응 확인 (외부 event에 대한 ego 반응인가) --------------------------
_EVT = ["lead_brake", "cut_in", "ped_cyclist", "cross_traffic", "obstacle", "yield_negotiation", "none"]
_SCHEMA = {"type": "object", "additionalProperties": False,
           "required": ["reactive", "score", "event_type", "reason"],
           "properties": {"reactive": {"type": "boolean"}, "score": {"type": "number"},
                          "event_type": {"type": "string", "enum": _EVT},
                          "reason": {"type": "string", "maxLength": 160}}}
_PROMPT = (
    "These are time-ordered frames from a ~20s driving clip (front camera). The static scene type is "
    "already fixed (this clip came from a specific condition folder), so DO NOT judge day/night/road-type.\n"
    "Judge only: is there an EXTERNAL agent or hazard that the EGO vehicle reacts to (brakes/slows/stops/"
    "swerves/yields)? e.g. a lead vehicle braking, a cut-in, a pedestrian/cyclist crossing, cross traffic "
    "at a junction, a road obstacle, or a yield/negotiation. A plain stop at a red light with NO other agent "
    "involved is NOT a reactive event.\n"
    "Return JSON: reactive (true if such an external event is clearly present), score 0-1 (how clear/strong "
    "the reactive event is), event_type (best label; none if not reactive), reason (short). Be calibrated: "
    "routine cruising or a lone red-light stop = low score, reactive=false.")


def vlm_reactive(client, clip_id):
    """VLM 몽타주 → {reactive, score, event_type, reason} (실패 시 {})."""
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
        return {}


# === 승격: 윈도우 + video 시간정렬 (정본) ===================================
LEAD_IN = 3.0
REACT_FLOOR = 1.0        # 이 미만 윈도우는 반응성 약 → VLM 스킵(비용 절감)
MAX_WINDOWS = 3          # clip당 VLM video 확인 윈도우 상한(반응성 강한 순)

_WPROMPT = (
    "This is a SHORT segment (front camera) where the ego vehicle's motion changes "
    "(brake/slow/stop/turn/lane-change). The static scene type is already fixed.\n"
    "Judge ONLY: is this ego motion change caused by an EXTERNAL agent or hazard IN the ego's path? "
    "e.g. a lead vehicle braking, a vehicle actually ENTERING the ego lane (cut-in), a pedestrian/"
    "cyclist crossing the ego path, cross traffic at a junction, a road obstacle, or yielding/negotiation.\n"
    "STRICT: a stop at a red light with NO agent involved is NOT reactive. A vehicle merely PRESENT in "
    "an adjacent lane WITHOUT crossing into the ego lane is NOT a cut-in and NOT reactive. Watch the "
    "MOTION across frames, not a single still.\n"
    "Return JSON: reactive (true only if clearly caused by such an external event), score 0-1, "
    "event_type, reason (short).")


def _win_ego(kinds, min_ax):
    """윈도우 kinds + 윈도우 내 최소 ax → 반응성 점수·기여항."""
    ks = set(kinds)
    harsh = max(0.0, -min_ax - 3.0)
    terms = {"harsh_brake": 2.5 * min(harsh, 4.0),
             "decel_repeat": 1.0 * min(sum(1 for k in kinds if k == "decelerate"), 4),
             "stop_go": 0.8 * int("stop" in ks and "accelerate" in ks),
             "stop": 1.0 * int("stop" in ks),
             "lane_change": 1.0 * int(bool({"lane_change_left", "lane_change_right"} & ks)),
             "turn": 0.6 * int(bool({"turn_left", "turn_right"} & ks))}
    return round(sum(terms.values()), 2), {k: round(v, 2) for k, v in terms.items() if v}


def clip_windows(clip_id):
    """ego 전이 윈도우 + 윈도우별 반응성. 반환 (dur, [{w0,w1,kinds,ego_react,terms}])."""
    ev = events.detect_events(clip_id)
    if not ev.get("ok"):
        return None, []
    try:
        dur = video_meta(P.video_path(clip_id))["duration_s"]
    except Exception:
        return None, []
    eg = events.load_egomotion_clip(clip_id)
    eps = C.consolidate_episodes(ev["events"])
    out = []
    for ep in eps:
        w0 = max(0.0, ep["onset"] - LEAD_IN); w1 = min(dur, ep["t1"] + 1.0)
        min_ax = 0.0
        if eg.get("ok"):
            m = (eg["t"] >= w0) & (eg["t"] <= w1)
            if m.sum():
                min_ax = float(np.min(eg["ax"][m]))
        sc, terms = _win_ego(ep["kinds"], min_ax)
        out.append({"w0": round(w0, 1), "w1": round(w1, 1), "kinds": ep["kinds"],
                    "ego_react": sc, "terms": terms})
    return dur, out


def vlm_window(client, path, w0, w1, tmp):
    """반응 윈도우 subclip(video) → VLM 반응 원인 확인 {reactive,score,event_type,reason}."""
    sub = tmp / f"w_{int(w0 * 10)}_{int(w1 * 10)}.mp4"
    try:
        write_subclip(path, w0, w1, sub, WINDOW_MAX_SIDE, SEND_FPS)
        uri = to_data_uri(sub); sub.unlink(missing_ok=True)
        r = client.chat.completions.create(
            model=MODEL, temperature=TEMPERATURE, max_tokens=512,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _WPROMPT},
                {"type": "video_url", "video_url": {"url": uri}}]}],
            extra_body={"guided_json": _SCHEMA})
        return json.loads(r.choices[0].message.content)
    except Exception:
        return {}


def score_clip_windowed(clip_id, client):
    """윈도우별 ego 반응성 × VLM(video, 같은 윈도우) 확인 → clip 점수(최강 반응 윈도우)."""
    dur, wins = clip_windows(clip_id)
    if not wins:
        return {"clip_id": clip_id, "clip_score": 0.0, "windows": [], "best": None, "n_react_win": 0}
    react_wins = sorted([w for w in wins if w["ego_react"] >= REACT_FLOOR],
                        key=lambda w: -w["ego_react"])[:MAX_WINDOWS]
    path = str(P.video_path(clip_id))
    scored = []
    tmp = Path(tempfile.mkdtemp(prefix="winsel_"))
    try:
        for w in react_wins:
            v = vlm_window(client, path, w["w0"], w["w1"], tmp)
            vs = float(v.get("score", 0) or 0); reactive = bool(v.get("reactive"))
            # 시간정렬 grounding: ego 반응(윈도우) × VLM(같은 윈도우 video). 미확인 시 대폭 감쇠.
            wscore = w["ego_react"] * (vs if reactive else vs * 0.2)
            scored.append({**w, "vlm": v, "vlm_score": round(vs, 2),
                           "reactive": reactive, "event_type": v.get("event_type"),
                           "reason": v.get("reason"), "wscore": round(wscore, 3)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not scored:
        return {"clip_id": clip_id, "clip_score": 0.0, "windows": [], "best": None, "n_react_win": 0}
    best = max(scored, key=lambda s: s["wscore"])
    clip_score = round(best["wscore"] + 0.15 * (len(scored) - 1), 3)  # 다중 반응 윈도우 소폭 가산
    return {"clip_id": clip_id, "clip_score": clip_score, "windows": scored,
            "best": best, "n_react_win": len(scored)}


def rank_folder_windowed(clip_ids, client_pool, top_k=None, workers=8):
    """윈도우+video 시간정렬 랭킹(정본). ego_react/clip_score는 이 폴더 max로 정규화."""
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_clip_windowed, c, client_pool[i % len(client_pool)]): c
                for i, c in enumerate(clip_ids)}
        for fut in as_completed(futs):
            rows.append(fut.result())
    max_s = max((r["clip_score"] for r in rows), default=1.0) or 1.0
    max_e = max((r["best"]["ego_react"] for r in rows if r["best"]), default=1.0) or 1.0
    for r in rows:
        r["score_norm"] = round(r["clip_score"] / max_s, 3)
        r["ego_norm"] = round((r["best"]["ego_react"] / max_e) if r["best"] else 0.0, 3)
        r["vlm_score"] = r["best"]["vlm_score"] if r["best"] else 0.0
    rows.sort(key=lambda r: -r["clip_score"])
    return rows[:top_k] if top_k else rows


# --- (legacy) clip-level montage 랭킹 --------------------------------------
def _score_one(clip_id, client):
    return {"clip_id": clip_id, "ego": react_ego_score(clip_id), "vlm": vlm_reactive(client, clip_id)}


def rank_folder(clip_ids, client_pool, top_k=None, workers=12):
    """한 폴더(조건) 안의 clip을 반응성 결합점수로 랭킹. ego_norm은 이 폴더 max 기준.

    반환: [{clip_id, ego, vlm, ego_norm, vlm_score, combined} ...정렬] (top_k 시 상위 K).
    """
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_score_one, c, client_pool[i % len(client_pool)]): c
                for i, c in enumerate(clip_ids)}
        for fut in as_completed(futs):
            rows.append(fut.result())
    max_ego = max((r["ego"]["score"] for r in rows), default=1.0) or 1.0
    for r in rows:
        en = r["ego"]["score"] / max_ego
        vs = float(r["vlm"].get("score", 0) or 0)
        r["ego_norm"] = round(en, 3); r["vlm_score"] = round(vs, 3)
        r["combined"] = round(combined_score(en, vs), 3)
    rows.sort(key=lambda r: -r["combined"])
    return rows[:top_k] if top_k else rows


def select_folders(groups, client_pool, top_k=None):
    """조건 폴더별로 독립 랭킹(윈도우+video 정본). groups={조건:[clip_ids]}. 반환 {조건: [rows]}."""
    return {cond: rank_folder_windowed(cids, client_pool, top_k) for cond, cids in groups.items()}


def groups_from_root(root):
    """디렉토리 트리 '루트/<조건>/<clip_id>/...' → {조건:[clip_ids]}."""
    root = Path(root)
    return {sub.name: sorted(d.name for d in sub.iterdir() if d.is_dir())
            for sub in root.iterdir() if sub.is_dir()}


def _verify(n=300, k=50, seed=42):
    """폴더 구분 없이 event 기반 선별 검증 — 윈도우+video 정본(run_selection과 동일 300 표본).
    산출: gold_label/select/{winevent_index.html, winevent_select{n}.json}."""
    from collections import Counter
    import review
    from client import build_client_pool
    OUT = Path("/home/daejun/vla-tagging/gold_label/select")
    sample, total, ngold, npool = review.sample_pool(n, seed)
    print(f"[verify·window] 전체 {total} · gold {ngold} 제외 · 유효 {npool} · 무작위 {len(sample)} (seed={seed})", flush=True)
    pool = build_client_pool()
    print(f"replica {len(pool)} · 윈도우+video 반응성 랭킹(폴더 무시, 단일 그룹)...", flush=True)
    rows = rank_folder_windowed(sample, pool)              # 전체를 한 그룹으로
    top = rows[:k]
    best_evt = lambda r: (r["best"]["event_type"] if r["best"] else None)
    n_react = sum(1 for r in top if r["best"] and r["best"]["reactive"])
    evt = Counter(best_evt(r) for r in top)
    nwin = [r["n_react_win"] for r in top]
    print(f"상위{k}: reactive={n_react}/{k} · event_type={dict(evt)} · 반응윈도우수 평균={round(sum(nwin)/max(len(nwin),1),1)}", flush=True)
    # 저장
    slim = [{"clip_id": r["clip_id"], "clip_score": r["clip_score"], "score_norm": r["score_norm"],
             "ego_norm": r["ego_norm"], "vlm_score": r["vlm_score"], "n_react_win": r["n_react_win"],
             "best": ({"win": [r["best"]["w0"], r["best"]["w1"]], "event_type": r["best"]["event_type"],
                       "reactive": r["best"]["reactive"], "reason": r["best"]["reason"],
                       "ego_react": r["best"]["ego_react"], "terms": r["best"]["terms"]}
                      if r["best"] else None)} for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"winevent_select{n}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1))
    (OUT / f"winevent_selected{k}.json").write_text(json.dumps([r["clip_id"] for r in top], ensure_ascii=False, indent=1))
    print(f"상위{k} 영상 트랜스코드...", flush=True)
    review.transcode_top([r["clip_id"] for r in top], OUT / "vids")
    def label(r):
        b = r["best"]
        if not b:
            return "—"
        return (b["event_type"] or "none") + (" ✓반응" if b["reactive"] else " (미확인)")
    def reason(r):
        b = r["best"]
        return f"@{b['w0']}–{b['w1']}s · {b['reason'] or ''}" if b else ""
    rrows = [{"clip_id": r["clip_id"], "combined": r["score_norm"], "ego_norm": r["ego_norm"],
              "vlm_score": r["vlm_score"], "label": label(r), "reason": reason(r),
              "arc": (r["best"]["kinds"] if r["best"] else [])} for r in rows]
    review.build_review_html(
        rrows, k, f"Event 선별 검증(윈도우+video) — 상위 {k}/{len(rows)}",
        f"윈도우별 ego반응×VLM(video,같은윈도우) 시간정렬 · 폴더 무시 · reactive {n_react}/{k} · {dict(evt)}",
        OUT / "winevent_index.html")
    print(f"완료 → {OUT}/winevent_index.html · winevent_select{n}.json", flush=True)


def _main():
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        _verify(int(sys.argv[2]) if len(sys.argv) > 2 else 300,
                int(sys.argv[3]) if len(sys.argv) > 3 else 50,
                int(sys.argv[4]) if len(sys.argv) > 4 else 42)
        return
    if len(sys.argv) < 2:
        raise SystemExit("usage: folder_selection.py verify [N K seed] | <groups.json|root_dir> [top_k]")
    src = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if os.path.isdir(src):
        groups = groups_from_root(src)
    else:
        groups = json.loads(Path(src).read_text())
    from client import build_client_pool
    pool = build_client_pool()
    print(f"폴더 {len(groups)}개 · replica {len(pool)} · top_k={top_k}", flush=True)
    out = {}
    for cond, cids in groups.items():
        rows = rank_folder(cids, pool, top_k)
        sel = [{"clip_id": r["clip_id"], "combined": r["combined"], "ego_norm": r["ego_norm"],
                "vlm_score": r["vlm_score"], "event_type": r["vlm"].get("event_type"),
                "reactive": r["vlm"].get("reactive"), "ego_terms": r["ego"].get("terms")} for r in rows]
        out[cond] = sel
        n_react = sum(1 for s in sel if s["reactive"])
        print(f"  [{cond}] {len(cids)}→{len(sel)} · reactive={n_react} · "
              f"top: {sel[0]['clip_id'][:8] if sel else '-'} C={sel[0]['combined'] if sel else '-'}", flush=True)
    res = Path("/home/daejun/vla-tagging/gold_label/select/folder_selection.json")
    res.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"완료 → {res}", flush=True)


if __name__ == "__main__":
    _main()

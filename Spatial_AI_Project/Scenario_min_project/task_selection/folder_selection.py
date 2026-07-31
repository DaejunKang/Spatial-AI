# -*- coding: utf-8 -*-
"""폴더별(정적환경 조건별) 선별 — 폴더 안에서 'ego motion에 영향을 준 event' 우선.

전제: 데이터가 조건 폴더로 분리 저장됨(주간/야간, 도심로/골목 등). 폴더가 정적환경을
고정하므로, 폴더 안에서는 **ego가 외부 event에 반응해 거동을 바꾼 clip**을 위로 올린다.
= Stage1(run_selection)과 별개 알고리즘: (조건 고정 → VLM 맥락 판정 불요, 대신 '반응 여부' 확인).

패러다임 (B) egomotion + VLM 융합, **반응성 이벤트 가중**:
  - ego 반응성 점수(egomotion): 급제동(agent 반응 대리) ≫ 정지-출발/감속반복 > 정지 > 차선변경 > 회전
  - VLM 확인: 그 거동이 **외부 agent/hazard 반응**인지(신호대기 routine 정지 배제)
  - combined = max(ego_norm, vlm) + 0.3·min   (한 채널만 강해도 유의미 + 합의 보너스)
  - 정규화(ego_norm)·랭킹은 **폴더 단위**.

입력: groups = {조건: [clip_id...]}  또는  groups_from_root("루트/<조건>/<clip_id>/...").
API: react_ego_score · vlm_reactive · rank_folder · select_folders.
실행: ./run.sh task_selection/folder_selection.py <groups.json> [top_k]
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import events
import paths as P
from config import MODEL, TEMPERATURE
from dataset import jpeg_to_data_uri, sample_montage
from selection import combined_score   # max(a,b)+0.3·min — Stage1과 동일 융합


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


# --- 폴더 단위 랭킹 ----------------------------------------------------------
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
    """조건 폴더별로 독립 랭킹. groups={조건:[clip_ids]}. 반환 {조건: [ranked rows]}."""
    return {cond: rank_folder(cids, client_pool, top_k) for cond, cids in groups.items()}


def groups_from_root(root):
    """디렉토리 트리 '루트/<조건>/<clip_id>/...' → {조건:[clip_ids]}."""
    root = Path(root)
    return {sub.name: sorted(d.name for d in sub.iterdir() if d.is_dir())
            for sub in root.iterdir() if sub.is_dir()}


def _verify(n=300, k=50, seed=42):
    """폴더 구분 없이 event 기반(반응성) 선별만 검증 — run_selection과 동일 300 표본.
    산출: gold_label/select/{event_index.html, event_select{n}.json} (Stage1 index.html과 비교용)."""
    from collections import Counter
    import review
    from client import build_client_pool
    OUT = Path("/home/daejun/vla-tagging/gold_label/select")
    sample, total, ngold, npool = review.sample_pool(n, seed)
    print(f"[verify] 전체 {total} · gold {ngold} 제외 · 유효 {npool} · 무작위 {len(sample)} (seed={seed})", flush=True)
    pool = build_client_pool()
    print(f"replica {len(pool)} · 반응성 랭킹(폴더 무시, 단일 그룹)...", flush=True)
    rows = rank_folder(sample, pool)                       # 전체를 한 그룹으로
    top = rows[:k]
    n_react = sum(1 for r in top if r["vlm"].get("reactive"))
    evt = Counter(r["vlm"].get("event_type") for r in top)
    print(f"상위{k}: reactive={n_react}/{k} · event_type={dict(evt)}", flush=True)
    # 저장 + 리뷰 rows 규약 변환
    slim = [{"clip_id": r["clip_id"], "combined": r["combined"], "ego_norm": r["ego_norm"],
             "vlm_score": r["vlm_score"], "reactive": r["vlm"].get("reactive"),
             "event_type": r["vlm"].get("event_type"), "reason": r["vlm"].get("reason"),
             "ego_terms": r["ego"].get("terms"), "ego_kinds": r["ego"].get("kinds", [])} for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"event_select{n}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1))
    (OUT / f"event_selected{k}.json").write_text(json.dumps([r["clip_id"] for r in top], ensure_ascii=False, indent=1))
    print(f"상위{k} 영상 트랜스코드...", flush=True)
    review.transcode_top([r["clip_id"] for r in top], OUT / "vids")
    rrows = [{"clip_id": r["clip_id"], "combined": r["combined"], "ego_norm": r["ego_norm"],
              "vlm_score": r["vlm_score"],
              "label": (r["vlm"].get("event_type") or "none") + (" ✓반응" if r["vlm"].get("reactive") else ""),
              "reason": r["vlm"].get("reason"), "arc": r["ego"].get("kinds", [])} for r in rows]
    review.build_review_html(
        rrows, k, f"Event 기반 선별 검증 — 상위 {k}/{len(rows)}",
        f"반응성 combined=max(ego,vlm)+0.3·min · 폴더 무시(단일 그룹) · reactive {n_react}/{k} · {dict(evt)}",
        OUT / "event_index.html")
    print(f"완료 → {OUT}/event_index.html · event_select{n}.json", flush=True)


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

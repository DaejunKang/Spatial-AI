# -*- coding: utf-8 -*-
"""전체 workflow(Stage1→Stage2 Track1/Track2) 실행 및 오류 점검 — decisions/DESIGN_LOG.md
[2026-09-08] "NVIDIA 데이터셋 전체 workflow" 조사 참고.

`tag_v08.tag_clip_v08()`(Track1)·`candidates.generate_candidates()`→`retrieve.index_clip()`
(Track2)를 실제로 호출하는 러너가 저장소에 없었다(호출자 0건, git log로 재확인) — 이
스크립트가 최초 실행이다. Stage1(`run_selection.py`)이 산출한 `selected50.json`을 입력
으로 받아(Stage1→Stage2 연결도 이번에 처음 구성) clip마다 S0(events)+Track1+Track2를
전부 돌리고, 예외·CJK 혼입·guided_json 파싱 실패·Track1/Track2 간 arc 불일치를 구조화
기록한다. 실패를 감추지 않고 전부 로그한다(무기록 fallback 금지).

실행: ./run.sh task_episode/run_workflow_audit.py [--n N]
       [--selected gold_label/select/selected50.json]
"""
import argparse
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paths as P
from dataset import video_meta
from client import build_client_pool
import events
import map_lane as M
from classify073 import consolidate_episodes
import tag_v08 as V08
import candidates as CA
import retrieve as R

CJK_RE = re.compile(r"[぀-ヿ一-鿿]")
OUT = Path("/home/daejun/vla-tagging/gold_label/workflow_audit")


def cjk_flag(text):
    return bool(CJK_RE.search(text or ""))


def _count_cjk(rec):
    fields = [rec.get("scene_description"), rec.get("ego_intent"), rec.get("chain_of_causation")]
    for c in rec.get("critical_components") or []:
        fields.append(c.get("description")); fields.append(c.get("why_critical"))
    return sum(1 for f in fields if cjk_flag(f))


def audit_clip(client, pool, clip_id):
    rec = {"clip_id": clip_id}
    path = P.video_path(clip_id)

    # --- S0: 전이검출(production 배선 그대로 — tag_v08.py:222-223과 동일) ---
    t0 = time.time()
    try:
        dur = video_meta(path)["duration_s"]
        curvature_fn = M.default_curvature_fn(clip_id, dur)
        lane_crossing_fn = M.default_lane_crossing_fn(clip_id, dur)
        det = events.detect_events(clip_id, curvature_fn=curvature_fn, lane_crossing_fn=lane_crossing_fn)
        s0 = {"ok": bool(det.get("ok")), "dur": dur, "map_valid": curvature_fn is not None,
              "n_raw_events": len(det.get("events", []))}
        eps = consolidate_episodes(det["events"]) if det.get("ok") else []
        s0["n_episodes"] = len(eps)
        s0["episode_kinds"] = [ep["kinds"] for ep in eps]
        if not det.get("ok"):
            s0["reason"] = det.get("reason")
    except Exception as e:
        s0 = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-400:]}
    s0["elapsed_s"] = round(time.time() - t0, 1)
    rec["s0"] = s0

    # --- Track1: tag_v08 (S1+S2+S3, 최초 실행) ---
    t0 = time.time()
    try:
        t1 = V08.tag_clip_v08(client, path, clip_id)
    except Exception as e:
        t1 = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-400:]}
    t1_summary = {"ok": bool(t1.get("ok")), "elapsed_s": round(time.time() - t0, 1), "error": t1.get("error")}
    if t1.get("ok"):
        recs = t1.get("records", [])
        t1_summary["n_records"] = len(recs)
        t1_summary["n_parse_fail"] = sum(1 for r in recs if r.get("scene_description") is None)
        t1_summary["n_cjk"] = sum(_count_cjk(r) for r in recs)
        t1_summary["arc_by_segment"] = [r["ego_context"]["arc"] for r in recs]
        t1_summary["cause_by_segment"] = [r["cause"] for r in recs]
    rec["track1"] = t1_summary

    # --- Track2: candidates -> retrieve (S1+S2, 최초 실행) ---
    t0 = time.time()
    cand = None
    try:
        cand = CA.generate_candidates(pool, path, clip_id)
        idx = R.index_clip(cand) if cand.get("ok") else None
    except Exception as e:
        cand = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-400:]}
        idx = None
    t2_summary = {"ok": bool(cand.get("ok")), "elapsed_s": round(time.time() - t0, 1), "error": cand.get("error")}
    if cand.get("ok"):
        t2_summary["n_episodes"] = len(cand.get("episodes", []))
        t2_summary["arc_by_segment"] = [ep.get("arc", []) for ep in cand.get("episodes", [])]
        if idx:
            t2_summary["n_tags_total"] = sum(len(e["tags"]) for e in idx["episodes"])
    rec["track2"] = t2_summary

    # --- Track1 vs Track2 arc 불일치 — 같은 clip인데 detect_events 호출 조건이 다름
    # (Track1=곡률보정 O, Track2=곡률보정·lane_crossing 미배선) — 설계 파이프라인 오류 실증용
    if t1_summary["ok"] and t2_summary["ok"]:
        a1 = sorted(set(k for arc in t1_summary["arc_by_segment"] for k in arc))
        a2 = sorted(set(k for arc in t2_summary["arc_by_segment"] for k in arc))
        rec["track_arc_mismatch"] = (a1 != a2)
        rec["track1_arc_union"] = a1
        rec["track2_arc_union"] = a2

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="gold_label/select/selected50.json")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4,
                     help="clip 단위 동시 실행 수(2026-09-08 병렬화 — clip 간 의존성 없음)")
    args = ap.parse_args()

    clip_ids = json.loads(Path(args.selected).read_text())
    if args.n:
        clip_ids = clip_ids[:args.n]
    print(f"대상 clip {len(clip_ids)}개 (source={args.selected})", flush=True)

    pool = build_client_pool()
    print(f"replica {len(pool)}개 가동 · workers={args.workers}", flush=True)

    # clip마다 Track1 client를 라운드로빈 배정(기존엔 pool[0] 고정 — 병렬화 김에 분산).
    # Track2는 audit_clip 내부에서 자체적으로 pool 전체를 n-vote 라운드로빈에 씀.
    results_by_id = {}
    t_wall0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(audit_clip, pool[i % len(pool)], pool, cid): cid
                for i, cid in enumerate(clip_ids)}
        done = 0
        for fut in as_completed(futs):
            cid = futs[fut]
            done += 1
            try:
                results_by_id[cid] = fut.result()
            except Exception as e:
                results_by_id[cid] = {
                    "clip_id": cid,
                    "s0": {"ok": False, "error": f"{type(e).__name__}: {e}"},
                    "track1": {"ok": False, "error": f"{type(e).__name__}: {e}"},
                    "track2": {"ok": False, "error": f"{type(e).__name__}: {e}"},
                }
            print(f"[{done}/{len(clip_ids)}] {cid[:8]} 완료", flush=True)
    wall_s = round(time.time() - t_wall0, 1)
    results = [results_by_id[cid] for cid in clip_ids]  # 원래 순서로 복원

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    n = len(results)
    s0_ok = sum(1 for r in results if r["s0"].get("ok"))
    t1_ok = sum(1 for r in results if r["track1"].get("ok"))
    t2_ok = sum(1 for r in results if r["track2"].get("ok"))
    n_cjk = sum(r["track1"].get("n_cjk", 0) for r in results)
    n_parse_fail = sum(r["track1"].get("n_parse_fail", 0) for r in results)
    n_mismatch = sum(1 for r in results if r.get("track_arc_mismatch"))
    n_map_valid = sum(1 for r in results if r["s0"].get("map_valid"))
    both_ok = sum(1 for r in results if r["track1"].get("ok") and r["track2"].get("ok"))
    print(f"\n=== 요약 ({n} clip) ===")
    print(f"S0 성공 {s0_ok}/{n} · Track1 성공 {t1_ok}/{n} · Track2 성공 {t2_ok}/{n}")
    print(f"map_valid {n_map_valid}/{n} · CJK혼입 {n_cjk}건 · guided_json 파싱실패 {n_parse_fail}건")
    print(f"Track1/Track2 arc 불일치 {n_mismatch}/{both_ok}(둘 다 성공한 clip 기준)")
    print(f"wall-clock {wall_s}s = {wall_s/60:.1f}분 (workers={args.workers})")
    print(f"결과 저장 -> {OUT / 'report.json'}")


if __name__ == "__main__":
    main()

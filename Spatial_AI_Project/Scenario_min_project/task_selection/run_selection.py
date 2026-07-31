# -*- coding: utf-8 -*-
"""Task「선별」(Stage 1) 러너 — 무작위 N clip → 상위 K + <video> 리뷰 HTML.

**기본(정본) = 윈도우+video 반응성**(2026-07-31 승격):
  consolidate_episodes로 ego 전이 윈도우 분할 → 윈도우별 ego 반응성 + 반응 윈도우 subclip을
  video로 VLM에 보내 원인(외부 agent) 확인(시간정렬 grounding). 몽타주 정지투영/오귀속 해소.
  실측(obj3d 대조): 몽타주 cut_in corroboration 6%(top50 0%) → 윈도우+video로 과검 대부분 제거.
  ※ Stage1은 obj3d 없이 egomotion(CAN)+VLM video 만 사용(전제 유지).

legacy: `montage` = 클립레벨 흥미도(selection.rank_clips, 값싸지만 투영 오류 많음).

실행:
  ./run.sh task_selection/run_selection.py                 # 기본 windowed, N=300 K=50 seed=42
  ./run.sh task_selection/run_selection.py 300 50 42
  ./run.sh task_selection/run_selection.py review           # 재랭킹 없이 select{N}.json→HTML
  ./run.sh task_selection/run_selection.py montage 300 50 42 # legacy 흥미도
산출: gold_label/select/{select{N}.json, selected{K}.json, index.html, vids/}
"""
import json
import sys
from pathlib import Path

import review
from client import build_client_pool

OUT = Path("/home/daejun/vla-tagging/gold_label/select")


# --- windowed(정본) 스키마·리뷰 변환 ---------------------------------------
def _win_slim(rows):
    return [{"clip_id": r["clip_id"], "clip_score": r["clip_score"], "score_norm": r["score_norm"],
             "ego_norm": r["ego_norm"], "vlm_score": r["vlm_score"], "n_react_win": r["n_react_win"],
             "best": ({"win": [r["best"]["w0"], r["best"]["w1"]], "kinds": r["best"]["kinds"],
                       "event_type": r["best"]["event_type"], "reactive": r["best"]["reactive"],
                       "reason": r["best"]["reason"], "ego_react": r["best"]["ego_react"],
                       "terms": r["best"]["terms"]} if r["best"] else None)} for r in rows]


def _win_review_rows(slim):
    def lab(b):
        return "—" if not b else (b["event_type"] or "none") + (" ✓반응" if b["reactive"] else " (미확인)")
    def rsn(b):
        return "" if not b else f"@{b['win'][0]}–{b['win'][1]}s · {b.get('reason') or ''}"
    return [{"clip_id": r["clip_id"], "combined": r["score_norm"], "ego_norm": r["ego_norm"],
             "vlm_score": r["vlm_score"], "label": lab(r["best"]), "reason": rsn(r["best"]),
             "arc": (r["best"].get("kinds", []) if r["best"] else [])} for r in slim]


def _win_stats(slim, k):
    from collections import Counter
    top = slim[:k]
    n_react = sum(1 for r in top if r["best"] and r["best"]["reactive"])
    evt = dict(Counter((r["best"]["event_type"] if r["best"] else None) for r in top))
    return n_react, evt


def cmd_windowed(n, k, seed):
    from folder_selection import rank_folder_windowed
    sample, total, ngold, npool = review.sample_pool(n, seed)
    print(f"전체 {total} · gold {ngold} 제외 · 유효 {npool} · 무작위 {len(sample)} (seed={seed})", flush=True)
    pool = build_client_pool()
    print(f"replica {len(pool)} · 윈도우+video 반응성 랭킹...", flush=True)
    rows = rank_folder_windowed(sample, pool)
    slim = _win_slim(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"select{n}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1))
    (OUT / f"selected{k}.json").write_text(json.dumps([r["clip_id"] for r in slim[:k]], ensure_ascii=False, indent=1))
    _render(slim, k, n)


def cmd_review(n, k):
    f = OUT / f"select{n}.json"
    if not f.exists():
        raise SystemExit(f"{f} 없음 — 먼저 windowed 랭킹 필요")
    slim = json.loads(f.read_text())
    print(f"[review] {f.name} 재사용 → 트랜스코드 + HTML", flush=True)
    _render(slim, k, n)


def _render(slim, k, n):
    n_react, evt = _win_stats(slim, k)
    print(f"상위{k}: reactive={n_react}/{k} · event_type={evt}", flush=True)
    print(f"상위{k} 영상 트랜스코드...", flush=True)
    review.transcode_top([r["clip_id"] for r in slim[:k]], OUT / "vids")
    review.build_review_html(
        _win_review_rows(slim), k, f"Stage1 선별(윈도우+video) — 상위 {k}/{len(slim)}",
        f"윈도우별 ego반응×VLM(video) 시간정렬 · reactive {n_react}/{k} · {evt}",
        OUT / "index.html")
    print(f"완료 → {OUT}/index.html · select{n}.json", flush=True)


# --- legacy: 클립레벨 몽타주 흥미도 -----------------------------------------
def cmd_montage(n, k, seed):
    import selection as S
    sample, total, ngold, npool = review.sample_pool(n, seed)
    print(f"[montage·legacy] 유효 {npool} · 무작위 {len(sample)} (seed={seed})", flush=True)
    pool = build_client_pool()
    rows = S.rank_clips(sample, pool)   # 흥미도(순차)
    slim = [{"clip_id": r["clip_id"], "combined": r["combined"],
             "ego_norm": round(r["ego"]["score"] / (max((x["ego"]["score"] for x in rows), default=1) or 1), 3),
             "vlm_score": float(r["vlm"].get("score", 0) or 0),
             "label": r["vlm"].get("category", "-"), "reason": r["vlm"].get("reason"),
             "arc": r["ego"].get("kinds", [])} for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"montage_select{n}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1))
    review.transcode_top([r["clip_id"] for r in slim[:k]], OUT / "vids")
    review.build_review_html(slim, k, f"Stage1 선별(montage·legacy) — 상위 {k}/{len(slim)}",
                             "클립레벨 흥미도 max(ego,vlm)+0.3·min (legacy)", OUT / "montage_index.html")
    print(f"완료 → {OUT}/montage_index.html", flush=True)


def main():
    a = sys.argv[1:]
    if a and a[0] == "review":
        cmd_review(int(a[1]) if len(a) > 1 else 300, int(a[2]) if len(a) > 2 else 50)
    elif a and a[0] == "montage":
        cmd_montage(int(a[1]) if len(a) > 1 else 300, int(a[2]) if len(a) > 2 else 50,
                    int(a[3]) if len(a) > 3 else 42)
    else:
        cmd_windowed(int(a[0]) if a else 300, int(a[1]) if len(a) > 1 else 50,
                     int(a[2]) if len(a) > 2 else 42)


if __name__ == "__main__":
    main()

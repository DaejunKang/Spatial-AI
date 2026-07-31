# -*- coding: utf-8 -*-
"""Task「선별」(Stage 1) 러너 — 무작위 N clip 랭킹 → 상위 K 선정 + 리뷰 HTML(영상 재생).

selection.py(ego arc 점수 + VLM 몽타주 흥미도 + 결합 랭킹)의 진입점.
전체 1966 clip에서 gold 제외 후 seed 고정 무작위 N을 뽑아 4-replica 병렬 랭킹.
리뷰 HTML은 상위 K를 **H.264 트랜스코드(vids/)** 한 <video>로 재생 + 몽타주 poster + 점수/근거.

실행:
  ./run.sh task_selection/run_selection.py             # 기본 N=300 K=50 seed=42 (랭킹→트랜스코드→HTML)
  ./run.sh task_selection/run_selection.py 300 50 42   # 파라미터 지정
  ./run.sh task_selection/run_selection.py review       # 재랭킹 없이 기존 selectN.json 재사용→트랜스코드→HTML
산출: gold_label/select/{select300.json, selected50.json, index.html, vids/}
"""
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paths as P
import selection as S
from client import build_client_pool
from dataset import jpeg_to_data_uri, sample_montage

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

OUT = Path("/home/daejun/vla-tagging/gold_label/select")
VIDS = OUT / "vids"
GOLD = Path("/home/daejun/vla-tagging/gold.json")

REVIEW = len(sys.argv) > 1 and sys.argv[1] == "review"
N = int(sys.argv[1]) if (len(sys.argv) > 1 and not REVIEW) else 300
K = int(sys.argv[2]) if len(sys.argv) > 2 else 50
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42


def valid_pool():
    """전체 clip 디렉토리 중 video+egomotion 존재. gold 50 제외."""
    gold = set(json.loads(GOLD.read_text()).keys()) if GOLD.exists() else set()
    cids = sorted(d for d in os.listdir(P.BASE) if (Path(P.BASE) / d).is_dir())
    pool = [c for c in cids
            if c not in gold and P.video_path(c).exists() and P.egomotion_path(c).exists()]
    return pool, len(cids), len(gold)


def score_one(cid, client):
    return {"clip_id": cid, "ego": S.ego_score(cid), "vlm": S.vlm_interest(client, cid) or {}}


def rank_parallel(sample, clients, workers=12):
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_one, cid, clients[i % len(clients)]): cid
                for i, cid in enumerate(sample)}
        for fut in as_completed(futs):
            rows.append(fut.result()); done += 1
            if done % 25 == 0:
                print(f"  ...{done}/{len(sample)}", flush=True)
    max_ego = max((r["ego"]["score"] for r in rows), default=1.0) or 1.0
    for r in rows:
        en = r["ego"]["score"] / max_ego
        vs = float(r["vlm"].get("score", 0) or 0)
        r["ego_norm"] = round(en, 3); r["vlm_score"] = round(vs, 3)
        r["combined"] = round(S.combined_score(en, vs), 3)
    rows.sort(key=lambda r: -r["combined"])
    return rows


def channel_stats(top):
    c = {"ego_dominant": 0, "vlm_dominant": 0, "both": 0, "weak": 0}
    hi = 0.45
    for r in top:
        e, v = r["ego_norm"], r["vlm_score"]
        if e >= hi and v >= hi: c["both"] += 1
        elif e >= hi: c["ego_dominant"] += 1
        elif v >= hi: c["vlm_dominant"] += 1
        else: c["weak"] += 1
    return c


def montage_uri(cid):
    try:
        return jpeg_to_data_uri(sample_montage(str(P.video_path(cid)), num_frames=12, cell_max_side=360))
    except Exception:
        return ""


def transcode(cid):
    """상위 K clip을 브라우저 재생용 H.264(640px)로 vids/<clip>.mp4 저장."""
    out = VIDS / f"{cid}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return cid, True
    try:
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(P.video_path(cid)),
                        "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "28", "-an", "-movflags", "+faststart", str(out)],
                       check=True, timeout=120)
        return cid, True
    except Exception as e:
        return cid, False


def transcode_top(top, workers=6):
    VIDS.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for cid, good in ex.map(transcode, [r["clip_id"] for r in top]):
            ok += int(good)
    print(f"  트랜스코드 {ok}/{len(top)} OK → {VIDS}", flush=True)


def build_html(rows, top, stats):
    cards = []
    for rank, r in enumerate(top, 1):
        cid = r["clip_id"]; v = r["vlm"]
        poster = montage_uri(cid)
        vid = VIDS / f"{cid}.mp4"
        media = (f'<video controls preload=none playsinline poster="{poster}">'
                 f'<source src="vids/{cid}.mp4" type="video/mp4"></video>'
                 if vid.exists() else
                 (f'<img src="{poster}" loading=lazy>' if poster else '<div class=noimg>영상/몽타주 실패</div>'))
        reason = (v.get("reason") or "").replace("<", "&lt;")
        kinds = ", ".join(r["ego"].get("kinds", [])) or "—"
        cards.append(f"""<div class=card>
  <div class=rk>#{rank}</div>{media}
  <div class=meta><div class=cid>{cid[:8]}</div>
    <div class=sc><b>{r['combined']}</b> <span class=mut>C</span>
      · <span class=ego>{r['ego_norm']}</span> ego · <span class=vlm>{r['vlm_score']}</span> vlm</div>
    <div class=cat>{v.get('category','-')}</div><div class=rsn>{reason}</div>
    <div class=arc>arc: {kinds}</div></div></div>""")
    trows = "".join(
        f"<tr class='{'sel' if i < len(top) else ''}'><td>{i+1}</td><td>{r['clip_id'][:8]}</td>"
        f"<td>{r['combined']}</td><td>{r['ego_norm']}</td><td>{r['vlm_score']}</td>"
        f"<td>{r['vlm'].get('category','-')}</td></tr>" for i, r in enumerate(rows))
    st = " · ".join(f"{k} {v}" for k, v in stats.items())
    return f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Stage1 선별 검증 (top {len(top)}/{len(rows)})</title>
<style>
:root{{--bg:#f6f8f9;--pn:#fff;--ink:#16202a;--mut:#6a7a86;--ln:#dde5e9;--ac:#0d7fa6;--ego:#2c9c68;--vlm:#b5641e}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d1417;--pn:#151f24;--ink:#e6eef2;--mut:#8598a2;--ln:#243138;--ac:#2ab0cc;--ego:#3fb27f;--vlm:#d89a52}}}}
:root[data-theme=dark]{{--bg:#0d1417;--pn:#151f24;--ink:#e6eef2;--mut:#8598a2;--ln:#243138;--ac:#2ab0cc;--ego:#3fb27f;--vlm:#d89a52}}
:root[data-theme=light]{{--bg:#f6f8f9;--pn:#fff;--ink:#16202a;--mut:#6a7a86;--ln:#dde5e9;--ac:#0d7fa6;--ego:#2c9c68;--vlm:#b5641e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,"Noto Sans KR",sans-serif;line-height:1.4}}
header{{padding:14px 20px;border-bottom:1px solid var(--ln)}}h1{{font-size:16px;margin:0 0 4px}}
.sub{{font-size:12.5px;color:var(--mut);font-family:ui-monospace,monospace}}
.wrap{{max-width:1180px;margin:0 auto;padding:16px 20px 60px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:var(--pn);border:1px solid var(--ln);border-radius:10px;overflow:hidden;position:relative}}
.card video,.card img{{width:100%;display:block;background:#000}}.noimg{{padding:30px;text-align:center;color:var(--mut)}}
.rk{{position:absolute;top:6px;left:6px;background:var(--ac);color:#fff;font:600 12px ui-monospace;padding:2px 7px;border-radius:6px;z-index:2}}
.meta{{padding:8px 10px}}.cid{{font:11px ui-monospace;color:var(--mut)}}
.sc{{font:13px ui-monospace;margin:2px 0}}.ego{{color:var(--ego)}}.vlm{{color:var(--vlm)}}.mut{{color:var(--mut);font-size:10px}}
.cat{{font-weight:600;font-size:13px;margin-top:3px}}.rsn{{font-size:12px;color:var(--mut);margin:2px 0}}
.arc{{font:10.5px ui-monospace;color:var(--mut)}}
table{{width:100%;border-collapse:collapse;font:12px ui-monospace;margin-top:10px}}
th,td{{text-align:left;padding:3px 8px;border-bottom:1px solid var(--ln)}}th{{color:var(--mut)}}
tr.sel td{{background:color-mix(in srgb,var(--ac) 10%,transparent)}}
details{{margin-top:20px}}summary{{cursor:pointer;font-weight:600;color:var(--ac)}}
</style></head><body>
<header><h1>Stage1 선별 검증 — 상위 {len(top)} / 무작위 {len(rows)}</h1>
<div class=sub>combined = max(ego,vlm)+0.3·min · ▶재생 클릭 · 채널기여(상위{len(top)}): {st}</div></header>
<div class=wrap><div class=grid>{''.join(cards)}</div>
<details><summary>전체 {len(rows)} 랭킹 테이블 (선정 {len(top)} 강조 = cutoff)</summary>
<table><tr><th>#</th><th>clip</th><th>combined</th><th>ego</th><th>vlm</th><th>category</th></tr>{trows}</table>
</details></div></body></html>"""


def load_saved():
    """review 모드: 기존 select{N}.json → rows 복원(랭킹·점수 그대로)."""
    f = OUT / f"select{N}.json"
    if not f.exists():
        raise SystemExit(f"{f} 없음 — 먼저 랭킹 실행 필요")
    slim = json.loads(f.read_text())
    return [{"clip_id": r["clip_id"], "combined": r["combined"], "ego_norm": r["ego_norm"],
             "vlm_score": r["vlm_score"], "ego": {"kinds": r.get("ego_kinds", [])},
             "vlm": {"category": r.get("vlm_category"), "reason": r.get("vlm_reason")}} for r in slim]


def main():
    if REVIEW:
        print(f"[review] 기존 select{N}.json 재사용 → 트랜스코드 + HTML", flush=True)
        rows = load_saved()
    else:
        pool, total, ngold = valid_pool()
        random.seed(SEED)
        sample = random.sample(pool, min(N, len(pool)))
        print(f"전체 {total} · gold {ngold} 제외 · 유효 {len(pool)} · 무작위 {len(sample)} (seed={SEED})", flush=True)
        clients = build_client_pool()
        print(f"VLM replica {len(clients)} · 랭킹...", flush=True)
        rows = rank_parallel(sample, clients)
        slim = [{"clip_id": r["clip_id"], "combined": r["combined"], "ego_norm": r["ego_norm"],
                 "vlm_score": r["vlm_score"], "ego_kinds": r["ego"].get("kinds", []),
                 "vlm_category": r["vlm"].get("category"), "vlm_reason": r["vlm"].get("reason")}
                for r in rows]
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"select{N}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1))
    top = rows[:K]
    (OUT / f"selected{K}.json").write_text(json.dumps([r["clip_id"] for r in top], ensure_ascii=False, indent=1))
    stats = channel_stats(top)
    print(f"채널기여(상위{K}): {stats}", flush=True)
    print(f"상위{K} 영상 트랜스코드(H.264)...", flush=True)
    transcode_top(top)
    print("몽타주 poster + HTML...", flush=True)
    (OUT / "index.html").write_text(build_html(rows, top, stats), encoding="utf-8")
    print(f"완료 → {OUT}/index.html", flush=True)


if __name__ == "__main__":
    main()

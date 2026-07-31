# -*- coding: utf-8 -*-
"""선별 검증 공용 유틸 — clip 풀 샘플링 · H.264 트랜스코드 · <video> 리뷰 HTML.

run_selection(전역 흥미도)·folder_selection(폴더별 반응성) 검증이 공유.
rows 규약: {clip_id, combined, ego_norm, vlm_score, label, reason, arc(list)}.
"""
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paths as P
from dataset import jpeg_to_data_uri, sample_montage

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

GOLD = Path("/home/daejun/vla-tagging/gold.json")


def valid_pool():
    """전체 clip 디렉토리 중 video+egomotion 존재. gold 제외. (run_selection과 동일 규약)."""
    gold = set(json.loads(GOLD.read_text()).keys()) if GOLD.exists() else set()
    cids = sorted(d for d in os.listdir(P.BASE) if (Path(P.BASE) / d).is_dir())
    pool = [c for c in cids
            if c not in gold and P.video_path(c).exists() and P.egomotion_path(c).exists()]
    return pool, len(cids), len(gold)


def sample_pool(n, seed):
    """seed 고정 무작위 n clip (run_selection과 동일 규약 → 동일 표본 재현)."""
    pool, total, ngold = valid_pool()
    random.seed(seed)
    return random.sample(pool, min(n, len(pool))), total, ngold, len(pool)


def montage_uri(cid):
    try:
        return jpeg_to_data_uri(sample_montage(str(P.video_path(cid)), num_frames=12, cell_max_side=360))
    except Exception:
        return ""


def _transcode(args):
    cid, vids = args
    out = vids / f"{cid}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return True
    try:
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(P.video_path(cid)),
                        "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "28", "-an", "-movflags", "+faststart", str(out)],
                       check=True, timeout=120)
        return True
    except Exception:
        return False


def transcode_top(clip_ids, vids_dir, workers=6):
    vids_dir = Path(vids_dir); vids_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = sum(ex.map(_transcode, [(c, vids_dir) for c in clip_ids]))
    print(f"  트랜스코드 {ok}/{len(clip_ids)} OK → {vids_dir}", flush=True)


def build_review_html(rows, K, title, subtitle, out_path, vids_dirname="vids"):
    """rows(정렬됨) → 상위 K는 <video> 카드 + 전체 랭킹 테이블. out_path 기준 vids_dirname 참조."""
    out_path = Path(out_path)
    vids = out_path.parent / vids_dirname
    top = rows[:K]
    cards = []
    for rank, r in enumerate(top, 1):
        cid = r["clip_id"]
        poster = montage_uri(cid)
        media = (f'<video controls preload=none playsinline poster="{poster}">'
                 f'<source src="{vids_dirname}/{cid}.mp4" type="video/mp4"></video>'
                 if (vids / f"{cid}.mp4").exists() else
                 (f'<img src="{poster}" loading=lazy>' if poster else '<div class=noimg>영상 실패</div>'))
        reason = (r.get("reason") or "").replace("<", "&lt;")
        arc = ", ".join(r.get("arc", [])) or "—"
        cards.append(f"""<div class=card><div class=rk>#{rank}</div>{media}
  <div class=meta><div class=cid>{cid[:8]}</div>
    <div class=sc><b>{r['combined']}</b> <span class=mut>C</span>
      · <span class=ego>{r['ego_norm']}</span> ego · <span class=vlm>{r['vlm_score']}</span> vlm</div>
    <div class=cat>{r.get('label','-')}</div><div class=rsn>{reason}</div>
    <div class=arc>arc: {arc}</div></div></div>""")
    trows = "".join(
        f"<tr class='{'sel' if i < K else ''}'><td>{i+1}</td><td>{r['clip_id'][:8]}</td>"
        f"<td>{r['combined']}</td><td>{r['ego_norm']}</td><td>{r['vlm_score']}</td>"
        f"<td>{r.get('label','-')}</td></tr>" for i, r in enumerate(rows))
    html = f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>{title}</title>
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
<header><h1>{title}</h1><div class=sub>{subtitle}</div></header>
<div class=wrap><div class=grid>{''.join(cards)}</div>
<details><summary>전체 {len(rows)} 랭킹 테이블 (선정 {K} 강조 = cutoff)</summary>
<table><tr><th>#</th><th>clip</th><th>combined</th><th>ego</th><th>vlm</th><th>label</th></tr>{trows}</table>
</details></div></body></html>"""
    out_path.write_text(html, encoding="utf-8")

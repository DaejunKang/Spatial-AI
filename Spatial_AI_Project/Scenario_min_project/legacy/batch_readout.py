"""배치 메타태깅 러너 (meta_tagging v0.7.1 segment).

split 의 클립들을 순회하며 클립당:
  - 세그먼트 태깅 → outputs/tags/<clip_id>.json (전체 결과)
  - 통합 outputs/tags.jsonl 한 줄
  - (옵션) outputs/overlay/<clip_id>.mp4 정성 확인용 오버레이

사용 예:
  ./run.sh batch_readout.py --limit 5
  ./run.sh batch_readout.py --split diverse_set-test.txt --start 0 --limit 10 --no-overlay
  ./run.sh batch_readout.py --index 3 --workers 1
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from config import CAMERA, FRAME_MODE, OUTPUT_DIR, OVERLAY_MAX_SIDE, SPLIT_FILE, TAG_MODE
from dataset import all_clip_ids, load_clip_ids, video_path
from overlay import render_overlay
from tagger import build_client_pool, tag_clip


def parse_args():
    p = argparse.ArgumentParser(description="배치 세그먼트 메타태깅")
    p.add_argument("--split", default=SPLIT_FILE,
                   help="curation/ 하위 split 파일, 또는 'all'(전체 1,966 클립)")
    p.add_argument("--camera", default=CAMERA)
    p.add_argument("--start", type=int, default=0, help="split 내 시작 인덱스")
    p.add_argument("--limit", type=int, default=5, help="처리할 클립 수 (0=전체)")
    p.add_argument("--index", type=int, default=None,
                   help="단일 클립 인덱스만 처리(지정 시 start/limit 무시)")
    p.add_argument("--frame-mode", default=FRAME_MODE, choices=["montage", "individual"])
    p.add_argument("--out", default=OUTPUT_DIR, help="결과 저장 루트")
    p.add_argument("--overlay", dest="overlay", action="store_true", default=True)
    p.add_argument("--no-overlay", dest="overlay", action="store_false")
    p.add_argument("--overlay-max-side", type=int, default=OVERLAY_MAX_SIDE)
    p.add_argument("--workers", type=int, default=2, help="동시 처리 클립 수")
    p.add_argument("--overwrite", action="store_true",
                   help="이미 tags/<clip_id>.json 있어도 재처리")
    return p.parse_args()


def select_clip_ids(args) -> list[str]:
    ids = all_clip_ids() if args.split == "all" else load_clip_ids(args.split)
    if args.index is not None:
        return [ids[args.index]]
    end = len(ids) if args.limit in (0, None) else args.start + args.limit
    return ids[args.start:end]


def process_clip(client, clip_id: str, args, dirs: dict) -> dict:
    """한 클립을 태깅·저장·(옵션)오버레이. 요약 dict 반환."""
    tags_json = dirs["tags"] / f"{clip_id}.json"
    if tags_json.exists() and not args.overwrite:
        r = json.loads(tags_json.read_text(encoding="utf-8"))
        r["_skipped"] = True
        return _summary(r, clip_id)

    path = video_path(clip_id, args.camera)
    if not path.exists():
        r = {"clip_id": clip_id, "ok": False, "error": f"영상 없음: {path}"}
        tags_json.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        return _summary(r, clip_id)

    r = tag_clip(client, path, clip_id, args.frame_mode)
    tags_json.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.overlay and r.get("ok"):
        try:
            out_mp4 = dirs["overlay"] / f"{clip_id}.mp4"
            render_overlay(path, r.get("segments", []), out_mp4,
                           clip_id=clip_id, max_side=args.overlay_max_side)
            r["overlay"] = str(out_mp4)
        except Exception as e:
            r["overlay_error"] = f"{type(e).__name__}: {e}"
    return _summary(r, clip_id)


def _summary(r: dict, clip_id: str) -> dict:
    return {
        "clip_id": clip_id,
        "ok": bool(r.get("ok")),
        "n_segments": len(r.get("segments", []) or []),
        "n_warnings": len(r.get("warnings", []) or []),
        "segments": r.get("segments", []),
        "error": r.get("error"),
        "skipped": r.get("_skipped", False),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.out)
    dirs = {"tags": root / "tags", "overlay": root / "overlay"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    clip_ids = select_clip_ids(args)
    print(f"split={args.split} camera={args.camera} clips={len(clip_ids)} "
          f"tag_mode={TAG_MODE} frame_mode={args.frame_mode} "
          f"overlay={args.overlay} workers={args.workers}")

    clients = build_client_pool()  # 살아있는 GPU 엔드포인트 자동 탐지(라운드로빈)
    print(f"endpoints={len(clients)}: {[c.base_url for c in clients]}")
    summaries: list[dict] = []

    def cli(i):  # 클립 i 를 엔드포인트에 라운드로빈 배정
        return clients[i % len(clients)]

    if args.workers <= 1:
        for i, cid in enumerate(tqdm(clip_ids, desc="tagging")):
            summaries.append(process_clip(cli(i), cid, args, dirs))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_clip, cli(i), cid, args, dirs): cid
                    for i, cid in enumerate(clip_ids)}
            for f in tqdm(as_completed(futs), total=len(futs), desc="tagging"):
                summaries.append(f.result())

    # 통합 jsonl (clip_id 순 정렬)
    summaries.sort(key=lambda s: s["clip_id"])
    jsonl = root / "tags.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for s in summaries:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    ok = sum(1 for s in summaries if s["ok"])
    fail = len(summaries) - ok
    seg_total = sum(s["n_segments"] for s in summaries)
    warned = sum(1 for s in summaries if s["n_warnings"] > 0)
    print("=" * 60)
    print(f"완료: ok={ok} fail={fail} | 세그먼트 총 {seg_total} | 검증경고 있는 클립 {warned}")
    print(f"결과: {root}/tags/<clip_id>.json, {jsonl}")
    if args.overlay:
        print(f"오버레이: {dirs['overlay']}/<clip_id>.mp4")
    for s in summaries:
        mark = "OK " if s["ok"] else "ERR"
        extra = f" ({s['error']})" if s.get("error") else ""
        wtag = f" ⚠{s['n_warnings']}" if s["n_warnings"] else ""
        print(f"  {mark} {s['clip_id']}  seg={s['n_segments']}{wtag}{extra}")


if __name__ == "__main__":
    main()

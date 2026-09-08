# -*- coding: utf-8 -*-
"""S0(앵커검출) 단독 검증용 clip 50개 추출 — decisions/DESIGN_LOG.md [2026-09-08] 참고.

기존 gold.json 50개를 제외한 순수 무작위 clip에 production Track1이 실제 쓰는 배선
(tag_v08.py 방식: map_lane.default_curvature_fn)으로 detect_events+consolidate_episodes를
돌려 gold_tool.py가 그대로 읽을 수 있는 episodes.json을 만든다. 곡률보정 유무 대조
(2026-08-28 37-clip on/off 방식을 이 표본으로 확장)도 s0_raw.json에 같이 남긴다.

감사 필드(transition_filters_passed·merged_from·window_*_reason·threshold_set_id)는
현재 코드에 계산 로직이 없어 이 스크립트도 새로 만들지 않는다 — 무엇이 없는지를 그대로
드러내는 쪽을 택한다(무기록 fallback 금지).

실행: ./run.sh task_episode/extract_s0_validation.py
출력: gold_label/s0_validation/{sample_clips.json, episodes.json, s0_raw.json, vids/*.mp4}
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paths as P
from dataset import video_meta
import events
import map_lane as M
from classify073 import consolidate_episodes
import taxonomy
import review  # task_selection — sample_pool 재사용

N = 50
SEED = 42
OUT = Path("/home/daejun/vla-tagging/gold_label/s0_validation")


def process(clip_id):
    """clip 1개 → (episodes.json용 레코드, s0_raw.json용 레코드, 오류메시지|None)."""
    path = P.video_path(clip_id)
    try:
        dur = video_meta(path)["duration_s"]
    except Exception as e:
        return None, None, f"video_meta 실패: {e}"

    ev_raw = events.detect_events(clip_id)
    if not ev_raw.get("ok"):
        return None, None, f"detect_events 실패: {ev_raw.get('reason')}"

    curvature_fn = M.default_curvature_fn(clip_id, dur)   # tag_v08.py:223 과 동일 배선
    lane_crossing_fn = M.default_lane_crossing_fn(clip_id, dur)  # 계측만, 판정 미반영(2026-09-08)
    ev_corr = events.detect_events(clip_id, curvature_fn=curvature_fn,
                                    lane_crossing_fn=lane_crossing_fn)

    eps = consolidate_episodes(ev_corr["events"])
    ep_out = [{"t0": ep["t0"], "t1": ep["t1"], "arc": ep["kinds"],
               "ego_action": ep["ego_action"], "axes": ep["axes"],
               "auto": taxonomy.auto_tags_from_arc(ep["kinds"])}
              for ep in eps]

    raw_kinds = sorted(e["kind"] for e in ev_raw["events"])
    corr_kinds = sorted(e["kind"] for e in ev_corr["events"])

    raw_dump = {
        "dur": dur,
        "map_valid": curvature_fn is not None,
        "events_raw": ev_raw["events"],
        "events_corrected": ev_corr["events"],
        "kind_diff": raw_kinds != corr_kinds,
    }
    return {"dur": dur, "episodes": ep_out}, raw_dump, None


def transcode_s0(clip_ids, vids_dir, workers=6):
    """gold_tool.py 규약(vids/{clip_id[:8]}.mp4)에 맞춰 트랜스코드.
    review._transcode와 동일 ffmpeg 설정이나 출력 파일명이 8자리 접두 — review.py는
    전체 clip_id로 명명하므로(gold_tool.py와 규약이 다름) 여기서 별도 구현한다."""
    vids_dir = Path(vids_dir); vids_dir.mkdir(parents=True, exist_ok=True)

    def _one(cid):
        out = vids_dir / f"{cid[:8]}.mp4"
        if out.exists() and out.stat().st_size > 0:
            return True
        try:
            subprocess.run([review.FFMPEG, "-y", "-loglevel", "error", "-i", str(P.video_path(cid)),
                            "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "28", "-an", "-movflags", "+faststart", str(out)],
                           check=True, timeout=120)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = sum(ex.map(_one, clip_ids))
    print(f"  트랜스코드 {ok}/{len(clip_ids)} 성공 -> {vids_dir}", flush=True)


def main():
    sample, total, ngold, npool = review.sample_pool(N, SEED)
    print(f"전체 {total} clip · gold 제외 {ngold} · 유효 풀 {npool} · 추출 {len(sample)}(seed={SEED})",
          flush=True)

    episodes_out, raw_out, ok_ids, errors = {}, {}, [], []
    for cid in sample:
        ep_json, raw_json, err = process(cid)
        if err:
            errors.append({"clip_id": cid, "error": err})
            continue
        episodes_out[cid] = ep_json
        raw_out[cid] = raw_json
        ok_ids.append(cid)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample_clips.json").write_text(
        json.dumps(ok_ids, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "episodes.json").write_text(
        json.dumps(episodes_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "s0_raw.json").write_text(
        json.dumps(raw_out, ensure_ascii=False, indent=1), encoding="utf-8")

    n_map_valid = sum(1 for r in raw_out.values() if r["map_valid"])
    n_diff = sum(1 for r in raw_out.values() if r["kind_diff"])
    print(f"S0 추출 완료: {len(ok_ids)}/{len(sample)} 성공, 실패 {len(errors)}건", flush=True)
    print(f"map_valid={n_map_valid}/{len(ok_ids)} · 곡률보정 유무로 kind 분류가 달라진 clip {n_diff}건",
          flush=True)
    if errors:
        print("실패 목록:", json.dumps(errors, ensure_ascii=False, indent=1), flush=True)

    transcode_s0(ok_ids, OUT / "vids")


if __name__ == "__main__":
    main()

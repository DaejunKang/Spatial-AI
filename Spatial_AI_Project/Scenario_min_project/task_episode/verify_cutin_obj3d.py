# -*- coding: utf-8 -*-
"""cut_in 검증 — VLM(선별)의 cut_in 판정을 obj3d GT로 대조 (검증 전용).

⚠️ task_selection(Stage1)은 obj3d 없이 도는 게 맞음. 이 스크립트는 그 산출물(cut_in 후보)을
**사후에** obj3d(3DOD)로 확인만 한다(파이프라인 배선 아님, task_episode 계층).

대조: taxo_detect.detect_taxonomy(clip) 의 obj3d corridor cut_in/cut_in_attempt 검출과 비교.
  - confirm  : obj3d 도 cut_in(또는 attempt) 검출 → VLM 판정 corroborated
  - no       : obj3d 트랙 있으나 cut_in 없음 → VLM 오검(투영) 의심
  - unavail  : obj3d 파일/트랙 없음 → 판정 불가

입력: gold_label/select/{event_select300.json(몽타주), winevent_select300.json(윈도우+video)}
실행: ./run.sh task_episode/verify_cutin_obj3d.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paths as P
import taxo_detect as D

SEL = Path("/home/daejun/vla-tagging/gold_label/select")


def obj3d_cutin(clip_id):
    """obj3d cut_in 계열 검출. 'cut_in'|'cut_in_attempt'|'no'|'unavail'."""
    if not P.obj3d_frames(clip_id).exists():
        return "unavail"
    try:
        det = D.detect_taxonomy(clip_id)
    except Exception:
        return "unavail"
    cats = det["cats"]
    if "cut_in" in cats:
        return "cut_in"
    if "cut_in_attempt" in cats:
        return "cut_in_attempt"
    return "no"


def verify(clip_ids, label):
    from collections import Counter
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = dict(zip(clip_ids, ex.map(obj3d_cutin, clip_ids)))
    c = Counter(res.values())
    n = len(clip_ids)
    conf = c.get("cut_in", 0) + c.get("cut_in_attempt", 0)
    decidable = n - c.get("unavail", 0)
    rate = (conf / decidable) if decidable else 0.0
    print(f"[{label}] VLM cut_in {n}개 → obj3d: "
          f"confirm(cut_in {c.get('cut_in',0)}+attempt {c.get('cut_in_attempt',0)})={conf} · "
          f"no={c.get('no',0)} · unavail={c.get('unavail',0)}  "
          f"→ corroboration {conf}/{decidable} ({rate:.0%})", flush=True)
    return res


def base_rate(all_ids):
    """전체 표본에서 obj3d가 cut_in을 찾는 기저율(희소성 맥락)."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(obj3d_cutin, all_ids))
    from collections import Counter
    c = Counter(res)
    conf = c.get("cut_in", 0) + c.get("cut_in_attempt", 0)
    print(f"[base] 전체 {len(all_ids)} 중 obj3d cut_in계열 = {conf} "
          f"(cut_in {c.get('cut_in',0)} + attempt {c.get('cut_in_attempt',0)}) · unavail {c.get('unavail',0)}", flush=True)


def main():
    mont = json.loads((SEL / "event_select300.json").read_text())
    win = json.loads((SEL / "winevent_select300.json").read_text())
    all300 = [r["clip_id"] for r in mont]

    mont_ci = [r["clip_id"] for r in mont if r.get("event_type") == "cut_in"]
    mont_ci_top = [r["clip_id"] for r in mont[:50] if r.get("event_type") == "cut_in"]
    win_ci = [r["clip_id"] for r in win if r.get("best") and r["best"].get("event_type") == "cut_in"]
    win_ci_top = [r["clip_id"] for r in win[:50] if r.get("best") and r["best"].get("event_type") == "cut_in"]

    print(f"VLM cut_in 후보: 몽타주 전체 {len(mont_ci)}(top50 {len(mont_ci_top)}) · "
          f"윈도우 전체 {len(win_ci)}(top50 {len(win_ci_top)})\n", flush=True)
    r1 = verify(mont_ci, "몽타주·전체300")
    verify(mont_ci_top, "몽타주·top50")
    r2 = verify(win_ci, "윈도우·전체300")
    verify(win_ci_top, "윈도우·top50")
    print()
    base_rate(all300)

    out = {"montage_cutin": {"clips": mont_ci, "obj3d": r1},
           "window_cutin": {"clips": win_ci, "obj3d": r2}}
    (SEL / "verify_cutin_obj3d.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n완료 → {SEL}/verify_cutin_obj3d.json")


if __name__ == "__main__":
    main()

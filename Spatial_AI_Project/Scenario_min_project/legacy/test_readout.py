"""단일 클립 세그먼트 메타태깅 테스트 (meta_tagging v0.7.1).

전체 클립을 몽타주로 모델에 전달하고, <answer> 세그먼트 배열을 파싱·검증해 출력한다.
"""

import json
import tempfile
import urllib.request
from pathlib import Path

from config import CLIP_INDEX, FRAME_MODE, VIDEO_SOURCE, VIDEO_URL
from dataset import get_clip
from tagger import build_client, tag_clip


def resolve_source() -> tuple[str, Path]:
    """(clip_id, 로컬 mp4 경로). url 모드면 임시 파일로 다운로드."""
    if VIDEO_SOURCE == "dataset":
        return get_clip(CLIP_INDEX)
    tmp = Path(tempfile.gettempdir()) / "vla_sample.mp4"
    urllib.request.urlretrieve(VIDEO_URL, tmp)
    return "sample_url", tmp


def fmt_segment(s: dict) -> str:
    """세그먼트 1건을 사람이 읽기 좋은 한 줄로."""
    span = f"[{s.get('t_start')}-{s.get('t_end')}s]"
    diff = s.get("difficulty")
    dtag = f" diff={diff}" if diff is not None else ""
    if s.get("subject") == "ego":
        body = f"EGO {s.get('ego_action')}"
    else:
        parts = [f"AGENT {s.get('object_type')}", f"role={s.get('role')}"]
        if s.get("longitudinal_action"):
            parts.append(f"lon={s['longitudinal_action']}")
        if s.get("relation"):
            parts.append(f"rel={s['relation']}")
        if s.get("vru_detail"):
            parts.append(f"vru={s['vru_detail']}")
        body = " ".join(parts)
    return f"  #{s.get('segment_id')} {span}{dtag}  {body}"


def main() -> None:
    clip_id, path = resolve_source()
    print(f"clip_id: {clip_id}")
    print(f"path:    {path}")

    client = build_client()
    r = tag_clip(client, path, clip_id, FRAME_MODE)

    v = r.get("video", {})
    if v:
        print(
            f"video:   {v['frames']} frames @ {v['fps']:.1f}fps "
            f"({v['duration_s']:.1f}s) {v['width']}x{v['height']}"
        )
    print(f"frames:  {r.get('frames')}")

    print("=" * 60)
    print("RAW OUTPUT:")
    print("=" * 60)
    print(r.get("raw", "(없음)"))
    print("=" * 60)
    print(f"finish_reason: {r.get('finish_reason')}")
    print(f"usage: {r.get('usage')}")
    print("=" * 60)

    if not r["ok"]:
        print(f"실패: {r.get('error')}")
        return

    segs = r["segments"]
    print(f"SEGMENTS ({len(segs)}):")
    for s in segs:
        print(fmt_segment(s))
    print("- - - JSON - - -")
    print(json.dumps(segs, ensure_ascii=False, indent=2))
    print("=" * 60)

    warns = r.get("warnings") or []
    if warns:
        print(f"검증 경고 {len(warns)}건:")
        for w in warns:
            print(f"  - {w}")
    else:
        print("검증 통과 — 모든 세그먼트가 스키마·규칙에 부합")


if __name__ == "__main__":
    main()

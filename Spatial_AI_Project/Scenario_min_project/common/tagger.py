"""클립 하나를 meta_tagging v0.7.1 세그먼트로 태깅하는 공용 추론 로직.

test_readout(단일) 과 batch_readout(배치) 가 함께 사용한다.
"""

import json
import re
from pathlib import Path

from openai import OpenAI

from config import (
    API_KEY,
    BASE_URL,
    FRAME_MODE,
    MAX_IMAGES,
    MAX_TOKENS,
    MODEL,
    MONTAGE_CELL_MAX_SIDE,
    MONTAGE_COLS,
    NUM_FRAMES,
    TAG_MODE,
    TEMPERATURE,
    USER_TEXT,
)
from dataset import jpeg_to_data_uri, sample_frames, sample_montage, video_meta
from prompts import SCHEMA, VOCAB, prompt_for
from vocab import validate_segments


def build_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def build_client_pool(timeout: float = 2.0) -> list:
    """BASE_URLS 중 헬스체크(200) 통과한 엔드포인트로 OpenAI 클라이언트 풀 생성.

    복제본이 안 떠 있으면 자동 제외되어 살아있는 것만 사용(다중 GPU 라운드로빈).
    """
    import urllib.request
    from config import BASE_URLS

    alive = []
    for url in BASE_URLS:
        try:
            urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout)
            alive.append(url)
        except Exception:
            pass
    if not alive:
        alive = [BASE_URL]
    return [OpenAI(base_url=u, api_key=API_KEY) for u in alive]


def extract_answer(text: str) -> list | None:
    """<answer>...</answer> 안의 JSON 배열을 파싱. 없으면 첫 JSON 배열 시도."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    chunk = m.group(1).strip() if m else None
    if chunk is None:
        # 폴백: </think> 이후 첫 대괄호 배열
        body = text.split("</think>")[-1]
        start = body.find("[")
        if start == -1:
            return None
        chunk = body[start:]
    # 균형 맞는 첫 배열만 잘라 파싱
    start = chunk.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(chunk)):
        if chunk[i] == "[":
            depth += 1
        elif chunk[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(chunk[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def build_content(path: Path | str, frame_mode: str = FRAME_MODE) -> tuple[list, str]:
    """user 메시지 content(텍스트+이미지) 와 프레임 요약 문자열."""
    content: list = [{"type": "text", "text": USER_TEXT}]
    if frame_mode == "montage":
        jpg = sample_montage(
            path, NUM_FRAMES, cols=MONTAGE_COLS, cell_max_side=MONTAGE_CELL_MAX_SIDE
        )
        content.append(
            {"type": "image_url", "image_url": {"url": jpeg_to_data_uri(jpg)}}
        )
        summary = f"{NUM_FRAMES}프레임 몽타주 1장 ({len(jpg):,} bytes)"
    else:
        frames = sample_frames(path, min(NUM_FRAMES, MAX_IMAGES))
        for jpg in frames:
            content.append(
                {"type": "image_url", "image_url": {"url": jpeg_to_data_uri(jpg)}}
            )
        summary = f"개별 {len(frames)}프레임"
    return content, summary


def tag_clip(
    client: OpenAI,
    path: Path | str,
    clip_id: str,
    frame_mode: str = FRAME_MODE,
) -> dict:
    """클립 하나를 태깅해 결과 dict 반환.

    반환 키: clip_id, ok, segments, warnings, raw, finish_reason, usage,
             frames, video, error

    TAG_MODE="windowed" 이면 window_tagger 로 위임(연속 윈도우 분할+통합).
    """
    if TAG_MODE == "events":
        from event_tagger import tag_clip_events  # 순환 import 회피
        return tag_clip_events(client, path, clip_id)
    if TAG_MODE == "windowed":
        from window_tagger import tag_clip_windowed
        return tag_clip_windowed(client, path, clip_id)

    result: dict = {"clip_id": clip_id, "ok": False}
    try:
        result["video"] = video_meta(path)
        content, summary = build_content(path, frame_mode)
        result["frames"] = summary

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt_for(frame_mode)},
                {"role": "user", "content": content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        result["raw"] = raw
        result["finish_reason"] = choice.finish_reason
        result["usage"] = resp.usage.model_dump() if resp.usage else None

        segs = extract_answer(raw)
        if segs is None:
            if choice.finish_reason == "length":
                result["error"] = "토큰 초과(length) — <answer> 미생성. MAX_TOKENS 상향 필요"
            else:
                result["error"] = "<answer> JSON 배열 파싱 실패"
            return result

        # clip_id 강제 주입(모델이 다르게 쓰면 교정)
        for s in segs:
            if isinstance(s, dict):
                s["clip_id"] = clip_id
        _, warns = validate_segments(segs, VOCAB, SCHEMA)
        result["segments"] = segs
        result["warnings"] = warns
        result["ok"] = True
    except Exception as e:  # 배치에서 개별 실패가 전체를 막지 않도록
        result["error"] = f"{type(e).__name__}: {e}"
    return result

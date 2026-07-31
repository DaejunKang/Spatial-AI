"""모델에 전달할 프롬프트 정의 (meta_tagging v0.7.1 segment).

어휘/스키마에서 동적으로 생성한다. 파일이 바뀌면 프롬프트도 따라간다.
"""

from vocab import (
    DEFAULT_FRAMES_DESC,
    MONTAGE_FRAMES_DESC,
    build_prompt,
    load_or_build_schema,
    load_vocab,
)

VOCAB = load_vocab()
SCHEMA = load_or_build_schema(VOCAB)

# 기본(몽타주) 시스템 프롬프트
PROMPT = build_prompt(VOCAB, SCHEMA, MONTAGE_FRAMES_DESC)


def prompt_for(frame_mode: str) -> str:
    desc = MONTAGE_FRAMES_DESC if frame_mode == "montage" else DEFAULT_FRAMES_DESC
    return build_prompt(VOCAB, SCHEMA, desc)

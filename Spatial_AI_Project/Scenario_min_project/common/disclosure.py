# -*- coding: utf-8 -*-
"""공개의무 필드 — 정책 이탈이 아니라 '무엇을 썼는지 기록'하는 계층.
vocab 대응: window.causal_judgment_scope / flags.s2_conditioning /
           flags.additional_frames_from_candidates / flags.view_expanded_from_scene

값은 스키마 선언 기본값이 아니라 **실제 실행 중인 코드의 사실**을 기록한다. 예:
S2_CONDITIONING의 스키마 선언 기본값은 "minimal"이지만, 실제 candidates.py/vlm_verify.py/
tag_v08.py는 GT 카테고리 힌트를 항상 주입하므로 실측 사실은 "rule_injected"다. 이
불일치는 CLAUDE.md §3 "VLM 프롬프트의 GT 힌트" 전환 규칙(실험 없이 기존 경로 안 바꿈)에
따른 것 — 코드 동작을 바꾸지 않고 사실만 정직하게 기록한다.
"""

# 현재 파이프라인의 실측 기본 동작(정책이 아니라 사실) — 바뀌면 여기만 수정
CAUSAL_JUDGMENT_SCOPE = "full_window"   # tag_v08/candidates 윈도우가 거동완료 이후까지 포함해 판정
S2_CONDITIONING = "rule_injected"       # candidates/vlm_verify/tag_v08 프롬프트가 GT 힌트 상시 주입


def stamp(*, additional_frames: bool = False, view_expanded: bool = False, **overrides) -> dict:
    """레코드에 동반할 공개의무 블록. 값은 실제 실행부에서 관측된 사실만 넣는다(추정 금지)."""
    d = {
        "causal_judgment_scope": CAUSAL_JUDGMENT_SCOPE,
        "s2_conditioning": S2_CONDITIONING,
        "additional_frames_from_candidates": additional_frames,
        "view_expanded_from_scene": view_expanded,
    }
    d.update(overrides)
    return d

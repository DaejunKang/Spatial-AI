# -*- coding: utf-8 -*-
"""S1→S2 GT 주입 배치 조회 (DV-8). tag_vocab_v0.4.json field_execution_policy.gt_injection이 정본
— 어휘를 코드에 복제하지 않고 여기서 로드만 한다.

미분류 필드(어휘 3-way 목록 어디에도 없는 필드) 기본값 = gt_before (B1, 2026-08-28 결정).
근거: gt_free(미주입)로 두면 grounding 실패 위험을 조용히 방치한다. gt_before(관측 사실
주입)는 최악의 경우도 "과잉 주입"일 뿐 정보 손실은 아니므로 안전한 쪽으로 강등한다.
"""
import json
from pathlib import Path

_VOCAB_PATH = Path(__file__).parent / "schema" / "tag_vocab_v0.4.json"
_INJ = json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))["field_execution_policy"]["gt_injection"]

DEFAULT_CLASS = "gt_before"  # B1 결정 — 미분류 필드의 안전 기본값


def placement_of(field: str) -> str:
    """필드명 → gt_only|gt_before|gt_free. 정의된 필드는 그 분류, 없으면 DEFAULT_CLASS."""
    for cls in ("gt_only", "gt_before", "gt_free"):
        if field in _INJ.get(cls, []):
            return cls
    return DEFAULT_CLASS

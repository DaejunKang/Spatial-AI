# -*- coding: utf-8 -*-
"""VLM 클라이언트 풀 (활성 유틸). Task: common.

BASE_URLS(8001–8004) 중 헬스체크 통과분으로 OpenAI 호환 클라이언트 풀 생성 → 라운드로빈.
(구 tagger.build_client_pool 에서 추출 — legacy 의존 없이 활성 파이프라인이 쓰도록 공용화.)
"""
from openai import OpenAI

from config import API_KEY, BASE_URL, BASE_URLS


def build_client_pool(timeout: float = 2.0) -> list:
    """살아있는 엔드포인트로 클라이언트 풀. 미기동 복제본은 자동 제외."""
    import urllib.request
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

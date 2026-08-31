#!/usr/bin/env python3
"""실험 B — 어휘 닫힌 집합 필드의 first-token 충돌 실측 (experiment_design_v0.1.md §실험B).

Vocab_lint.py의 check_first_token()은 문자열 3자 프록시(근사)로 충돌 후보를 찾을 뿐이다.
이 스크립트는 실제 서빙 토크나이저로 각 필드 값의 first-token id를 비교해, 필드별
scoring_method(first_token 허용 / sequence 강제) 배정표를 산출한다.

**엔드포인트 정정(2026-08-28 실측)**: 이 NIM 빌드(nvidia/cosmos3-nano-reasoner)는
/tokenize·/v1/tokenize를 노출하지 않는다(openapi.json에 경로 자체가 없음, 404 확인).
따라서 서빙 컨테이너 내부의 실제 토크나이저 파일(HF hub 캐시, symlink→blobs 구조)을
추출해 로컬 로드하는 방식으로 대체한다 — Qwen2Tokenizer로 확인됨.

캐시 추출(컨테이너가 떠 있을 때 1회):
    TOKDIR=common/schema/.tokenizer_cache; mkdir -p "$TOKDIR"
    SRC=/opt/nim/.cache/ngc/hub/models--nim--nvidia--cosmos3-nano-reasoner/snapshots/modelopt-fp8-final_format_fix
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt config.json; do
        docker exec cosmos_dj cat "$SRC/$f" > "$TOKDIR/$f"
    done
    # docker cp는 symlink를 그대로 복사해 깨지므로 사용 금지 — cat으로 내용을 뽑아야 함

결과는 tag_vocab_v0.4.json의 field_execution_policy.scoring_method에 필드 단위로
수동 반영한다(자동 반영 안 함 — 하이퍼파라미터와 마찬가지로 "실측 후 선언").

사용:
    ./run.sh common/schema/scoring_probe.py [vocab_path] [tokenizer_dir]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TOKENIZER_DIR = Path(__file__).parent / ".tokenizer_cache"


def walk_value_fields(node, path=""):
    """(field_path, values) 산출 — 'values' 키를 가진 문자열 배열 전부(닫힌 집합 필드)."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if k == "values" and isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                yield path, v
            yield from walk_value_fields(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_value_fields(v, f"{path}[{i}]")


def load_first_token_fn(tokenizer_dir: Path):
    """로컬 토크나이저 캐시에서 first_token_fn(str)->int 생성. 캐시 없으면 None."""
    if not (tokenizer_dir / "tokenizer.json").exists():
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return lambda text: tok.encode(text, add_special_tokens=False)[0]


def assign_scoring_methods(fields: dict, first_token_fn) -> dict:
    """fields: {path: [values]}. first_token_fn(str) -> token_id | None.
    반환: {path: {"scoring_method": "first_token"|"sequence"|"unmeasured", ...}}"""
    out = {}
    for path, values in fields.items():
        if first_token_fn is None:
            out[path] = {"scoring_method": "unmeasured", "note": "토크나이저 캐시 없음 — 상단 docstring 추출 절차 실행"}
            continue
        ids = {v: first_token_fn(v) for v in values}
        groups = defaultdict(list)
        for v, t in ids.items():
            groups[t].append(v)
        collisions = {t: vs for t, vs in groups.items() if len(vs) > 1}
        if collisions:
            out[path] = {"scoring_method": "sequence", "collisions": collisions}
        else:
            out[path] = {"scoring_method": "first_token"}
    return out


def probe(vocab_path: str, tokenizer_dir: Path) -> dict:
    d = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    fields = dict(walk_value_fields(d))
    return assign_scoring_methods(fields, load_first_token_fn(tokenizer_dir))


def main() -> int:
    vocab_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "tag_vocab_v0.4.json")
    tokenizer_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOKENIZER_DIR

    result = probe(vocab_path, tokenizer_dir)

    n_unmeasured = sum(1 for a in result.values() if a["scoring_method"] == "unmeasured")
    n_first_token = sum(1 for a in result.values() if a["scoring_method"] == "first_token")
    n_sequence = sum(1 for a in result.values() if a["scoring_method"] == "sequence")

    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n필드 {len(result)}개 · first_token 허용 {n_first_token} · "
          f"sequence 강제 {n_sequence} · 미측정 {n_unmeasured}", file=sys.stderr)
    if n_unmeasured:
        print(f"[경고] {n_unmeasured}개 필드 미측정 — 토크나이저 캐시 추출 필요(상단 docstring 참고)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

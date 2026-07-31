#!/usr/bin/env bash
# 로컬(이 저장소)의 추적 파일을 Spatial-AI 모노레포 서브폴더로 동기화·push.
# 중요 변경사항이 있을 때만 실행. .venv/영상/outputs 등은 .gitignore로 자동 제외.
#
# 사용법:  ./push_to_monorepo.sh "커밋 메시지"
set -euo pipefail
cd "$(dirname "$0")"
SRC="$PWD"
MSG="${1:?사용법: ./push_to_monorepo.sh \"커밋 메시지\"}"
REPO="git@github.com:DaejunKang/Spatial-AI.git"
PREFIX="Spatial_AI_Project/Scenario_min_project"

# 1) 로컬 커밋 (변경 있을 때만)
git add -A
git commit -q -m "$MSG" || echo "  (로컬 변경 없음 — 기존 HEAD로 동기화)"

# 2) 모노레포 얕은 클론 → 서브폴더를 로컬 HEAD로 미러 (추적파일만)
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 -q "$REPO" "$TMP/mono"
SUB="$TMP/mono/$PREFIX"
mkdir -p "$SUB"
find "$SUB" -mindepth 1 -not -name "__init__.py" -delete 2>/dev/null || true   # __init__.py는 보존
git archive HEAD | tar -x -C "$SUB"

# 3) 모노레포 커밋·push (변경 있을 때만)
cd "$TMP/mono"
git config user.name "DaejunKang"; git config user.email "djkang@katech.re.kr"
git add "$PREFIX"
if git diff --cached --quiet; then
  echo "모노레포 변경 없음 — push 생략."
else
  git commit -q -m "update Scenario_min_project: $MSG

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  git push -q origin HEAD
  echo "push 완료: $(git rev-parse --short HEAD)  ($(git diff --cached --name-only | wc -l) 파일)"
fi

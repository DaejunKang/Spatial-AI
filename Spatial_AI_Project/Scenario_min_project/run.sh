#!/usr/bin/env bash
# 실행 출력을 화면 + logs/ 에 동시에 남기는 래퍼.
#
# 사용법:
#   ./run.sh test_readout.py            # 기본: venv 파이썬으로 실행
#   ./run.sh <스크립트> [인자...]
#
# 결과 확인:
#   tail -f logs/latest.log             # 다른 창/페인에서 실시간 모니터링
#   less logs/latest.log                # 지난 실행 검토

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/common:$PWD/task_selection:$PWD/task_episode:${PYTHONPATH:-}"

PY=".venv/bin/python"
SCRIPT="${1:-test_readout.py}"
shift || true

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="$(basename "${SCRIPT%.py}")"
LOG="logs/${BASE}_${STAMP}.log"

# latest.log 를 이번 실행 로그로 갱신 (실시간 tail 대상)
ln -sf "$(basename "$LOG")" logs/latest.log

echo "=== $(date '+%F %T')  run: $SCRIPT $*" | tee "$LOG"
# stdbuf: 파이썬 출력 버퍼링 해제 → tail -f 로 실시간 확인 가능
PYTHONUNBUFFERED=1 stdbuf -oL -eL "$PY" "$SCRIPT" "$@" 2>&1 | tee -a "$LOG"
echo "=== done: exit ${PIPESTATUS[0]}  log: $LOG" | tee -a "$LOG"

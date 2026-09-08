"""훅 공용 유틸. 저장소 배치가 다르면 이 파일의 경로만 고친다."""
import subprocess, sys, json
from pathlib import Path

# ── 저장소 경로 (실제 배치에 맞게 수정)
# 2026-09-08 정정: 아래 경로들이 이 저장소의 실제 배치와 달라 관련 훅 4/6이 상시
# 무력화(대상 없음→항상 통과)돼 있었음. GOLD/LINT는 실제 경로로 고쳤다. VERIFY_DIRS·
# CARDS_DIR·RESULTS_DIR은 이 저장소에 아직 그 역할의 디렉토리가 없어 플레이스홀더로
# 남겨두되, 의존 훅(check_import_separation·check_experiment_card)이 "대상 없음"을
# 침묵 통과 대신 명시적으로 출력하도록 각 훅 파일에서 처리한다.
VOCAB_GLOB   = "common/schema/tag_vocab_v*.json"
GUIDE_GLOB   = "docs/design/pipeline_design_guide_v*.md"
LINT         = "common/schema/Vocab_lint.py"
THRESHOLDS   = "common/thresholds.py"          # 임계 상수 유일 허용 파일
PIPELINE_DIRS = ["task_episode/", "task_selection/", "common/"]   # 생산 측
VERIFY_DIRS   = ["verification/"]                                 # 검증 측(플레이스홀더 — 아직 미실재)
GOLD_DIR      = "gold_label/"                                     # 실제 gold 라벨링 산출물 위치
GOLD_FILE     = "gold.json"                                       # 최상위 gold 정답 파일
GOLD_ALLOWED  = ["task_selection/review.py", "task_episode/gold_tool.py",
                 "task_episode/extract_s0_validation.py", "task_episode/run_workflow_audit.py"]
# 위 4개: gold_label/ 이 실제 gold 정답(gold.json, sample_clips/episodes.json)과
# Stage1/Stage2 일반 산출물 저장 위치를 겸하고 있어(이 저장소 관행) 함께 걸림 — 전부
# 표본제외·산출물 read/write일 뿐 gold 정답을 학습·튜닝에 쓰는 경로가 아님(합법 참조)
RESULTS_DIR   = "experiments/results/"                             # 플레이스홀더 — 아직 미실재
CARDS_DIR     = "experiments/cards/"                                # 플레이스홀더 — 아직 미실재

def staged_files():
    out = subprocess.run(["git","diff","--cached","--name-only","--diff-filter=ACMR"],
                         capture_output=True, text=True).stdout
    return [Path(p) for p in out.split() if p]

def staged_content(path):
    r = subprocess.run(["git","show",f":{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""

def latest(glob):
    hits = sorted(Path(".").glob(glob))
    return hits[-1] if hits else None

def under(path, dirs):
    s = str(path).replace("\\","/")
    return any(s.startswith(d) for d in dirs)

def fail(msg):
    print(f"\n[pre-commit 차단] {msg}\n", file=sys.stderr); sys.exit(1)
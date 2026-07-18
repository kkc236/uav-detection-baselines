#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-$(dirname "$REPO_DIR")/btdse-storage}"
PYTHON="${PYTHON:-$STORAGE_ROOT/venv/bin/python}"
TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"
PROJECT_DIR="${PROJECT_DIR:-$STORAGE_ROOT/runs/btdse}"
RUN_NAME="${RUN_NAME:-scratch-rtdetr-l-btdse-100ep-4090}"
RUN_DIR="$PROJECT_DIR/$RUN_NAME"
RESULTS_REPO="${RESULTS_REPO:-$STORAGE_ROOT/results-checkout}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
MAX_RESTARTS="${MAX_RESTARTS:-50}"
RESTART_DELAY="${RESTART_DELAY:-30}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
TAG="${TAG:-btdse-v2.5-s-4090-live}"

mkdir -p "$PROJECT_DIR" "$LOG_DIR"
cd "$REPO_DIR"

exec 9>"$LOG_DIR/btdse_4090.lock"
if ! flock -n 9; then
  printf 'Another BTD-SE supervisor already holds %s.\n' "$LOG_DIR/btdse_4090.lock" >&2
  exit 1
fi

[[ -x "$PYTHON" ]] || { printf 'Python environment not found: %s\n' "$PYTHON" >&2; exit 1; }
[[ "$BATCH" == "8" ]] || { printf 'The paper protocol requires BATCH=8.\n' >&2; exit 2; }
[[ -s "$TOKEN_FILE" ]] || { printf 'GitHub token file is missing or empty: %s\n' "$TOKEN_FILE" >&2; exit 1; }
if (( $(stat -c '%a' "$TOKEN_FILE") % 100 != 0 )); then
  printf 'GitHub token file must not be readable by group or others: %s\n' "$TOKEN_FILE" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONUNBUFFERED=1

printf '%s\n' "$$" > "$LOG_DIR/btdse_4090_supervisor.pid"

free_gib() {
  df -Pk "$STORAGE_ROOT" | awk 'NR==2 {printf "%d", $4 / 1024 / 1024}'
}

training_complete() {
  "$PYTHON" - "$RUN_DIR/results.csv" "$EPOCHS" <<'PY' >/dev/null 2>&1
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
with path.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
raise SystemExit(0 if rows and int(float(rows[-1]["epoch"])) >= target else 1)
PY
}

find_checkpoint() {
  "$PYTHON" scripts/find_btdse_checkpoint.py --run-dir "$RUN_DIR" 2>/dev/null
}

sync_arguments=(
  scripts/sync_btdse_checkpoint.py
  --run-dir "$RUN_DIR"
  --token-file "$TOKEN_FILE"
  --results-repo "$RESULTS_REPO"
  --run-name "$RUN_NAME"
  --tag "$TAG"
  --status-file "$LOG_DIR/btdse_github_sync.json"
  --retain 3
  --interval 60
)

"$PYTHON" -u "${sync_arguments[@]}" >> "$LOG_DIR/btdse_github_sync.log" 2>&1 &
sync_pid=$!
training_pid=""

cleanup() {
  if [[ -n "$training_pid" ]] && kill -0 "$training_pid" 2>/dev/null; then
    kill -TERM "$training_pid" 2>/dev/null || true
    wait "$training_pid" 2>/dev/null || true
  fi
  kill "$sync_pid" 2>/dev/null || true
  wait "$sync_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

resume_checkpoint="$(find_checkpoint || true)"
restart_count=0

if training_complete; then
  printf '[%s] Training already contains %s epochs.\n' "$(date '+%F %T')" "$EPOCHS" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
else
  while true; do
    available=$(free_gib)
    if (( available < MIN_FREE_GIB )); then
      printf '[%s] Stopping safely: only %s GiB is free.\n' "$(date '+%F %T')" "$available" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
      exit 2
    fi

    attempt_log="$LOG_DIR/btdse_attempt_$(date '+%Y%m%d_%H%M%S').log"
    train_arguments=(
      scripts/train_rtdetr_btdse.py
      --epochs "$EPOCHS"
      --batch "$BATCH"
      --imgsz 640
      --workers "$WORKERS"
      --device 0
      --project "$PROJECT_DIR"
      --name "$RUN_NAME"
    )
    if [[ -n "$resume_checkpoint" ]]; then
      train_arguments+=(--resume "$resume_checkpoint")
      printf '[%s] Resume attempt %s from %s with batch=%s.\n' \
        "$(date '+%F %T')" "$((restart_count + 1))" "$resume_checkpoint" "$BATCH" \
        | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
    else
      printf '[%s] Scratch attempt %s with batch=%s.\n' \
        "$(date '+%F %T')" "$((restart_count + 1))" "$BATCH" \
        | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
    fi

    set +e
    "$PYTHON" -u "${train_arguments[@]}" > >(tee "$attempt_log" | tee -a "$LOG_DIR/btdse_4090_training.log") 2>&1 &
    training_pid=$!
    wait "$training_pid"
    train_rc=$?
    training_pid=""
    set -e
    sleep 1

    if (( train_rc == 0 )) || training_complete; then
      printf '[%s] Training completed successfully.\n' "$(date '+%F %T')" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
      break
    fi

    restart_count=$((restart_count + 1))
    if (( restart_count > MAX_RESTARTS )); then
      printf '[%s] Restart limit reached; data remains on persistent storage.\n' "$(date '+%F %T')" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
      exit "$train_rc"
    fi

    if grep -Eq 'CUDA out of memory|NONFINITE_LOSS|non-finite|NaN|Inf' "$attempt_log"; then
      printf '[%s] Fixed protocol violation detected; stopping without changing batch or AMP.\n' \
        "$(date '+%F %T')" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
      exit 2
    fi

    resume_checkpoint="$(find_checkpoint || true)"
    printf '[%s] Restarting in %s seconds.\n' "$(date '+%F %T')" "$RESTART_DELAY" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
    sleep "$RESTART_DELAY"
  done
fi

kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
for attempt in 1 2 3 4 5; do
  if "$PYTHON" -u "${sync_arguments[@]}" --once >> "$LOG_DIR/btdse_github_sync.log" 2>&1; then
    printf '[%s] Final GitHub checkpoint publication verified.\n' "$(date '+%F %T')" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
    exit 0
  fi
  printf '[%s] Final publication attempt %s failed; retrying.\n' "$(date '+%F %T')" "$attempt" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
  sleep 60
done

printf '[%s] Training is complete, but final GitHub publication needs manual retry. Local data is intact.\n' \
  "$(date '+%F %T')" | tee -a "$LOG_DIR/btdse_4090_supervisor.log"
exit 3

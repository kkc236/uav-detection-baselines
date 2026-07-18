#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-$(dirname "$REPO_DIR")/matched-baseline-storage}"
PYTHON="${PYTHON:-$STORAGE_ROOT/venv/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-$STORAGE_ROOT/runs/matched-baseline}"
RUN_NAME="${RUN_NAME:-scratch-rtdetr-l-btdse-matched-baseline-100ep}"
RUN_DIR="$PROJECT_DIR/$RUN_NAME"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"
RESULTS_REPO="${RESULTS_REPO:-$STORAGE_ROOT/results-checkout}"
SOURCE_BRANCH="${SOURCE_BRANCH:-codex/matched-baseline}"
TAG="${TAG:-rtdetr-l-btdse-matched-baseline-live}"
ASSET_PREFIX="${ASSET_PREFIX:-matched-baseline-last}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
RESTART_DELAY="${RESTART_DELAY:-30}"
ENABLE_GITHUB_SYNC="${ENABLE_GITHUB_SYNC:-1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

[[ "$BATCH" == "8" ]] || { printf 'The matched protocol requires BATCH=8.\n' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Python environment not found: %s\n' "$PYTHON" >&2; exit 1; }

mkdir -p "$RUN_DIR" "$LOG_DIR" "$RESULTS_REPO"
cd "$REPO_DIR"

exec 9>"$LOG_DIR/matched_baseline.lock"
if ! flock -n 9; then
  printf 'Another matched baseline run already owns the lock.\n' >&2
  exit 1
fi

if [[ "$ENABLE_GITHUB_SYNC" == "1" ]]; then
  [[ -s "$TOKEN_FILE" ]] || { printf 'GitHub token file is missing or empty: %s\n' "$TOKEN_FILE" >&2; exit 1; }
  if (( $(stat -c '%a' "$TOKEN_FILE") % 100 != 0 )); then
    printf 'GitHub token file must have mode 600: %s\n' "$TOKEN_FILE" >&2
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONUNBUFFERED=1

TRAINING_LOG="$LOG_DIR/matched_baseline_training.log"
SUPERVISOR_LOG="$LOG_DIR/matched_baseline_supervisor.log"
SYNC_LOG="$LOG_DIR/matched_baseline_github_sync.log"
SYNC_STATUS="$LOG_DIR/matched_baseline_github_sync.json"
printf '%s\n' "$$" > "$LOG_DIR/matched_baseline_launcher.pid"

sync_arguments=(
  scripts/sync_experiment_checkpoint.py
  --run-dir "$RUN_DIR"
  --token-file "$TOKEN_FILE"
  --results-repo "$RESULTS_REPO"
  --run-name "$RUN_NAME"
  --tag "$TAG"
  --source-branch "$SOURCE_BRANCH"
  --status-file "$SYNC_STATUS"
  --retain 3
  --asset-prefix "$ASSET_PREFIX"
  --release-name "BTD-SE-Matched RT-DETR-L Baseline Checkpoints"
  --release-body "Rolling resumable checkpoints for the fixed batch-8 matched RT-DETR-L baseline."
  --interval 60
)

sync_pid=""
train_pid=""
if [[ "$ENABLE_GITHUB_SYNC" == "1" ]]; then
  "$PYTHON" -u "${sync_arguments[@]}" >> "$SYNC_LOG" 2>&1 &
  sync_pid=$!
fi

cleanup() {
  if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
    kill -TERM -- "-$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true
  fi
  if [[ -n "$sync_pid" ]]; then
    kill "$sync_pid" 2>/dev/null || true
    wait "$sync_pid" 2>/dev/null || true
  fi
  rm -f "$LOG_DIR/matched_baseline_launcher.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

find_checkpoint() {
  "$PYTHON" scripts/find_btdse_checkpoint.py --run-dir "$RUN_DIR" 2>/dev/null || true
}

restart_count=0
while true; do
  resume_checkpoint="$(find_checkpoint)"
  command=(
    "$PYTHON" -u scripts/train_rtdetr_matched_baseline.py
    --project "$PROJECT_DIR"
    --name "$RUN_NAME"
    --device "$DEVICE"
  )
  if [[ -n "$resume_checkpoint" ]]; then
    command+=(--resume "$resume_checkpoint")
  fi

  attempt_log="$LOG_DIR/matched_baseline_attempt_$((restart_count + 1)).log"
  printf '[%s] Starting fixed protocol attempt %s; checkpoint=%s.\n' \
    "$(date '+%F %T')" "$((restart_count + 1))" "${resume_checkpoint:-scratch}" | tee -a "$SUPERVISOR_LOG"

  set +e
  setsid "${command[@]}" > >(tee -a "$TRAINING_LOG" "$attempt_log") 2>&1 &
  train_pid=$!
  wait "$train_pid"
  rc=$?
  train_pid=""
  set -e

  if (( rc == 0 )); then
    break
  fi
  if grep -Eq 'CUDA out of memory|NONFINITE_LOSS|non-finite|NaN|Inf' "$attempt_log"; then
    printf '[%s] Fixed protocol violation detected; stopping with all training parameters unchanged.\n' \
      "$(date '+%F %T')" | tee -a "$SUPERVISOR_LOG"
    exit 2
  fi

  restart_count=$((restart_count + 1))
  if (( restart_count > MAX_RESTARTS )); then
    printf '[%s] Maximum same-protocol restarts exceeded.\n' "$(date '+%F %T')" | tee -a "$SUPERVISOR_LOG"
    exit 3
  fi
  printf '[%s] Abnormal external stop; retrying the same protocol in %s seconds.\n' \
    "$(date '+%F %T')" "$RESTART_DELAY" | tee -a "$SUPERVISOR_LOG"
  sleep "$RESTART_DELAY"
done

if [[ -n "$sync_pid" ]]; then
  kill "$sync_pid" 2>/dev/null || true
  wait "$sync_pid" 2>/dev/null || true
  sync_pid=""
fi

if [[ "$ENABLE_GITHUB_SYNC" == "1" ]]; then
  "$PYTHON" -u "${sync_arguments[@]}" --once >> "$SYNC_LOG" 2>&1
fi

printf '[%s] Matched baseline completed with the fixed batch-8 AMP protocol.\n' \
  "$(date '+%F %T')" | tee -a "$SUPERVISOR_LOG"

if [[ "$AUTO_SHUTDOWN" == "1" && "$ENABLE_GITHUB_SYNC" == "1" ]]; then
  shutdown -h now
fi

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-$(dirname "$REPO_DIR")/ioqc-sa-storage}"
PYTHON="${PYTHON:-$STORAGE_ROOT/venv/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-$STORAGE_ROOT/runs/ioqc-sa}"
RUN_NAME="${RUN_NAME:-scratch-rtdetr-l-ioqc-sa-100ep}"
RUN_DIR="$PROJECT_DIR/$RUN_NAME"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"
RESULTS_REPO="${RESULTS_REPO:-$STORAGE_ROOT/results-checkout}"
SOURCE_BRANCH="${SOURCE_BRANCH:-codex/ioqc-sa}"
TAG="${TAG:-ioqc-sa-rtdetr-l-live}"
EPOCHS="${EPOCHS:-100}"
WORKERS="${WORKERS:-8}"
if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
INITIAL_BATCH="${INITIAL_BATCH:-}"
BATCH_LEVELS="${BATCH_LEVELS:-}"
SAVE_PERIOD="${SAVE_PERIOD:-1}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"
ENABLE_GITHUB_SYNC="${ENABLE_GITHUB_SYNC:-1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

mkdir -p "$PROJECT_DIR" "$RUN_DIR" "$LOG_DIR" "$RESULTS_REPO" "$STORAGE_ROOT/state"
cd "$REPO_DIR"

exec 9>"$LOG_DIR/ioqc_sa.lock"
if ! flock -n 9; then
  printf 'Another IOQC-SA server run already holds %s.\n' "$LOG_DIR/ioqc_sa.lock" >&2
  exit 1
fi

[[ -x "$PYTHON" ]] || { printf 'Python environment not found: %s\n' "$PYTHON" >&2; exit 1; }
if [[ "$ENABLE_GITHUB_SYNC" == "1" ]]; then
  [[ -s "$TOKEN_FILE" ]] || { printf 'GitHub token file is missing or empty: %s\n' "$TOKEN_FILE" >&2; exit 1; }
  if (( $(stat -c '%a' "$TOKEN_FILE") % 100 != 0 )); then
    printf 'GitHub token file must not be readable by group or others: %s\n' "$TOKEN_FILE" >&2
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$LOG_DIR/ioqc_sa_launcher.pid"

STATE_FILE="$STORAGE_ROOT/state/ioqc_sa_adaptive_state.json"
STATUS_FILE="$LOG_DIR/ioqc_sa_status.json"
SUPERVISOR_LOG="$LOG_DIR/ioqc_sa_training.log"
PID_LOCK="$LOG_DIR/ioqc_sa_supervisor.pid"
SYNC_STATUS="$LOG_DIR/ioqc_sa_github_sync.json"

sync_arguments=(
  scripts/sync_btdse_checkpoint.py
  --run-dir "$RUN_DIR"
  --token-file "$TOKEN_FILE"
  --results-repo "$RESULTS_REPO"
  --run-name "$RUN_NAME"
  --tag "$TAG"
  --source-branch "$SOURCE_BRANCH"
  --status-file "$SYNC_STATUS"
  --retain 3
  --asset-prefix ioqc-sa-last
  --release-name "IOQC-SA RT-DETR-L Live Checkpoints"
  --release-body "Rolling resumable checkpoints for standalone IOQC-SA training."
  --interval 60
)

sync_pid=""
if [[ "$ENABLE_GITHUB_SYNC" == "1" ]]; then
  "$PYTHON" -u "${sync_arguments[@]}" >> "$LOG_DIR/ioqc_sa_github_sync.log" 2>&1 &
  sync_pid=$!
fi
supervisor_pid=""
cleanup() {
  if [[ -n "$supervisor_pid" ]] && kill -0 "$supervisor_pid" 2>/dev/null; then
    kill -TERM "$supervisor_pid" 2>/dev/null || true
    wait "$supervisor_pid" 2>/dev/null || true
  fi
  if [[ -n "$sync_pid" ]]; then
    kill "$sync_pid" 2>/dev/null || true
    wait "$sync_pid" 2>/dev/null || true
  fi
  rm -f "$LOG_DIR/ioqc_sa_launcher.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

supervisor_arguments=(
  scripts/supervise_ioqc_sa.py
  --project "$PROJECT_DIR"
  --name "$RUN_NAME"
  --state "$STATE_FILE"
  --status "$STATUS_FILE"
  --log "$SUPERVISOR_LOG"
  --lock "$PID_LOCK"
  --epochs "$EPOCHS"
  --workers "$WORKERS"
  --device "$DEVICE"
  --save-period "$SAVE_PERIOD"
  --optimizer "$OPTIMIZER"
  --min-free-gib "$MIN_FREE_GIB"
)
if [[ -n "$INITIAL_BATCH" ]]; then
  supervisor_arguments+=(--initial-batch "$INITIAL_BATCH")
fi
if [[ -n "$BATCH_LEVELS" ]]; then
  supervisor_arguments+=(--batch-levels "$BATCH_LEVELS")
fi

set +e
"$PYTHON" -u "${supervisor_arguments[@]}" &
supervisor_pid=$!
wait "$supervisor_pid"
supervisor_rc=$?
supervisor_pid=""
set -e

if [[ -n "$sync_pid" ]]; then
  kill "$sync_pid" 2>/dev/null || true
  wait "$sync_pid" 2>/dev/null || true
fi
rm -f "$LOG_DIR/ioqc_sa_launcher.pid"
trap - EXIT

if (( supervisor_rc != 0 )); then
  printf 'Supervisor stopped with code %s. Local checkpoints and state are intact.\n' "$supervisor_rc" >&2
  exit "$supervisor_rc"
fi

if [[ "$ENABLE_GITHUB_SYNC" == "1" ]]; then
  published=0
  for attempt in 1 2 3 4 5; do
    if "$PYTHON" -u "${sync_arguments[@]}" --once >> "$LOG_DIR/ioqc_sa_github_sync.log" 2>&1; then
      published=1
      break
    fi
    printf 'Final GitHub publication attempt %s failed; retrying in 60 seconds.\n' "$attempt" >&2
    sleep 60
  done

  if (( published != 1 )); then
    printf 'Training completed, but final GitHub publication still needs retry. Local data is intact.\n' >&2
    exit 3
  fi
  printf 'IOQC-SA training and final GitHub publication are verified.\n'
else
  printf 'IOQC-SA training completed with GitHub synchronization disabled.\n'
fi
if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  if [[ "$ENABLE_GITHUB_SYNC" == "1" ]]; then
    shutdown -h now
  else
    printf 'Automatic shutdown skipped because GitHub synchronization is disabled.\n' >&2
  fi
fi

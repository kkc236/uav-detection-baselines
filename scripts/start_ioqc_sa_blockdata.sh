#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/root/blockdata/ioqc-sa}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/repo}"
STORAGE_ROOT="${STORAGE_ROOT:-$WORK_ROOT/storage}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"
LOG_DIR="$STORAGE_ROOT/logs"
PID_FILE="$LOG_DIR/ioqc_sa_launcher.pid"

[[ -d "$REPO_DIR" ]] || { printf 'Repository not found: %s\n' "$REPO_DIR" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { printf 'Python not found: %s\n' "$PYTHON" >&2; exit 1; }
mkdir -p "$LOG_DIR"

if [[ -s "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    printf 'IOQC-SA is already running with launcher PID %s.\n' "$existing_pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

printf 'Persistent root: %s\n' "$STORAGE_ROOT"
df -h "$STORAGE_ROOT"

nohup env \
  WORK_ROOT="$WORK_ROOT" \
  REPO_DIR="$REPO_DIR" \
  STORAGE_ROOT="$STORAGE_ROOT" \
  PYTHON="$PYTHON" \
  OPTIMIZER="$OPTIMIZER" \
  MIN_FREE_GIB="$MIN_FREE_GIB" \
  bash "$REPO_DIR/scripts/run_ioqc_sa_server.sh" \
  > "$LOG_DIR/ioqc_sa_launcher.log" 2>&1 < /dev/null &

launcher_pid=$!
printf 'IOQC-SA launcher started with PID %s.\n' "$launcher_pid"
printf 'Training log: %s/ioqc_sa_training.log\n' "$LOG_DIR"

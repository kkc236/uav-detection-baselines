#!/usr/bin/env bash
set -u

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CURRENT_PID DATASET_ROOT INITIAL_STATE OUTPUT_ROOT LOG_ROOT" >&2
  exit 64
fi

current_pid="$1"
dataset_root="$2"
initial_state="$3"
output_root="$4"
log_root="$5"
run_dir="$output_root/formal-seed0-lrs_gfdr-v1"
fia_dir="$output_root/formal-seed0-lrs_gfdr_fia-v1"
mkdir -p "$log_root"

while kill -0 "$current_pid" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "$run_dir/results.csv" ]] || [[ "$(wc -l < "$run_dir/results.csv")" -ne 101 ]] || \
   [[ ! -f "$run_dir/weights/best.pt" ]] || [[ ! -f "$run_dir/weights/last.pt" ]]; then
  printf 'LRS-GFDR did not satisfy completion criteria; FIA was not started.\n' \
    > "$log_root/lrs_gfdr_fia_chain.blocked"
  exit 2
fi

if [[ -e "$fia_dir/results.csv" ]]; then
  printf 'LRS-GFDR-FIA output already exists; refusing duplicate launch.\n' \
    > "$log_root/lrs_gfdr_fia_chain.blocked"
  exit 3
fi

source /root/miniconda3/bin/activate lrs-v2
nohup python scripts/train_lrs_gfdr_fia.py \
  --dataset-root "$dataset_root" \
  --initial-state "$initial_state" \
  --output-root "$output_root" \
  > "$log_root/lrs_gfdr_fia_20260909.log" 2>&1 &
fia_pid=$!
printf '%s\n' "$fia_pid" > "$log_root/lrs_gfdr_fia.pid"
printf 'LRS-GFDR complete; started LRS-GFDR-FIA with PID %s.\n' "$fia_pid" \
  > "$log_root/lrs_gfdr_fia_chain.started"

#!/usr/bin/env bash
set -euo pipefail

gate="${1:?usage: supervise_sqda_sgc_server.sh g2|formal [token-file] [target-epochs]}"
token_file="${2:-/root/.config/sqda-sgc/github-token}"
target_epochs="${3:-}"
root="/root/data/uav"
repo="$root/sqda-sgc"
project="$root/runs/sqda-sgc"
venv="$root/venv"

case "$gate" in
  g1) run_name="sqda-sgc-g1-seed0-3ep"; default_epochs=3 ;;
  g1r) run_name="sqda-sgc-g1r-seed0-3ep"; default_epochs=3 ;;
  g2) run_name="sqda-sgc-g2-seed0-10ep"; default_epochs=10 ;;
  formal) run_name="sqda-sgc-formal-seed0-100ep"; default_epochs=100 ;;
  *) echo "unsupported SQDA-SGC gate: $gate" >&2; exit 2 ;;
esac
target_epochs="${target_epochs:-$default_epochs}"
run_dir="$project/$run_name"
log_file="$project/${gate}-supervisor.log"

cd "$repo"
exec >>"$log_file" 2>&1
echo "supervisor started gate=$gate target_epochs=$target_epochs"

while true; do
  if pgrep -af "train_rtdetr_sqda_sgc.py.*--gate[= ]$gate" >/dev/null; then
    sleep 60
    continue
  fi

  completed="$("$venv/bin/python" -c \
    'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(json.loads(p.read_text()).get("completed_epoch", 0) if p.is_file() else 0)' \
    "$run_dir/stage-status.json")"
  if (( completed >= target_epochs )); then
    echo "target complete gate=$gate completed=$completed"
    exit 0
  fi

  resume_from="$("$venv/bin/python" -c \
    'from src.checkpoint_recovery import find_resume_checkpoint; import sys; p=find_resume_checkpoint(sys.argv[1]); print(p or "")' \
    "$run_dir")"
  if [[ -z "$resume_from" && "$completed" != "0" ]]; then
    echo "no valid resume checkpoint yet; retrying completed=$completed"
    sleep 60
    continue
  fi

  echo "launching gate=$gate completed=$completed resume=${resume_from:-none}"
  bash scripts/run_sqda_sgc_server.sh "$gate" "$token_file" "$resume_from" "$target_epochs"
  sleep 60
done

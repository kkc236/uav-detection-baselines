#!/usr/bin/env bash
set -euo pipefail

gate="${1:?usage: run_sqda_sgc_server.sh g1|g1r|g2 [token-file]}"
if [[ "$gate" != "g1" && "$gate" != "g1r" && "$gate" != "g2" ]]; then
  echo "gate must be g1, g1r, or g2" >&2
  exit 2
fi

root="/root/data/uav"
repo="$root/sqda-sgc"
venv="$root/venv"
project="$root/runs/sqda-sgc"
checkpoint="$root/checkpoints/matched-baseline-best-epoch-0100.pt"
data="$root/protocols/tsgr-p2-e1/source-VisDrone-full.yaml"
token_file="${2:-/root/.config/sqda-sgc/github-token}"
branch="codex/sqda-sgc"

case "$gate" in
  g1) run_name="sqda-sgc-g1-seed0-3ep" ;;
  g1r) run_name="sqda-sgc-g1r-seed0-3ep" ;;
  g2) run_name="sqda-sgc-g2-seed0-10ep" ;;
esac
retain=3
run_dir="$project/$run_name"
pending_log="$project/.${gate}-console.pending.log"
sync_log="$project/${gate}-github-sync.log"
status_file="$project/${gate}-github-sync-status.json"
tag="sqda-sgc-${gate}-live"

mkdir -p "$project"
if pgrep -af "train_rtdetr_sqda_sgc.py.*--gate[= ]$gate" >/dev/null; then
  echo "an SQDA-SGC $gate training process is already running" >&2
  exit 3
fi

cd "$repo"
nohup "$venv/bin/python" -u scripts/train_rtdetr_sqda_sgc.py \
  --gate "$gate" \
  --checkpoint "$checkpoint" \
  --data "$data" \
  --project "$project" \
  --device 0 \
  --workers 8 \
  > "$pending_log" 2>&1 &
train_pid=$!

for _ in $(seq 1 120); do
  if [[ -f "$run_dir/run-manifest.json" ]]; then
    mv "$pending_log" "$run_dir/${gate}-console.log"
    break
  fi
  if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "training exited before creating its run manifest; inspect $pending_log" >&2
    exit 4
  fi
  sleep 1
done
if [[ ! -f "$run_dir/run-manifest.json" ]]; then
  echo "training did not create its run manifest within 120 seconds" >&2
  exit 5
fi

nohup "$venv/bin/python" -u scripts/sync_experiment_checkpoint.py \
  --run-dir "$run_dir" \
  --token-file "$token_file" \
  --repo kkc236/uav-detection-baselines \
  --repo-url https://github.com/kkc236/uav-detection-baselines.git \
  --tag "$tag" \
  --source-branch "$branch" \
  --results-branch sqda-sgc-results \
  --results-repo "$root/sqda-sgc-results" \
  --run-name "$run_name" \
  --retain "$retain" \
  --asset-prefix "sqda-sgc-${gate}" \
  --release-name "SQDA-SGC ${gate^^} RTX 4090 Stage Checkpoints" \
  --release-body "Validated SQDA-SGC ${gate^^} frozen-stock checkpoints and stage evidence." \
  --interval 60 \
  --status-file "$status_file" \
  > "$sync_log" 2>&1 &
sync_pid=$!

printf 'TRAIN_PID=%s\nSYNC_PID=%s\nRUN_DIR=%s\n' "$train_pid" "$sync_pid" "$run_dir"

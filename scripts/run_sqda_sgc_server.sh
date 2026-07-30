#!/usr/bin/env bash
set -euo pipefail

gate="${1:?usage: run_sqda_sgc_server.sh g1|g1r|g2|formal [token-file] [resume-from] [target-epochs]}"
if [[ "$gate" != "g1" && "$gate" != "g1r" && "$gate" != "g2" && "$gate" != "formal" ]]; then
  echo "gate must be g1, g1r, g2, or formal" >&2
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
  formal) run_name="sqda-sgc-formal-seed0-100ep" ;;
esac
resume_from="${3:-}"
target_epochs="${4:-}"
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
if [[ -z "$resume_from" && -d "$run_dir" ]]; then
  resume_from="$("$venv/bin/python" -c \
    'from src.checkpoint_recovery import find_resume_checkpoint; import sys; p=find_resume_checkpoint(sys.argv[1]); print(p or "")' \
    "$run_dir")"
fi

train_args=(
  scripts/train_rtdetr_sqda_sgc.py
  --gate "$gate" \
  --checkpoint "$checkpoint" \
  --data "$data" \
  --project "$project" \
  --device 0 \
  --workers 8 \
)
if [[ -n "$resume_from" ]]; then
  train_args+=(--resume-from "$resume_from")
fi
if [[ -n "$target_epochs" ]]; then
  train_args+=(--target-epochs "$target_epochs")
fi
nohup "$venv/bin/python" -u "${train_args[@]}" > "$pending_log" 2>&1 &
train_pid=$!

for _ in $(seq 1 120); do
  if [[ -f "$run_dir/run-manifest.json" ]]; then
    if [[ -n "$resume_from" ]]; then
      mv "$pending_log" "$run_dir/${gate}-console-resume-$(date +%s).log"
    else
      mv "$pending_log" "$run_dir/${gate}-console.log"
    fi
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

sync_pid="$(pgrep -f "sync_experiment_checkpoint.py.*--run-dir $run_dir" | head -n 1 || true)"
if [[ -z "$sync_pid" ]]; then
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
fi

printf 'TRAIN_PID=%s\nSYNC_PID=%s\nRUN_DIR=%s\n' "$train_pid" "$sync_pid" "$run_dir"

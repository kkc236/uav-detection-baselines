#!/usr/bin/env bash
set -euo pipefail

gate="${1:?usage: run_sqda_geometry_gate_server.sh g1|g2 [token-file] [resume-from]}"
if [[ "$gate" != "g1" && "$gate" != "g2" ]]; then
  echo "gate must be g1 or g2" >&2
  exit 2
fi

root="/root/data/uav"
repo="$root/sqda-sgc"
venv="$root/venv"
project="$root/runs/sqda-geometry-gate"
baseline="$root/checkpoints/matched-baseline-best-epoch-0100.pt"
adapter="$root/runs/sqda-sgc/sqda-sgc-g2-seed0-10ep/weights/best.pt"
data="$root/protocols/tsgr-p2-e1/source-VisDrone-full.yaml"
token_file="${2:-/root/.config/sqda-sgc/github-token}"
branch="codex/sqda-sgc"
admission="$project/geometry-branch-diagnosis/g1-admission.json"
run_name="sqda-geometry-gate-$gate-seed0-$([[ "$gate" == "g1" ]] && echo 3 || echo 10)ep"
run_dir="$project/$run_name"
pending_log="$project/.${gate}-console.pending.log"
sync_log="$project/${gate}-github-sync.log"
status_file="$project/${gate}-github-sync-status.json"
resume_from="${3:-}"

if [[ ! -f "$admission" ]]; then
  echo "missing diagnosis admission decision: $admission" >&2
  exit 3
fi
if [[ "$("$venv/bin/python" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["passed"])).lower())' "$admission")" != "true" ]]; then
  echo "geometry G1 was not admitted by retained-G2 evidence; no training started." >&2
  exit 4
fi
if [[ "$gate" == "g2" ]]; then
  g1_decision="$project/sqda-geometry-gate-g1-seed0-3ep/final-gate-decision.json"
  if [[ ! -f "$g1_decision" ]] || [[ "$("$venv/bin/python" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["passed"])).lower())' "$g1_decision")" != "true" ]]; then
    echo "a passed G1 final decision is required before G2." >&2
    exit 5
  fi
fi
if pgrep -af "train_rtdetr_sqda_geometry_gate.py.*--gate[= ]$gate" >/dev/null; then
  echo "a geometry-gate $gate training process is already running" >&2
  exit 6
fi

cd "$repo"
git fetch origin "$branch"
git switch "$branch"
git pull --ff-only origin "$branch"
if [[ -z "$resume_from" && -d "$run_dir" ]]; then
  resume_from="$("$venv/bin/python" -c \
    'from src.checkpoint_recovery import find_resume_checkpoint; import sys; print(find_resume_checkpoint(sys.argv[1]) or "")' \
    "$run_dir")"
fi

train_args=(
  scripts/train_rtdetr_sqda_geometry_gate.py
  --gate "$gate"
  --checkpoint "$baseline"
  --adapter-checkpoint "$adapter"
  --data "$data"
  --project "$project"
  --device 0
  --workers 8
)
if [[ -n "$resume_from" ]]; then
  train_args+=(--resume-from "$resume_from")
fi
nohup "$venv/bin/python" -u "${train_args[@]}" > "$pending_log" 2>&1 &
train_pid=$!

for _ in $(seq 1 120); do
  if [[ -f "$run_dir/run-manifest.json" ]]; then
    mv "$pending_log" "$run_dir/${gate}-console.log"
    break
  fi
  if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "geometry-gate training exited before creating its manifest; inspect $pending_log" >&2
    exit 7
  fi
  sleep 1
done
if [[ ! -f "$run_dir/run-manifest.json" ]]; then
  echo "geometry-gate training did not create its manifest within 120 seconds" >&2
  exit 8
fi

sync_pid="$(pgrep -f "sync_experiment_checkpoint.py.*--run-dir $run_dir" | head -n 1 || true)"
if [[ -z "$sync_pid" ]]; then
  nohup "$venv/bin/python" -u scripts/sync_experiment_checkpoint.py \
    --run-dir "$run_dir" \
    --token-file "$token_file" \
    --repo kkc236/uav-detection-baselines \
    --repo-url https://github.com/kkc236/uav-detection-baselines.git \
    --tag "sqda-geometry-gate-${gate}-live" \
    --source-branch "$branch" \
    --results-branch sqda-geometry-gate-results \
    --results-repo "$root/sqda-geometry-gate-results" \
    --run-name "$run_name" \
    --retain 3 \
    --asset-prefix "sqda-geometry-gate-${gate}" \
    --release-name "SQDA Geometry Gate ${gate^^} RTX 4090" \
    --release-body "Frozen inherited SQDA adapter; only geometry trust MLP is trainable." \
    --interval 60 \
    --status-file "$status_file" \
    > "$sync_log" 2>&1 &
  sync_pid=$!
fi

printf 'TRAIN_PID=%s\nSYNC_PID=%s\nRUN_DIR=%s\n' "$train_pid" "$sync_pid" "$run_dir"

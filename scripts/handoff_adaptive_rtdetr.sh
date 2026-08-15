#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/uav-detection-baselines
PYTHON=/root/miniconda3/bin/python
CHECKPOINT="$ROOT/runs/detect/runs/baselines/scratch-rtdetr-100ep-5090-b24-max/weights/last.pt"
OLD_PID_FILE="$ROOT/logs/rtdetr_b16_true_resume.pid"
HANDOFF_LOG="$ROOT/logs/adaptive_handoff.log"

cd "$ROOT"
old_pid=$(<"$OLD_PID_FILE")

checkpoint_epoch() {
  "$PYTHON" - "$CHECKPOINT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)) + 1)
PY
}

starting_epoch=$(checkpoint_epoch)
target_epoch=$((starting_epoch + 1))

while kill -0 "$old_pid" 2>/dev/null; do
  completed=$(checkpoint_epoch)
  if (( completed >= target_epoch )); then
    break
  fi
  sleep 5
done

completed=$(checkpoint_epoch)
if (( completed < target_epoch )); then
  printf 'Trainer ended before handoff target epoch %s was saved.\n' "$target_epoch" >> "$HANDOFF_LOG"
  exit 1
fi

kill -TERM "$old_pid" 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 "$old_pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$old_pid" 2>/dev/null; then
  kill -KILL "$old_pid"
fi

completed=$(checkpoint_epoch)
"$PYTHON" - "$CHECKPOINT" "$completed" <<'PY'
import sys
from pathlib import Path

from src.adaptive_batch import AdaptiveBatchState, save_state

checkpoint = str(Path(sys.argv[1]).resolve())
completed_epoch = int(sys.argv[2])
state = AdaptiveBatchState(
    current_batch=16,
    completed_epoch=completed_epoch,
    checkpoint=checkpoint,
    last_event="handoff",
)
save_state(Path("logs/adaptive_rtdetr_state.json"), state)
PY

rm -f logs/adaptive_rtdetr.lock
export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup bash scripts/run_adaptive_rtdetr_to_completion.sh \
  > logs/adaptive_supervisor_launcher.log 2>&1 &
supervisor_pid=$!
printf '%s\n' "$supervisor_pid" > logs/adaptive_supervisor.pid
printf 'handoff complete epoch=%s old_pid=%s supervisor_pid=%s\n' "$completed" "$old_pid" "$supervisor_pid" >> "$HANDOFF_LOG"

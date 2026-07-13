#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/uav-detection-baselines
PYTHON=/root/miniconda3/bin/python
CHECKPOINT="$ROOT/runs/detect/runs/baselines/scratch-rtdetr-100ep-5090-b24-max/weights/last.pt"
PID_FILE="$ROOT/logs/adaptive_supervisor.pid"
RELOAD_LOG="$ROOT/logs/adaptive_reload.log"

cd "$ROOT"
root_pid=$(<"$PID_FILE")
supervisor_pid=$(pgrep -P "$root_pid" -f 'train_rtdetr_adaptive.py --checkpoint' | head -n 1)
trainer_pid=$(pgrep -P "$supervisor_pid" -f 'train_rtdetr_adaptive.py --child' | head -n 1)

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
printf 'waiting for epoch=%s root=%s supervisor=%s trainer=%s\n' \
  "$target_epoch" "$root_pid" "$supervisor_pid" "$trainer_pid" >> "$RELOAD_LOG"

while kill -0 "$trainer_pid" 2>/dev/null; do
  completed=$(checkpoint_epoch)
  if (( completed >= target_epoch )); then
    break
  fi
  sleep 5
done

completed=$(checkpoint_epoch)
if (( completed < target_epoch )); then
  printf 'trainer ended before reload target epoch=%s\n' "$target_epoch" >> "$RELOAD_LOG"
  exit 1
fi

kill -TERM "$supervisor_pid" 2>/dev/null || true
pkill -TERM -P "$trainer_pid" 2>/dev/null || true
kill -TERM "$trainer_pid" 2>/dev/null || true
kill -TERM "$root_pid" 2>/dev/null || true

for _ in $(seq 1 30); do
  if ! kill -0 "$root_pid" 2>/dev/null && ! kill -0 "$supervisor_pid" 2>/dev/null && ! kill -0 "$trainer_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
kill -KILL "$supervisor_pid" "$trainer_pid" "$root_pid" 2>/dev/null || true

"$PYTHON" - "$CHECKPOINT" "$completed" <<'PY'
import sys
from pathlib import Path

from src.adaptive_batch import load_state, save_state

state_path = Path("logs/adaptive_rtdetr_state.json")
state = load_state(state_path)
state.checkpoint = str(Path(sys.argv[1]).resolve())
state.completed_epoch = max(state.completed_epoch, int(sys.argv[2]))
state.stable_epochs = 0
state.last_event = "policy_reload"
save_state(state_path, state)
PY

rm -f logs/adaptive_rtdetr.lock
export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup bash scripts/run_adaptive_rtdetr_to_completion.sh \
  > logs/adaptive_supervisor_launcher.log 2>&1 &
new_root_pid=$!
printf '%s\n' "$new_root_pid" > "$PID_FILE"
printf 'reload complete epoch=%s new_root=%s\n' "$completed" "$new_root_pid" >> "$RELOAD_LOG"

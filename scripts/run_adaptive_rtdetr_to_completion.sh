#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/uav-detection-baselines
PYTHON=/root/miniconda3/bin/python
CHECKPOINT="$ROOT/runs/detect/runs/baselines/scratch-rtdetr-100ep-5090-b24-max/weights/last.pt"
RUN_DIR="$ROOT/runs/detect/runs/baselines/scratch-rtdetr-100ep-5090-b24-max"

cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u scripts/train_rtdetr_adaptive.py \
  --checkpoint "$CHECKPOINT" \
  --state logs/adaptive_rtdetr_state.json \
  --log logs/adaptive_rtdetr.log \
  --status logs/adaptive_rtdetr_status.json \
  --lock logs/adaptive_rtdetr.lock \
  --workdir "$ROOT" \
  --batch 16 \
  --target-epoch 100

while ! "$PYTHON" -u scripts/publish_rtdetr_results.py \
  --run-dir "$RUN_DIR" \
  --repo-dir "$ROOT" \
  --token-file /root/autodl-tmp/github_token \
  --results-dir results/rtdetr-100ep-5090-adaptive \
  --tag rtdetr-100ep-5090-adaptive \
  --target-epoch 100 \
  >> logs/adaptive_publisher.log 2>&1; do
  printf '[%s] publication failed; retrying in 300 seconds\n' "$(date '+%F %T')" >> logs/adaptive_publisher.log
  sleep 300
done

printf '[%s] publication verified; powering off\n' "$(date '+%F %T')" >> logs/adaptive_publisher.log
sync
/sbin/poweroff

#!/bin/bash
set +e
LOGDIR=/home/ubuntu/saded-fresh100-logs/final-saded-fresh100-c5c35374
TMP_RUN=/home/ubuntu/saded-fresh100-tmp/final-saded-fresh100-c5c35374
mkdir -p "$TMP_RUN"
printf '%s\n' "$$" > "$LOGDIR/driver-runtime.pid.tmp"
mv "$LOGDIR/driver-runtime.pid.tmp" "$LOGDIR/driver-runtime.pid"
printf '%s\n' RUNNING > "$LOGDIR/status.tmp"
mv "$LOGDIR/status.tmp" "$LOGDIR/status"
cd /home/ubuntu/repo-saded-fresh100-c5c35374 || exit 90
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TMPDIR="$TMP_RUN"
/mnt/uav/venv/bin/python -u scripts/train_rtdetr_saded_stock.py \
  --protocol-manifest /home/ubuntu/saded-fresh100-protocols/final-saded-fresh100-c5c35374/protocol_manifest.json \
  --initial-state /mnt/uav/protocols/ebc-qp-d2-musgd-seed0/initial-state-seed0.pt \
  --data /home/ubuntu/tascv-protocols/final-tascv-dbf84670/tascv_full_train_only.yaml \
  --project /home/ubuntu/saded-fresh100-runs/final-saded-fresh100-c5c35374 \
  --name seed0 --device 0 --seed 0 \
  > "$LOGDIR/train.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$LOGDIR/exit_code.tmp"
mv "$LOGDIR/exit_code.tmp" "$LOGDIR/exit_code"
if [ "$rc" -eq 0 ]; then final=TRAIN_COMPLETE; else final=TRAIN_INVALID; fi
printf '%s\n' "$final" > "$LOGDIR/status.tmp"
mv "$LOGDIR/status.tmp" "$LOGDIR/status"
exit "$rc"

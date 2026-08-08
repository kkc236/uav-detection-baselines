#!/usr/bin/env bash
set -Eeuo pipefail

export REPO_DIR=/home/ubuntu/glgm/source/ultralytics-main
export DATA_YAML=/home/ubuntu/glgm/datasets/VisDrone-official-v2/data.yaml
export WORK_ROOT=/home/ubuntu/glgm/work/screen30-seed0-v3
export PYTHON=/home/ubuntu/glgm/env/venv/bin/python
export BASE_WEIGHTS=/home/ubuntu/glgm/weights/rtdetr-x.pt
export EPOCHS=30
export IMGSZ=640
export BATCH=4
export WORKERS=4
export SEED=0
export FRACTION=1.0
export SAVE_PERIOD=1
export STRICT_PAIR=1
export PARALLEL=0
export CONTROL_DEVICE=0
export GLGM_DEVICE=0
export EVAL_DEVICE=0

test ! -e "$WORK_ROOT"
cd /home/ubuntu/glgm/source/glgm-experiment-package
test "$(sha256sum yolo26n.pt | cut -d ' ' -f 1)" = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
exec /bin/bash scripts/run_glgm_pair.sh

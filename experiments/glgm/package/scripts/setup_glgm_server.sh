#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPO_DIR:?Set REPO_DIR to the Ultralytics source checkout}"
: "${WORK_ROOT:=/root/data/glgm-experiment}"
: "${PYTHON_BIN:=python3}"

VENV_DIR="$WORK_ROOT/venv"
ENV_DIR="$WORK_ROOT/environment"
mkdir -p "$WORK_ROOT" "$ENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
"$VENV_DIR/bin/python" -m pip install -e "$REPO_DIR"
"$VENV_DIR/bin/python" - <<'PY'
import torch
import ultralytics
import polars

print("torch", torch.__version__)
print("ultralytics", ultralytics.__version__)
print("polars", polars.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if ultralytics.__version__ != "8.4.116":
    raise SystemExit("Expected Ultralytics 8.4.116")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

"$VENV_DIR/bin/python" -m pip freeze > "$ENV_DIR/pip-freeze.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$ENV_DIR/nvidia-smi-q.txt"
fi
sha256sum \
  "$REPO_DIR/ultralytics/nn/modules/block.py" \
  "$REPO_DIR/ultralytics/nn/modules/__init__.py" \
  "$REPO_DIR/ultralytics/nn/tasks.py" \
  > "$ENV_DIR/source-SHA256SUMS.txt"

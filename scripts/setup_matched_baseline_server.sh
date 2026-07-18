#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-$(dirname "$REPO_DIR")/matched-baseline-storage}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$STORAGE_ROOT/venv}"
TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"

command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is required.\n' >&2; exit 1; }
command -v git >/dev/null || { printf 'git is required.\n' >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { printf '%s is required.\n' "$PYTHON_BIN" >&2; exit 1; }

mkdir -p "$STORAGE_ROOT/datasets" "$STORAGE_ROOT/runs/matched-baseline" \
  "$STORAGE_ROOT/logs" "$STORAGE_ROOT/secrets" "$STORAGE_ROOT/results-checkout"

free_kib=$(df -Pk "$STORAGE_ROOT" | awk 'NR==2 {print $4}')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( free_kib < required_kib )); then
  printf 'At least %s GiB free space is required on %s.\n' "$MIN_FREE_GIB" "$STORAGE_ROOT" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -n 1)
case "$gpu_name" in
  *5090*)
    "$VENV_DIR/bin/python" -m pip install \
      torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128
    ;;
  *)
    "$VENV_DIR/bin/python" -m pip install \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121
    ;;
esac
"$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<PY
from ultralytics import settings
settings.update({"datasets_dir": r"$STORAGE_ROOT/datasets"})
PY

cd "$REPO_DIR"
"$VENV_DIR/bin/python" - <<'PY'
import torch
import ultralytics

assert torch.cuda.is_available(), "CUDA is not available"
assert ultralytics.__version__ == "8.4.90", ultralytics.__version__
name = torch.cuda.get_device_name(0)
if "5090" in name:
    assert "sm_120" in torch.cuda.get_arch_list(), "Installed PyTorch wheel lacks RTX 5090 support"
print({"gpu": name, "torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__})
PY

if [[ ! -d "$STORAGE_ROOT/datasets/VisDrone/images/train" ]]; then
  "$VENV_DIR/bin/python" scripts/prepare_visdrone.py \
    --dataset-dir "$STORAGE_ROOT/datasets/VisDrone" \
    --splits train val
fi

if [[ ! -e "$TOKEN_FILE" ]]; then
  install -m 600 /dev/null "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

printf '\nSetup complete. Store the GitHub token in:\n  %s\n' "$TOKEN_FILE"
printf 'The token needs Contents read/write permission only for kkc236/uav-detection-baselines.\n'


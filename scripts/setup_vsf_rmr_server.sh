#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-$(dirname "$REPO_DIR")/vsf-rmr-storage}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$STORAGE_ROOT/venv}"
TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"
MIN_FREE_GIB="${MIN_FREE_GIB:-30}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is required.\n' >&2; exit 1; }
command -v git >/dev/null || { printf 'git is required.\n' >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { printf '%s is required.\n' "$PYTHON_BIN" >&2; exit 1; }

mkdir -p "$STORAGE_ROOT/datasets" "$STORAGE_ROOT/runs/vsf-rmr" "$STORAGE_ROOT/logs" \
  "$STORAGE_ROOT/state" "$STORAGE_ROOT/secrets" "$STORAGE_ROOT/results-checkout"

free_kib=$(df -Pk "$STORAGE_ROOT" | awk 'NR==2 {print $4}')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( free_kib < required_kib )); then
  printf 'At least %s GiB free space is required on %s.\n' "$MIN_FREE_GIB" "$STORAGE_ROOT" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
export PIP_INDEX_URL="$PYPI_INDEX_URL"
export PIP_TRUSTED_HOST
"$VENV_DIR/bin/python" -m pip install --index-url "$PYPI_INDEX_URL" --upgrade pip wheel setuptools
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -n 1)
case "$gpu_name" in
  *5090*)
    "$VENV_DIR/bin/python" -m pip install \
      torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url "$PYPI_INDEX_URL"
    ;;
  *)
    "$VENV_DIR/bin/python" -m pip install \
      torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
      --extra-index-url "$PYPI_INDEX_URL"
    ;;
esac
"$VENV_DIR/bin/python" -m pip install --index-url "$PYPI_INDEX_URL" -r "$REPO_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<PY
from ultralytics import settings
settings.update({"datasets_dir": r"$STORAGE_ROOT/datasets"})
PY

cd "$REPO_DIR"
"$VENV_DIR/bin/python" - <<'PY'
from dataclasses import asdict
from src.gpu_adaptive_batch import batch_policy_for_vram, detect_gpu_profile

profile = detect_gpu_profile()
policy = batch_policy_for_vram(total_gib=profile.total_gib, free_gib=profile.free_gib)
print("GPU profile:", asdict(profile))
print("Per-GPU adaptive batch policy:", policy)
if "5090" in profile.name:
    assert "sm_120" in __import__("torch").cuda.get_arch_list(), "PyTorch wheel lacks RTX 5090 support"
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

printf '\nSetup complete. Put a fine-grained GitHub token in:\n  %s\n' "$TOKEN_FILE"
printf 'The token needs Contents read/write access only to kkc236/uav-detection-baselines.\n'

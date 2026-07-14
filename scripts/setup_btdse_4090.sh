#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-$(dirname "$REPO_DIR")/btdse-storage}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-$STORAGE_ROOT/venv}"
TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"

printf 'Repository: %s\nPersistent storage: %s\n' "$REPO_DIR" "$STORAGE_ROOT"

command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is required.\n' >&2; exit 1; }
command -v git >/dev/null || { printf 'git is required.\n' >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { printf '%s is required.\n' "$PYTHON_BIN" >&2; exit 1; }

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$gpu_name" in
  *4090*) ;;
  *) printf 'Expected an RTX 4090, found: %s\n' "$gpu_name" >&2; exit 1 ;;
esac

mkdir -p "$STORAGE_ROOT" "$STORAGE_ROOT/datasets" "$STORAGE_ROOT/runs/btdse" \
  "$STORAGE_ROOT/logs" "$STORAGE_ROOT/secrets" "$STORAGE_ROOT/results-checkout"

free_kib=$(df -Pk "$STORAGE_ROOT" | awk 'NR==2 {print $4}')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( free_kib < required_kib )); then
  printf 'At least %s GiB free space is required on %s.\n' "$MIN_FREE_GIB" "$STORAGE_ROOT" >&2
  exit 1
fi

if [[ -L "$REPO_DIR/datasets" ]]; then
  current_target=$(readlink -f "$REPO_DIR/datasets")
  [[ "$current_target" == "$(readlink -f "$STORAGE_ROOT/datasets")" ]] || {
    printf 'Existing datasets symlink points to %s, not persistent storage.\n' "$current_target" >&2
    exit 1
  }
elif [[ -e "$REPO_DIR/datasets" ]]; then
  printf '%s already exists and is not a symlink; move it to persistent storage first.\n' "$REPO_DIR/datasets" >&2
  exit 1
else
  ln -s "$STORAGE_ROOT/datasets" "$REPO_DIR/datasets"
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/python" -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
"$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<PY
from ultralytics import settings
settings.update({"datasets_dir": r"$STORAGE_ROOT/datasets"})
PY

if [[ ! -e "$TOKEN_FILE" ]]; then
  install -m 600 /dev/null "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

cd "$REPO_DIR"
"$VENV_DIR/bin/python" - <<'PY'
import torch
from ultralytics import __version__ as ultralytics_version

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print(f"torch={torch.__version__} ultralytics={ultralytics_version}")
print(f"gpu={torch.cuda.get_device_name(0)} vram={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

if [[ ! -d "$STORAGE_ROOT/datasets/VisDrone/images/train" ]]; then
  "$VENV_DIR/bin/python" scripts/prepare_visdrone.py \
    --dataset-dir "$STORAGE_ROOT/datasets/VisDrone" \
    --splits train val
fi

printf '\nSetup complete. Store a NEW fine-grained GitHub token in:\n  %s\n' "$TOKEN_FILE"
printf 'Then run: chmod 600 %q\n' "$TOKEN_FILE"
printf 'Do not reuse any token that has appeared in chat or logs.\n'

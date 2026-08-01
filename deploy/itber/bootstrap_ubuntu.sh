#!/usr/bin/env bash
set -euo pipefail

source_root="${1:-/data/uav/source/uav-detection-baselines}"
lock_path="${source_root}/requirements-itber.lock"
venv_root="/data/uav/venvs/itber-v1.1"
wheelhouse_root="/data/uav/staging/itber-v1.1-wheelhouse"
marker_root="/data/uav/deploy/markers"
secret_path="/data/uav/HANDOFFS/secrets/github_token"
freeze_path="/data/uav/deploy/itber-v1.1-pip-freeze.txt"
mirror_index="https://mirrors.aliyun.com/pypi/simple"
pytorch_index="https://download.pytorch.org/whl/cu121"
expected_gpu="NVIDIA GeForce RTX 4090"
baseline_reference_driver="550.142"
expected_driver="570.133.07"
minimum_disk_kib=$((80 * 1024 * 1024))

if [[ ! -f "$lock_path" ]]; then
    printf 'missing lock file: %s\n' "$lock_path" >&2
    exit 2
fi
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | xargs)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | xargs)"
if [[ "$gpu" != "$expected_gpu" || "$driver" != "$expected_driver" ]]; then
    printf 'GPU contract mismatch: gpu=%s driver=%s\n' "$gpu" "$driver" >&2
    exit 3
fi
printf 'Runtime driver amendment: baseline=%s execution=%s\n' "$baseline_reference_driver" "$expected_driver"
available_kib="$(df -Pk /data | awk 'NR==2 {print $4}')"
if (( available_kib < minimum_disk_kib )); then
    printf 'insufficient /data disk: available_kib=%s\n' "$available_kib" >&2
    exit 4
fi
if [[ -e "$secret_path" ]]; then
    secret_mode="$(stat -c %a "$secret_path")"
    if [[ "$secret_mode" != "600" ]]; then
        printf 'GitHub token must have mode 600, got %s\n' "$secret_mode" >&2
        exit 5
    fi
fi

lock_sha="$(sha256sum "$lock_path" | awk '{print $1}')"
marker_path="${marker_root}/bootstrap-${lock_sha}.complete"
if [[ -f "$marker_path" ]]; then
    printf 'bootstrap already complete for lock %s\n' "$lock_sha"
    exit 0
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git rsync python3.10 python3.10-venv python3-pip build-essential
sudo install -d -o "$(id -u)" -g "$(id -g)" \
    /data/uav/venvs /data/uav/staging "$marker_root" /data/uav/deploy \
    /data/uav/config/Ultralytics /data/uav/HANDOFFS/secrets

if [[ ! -x "${venv_root}/bin/python" ]]; then
    python3.10 -m venv "$venv_root"
fi
export YOLO_CONFIG_DIR="/data/uav/config/Ultralytics"
"${venv_root}/bin/python" -m pip install --upgrade pip setuptools wheel \
    --index-url "$mirror_index"

wheelhouse_marker="${marker_root}/wheelhouse-${lock_sha}.complete"
if [[ -f "$wheelhouse_marker" ]]; then
    "${venv_root}/bin/python" -m pip install --no-index --no-deps \
        --find-links "$wheelhouse_root" --requirement "$lock_path"
else
    "${venv_root}/bin/python" -m pip install \
        torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
        --index-url "$pytorch_index"
    install_args=(
        -m pip install --no-deps --requirement "$lock_path"
        --extra-index-url "$pytorch_index"
    )
    if ! "${venv_root}/bin/python" "${install_args[@]}" --index-url "$mirror_index"; then
        "${venv_root}/bin/python" "${install_args[@]}" --index-url "https://pypi.org/simple"
    fi
fi

"${venv_root}/bin/python" - <<'PY'
import json
import platform
import torch
import torchvision
import ultralytics

actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "ultralytics": ultralytics.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
expected = {
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "ultralytics": "8.4.90",
    "cuda": "12.1",
    "gpu": "NVIDIA GeForce RTX 4090",
}
if actual != expected:
    raise SystemExit(f"runtime contract mismatch: expected={expected}, actual={actual}")
print(json.dumps(actual, sort_keys=True, separators=(",", ":")))
PY
"${venv_root}/bin/python" -m pip freeze | LC_ALL=C sort > "$freeze_path"
printf '%s\n' "$lock_sha" > "${marker_path}.tmp"
mv "${marker_path}.tmp" "$marker_path"

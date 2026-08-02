#!/usr/bin/env bash
set -euo pipefail

source_commit="${1:-}"
source_remote="https://github.com/kkc236/uav-detection-baselines.git"
if (( $# > 1 )); then
    printf 'bootstrap accepts only source_commit; the public source remote is fixed\n' >&2
    exit 2
fi
if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'source_commit must be exactly 40 lowercase hexadecimal characters\n' >&2
    exit 2
fi
source_short_sha="${source_commit:0:12}"
source_root="/data/uav/source/uav-detection-baselines-${source_short_sha}"
venv_root="/data/uav/venvs/iber-be-v1"
wheelhouse_root="/data/uav/staging/iber-be-v1-wheelhouse"
marker_root="/data/uav/deploy/iber-be-v1/markers"
secret_root="/data/uav/HANDOFFS/secrets"
secret_path="/data/uav/HANDOFFS/secrets/github_token"
publication_config="/data/uav/config/iber-be-v1/publication-screen.json"
git_http_env="/data/uav/config/iber-be-v1/git-http.env"
freeze_path="/data/uav/deploy/iber-be-v1/pip-freeze-${source_short_sha}.txt"
mirror_index="https://mirrors.aliyun.com/pypi/simple"
pytorch_index="https://download.pytorch.org/whl/cu121"
expected_gpu="NVIDIA GeForce RTX 4090"
expected_driver="570.133.07"
expected_gpu_memory_mib=49140
minimum_disk_kib=$((80 * 1024 * 1024))
cache_root="/data/uav/cache/iber-be-v1-${source_short_sha}"
run_root="/data/uav/runs/iber-be-v1/${source_short_sha}-seed0-amended"
results_root="/data/uav/results/iber-be-v1-${source_short_sha}"
log_path="/data/uav/logs/iber-be-v1-${source_short_sha}-pipeline.log"
pinned_requirements=(
    "torch==2.5.1+cu121"
    "torchvision==0.20.1+cu121"
    "ultralytics==8.4.90"
    "pytest==9.1.1"
    "requests==2.34.2"
    "numpy==2.2.6"
    "scipy==1.15.3"
    "pandas==2.3.1"
    "opencv-python-headless==4.12.0.88"
    "PyYAML==6.0.3"
    "psutil==7.2.2"
    "thop==0.1.1.post2209072238"
)
requirements_sha="$(printf '%s\n' "${pinned_requirements[@]}" | sha256sum | awk '{print $1}')"
marker_path="${marker_root}/bootstrap-${source_commit}-${requirements_sha}.complete"

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | xargs)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | xargs)"
gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p' | xargs)"
if [[ "$gpu" != "$expected_gpu" || "$driver" != "$expected_driver" || "$gpu_memory_mib" != "$expected_gpu_memory_mib" ]]; then
    printf 'amended GPU contract mismatch: gpu=%s driver=%s memory=%s\n' \
        "$gpu" "$driver" "$gpu_memory_mib" >&2
    exit 3
fi
available_kib="$(df -Pk /data | awk 'NR==2 {print $4}')"
if (( available_kib < minimum_disk_kib )); then
    printf 'insufficient /data disk: available_kib=%s\n' "$available_kib" >&2
    exit 4
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git rsync python3.10 python3.10-venv python3-pip build-essential
sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" \
    /data/uav/source /data/uav/venvs /data/uav/staging /data/uav/deploy/iber-be-v1 \
    "$marker_root" /data/uav/config/Ultralytics /data/uav/cache /data/uav/runs/iber-be-v1 \
    /data/uav/results /data/uav/logs
sudo install -d -m 0700 -o "$(id -u)" -g "$(id -g)" "$secret_root"
sudo install -d -m 0700 -o "$(id -u)" -g "$(id -g)" /data/uav/config/iber-be-v1

if [[ ! -e "$git_http_env" ]]; then
    umask 077
    printf '%s\n' \
        'export GIT_CONFIG_COUNT=1' \
        'export GIT_CONFIG_KEY_0=http.version' \
        'export GIT_CONFIG_VALUE_0=HTTP/1.1' \
        > "${git_http_env}.tmp"
    chmod 600 "${git_http_env}.tmp"
    mv "${git_http_env}.tmp" "$git_http_env"
fi
git_http_mode="$(stat -c %a "$git_http_env")"
if [[ "$git_http_mode" != "600" ]]; then
    printf 'Git HTTP environment must have mode 600, got %s\n' "$git_http_mode" >&2
    exit 5
fi
expected_git_http_env="$(printf '%s\n' \
    'export GIT_CONFIG_COUNT=1' \
    'export GIT_CONFIG_KEY_0=http.version' \
    'export GIT_CONFIG_VALUE_0=HTTP/1.1')"
actual_git_http_env="$(sed -n '1,3p' "$git_http_env")"
if [[ "$actual_git_http_env" != "$expected_git_http_env" ]]; then
    printf 'Git HTTP environment drift: %s\n' "$git_http_env" >&2
    exit 5
fi

if [[ -e "$secret_path" ]]; then
    chmod 600 "$secret_path"
    secret_mode="$(stat -c %a "$secret_path")"
    if [[ "$secret_mode" != "600" ]]; then
        printf 'GitHub token must have mode 600, got %s\n' "$secret_mode" >&2
        exit 6
    fi
fi
if [[ -e "$publication_config" ]]; then
    chmod 600 "$publication_config"
    publication_config_mode="$(stat -c %a "$publication_config")"
    if [[ "$publication_config_mode" != "600" ]]; then
        printf 'publication config must have mode 600, got %s\n' \
            "$publication_config_mode" >&2
        exit 7
    fi
fi

if [[ -d "$source_root/.git" ]]; then
    checked_out="$(git -C "$source_root" rev-parse HEAD)"
    if [[ "$checked_out" != "$source_commit" ]]; then
        printf 'immutable source checkout mismatch: expected=%s actual=%s\n' \
            "$source_commit" "$checked_out" >&2
        exit 8
    fi
elif [[ -e "$source_root" ]]; then
    printf 'immutable source path exists but is not a checkout: %s\n' "$source_root" >&2
    exit 9
else
    git -c http.version=HTTP/1.1 clone --no-checkout "$source_remote" "$source_root"
    git -c http.version=HTTP/1.1 -C "$source_root" fetch --no-tags origin "$source_commit"
    git -c http.version=HTTP/1.1 -C "$source_root" checkout --detach "$source_commit"
    checked_out="$(git -C "$source_root" rev-parse HEAD)"
    if [[ "$checked_out" != "$source_commit" ]]; then
        printf 'source verification failed: expected=%s actual=%s\n' \
            "$source_commit" "$checked_out" >&2
        exit 10
    fi
fi
source_status="$(git -C "$source_root" status --porcelain --untracked-files=all)"
if [[ -n "$source_status" ]]; then
    printf 'immutable source checkout is dirty: %s\n' "$source_root" >&2
    exit 11
fi
chmod -R a-w "$source_root"

bootstrap_required=true
if [[ -f "$marker_path" ]]; then
    bootstrap_required=false
fi
if [[ ! -x "${venv_root}/bin/python" ]]; then
    bootstrap_required=true
    python3.10 -m venv "$venv_root"
fi
export YOLO_CONFIG_DIR="/data/uav/config/Ultralytics"
if [[ "$bootstrap_required" == true ]]; then
    "${venv_root}/bin/python" -m pip install --upgrade pip setuptools wheel \
        --index-url "$mirror_index"

    wheelhouse="${wheelhouse_root}/${requirements_sha}"
    if [[ ! -d "$wheelhouse" ]]; then
        bash "${source_root}/deploy/iber/build_wheelhouse.sh"
    fi
    "${venv_root}/bin/python" -m pip install --no-index \
        --find-links "$wheelhouse" "${pinned_requirements[@]}"
fi

PYTHONDONTWRITEBYTECODE=1 "${venv_root}/bin/python" - <<'PY'
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
if [[ "$bootstrap_required" == true ]]; then
    printf '%s\n' "$source_commit" > "${marker_path}.tmp"
    mv "${marker_path}.tmp" "$marker_path"
fi
printf 'immutable source checkout: %s\n' "$source_root"
printf 'reserved cache root: %s\n' "$cache_root"
printf 'reserved run root: %s\n' "$run_root"
printf 'reserved results root: %s\n' "$results_root"
printf 'reserved pipeline log: %s\n' "$log_path"

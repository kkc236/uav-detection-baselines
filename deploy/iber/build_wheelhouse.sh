#!/usr/bin/env bash
set -euo pipefail

# This script runs in the foreground; a remote launcher may wrap it with nohup.
wheelhouse_root="/data/uav/staging/iber-be-v1-wheelhouse"
marker_root="/data/uav/deploy/iber-be-v1/markers"
mirror_index="https://mirrors.aliyun.com/pypi/simple"
pytorch_index="https://download.pytorch.org/whl/cu121"
python_command="python3.10"
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
    "ultralytics-thop==2.0.18"
)

if [[ -x /data/uav/venvs/iber-be-v1/bin/python ]]; then
    python_command="/data/uav/venvs/iber-be-v1/bin/python"
fi
requirements_sha="$(printf '%s\n' "${pinned_requirements[@]}" | sha256sum | awk '{print $1}')"
wheelhouse="${wheelhouse_root}/${requirements_sha}"
marker_path="${marker_root}/wheelhouse-${requirements_sha}.complete"
mkdir -p "$wheelhouse" "$marker_root"
if [[ -f "$marker_path" ]]; then
    printf 'IBER-BE wheelhouse already complete for %s\n' "$requirements_sha"
    exit 0
fi

download_args=(
    -m pip download --only-binary=:all:
    --dest "$wheelhouse"
    "${pinned_requirements[@]}"
)
if ! "$python_command" "${download_args[@]}" \
    --index-url "$mirror_index" --extra-index-url "$pytorch_index"; then
    "$python_command" "${download_args[@]}" \
        --index-url "$pytorch_index" --extra-index-url "https://pypi.org/simple"
fi

find "$wheelhouse" -maxdepth 1 -type f -printf '%f\n' \
    | LC_ALL=C sort > "${wheelhouse}/files-${requirements_sha}.txt"
find "$wheelhouse" -maxdepth 1 -type f ! -name 'sha256-*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum \
    > "${wheelhouse}/sha256-${requirements_sha}.txt"
printf '%s\n' "$requirements_sha" > "${marker_path}.tmp"
mv "${marker_path}.tmp" "$marker_path"

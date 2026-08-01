#!/usr/bin/env bash
set -euo pipefail

# This script runs in the foreground; a remote launcher may wrap it with nohup.
source_root="${1:-/data/uav/source/uav-detection-baselines}"
lock_path="${source_root}/requirements-itber.lock"
wheelhouse_root="/data/uav/staging/itber-v1.1-wheelhouse"
marker_root="/data/uav/deploy/markers"
mirror_index="https://mirrors.aliyun.com/pypi/simple"
pytorch_index="https://download.pytorch.org/whl/cu121"

if [[ ! -f "$lock_path" ]]; then
    printf 'missing lock file: %s\n' "$lock_path" >&2
    exit 2
fi
lock_sha="$(sha256sum "$lock_path" | awk '{print $1}')"
marker_path="${marker_root}/wheelhouse-${lock_sha}.complete"
mkdir -p "$wheelhouse_root" "$marker_root"
if [[ -f "$marker_path" ]]; then
    printf 'wheelhouse already complete for lock %s\n' "$lock_sha"
    exit 0
fi

python_command="python3.10"
if [[ -x /data/uav/venvs/itber-v1.1/bin/python ]]; then
    python_command="/data/uav/venvs/itber-v1.1/bin/python"
fi

download_args=(
    -m pip download --no-deps --only-binary=:all:
    --dest "$wheelhouse_root"
    --requirement "$lock_path"
    --extra-index-url "$pytorch_index"
)
if ! "$python_command" "${download_args[@]}" --index-url "$mirror_index"; then
    "$python_command" "${download_args[@]}" --index-url "https://pypi.org/simple"
fi

find "$wheelhouse_root" -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort > "${wheelhouse_root}/files-${lock_sha}.txt"
sha256sum "$wheelhouse_root"/* | LC_ALL=C sort -k2 > "${wheelhouse_root}/sha256-${lock_sha}.txt"
printf '%s\n' "$lock_sha" > "${marker_path}.tmp"
mv "${marker_path}.tmp" "$marker_path"

#!/usr/bin/env bash
set -euo pipefail

# Read-only verification for the baseline-aligned IBER-BE v1.0 execution contract.
expected_gpu="NVIDIA GeForce RTX 4090"
expected_gpu_memory_mib=24564
baseline_reference_driver="550.142"
expected_driver="550.142"
expected_python="Python 3.10.12"
expected_torch="torch==2.5.1+cu121"
expected_torchvision="torchvision==0.20.1+cu121"
expected_ultralytics="ultralytics==8.4.90"
expected_cuda="CUDA 12.1"
venv_python="/data/uav/venvs/iber-be-v1/bin/python"
minimum_disk_kib=$((70 * 1024 * 1024))
minimum_memory_kib=$((31 * 1024 * 1024))

os_id="$(. /etc/os-release; printf '%s' "${ID:-unknown}")"
os_version="$(. /etc/os-release; printf '%s' "${VERSION_ID:-unknown}")"
architecture="$(uname -m)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | xargs)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | xargs)"
gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p' | xargs)"
available_kib="$(df -Pk /data | awk 'NR==2 {print $4}')"
memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
system_python="$(python3 --version 2>&1)"
git_version="$(git --version 2>&1)"

runtime_json='{}'
if [[ -x "$venv_python" ]]; then
    runtime_json="$(PYTHONDONTWRITEBYTECODE=1 "$venv_python" - <<'PY'
import json
import platform

import torch
import torchvision
import ultralytics

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "ultralytics": ultralytics.__version__,
    "cuda": torch.version.cuda,
}, sort_keys=True, separators=(",", ":")))
PY
)"
fi

github_reachable=false
mirror_reachable=false
pytorch_reachable=false
if curl -fsSI --connect-timeout 3 --max-time 5 -o /dev/null https://api.github.com; then
    github_reachable=true
fi
if curl -fsSI --connect-timeout 3 --max-time 5 -o /dev/null https://mirrors.aliyun.com/pypi/simple; then
    mirror_reachable=true
fi
if curl -fsSI --connect-timeout 3 --max-time 5 -o /dev/null https://download.pytorch.org/whl/cu121; then
    pytorch_reachable=true
fi

python3 - \
    "$os_id" "$os_version" "$architecture" "$gpu" "$driver" \
    "$gpu_memory_mib" "$available_kib" "$memory_kib" "$system_python" "$git_version" \
    "$github_reachable" "$mirror_reachable" "$pytorch_reachable" "$runtime_json" \
    "$expected_gpu" "$expected_driver" "$baseline_reference_driver" \
    "$expected_gpu_memory_mib" "$minimum_disk_kib" "$minimum_memory_kib" <<'PY'
import json
import sys

(
    os_id,
    os_version,
    architecture,
    gpu,
    driver,
    gpu_memory_mib,
    available_kib,
    memory_kib,
    system_python,
    git_version,
    github_reachable,
    mirror_reachable,
    pytorch_reachable,
    runtime_json,
    expected_gpu,
    expected_driver,
    baseline_reference_driver,
    expected_gpu_memory_mib,
    minimum_disk_kib,
    minimum_memory_kib,
) = sys.argv[1:]
runtime = json.loads(runtime_json)
actual = {
    "os_id": os_id,
    "os_version": os_version,
    "architecture": architecture,
    "gpu": gpu,
    "driver": driver,
    "reported_memory_mib": int(gpu_memory_mib),
    "available_kib": int(available_kib),
    "memory_kib": int(memory_kib),
    "system_python": system_python,
    "git": git_version,
    "runtime": runtime,
    "github_reachable": github_reachable == "true",
    "mirror_reachable": mirror_reachable == "true",
    "pytorch_reachable": pytorch_reachable == "true",
}
expected = {
    "os_id": "ubuntu",
    "os_version": "22.04",
    "architecture": "x86_64",
    "gpu": expected_gpu,
    "driver": expected_driver,
    "reported_memory_mib": int(expected_gpu_memory_mib),
    "baseline_reference_driver": baseline_reference_driver,
    "minimum_disk_kib": int(minimum_disk_kib),
    "minimum_memory_kib": int(minimum_memory_kib),
    "runtime": {
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "ultralytics": "8.4.90",
        "cuda": "12.1",
    },
}
violations = {}
for key in (
    "os_id",
    "os_version",
    "architecture",
    "gpu",
    "driver",
    "reported_memory_mib",
    "runtime",
):
    if actual[key] != expected[key]:
        violations[key] = {"expected": expected[key], "actual": actual[key]}
if actual["available_kib"] < expected["minimum_disk_kib"]:
    violations["available_kib"] = {
        "expected_minimum": expected["minimum_disk_kib"],
        "actual": actual["available_kib"],
    }
if actual["memory_kib"] < expected["minimum_memory_kib"]:
    violations["memory_kib"] = {
        "expected_minimum": expected["minimum_memory_kib"],
        "actual": actual["memory_kib"],
    }
status = "passed_with_runtime_amendment" if not violations else "engineering_invalid"
print(json.dumps(
    {"status": status, "actual": actual, "expected": expected, "violations": violations},
    sort_keys=True,
    separators=(",", ":"),
))
raise SystemExit(0 if not violations else 1)
PY

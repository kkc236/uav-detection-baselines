#!/usr/bin/env bash
set -euo pipefail

# Read-only preflight for the frozen I-TBER v1.1 host contract.
expected_gpu="NVIDIA GeForce RTX 4090"
expected_driver="550.142"
minimum_disk_kib=$((80 * 1024 * 1024))
minimum_memory_kib=$((32 * 1024 * 1024))

os_id="$(. /etc/os-release; printf '%s' "${ID:-unknown}")"
os_version="$(. /etc/os-release; printf '%s' "${VERSION_ID:-unknown}")"
architecture="$(uname -m)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | xargs)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | xargs)"
available_kib="$(df -Pk /data | awk 'NR==2 {print $4}')"
memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
python_version="$(python3 --version 2>&1)"
git_version="$(git --version 2>&1)"

github_reachable=false
pypi_reachable=false
if curl -fsSI --connect-timeout 8 --max-time 20 -o /dev/null https://github.com; then
    github_reachable=true
fi
if curl -fsSI --connect-timeout 8 --max-time 20 -o /dev/null https://pypi.org; then
    pypi_reachable=true
fi

python3 - \
    "$os_id" "$os_version" "$architecture" "$gpu" "$driver" \
    "$available_kib" "$memory_kib" "$python_version" "$git_version" \
    "$github_reachable" "$pypi_reachable" "$expected_gpu" "$expected_driver" \
    "$minimum_disk_kib" "$minimum_memory_kib" <<'PY'
import json
import sys

(
    os_id,
    os_version,
    architecture,
    gpu,
    driver,
    available_kib,
    memory_kib,
    python_version,
    git_version,
    github_reachable,
    pypi_reachable,
    expected_gpu,
    expected_driver,
    minimum_disk_kib,
    minimum_memory_kib,
) = sys.argv[1:]
actual = {
    "os_id": os_id,
    "os_version": os_version,
    "architecture": architecture,
    "gpu": gpu,
    "driver": driver,
    "available_kib": int(available_kib),
    "memory_kib": int(memory_kib),
    "python": python_version,
    "git": git_version,
    "github_reachable": github_reachable == "true",
    "pypi_reachable": pypi_reachable == "true",
}
expected = {
    "os_id": "ubuntu",
    "os_version": "22.04",
    "architecture": "x86_64",
    "gpu": expected_gpu,
    "driver": expected_driver,
    "minimum_disk_kib": int(minimum_disk_kib),
    "minimum_memory_kib": int(minimum_memory_kib),
}
violations = {}
for key in ("os_id", "os_version", "architecture", "gpu", "driver"):
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
for key in ("github_reachable", "pypi_reachable"):
    if not actual[key]:
        violations[key] = {"expected": True, "actual": False}
print(json.dumps({"status": "passed" if not violations else "invalid", "actual": actual, "violations": violations}, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if not violations else 1)
PY

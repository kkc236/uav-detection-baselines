#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

root="/root/data/uav"
repo="$root/sqda-sgc"
venv="$root/venv"
project="$root/runs/sqda-geometry-gate"
baseline="$root/checkpoints/matched-baseline-best-epoch-0100.pt"
adapter="$root/runs/sqda-sgc/sqda-sgc-g2-seed0-10ep/weights/best.pt"
data="$root/protocols/tsgr-p2-e1/source-VisDrone-full.yaml"
images="$root/datasets/VisDrone/images/val"
labels="$root/datasets/VisDrone/labels/val"
diagnosis_dir="$project/geometry-branch-diagnosis"
diagnosis="$diagnosis_dir/geometry-branch-diagnosis.json"
admission="$diagnosis_dir/g1-admission.json"
log="$project/geometry-branch-diagnosis.log"
branch="codex/sqda-sgc"

mkdir -p "$project"
if pgrep -af "diagnose_sqda_geometry_branches.py" >/dev/null; then
  echo "a geometry-branch diagnosis is already running" >&2
  exit 3
fi
for required in "$baseline" "$adapter" "$data"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 4
  fi
done
for required in "$images" "$labels"; do
  if [[ ! -d "$required" ]]; then
    echo "missing required directory: $required" >&2
    exit 5
  fi
done

cd "$repo"
git fetch origin "$branch"
git switch "$branch"
git pull --ff-only origin "$branch"

(
  set -euo pipefail
  "$venv/bin/python" -u scripts/verify_sqda_sgc_g0.py \
    --checkpoint "$baseline" \
    --data "$data" \
    --device 0 \
    --output "$project/g0-equivalence.json"
  "$venv/bin/python" -u scripts/diagnose_sqda_geometry_branches.py \
    --checkpoint "$baseline" \
    --adapter-checkpoint "$adapter" \
    --data "$data" \
    --images "$images" \
    --labels "$labels" \
    --output "$diagnosis_dir" \
    --device 0 \
    --workers 8
  "$venv/bin/python" -u scripts/decide_sqda_geometry_admission.py \
    --diagnosis "$diagnosis" \
    --output "$admission"
) > "$log" 2>&1 &
pid=$!

printf 'DIAGNOSIS_PID=%s\nDIAGNOSIS_DIR=%s\nLOG=%s\n' "$pid" "$diagnosis_dir" "$log"

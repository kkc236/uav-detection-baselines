#!/bin/bash
set +e

RUN=final-saded-fresh-eval-f9d01f65
REPO=/home/ubuntu/repo-saded-fresh-postprocess-f9d01f65
PROTOCOL=/home/ubuntu/saded-fresh-eval-protocols/${RUN}/protocol_manifest.json
LOG_ROOT=/home/ubuntu/saded-fresh-eval-logs/${RUN}
TMP_ROOT=/home/ubuntu/saded-fresh-eval-tmp/${RUN}
PYTHON=/mnt/uav/venv/bin/python

mkdir -p "$LOG_ROOT" "$TMP_ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TMPDIR="$TMP_ROOT"

atomic_text() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" > "${path}.tmp"
  mv "${path}.tmp" "$path"
}

finish_invalid() {
  local stage="$1"
  local code="$2"
  atomic_text "$LOG_ROOT/failed_stage" "$stage"
  atomic_text "$LOG_ROOT/exit_code" "$code"
  atomic_text "$LOG_ROOT/status" "PIPELINE_INVALID"
  exit "$code"
}

atomic_text "$LOG_ROOT/driver.pid" "$$"
atomic_text "$LOG_ROOT/status" "CACHE_RUNNING"
cd "$REPO" || finish_invalid "checkout" 90

"$PYTHON" scripts/cache_saded_stock_endpoint.py \
  --evaluation-protocol "$PROTOCOL" \
  --device 0 \
  > "$LOG_ROOT/cache.log" 2>&1
cache_rc=$?
atomic_text "$LOG_ROOT/cache_exit_code" "$cache_rc"
if [ "$cache_rc" -ne 0 ]; then
  finish_invalid "cache" "$cache_rc"
fi

atomic_text "$LOG_ROOT/status" "ROUTE_RUNNING"
"$PYTHON" scripts/route_saded_stock_single.py \
  --evaluation-protocol "$PROTOCOL" \
  > "$LOG_ROOT/route.log" 2>&1
route_rc=$?
atomic_text "$LOG_ROOT/route_exit_code" "$route_rc"
if [ "$route_rc" -ne 0 ]; then
  finish_invalid "route" "$route_rc"
fi

atomic_text "$LOG_ROOT/status" "ROUTE_PREFLIGHT_RUNNING"
"$PYTHON" - "$PROTOCOL" > "$LOG_ROOT/route_preflight.log" 2>&1 <<'PY'
from pathlib import Path
import sys

from scripts.evaluate_saded_stock_single import _verify_route
from src.saded_stock_evaluation_protocol import validate_evaluation_protocol

protocol_path = Path(sys.argv[1]).resolve()
protocol = validate_evaluation_protocol(
    protocol_path,
    repo_root=Path.cwd(),
    verify_images=False,
)
_verify_route(protocol, protocol_path)
print("ROUTE_PREFLIGHT_PASS")
PY
preflight_rc=$?
atomic_text "$LOG_ROOT/route_preflight_exit_code" "$preflight_rc"
if [ "$preflight_rc" -ne 0 ]; then
  finish_invalid "route_preflight" "$preflight_rc"
fi

atomic_text "$LOG_ROOT/status" "SEALED_EVALUATION_RUNNING"
"$PYTHON" scripts/evaluate_saded_stock_single.py \
  --evaluation-protocol "$PROTOCOL" \
  > "$LOG_ROOT/evaluation.log" 2>&1
evaluation_rc=$?
atomic_text "$LOG_ROOT/evaluation_exit_code" "$evaluation_rc"
if [ "$evaluation_rc" -ne 0 ]; then
  finish_invalid "evaluation" "$evaluation_rc"
fi

atomic_text "$LOG_ROOT/status" "ADJUDICATION_RUNNING"
"$PYTHON" scripts/adjudicate_saded_stock_fresh.py \
  --evaluation-protocol "$PROTOCOL" \
  > "$LOG_ROOT/adjudication.log" 2>&1
adjudication_rc=$?
atomic_text "$LOG_ROOT/adjudication_exit_code" "$adjudication_rc"

decision=$(
  "$PYTHON" - "$PROTOCOL" <<'PY'
import json
from pathlib import Path
import sys

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path = Path(protocol["outputs"]["adjudication"]) / "adjudication.json"
if path.is_file():
    print(json.loads(path.read_text(encoding="utf-8")).get("decision", ""))
PY
)
atomic_text "$LOG_ROOT/decision" "$decision"

if [ "$adjudication_rc" -eq 0 ] && \
   [ "$decision" = "SADED_SINGLE_SEED_GO" ]; then
  atomic_text "$LOG_ROOT/exit_code" "0"
  atomic_text "$LOG_ROOT/status" "PIPELINE_GO"
  exit 0
fi
if [ "$adjudication_rc" -eq 1 ] && \
   [ "$decision" = "SADED_SINGLE_SEED_STOP" ]; then
  atomic_text "$LOG_ROOT/exit_code" "1"
  atomic_text "$LOG_ROOT/status" "PIPELINE_STOP"
  exit 1
fi

finish_invalid "adjudication" "${adjudication_rc:-2}"

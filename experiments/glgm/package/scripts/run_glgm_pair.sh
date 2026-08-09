#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${REPO_DIR:?Set REPO_DIR to the Ultralytics source checkout}"
: "${DATA_YAML:?Set DATA_YAML to the VisDrone data.yaml}"
: "${WORK_ROOT:=$HOME/glgm/work/glgm-experiment}"
: "${PYTHON:=$WORK_ROOT/venv/bin/python}"
: "${DEVICE:=0}"
: "${EPOCHS:=100}"
: "${IMGSZ:=640}"
: "${BATCH:=4}"
: "${WORKERS:=4}"
: "${SEED:=0}"
: "${FRACTION:=1.0}"
: "${BASE_WEIGHTS:=}"
: "${PRIVATE_SEED:=$((10000 + SEED))}"
: "${SAVE_PERIOD:=5}"
: "${ARM_ORDER:=control glgm}"
: "${PARALLEL:=0}"
: "${CONTROL_DEVICE:=$DEVICE}"
: "${GLGM_DEVICE:=$DEVICE}"
: "${EVAL_DEVICE:=$DEVICE}"
: "${STRICT_PAIR:=1}"
: "${GLGM_CONTROL_CONFIG:=$PACKAGE_ROOT/configs/rtdetr-x-glgm-control.yaml}"
: "${GLGM_METHOD_CONFIG:=$PACKAGE_ROOT/configs/rtdetr-x-glgm-only.yaml}"
: "${GLGM_METHOD_MODULE:=GLGM}"
: "${GLGM_PAIRED_HEAD_INDEX:=2}"
: "${GLGM_PAIRED_MODEL_INDEX:=16}"
: "${GLGM_PAIRED_SPATIAL_SIZE:=20}"
: "${GLGM_EXPERIMENT_VARIANT:=glgm-v1-p5}"

if [[ "$PARALLEL" != 0 && "$PARALLEL" != 1 ]]; then
  echo "PARALLEL must be 0 or 1" >&2
  exit 2
fi
if [[ "$STRICT_PAIR" != 0 && "$STRICT_PAIR" != 1 ]]; then
  echo "STRICT_PAIR must be 0 or 1" >&2
  exit 2
fi
if [[ "$STRICT_PAIR" == 1 ]]; then
  if [[ "$PARALLEL" != 0 ]]; then
    echo "Strict paired mode requires sequential training on one GPU; set STRICT_PAIR=0 for exploratory parallel runs" >&2
    exit 2
  fi
  if [[ "$CONTROL_DEVICE" != "$GLGM_DEVICE" || ! "$CONTROL_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "Strict paired mode requires the same single numeric GPU for both arms" >&2
    exit 2
  fi
elif [[ "$PARALLEL" == 1 && "$CONTROL_DEVICE" == "$GLGM_DEVICE" ]]; then
  echo "Exploratory parallel training requires distinct GPUs" >&2
  exit 2
fi

read -r -a ARMS <<< "$ARM_ORDER"
if [[ " ${ARMS[*]} " != *" control "* || " ${ARMS[*]} " != *" glgm "* || "${#ARMS[@]}" -ne 2 ]]; then
  echo "ARM_ORDER must contain control and glgm exactly once" >&2
  exit 2
fi

ARTIFACT_DIR="$WORK_ROOT/artifacts"
RUNS_DIR="$WORK_ROOT/runs"
LOG_DIR="$WORK_ROOT/logs"
if [[ -d "$WORK_ROOT" && -n "$(find "$WORK_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty WORK_ROOT: $WORK_ROOT" >&2
  exit 2
fi
mkdir -p "$ARTIFACT_DIR" "$RUNS_DIR" "$LOG_DIR"
STATUS_FILE="$WORK_ROOT/supervisor-status.tsv"
if [[ "$STRICT_PAIR" == 1 ]]; then
  COMPLETION_PATH="$WORK_ROOT/COMPLETED"
else
  COMPLETION_PATH="$WORK_ROOT/EXPLORATORY_COMPLETED"
fi
COMPLETION_TMP="$WORK_ROOT/.completion.tmp"
FINALIZE_READY=0
printf 'state\tstarted\tutc\t%s\n' "$(date -u +%FT%TZ)" > "$STATUS_FILE"
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -eq 0 && "$FINALIZE_READY" -ne 1 ]]; then
    code=1
  fi
  if ! printf 'state\texit\tcode\t%s\tutc\t%s\n' "$code" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"; then
    code=1
  fi
  if [[ "$code" -eq 0 ]]; then
    mv -f "$COMPLETION_TMP" "$COMPLETION_PATH" || code=1
  fi
  exit "$code"
}
trap on_exit EXIT

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export YOLO_AUTOINSTALL=false
export GLGM_CONTROL_CONFIG GLGM_METHOD_CONFIG GLGM_METHOD_MODULE
export GLGM_PAIRED_HEAD_INDEX GLGM_PAIRED_MODEL_INDEX GLGM_PAIRED_SPATIAL_SIZE
export GLGM_EXPERIMENT_VARIANT

"$PYTHON" "$SCRIPT_DIR/audit_visdrone.py" \
  --data "$DATA_YAML" \
  --output "$ARTIFACT_DIR/visdrone-audit.json" \
  --expected-train 6471 \
  --expected-val 548 \
  --hash-content \
  2>&1 | tee "$LOG_DIR/visdrone-audit.log"

PREFLIGHT_ARGS=(
  --repo-dir "$REPO_DIR"
  preflight
  --artifact-dir "$ARTIFACT_DIR"
  --public-seed "$SEED"
  --private-seed "$PRIVATE_SEED"
  --device "$EVAL_DEVICE"
  --full-forward
)
if [[ -n "$BASE_WEIGHTS" ]]; then
  PREFLIGHT_ARGS+=(--base-weights "$BASE_WEIGHTS")
fi
"$PYTHON" "$SCRIPT_DIR/glgm_experiment.py" "${PREFLIGHT_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/preflight.log"

audit_arm() {
  local arm="$1"
  "$PYTHON" "$SCRIPT_DIR/audit_visdrone.py" \
    --data "$DATA_YAML" \
    --output "$ARTIFACT_DIR/visdrone-audit-${arm}-pretrain.json" \
    --reference "$ARTIFACT_DIR/visdrone-audit.json" \
    --expected-train 6471 \
    --expected-val 548 \
    --hash-content \
    2>&1 | tee "$LOG_DIR/visdrone-audit-${arm}-pretrain.log"
}

arm_device() {
  if [[ "$1" == control ]]; then
    printf '%s\n' "$CONTROL_DEVICE"
  else
    printf '%s\n' "$GLGM_DEVICE"
  fi
}

train_arm() {
  local arm="$1"
  local run_name="${arm}-seed${SEED}-e${EPOCHS}"
  local arm_gpu
  arm_gpu="$(arm_device "$arm")"
  "$PYTHON" "$SCRIPT_DIR/glgm_experiment.py" \
    --repo-dir "$REPO_DIR" \
    train \
    --arm "$arm" \
    --artifact-dir "$ARTIFACT_DIR" \
    --data "$DATA_YAML" \
    --runs-dir "$RUNS_DIR" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$arm_gpu" \
    --seed "$SEED" \
    --fraction "$FRACTION" \
    --save-period "$SAVE_PERIOD" \
    --name "$run_name" \
    2>&1 | tee "$LOG_DIR/${run_name}.log"
}

if [[ "$PARALLEL" == 1 ]]; then
  for ARM in "${ARMS[@]}"; do
    audit_arm "$ARM"
  done
  train_arm control &
  CONTROL_PID=$!
  train_arm glgm &
  GLGM_PID=$!
  printf 'state\ttraining\tcontrol_pid\t%s\tglgm_pid\t%s\n' "$CONTROL_PID" "$GLGM_PID" >> "$STATUS_FILE"
  set +e
  wait "$CONTROL_PID"
  CONTROL_STATUS=$?
  wait "$GLGM_PID"
  GLGM_STATUS=$?
  set -e
  if [[ "$CONTROL_STATUS" -ne 0 || "$GLGM_STATUS" -ne 0 ]]; then
    echo "paired training failed: control=$CONTROL_STATUS glgm=$GLGM_STATUS" >&2
    exit 1
  fi
else
  for ARM in "${ARMS[@]}"; do
    audit_arm "$ARM"
    train_arm "$ARM"
  done
fi

"$PYTHON" "$SCRIPT_DIR/audit_visdrone.py" \
  --data "$DATA_YAML" \
  --output "$ARTIFACT_DIR/visdrone-audit-pre-eval.json" \
  --reference "$ARTIFACT_DIR/visdrone-audit.json" \
  --expected-train 6471 \
  --expected-val 548 \
  --hash-content \
  2>&1 | tee "$LOG_DIR/visdrone-audit-pre-eval.log"

for ARM in "${ARMS[@]}"; do
  RUN_NAME="${ARM}-seed${SEED}-e${EPOCHS}"
  for KIND in last best; do
    WEIGHTS="$RUNS_DIR/$RUN_NAME/weights/$KIND.pt"
    "$PYTHON" "$SCRIPT_DIR/glgm_experiment.py" \
      --repo-dir "$REPO_DIR" \
      eval \
      --arm "$ARM" \
      --manifest "$ARTIFACT_DIR/paired_preflight_manifest.json" \
      --train-receipt "$ARTIFACT_DIR/${ARM}-train-receipt.json" \
      --checkpoint-kind "$KIND" \
      --weights "$WEIGHTS" \
      --data "$DATA_YAML" \
      --output "$ARTIFACT_DIR/${ARM}-${KIND}-metrics.json" \
      --project "$WORK_ROOT/eval" \
      --split val \
      --imgsz "$IMGSZ" \
      --batch "$BATCH" \
      --workers "$WORKERS" \
      --device "$EVAL_DEVICE" \
      2>&1 | tee "$LOG_DIR/${ARM}-${KIND}-eval.log"
  done
done

for ARM in "${ARMS[@]}"; do
  RUN_NAME="${ARM}-seed${SEED}-e${EPOCHS}"
  "$PYTHON" "$SCRIPT_DIR/glgm_experiment.py" \
    --repo-dir "$REPO_DIR" \
    benchmark \
    --arm "$ARM" \
    --manifest "$ARTIFACT_DIR/paired_preflight_manifest.json" \
    --train-receipt "$ARTIFACT_DIR/${ARM}-train-receipt.json" \
    --checkpoint-kind best \
    --weights "$RUNS_DIR/$RUN_NAME/weights/best.pt" \
    --output "$ARTIFACT_DIR/${ARM}-best-benchmark.json" \
    --imgsz "$IMGSZ" \
    --device "$EVAL_DEVICE" \
    --warmup 50 \
    --iterations 200 \
    --half \
    2>&1 | tee "$LOG_DIR/${ARM}-best-benchmark.log"
done

COMPARE_MODE_ARGS=()
if [[ "$STRICT_PAIR" == 0 ]]; then
  COMPARE_MODE_ARGS+=(--exploratory)
fi
for KIND in last best; do
  "$PYTHON" "$SCRIPT_DIR/glgm_experiment.py" \
    --repo-dir "$REPO_DIR" \
    compare \
    --control "$ARTIFACT_DIR/control-${KIND}-metrics.json" \
    --glgm "$ARTIFACT_DIR/glgm-${KIND}-metrics.json" \
    --output "$ARTIFACT_DIR/comparison-${KIND}.json" \
    "${COMPARE_MODE_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/comparison-${KIND}.log"
done

find "$ARTIFACT_DIR" -maxdepth 1 -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ARTIFACT_DIR/SHA256SUMS.txt"
sha256sum -c "$ARTIFACT_DIR/SHA256SUMS.txt" > "$LOG_DIR/artifact-sha256-check.log"
printf 'mode\t%s\nvariant\t%s\nutc\t%s\n' \
  "$([[ "$STRICT_PAIR" == 1 ]] && printf strict || printf exploratory)" \
  "$GLGM_EXPERIMENT_VARIANT" \
  "$(date -u +%FT%TZ)" > "$COMPLETION_TMP"
echo "GLGM paired experiment completed: $WORK_ROOT"
FINALIZE_READY=1

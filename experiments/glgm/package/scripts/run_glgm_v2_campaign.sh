#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${REPO_DIR:?Set REPO_DIR to the fresh GLGM-v2 Ultralytics checkout}"
: "${DATA_YAML:?Set DATA_YAML to the audited VisDrone data.yaml}"
: "${BASE_WEIGHTS:?Set BASE_WEIGHTS to the audited stock RT-DETR-X checkpoint}"
: "${CAMPAIGN_ROOT:=$HOME/glgm-v2/work/campaign-v1}"
: "${PYTHON:=$HOME/glgm/env/venv/bin/python}"
: "${BATCH:=4}"
: "${WORKERS:=4}"
: "${SAVE_PERIOD:=5}"

VARIANTS=(v2-lite-equal-p5 v3-lite-gated-p5 v4-lite-gated-p4 v5-lite-gated-p3)
STATUS_FILE="$CAMPAIGN_ROOT/campaign-status.tsv"
mkdir -p "$CAMPAIGN_ROOT"
exec 9>"$CAMPAIGN_ROOT/campaign.lock"
if ! flock -n 9; then
  echo "Another GLGM-v2 campaign supervisor owns $CAMPAIGN_ROOT" >&2
  exit 2
fi

status() {
  printf 'utc\t%s\tstage\t%s\tmessage\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$STATUS_FILE"
}

variant_contract() {
  case "$1" in
    v2-lite-equal-p5)
      CONTROL_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-control.yaml"
      METHOD_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-lite-equal-p5.yaml"
      HEAD_INDEX=2 MODEL_INDEX=16 SPATIAL_SIZE=20
      ;;
    v3-lite-gated-p5)
      CONTROL_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-control.yaml"
      METHOD_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-lite-gated-p5.yaml"
      HEAD_INDEX=2 MODEL_INDEX=16 SPATIAL_SIZE=20
      ;;
    v4-lite-gated-p4)
      CONTROL_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-control-p4.yaml"
      METHOD_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-lite-gated-p4.yaml"
      HEAD_INDEX=7 MODEL_INDEX=21 SPATIAL_SIZE=40
      ;;
    v5-lite-gated-p3)
      CONTROL_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-control-p3.yaml"
      METHOD_CONFIG="$PACKAGE_ROOT/configs/rtdetr-x-glgm-lite-gated-p3.yaml"
      HEAD_INDEX=12 MODEL_INDEX=26 SPATIAL_SIZE=80
      ;;
    *)
      echo "Unknown preregistered variant: $1" >&2
      return 2
      ;;
  esac
}

run_pair() {
  local stage="$1" variant="$2" epochs="$3" fraction="$4" gpu="$5" seed="$6" order="$7"
  local work_root="$CAMPAIGN_ROOT/$stage/${variant}-seed${seed}"
  if [[ -f "$work_root/COMPLETED" ]]; then
    status "$stage" "skip_completed:$variant:seed$seed"
    return 0
  fi
  if [[ -e "$work_root" ]]; then
    echo "Refusing to reuse incomplete authority: $work_root" >&2
    return 2
  fi
  variant_contract "$variant"
  status "$stage" "start:$variant:seed$seed:gpu$gpu"
  REPO_DIR="$REPO_DIR" \
  DATA_YAML="$DATA_YAML" \
  BASE_WEIGHTS="$BASE_WEIGHTS" \
  WORK_ROOT="$work_root" \
  PYTHON="$PYTHON" \
  EPOCHS="$epochs" \
  FRACTION="$fraction" \
  BATCH="$BATCH" \
  WORKERS="$WORKERS" \
  SAVE_PERIOD="$SAVE_PERIOD" \
  SEED="$seed" \
  DEVICE="$gpu" \
  CONTROL_DEVICE="$gpu" \
  GLGM_DEVICE="$gpu" \
  EVAL_DEVICE="$gpu" \
  STRICT_PAIR=1 \
  PARALLEL=0 \
  ARM_ORDER="$order" \
  GLGM_CONTROL_CONFIG="$CONTROL_CONFIG" \
  GLGM_METHOD_CONFIG="$METHOD_CONFIG" \
  GLGM_METHOD_MODULE=GLGMLite \
  GLGM_PAIRED_HEAD_INDEX="$HEAD_INDEX" \
  GLGM_PAIRED_MODEL_INDEX="$MODEL_INDEX" \
  GLGM_PAIRED_SPATIAL_SIZE="$SPATIAL_SIZE" \
  GLGM_EXPERIMENT_VARIANT="$variant" \
  bash "$SCRIPT_DIR/run_glgm_pair.sh"
  status "$stage" "complete:$variant:seed$seed:gpu$gpu"
}

run_wave() {
  local stage="$1" epochs="$2" fraction="$3" first="$4" second="$5"
  run_pair "$stage" "$first" "$epochs" "$fraction" 0 0 "control glgm" &
  local first_pid=$!
  run_pair "$stage" "$second" "$epochs" "$fraction" 1 0 "glgm control" &
  local second_pid=$!
  local first_status=0 second_status=0
  wait "$first_pid" || first_status=$?
  wait "$second_pid" || second_status=$?
  if [[ "$first_status" -ne 0 || "$second_status" -ne 0 ]]; then
    echo "Campaign wave failed: $first=$first_status $second=$second_status" >&2
    return 1
  fi
}

gate_stage() {
  local stage="$1"
  shift
  local variant root
  for variant in "$@"; do
    root="$CAMPAIGN_ROOT/$stage/${variant}-seed0"
    "$PYTHON" "$SCRIPT_DIR/evaluate_glgm_v2_gate.py" \
      --work-root "$root" \
      --stage "$stage" \
      --output "$root/artifacts/promotion-gate.json"
  done
}

select_candidates() {
  local stage="$1" maximum="$2" output="$3"
  shift 3
  local arguments=() variant
  for variant in "$@"; do
    arguments+=(--gate "$CAMPAIGN_ROOT/$stage/${variant}-seed0/artifacts/promotion-gate.json")
  done
  "$PYTHON" "$SCRIPT_DIR/select_glgm_v2_candidates.py" \
    "${arguments[@]}" --maximum "$maximum" --output "$output"
}

status source_install start
"$PYTHON" "$SCRIPT_DIR/install_glgm_v2_source.py" \
  --repo-dir "$REPO_DIR" \
  --receipt "$CAMPAIGN_ROOT/glgm-v2-source-install-receipt.json"
status source_install complete

run_wave smoke2 2 0.05 "${VARIANTS[0]}" "${VARIANTS[1]}"
run_wave smoke2 2 0.05 "${VARIANTS[2]}" "${VARIANTS[3]}"

run_wave screen10 10 1.0 "${VARIANTS[0]}" "${VARIANTS[1]}"
run_wave screen10 10 1.0 "${VARIANTS[2]}" "${VARIANTS[3]}"
gate_stage screen10 "${VARIANTS[@]}"
mapfile -t SCREEN30_VARIANTS < <(
  select_candidates screen10 2 "$CAMPAIGN_ROOT/screen10-selection.json" "${VARIANTS[@]}"
)
if [[ "${#SCREEN30_VARIANTS[@]}" -eq 0 ]]; then
  status stopped no_screen10_candidate_passed
  touch "$CAMPAIGN_ROOT/STOPPED_NO_SCREEN10_CANDIDATE"
  exit 3
fi

if [[ "${#SCREEN30_VARIANTS[@]}" -eq 2 ]]; then
  run_wave screen30 30 1.0 "${SCREEN30_VARIANTS[0]}" "${SCREEN30_VARIANTS[1]}"
else
  run_pair screen30 "${SCREEN30_VARIANTS[0]}" 30 1.0 0 0 "control glgm"
fi
gate_stage screen30 "${SCREEN30_VARIANTS[@]}"
mapfile -t FORMAL_VARIANT < <(
  select_candidates screen30 1 "$CAMPAIGN_ROOT/screen30-selection.json" "${SCREEN30_VARIANTS[@]}"
)
if [[ "${#FORMAL_VARIANT[@]}" -eq 0 ]]; then
  status stopped no_screen30_candidate_passed
  touch "$CAMPAIGN_ROOT/STOPPED_NO_SCREEN30_CANDIDATE"
  exit 4
fi
WINNER="${FORMAL_VARIANT[0]}"
status formal100 "winner:$WINNER"

run_pair formal100 "$WINNER" 100 1.0 0 0 "control glgm" &
seed0_pid=$!
run_pair formal100 "$WINNER" 100 1.0 1 1 "glgm control" &
seed1_pid=$!
seed0_status=0 seed1_status=0
wait "$seed0_pid" || seed0_status=$?
wait "$seed1_pid" || seed1_status=$?
if [[ "$seed0_status" -ne 0 || "$seed1_status" -ne 0 ]]; then
  echo "Formal100 first wave failed: seed0=$seed0_status seed1=$seed1_status" >&2
  exit 1
fi

run_pair formal100 "$WINNER" 100 1.0 0 2 "control glgm" &
seed2_pid=$!
run_pair formal100 "$WINNER" 100 1.0 1 3 "glgm control" &
seed3_pid=$!
seed2_status=0 seed3_status=0
wait "$seed2_pid" || seed2_status=$?
wait "$seed3_pid" || seed3_status=$?
if [[ "$seed2_status" -ne 0 || "$seed3_status" -ne 0 ]]; then
  echo "Formal100 second wave failed: seed2=$seed2_status seed3=$seed3_status" >&2
  exit 1
fi

printf 'winner\t%s\nutc\t%s\n' "$WINNER" "$(date -u +%FT%TZ)" > "$CAMPAIGN_ROOT/COMPLETED"
status completed "$WINNER"

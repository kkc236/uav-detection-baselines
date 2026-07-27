#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 REPO BASELINE DATA_YAML SOURCE_COMMIT OUTPUT_ROOT" >&2
  exit 2
fi

repo="$(realpath "$1")"
baseline="$(realpath "$2")"
data_yaml="$(realpath "$3")"
source_commit="$4"
output_root="$(realpath -m "$5")"
module_artifact="${output_root}/artifacts/calibrated-module.pt"
runs="${output_root}/runs"
logs="${output_root}/logs"
evaluation="${output_root}/evaluation/three-state.json"

mkdir -p "${output_root}/artifacts" "${runs}" "${logs}" \
  "${output_root}/evaluation"

status_file="${output_root}/PIPELINE_STATUS"
failure_file="${output_root}/PIPELINE_FAILED"
complete_file="${output_root}/PIPELINE_COMPLETE"

on_error() {
  code="$?"
  printf 'failed exit=%s time=%s\n' \
    "${code}" "$(date --iso-8601=seconds)" >"${failure_file}"
  printf 'failed\n' >"${status_file}"
  exit "${code}"
}
trap on_error ERR

cd "${repo}"
printf 'calibration\n' >"${status_file}"
python -m scripts.train_rtdetr_gcmv_warmstart \
  --stage calibration \
  --baseline "${baseline}" \
  --module-artifact "${module_artifact}" \
  --data "${data_yaml}" \
  --project "${runs}" \
  --name calibration \
  --source-commit "${source_commit}" \
  --device 0 \
  --seed 0 >"${logs}/calibration.log" 2>&1

printf 'control\n' >"${status_file}"
python -m scripts.train_rtdetr_gcmv_warmstart \
  --stage control \
  --baseline "${baseline}" \
  --module-artifact "${module_artifact}" \
  --data "${data_yaml}" \
  --project "${runs}" \
  --name control-finetune-10ep \
  --source-commit "${source_commit}" \
  --device 0 \
  --seed 0 >"${logs}/control.log" 2>&1

printf 'method\n' >"${status_file}"
python -m scripts.train_rtdetr_gcmv_warmstart \
  --stage method \
  --baseline "${baseline}" \
  --module-artifact "${module_artifact}" \
  --data "${data_yaml}" \
  --project "${runs}" \
  --name method-finetune-10ep \
  --source-commit "${source_commit}" \
  --device 0 \
  --seed 0 >"${logs}/method.log" 2>&1

control_checkpoint="${runs}/control-finetune-10ep/weights/last.pt"
method_checkpoint="${runs}/method-finetune-10ep/weights/last.pt"
test -s "${control_checkpoint}"
test -s "${method_checkpoint}"

printf 'evaluation\n' >"${status_file}"
python -m scripts.evaluate_gcmv_warmstart \
  --control-checkpoint "${control_checkpoint}" \
  --method-checkpoint "${method_checkpoint}" \
  --data "${data_yaml}" \
  --output "${evaluation}" \
  --batch 4 \
  --workers 4 \
  --device 0 >"${logs}/evaluation.log" 2>&1

sha256sum \
  "${baseline}" \
  "${module_artifact}" \
  "${control_checkpoint}" \
  "${method_checkpoint}" \
  "${evaluation}" >"${output_root}/checksums.sha256"

printf 'completed\n' >"${status_file}"
printf 'completed time=%s\n' "$(date --iso-8601=seconds)" >"${complete_file}"
trap - ERR

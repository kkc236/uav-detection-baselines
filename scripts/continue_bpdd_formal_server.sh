#!/usr/bin/env bash
set -euo pipefail

PYTHON=/data/uav/venvs/iber-be-v1/bin/python
SOURCE=/data/uav/source/uav-detection-baselines-848f00cb
PROTOCOL=/data/uav/protocols/bpdd-848f00cb/protocol.json
INITIAL=/data/uav/protocols/fdr-d97e1eb7/initial-state.pt
DATASET=/data/uav/datasets/VisDrone
FORMAL_ROOT=/data/uav/runs/bpdd-formal-848f00cb
LOG_ROOT=/data/uav/logs/bpdd-formal-848f00cb
PUBLICATION_ROOT=/data/uav/publication/bpdd-formal-848f00cb
PIPELINE_STATE=/data/uav/logs/bpdd-screen-848f00cb/pipeline-state.txt
FDR_RUN=${FORMAL_ROOT}/formal-seed0-fdr-bpdd-v1
BPDD_RUN=${FORMAL_ROOT}/formal-seed0-fdr_bpdd-bpdd-v1

mkdir -p "${LOG_ROOT}" "${PUBLICATION_ROOT}" "${FORMAL_ROOT}"
exec 9>"${LOG_ROOT}/formal-continuation.lock"
flock -n 9

state() {
  printf '%s\n' "$1" > "${PIPELINE_STATE}"
}

failed() {
  rc=$?
  if (( rc != 0 )); then
    state formal_engineering_failed
  fi
  exit "${rc}"
}
trap failed EXIT

fdr_pid=$(cat "${LOG_ROOT}/fdr.pid")
while kill -0 "${fdr_pid}" 2>/dev/null; do
  sleep 30
done

test -f "${FDR_RUN}/bpdd-run.json"
test -f "${FDR_RUN}/bpdd-epochs.jsonl"
test "$(wc -l < "${FDR_RUN}/bpdd-epochs.jsonl")" -eq 100
test -f "${FDR_RUN}/weights/epoch99.pt"
jq -e \
  '.run_identity.stage == "formal" and
   .run_identity.variant == "fdr" and
   .run_identity.seed == 0 and
   .source.git_commit == "848f00cb7a40907e3884885ecd5bbd474450758a" and
   .initial_state.sha256 == "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"' \
  "${FDR_RUN}/bpdd-run.json" >/dev/null

state formal_bpdd_starting
test ! -e "${BPDD_RUN}"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON}" "${SOURCE}/scripts/train_rtdetr_bpdd.py" \
  --variant fdr_bpdd \
  --stage formal \
  --protocol-manifest "${PROTOCOL}" \
  --initial-state "${INITIAL}" \
  --dataset-root "${DATASET}" \
  --output-root "${FORMAL_ROOT}" \
  --publication-queue "${PUBLICATION_ROOT}/bpdd-queue.jsonl" \
  > "${LOG_ROOT}/bpdd.log" 2>&1 &
bpdd_pid=$!
printf '%s\n' "${bpdd_pid}" > "${LOG_ROOT}/bpdd.pid"
state formal_bpdd_running
wait "${bpdd_pid}"

test -f "${BPDD_RUN}/bpdd-run.json"
test -f "${BPDD_RUN}/bpdd-epochs.jsonl"
test "$(wc -l < "${BPDD_RUN}/bpdd-epochs.jsonl")" -eq 100
test -f "${BPDD_RUN}/weights/epoch99.pt"
jq -e \
  '.run_identity.stage == "formal" and
   .run_identity.variant == "fdr_bpdd" and
   .run_identity.seed == 0 and
   .source.git_commit == "848f00cb7a40907e3884885ecd5bbd474450758a" and
   .initial_state.sha256 == "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"' \
  "${BPDD_RUN}/bpdd-run.json" >/dev/null

state formal_pair_training_complete
trap - EXIT

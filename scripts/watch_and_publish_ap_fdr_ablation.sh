#!/usr/bin/env bash
set -u

BASE_DIR=/home/ubuntu/ap-fdr-ablation
SOURCE_DIR="$BASE_DIR/publisher-source"
PYTHON=/data/uav/venvs/iber-be-v1/bin/python
TOKEN_FILE="$BASE_DIR/github_token"
LOG_FILE="$BASE_DIR/logs/auto-publication.log"
SUCCESS_MARKER="$BASE_DIR/publication-succeeded"
FAILURE_MARKER="$BASE_DIR/publication-failed"

mkdir -p "$BASE_DIR/logs"

if [[ -f "$SUCCESS_MARKER" ]]; then
  exit 0
fi

while [[ ! -f "$BASE_DIR/all.completed" ]]; do
  sleep 60
done

delays=(0 60 180 300 600)
for delay in "${delays[@]}"; do
  if (( delay > 0 )); then
    sleep "$delay"
  fi
  date -Is >>"$LOG_FILE"
  if "$PYTHON" "$SOURCE_DIR/scripts/publish_ap_fdr_ablation.py" \
      --base-dir "$BASE_DIR" \
      --token-file "$TOKEN_FILE" \
      --repository kkc236/icassp2027-fdr-bpdd-fia-material \
      --tag ap-fdr-internal-ablation-seed0-20260820 \
      --target-commitish main >>"$LOG_FILE" 2>&1; then
    touch "$SUCCESS_MARKER"
    rm -f "$FAILURE_MARKER" "$TOKEN_FILE"
    exit 0
  fi
done

touch "$FAILURE_MARKER"
exit 1

# Source authority: ebb349aeb2cf092d4880751e165e22614c3c9d8c

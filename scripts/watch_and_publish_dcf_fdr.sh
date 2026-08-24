#!/usr/bin/env bash
set -u -o pipefail

SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/data/uav/venvs/iber-be-v1/bin/python
PUBLISHER="$SOURCE_DIR/scripts/publish_dcf_fdr_results.py"
PUBLICATION_ROOT=/data/uav/publication/clean-dcf-fdr-formal100-seed0-20260824
CLEAN_ROOT=/data/uav/runs/dcf-fdr-ec4e2a46-clean
DCF_ROOT=/data/uav/runs/dcf-fdr-ec4e2a46-dcf
TOKEN_FILE=/data/uav/secrets/github_material_token
MATERIAL_CHECKOUT="$PUBLICATION_ROOT/material-repo"
LOG_FILE="$PUBLICATION_ROOT/watcher.log"
SUCCESS_MARKER="$PUBLICATION_ROOT/publication-succeeded.json"
FAILURE_MARKER="$PUBLICATION_ROOT/publication-failed.json"
PREFLIGHT_OUTPUT="$PUBLICATION_ROOT/preflight.json"
ATTEMPT_OUTPUT="$PUBLICATION_ROOT/publication-attempt.json"

mkdir -p "$PUBLICATION_ROOT"

if [[ -f "$SUCCESS_MARKER" ]]; then
  exit 0
fi

PUBLISH_ARGS=(
  --clean-root "$CLEAN_ROOT"
  --dcf-root "$DCF_ROOT"
  --staging-root "$PUBLICATION_ROOT"
  --material-checkout "$MATERIAL_CHECKOUT"
  --token-file "$TOKEN_FILE"
  --repository kkc236/icassp2027-fdr-bpdd-fia-material
  --branch main
  --tag clean-dcf-fdr-formal100-seed0-20260824
)

date -Is >>"$LOG_FILE"
while pgrep -f -- "train_dcf_fdr.py --arm dcf .*--output-root $DCF_ROOT" >/dev/null; do
  sleep 60
done

date -Is >>"$LOG_FILE"
if ! "$PYTHON" "$PUBLISHER" "${PUBLISH_ARGS[@]}" --check-only     >"$PREFLIGHT_OUTPUT" 2>>"$LOG_FILE"; then
  "$PYTHON" - "$FAILURE_MARKER" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "published": False,
            "stage": "completion_gate",
            "error": "Formal100 evidence did not pass completion validation; inspect watcher.log",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  exit 2
fi

attempt=1
while (( attempt <= 10 )); do
  date -Is >>"$LOG_FILE"
  if "$PYTHON" "$PUBLISHER" "${PUBLISH_ARGS[@]}"       >"$ATTEMPT_OUTPUT" 2>>"$LOG_FILE"; then
    mv -f "$ATTEMPT_OUTPUT" "$SUCCESS_MARKER"
    exit 0
  fi
  attempt=$((attempt + 1))
  if (( attempt <= 10 )); then
    sleep 60
  fi
done

"$PYTHON" - "$FAILURE_MARKER" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "published": False,
            "stage": "remote_publication",
            "attempts": 10,
            "error": "GitHub publication retries exhausted; local evidence is retained",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
exit 1


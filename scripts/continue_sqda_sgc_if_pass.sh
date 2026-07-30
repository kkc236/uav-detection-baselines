#!/usr/bin/env bash
set -euo pipefail

source_gate="${1:?usage: continue_sqda_sgc_if_pass.sh g2 [token-file]}"
token_file="${2:-/root/.config/sqda-sgc/github-token}"
root="/root/data/uav"
project="$root/runs/sqda-sgc"
decision="$project/sqda-sgc-${source_gate}-seed0-10ep/final-gate-decision.json"
source_run="$project/sqda-sgc-${source_gate}-seed0-10ep"
cd "$root/sqda-sgc"

if [[ ! -f "$decision" ]]; then
  echo "final gate decision is missing: $decision" >&2
  exit 4
fi
passed="$("$root/venv/bin/python" -c \
  'import json, sys; print(str(bool(json.loads(open(sys.argv[1]).read()).get("passed", False))).lower())' \
  "$decision")"
if [[ "$passed" != "true" ]]; then
  echo "SQDA-SGC $source_gate did not pass; formal continuation is not started."
  exit 0
fi

resume_from="$("$root/venv/bin/python" -c \
  'from src.checkpoint_recovery import find_resume_checkpoint; import sys; p=find_resume_checkpoint(sys.argv[1]); print(p or "")' \
  "$source_run")"
if [[ -z "$resume_from" ]]; then
  echo "no valid optimizer-bearing checkpoint is available for formal continuation" >&2
  exit 5
fi

echo "SQDA-SGC $source_gate passed; starting formal 100-epoch continuation from $resume_from"
exec bash "$root/sqda-sgc/scripts/run_sqda_sgc_server.sh" formal "$token_file" "$resume_from" 100

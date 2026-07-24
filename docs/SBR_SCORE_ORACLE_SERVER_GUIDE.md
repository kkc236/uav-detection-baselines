# SBR Score-Oracle Authoritative Server Guide

This guide runs the one frozen score-only causal oracle used to decide the
next route for SBR-RTDETR innovation point 1. The oracle is a feasibility
screen, not a deployable paper method or a reportable method result.

## Frozen inputs and locations

- Repository: `/mnt/uav/repo-sbr-rtdetr-g0`
- Python: `/mnt/uav/venv/bin/python`
- Immutable G0 evidence: `/mnt/uav/evidence/sbr-g0a-51ee6c44`
- Approved design:
  `docs/superpowers/specs/2026-07-24-sbr-score-oracle-design.md`
- Trusted V2 audit manifest:
  `/mnt/uav/evidence/sbr-v2-audit-b6a10f16-20260723T204530Z/audit_manifest.json`
- Upstream input: resolved deterministically from
  `audit_manifest.json[input_manifest][uri]`
- Protocol input: one new `sbr-score-oracle-input/v1` wrapper that hashes the
  trusted input, approved specification, exact clean commit/tree, and frozen
  scientific rule
- Output: `/mnt/uav/evidence/sbr-score-oracle-${COMMIT8}-${UTC}`

## Authoritative command

Run this block once from a clean, reviewed commit:

```bash
set -euo pipefail
REPO=/mnt/uav/repo-sbr-rtdetr-g0
PYTHON=/mnt/uav/venv/bin/python
TRUSTED_AUDIT=/mnt/uav/evidence/sbr-v2-audit-b6a10f16-20260723T204530Z/audit_manifest.json
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT8="${COMMIT:0:8}"
UTC="$(date -u +%Y%m%dT%H%M%SZ)"
PROTOCOL="/mnt/uav/manifests/sbr-score-oracle-${COMMIT8}.json"
OUTPUT="/mnt/uav/evidence/sbr-score-oracle-${COMMIT8}-${UTC}"
UPSTREAM_INPUT="$("$PYTHON" -c 'import json, pathlib, sys, urllib.parse; p=pathlib.Path(sys.argv[1]); u=json.loads(p.read_text(encoding="utf-8"))["input_manifest"]["uri"]; q=urllib.parse.urlparse(u); print(pathlib.Path(urllib.parse.unquote(q.path)) if q.scheme=="file" else pathlib.Path(u))' "$TRUSTED_AUDIT")"

/mnt/uav/venv/bin/python -m pytest \
  tests/test_sbr_score_oracle.py \
  tests/test_sbr_score_oracle_cli.py \
  tests/test_sbr_score_oracle_adjudicator.py -q

"$PYTHON" scripts/prepare_sbr_score_oracle_protocol.py \
  --upstream-input "$UPSTREAM_INPUT" \
  --spec docs/superpowers/specs/2026-07-24-sbr-score-oracle-design.md \
  --repo "$REPO" \
  --output "$PROTOCOL"

"$PYTHON" scripts/run_sbr_score_oracle.py \
  --input-manifest "$PROTOCOL" \
  --spec docs/superpowers/specs/2026-07-24-sbr-score-oracle-design.md \
  --output "$OUTPUT" \
  --workers 8

test -z "$(find "$OUTPUT/primary" -perm /222 -print)"
PRIMARY_ANCHOR="$(sha256sum "$OUTPUT/primary/checksums.sha256" | cut -d' ' -f1)"
"$PYTHON" scripts/adjudicate_sbr_score_oracle.py \
  --evidence "$OUTPUT" \
  --primary-checksums-sha256 "$PRIMARY_ANCHOR"
(cd "$OUTPUT" && sha256sum -c checksums.sha256)
printf 'AUTHORITATIVE_OUTPUT=%s\n' "$OUTPUT"
```

The authoritative status is read only from `final_status.json` after the root
checksum verification succeeds. Both `SBR_SCORE_ORACLE_GO` and
`SBR_SCORE_ORACLE_STOP` are valid scientific outcomes. An
`SBR_SCORE_ORACLE_INVALID` result is a software or integrity failure and may
be rerun only after fixing the demonstrated implementation defect without
changing the frozen scientific rule.

## Prohibited actions

- Do not inspect or use `test-dev` during oracle development or execution.
- Do not add an external or second dataset to the oracle.
- Do not change confidence, IoS, max-det, size, demotion, selection, metric,
  or five-gate thresholds.
- Do not search subsets, retry with revised scientific rules, or use metric
  feedback to relabel groups.
- Do not overwrite an existing protocol wrapper or evidence directory.
- Do not delete, mutate, or replace the failed V2 evidence or immutable G0
  evidence.
- Do not run new GPU inference, training, or a broader data audit in this
  oracle phase.

If the authoritative result is `GO`, freeze a separate GT-free learnable
score-calibration design before training. If it is `STOP`, permanently abandon
score calibration and design the predeclared training-time cross-view
consistency route. Neither outcome authorizes test-dev access by itself.

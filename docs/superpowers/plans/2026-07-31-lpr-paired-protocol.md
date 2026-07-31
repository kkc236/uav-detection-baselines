# LPR strict paired-protocol implementation plan

**Goal:** Replace historical single-arm screening with exact baseline/LPR paired
screening under MuSGD, fixed AMP128, the frozen 10% subset, and seeds 0/1/2.

## Task 1: Protocol utilities and immutable artifacts

- Add `src/lpr_protocol.py` with frozen constants, dataset/subset signatures,
  state fingerprints, environment audit, and atomic artifact helpers.
- Add tests for the exact 647-image selector/hash, dataset semantic hash
  algorithm, category mapping, environment mismatch reporting, and state
  fingerprint corruption detection.
- Add `scripts/prepare_lpr_protocol.py` to create a locked subset list, screen
  YAML, full YAML, seed-specific common/LPR initial states, and signed manifests.
- Refuse to replace any changed artifact.

## Task 2: Exact trainer contract

- Add a shared trainer mixin that explicitly resolves MuSGD with momentum 0.937.
- Replace GradScaler after setup with fixed scale 128 and growth interval
  `2**31 - 1`.
- Audit every optimizer attempt; abort on skip, scale drift, or non-finite data.
- Add a stock paired-control trainer using the exact same mixin and initial-state loader.
- Make LPR-private initialization seed-specific without advancing global RNG.

## Task 3: Frozen paired CLI

- Replace the old CLI with `--variant {control,lpr}`, `--stage {screen,formal}`,
  `--seed {0,1,2}`, and required protocol/initial-state paths.
- Screen means 10 epochs and the frozen subset YAML; formal means fresh 100
  epochs and the full YAML.
- Freeze all scientific parameters and reject resume across stage/dataset/scheduler changes.
- Record commit, environment, dataset, subset, initialization, optimizer,
  AMP, diagnostics, and checkpoint evidence.

## Task 4: Three-seed comparator

- Replace hardcoded historical constants with six-arm paired comparison.
- Enforce final/tail three-seed gates, the 80% floor, localization-loss gate,
  and gate/gradient activity.
- Preserve every failed arm and recommend at most one pre-registered fallback.

## Task 5: Verification and deployment

- Run the complete local and server test suites and direct CLI checks.
- Generate strict dataset and protocol artifacts and verify all frozen hashes.
- Do not launch while driver, Python, source, dataset, or initial-state audits fail.
- Run a one-step control/LPR preflight, then six 10-epoch arms in frozen order.
- If the comparator passes, start fresh full-data seed-0 control/LPR 100-epoch arms.

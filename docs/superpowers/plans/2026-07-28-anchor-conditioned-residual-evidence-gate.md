# Anchor-Conditioned Residual Evidence Gate Implementation Plan

> Execute this plan in the current branch. The sealed validation cache and
> all prior output directories are read-only inputs.

## Goal

Replace the third GCQF stage's independent threshold admission with a
trainable anchor-conditioned admission/ranking head that preserves Fixed
evidence, fills safe capacity, and keeps the existing global protection
invariants.

## Constraints

- Only seed0 is authorized.
- Do not mutate `/home/ubuntu/gcte-g0-output-0e10f1f1`,
  `/home/ubuntu/gcmv-warmstart-output-7d44a725`, or any previous output.
- Use a new source directory and new output directory on the server.
- Do not tune thresholds on the sealed validation set.
- Preserve exact Global, Fixed-SADED, Residual-Off, and Full evaluation
  semantics.
- Formal training is allowed only after the corrected seed0 gate passes.

## Step 1: Add failing module-contract tests

Files:

- `tests/test_sr_peg.py`
- `tests/test_gcqf.py`
- `tests/test_gcqf_loss.py`

Tests to add before implementation:

1. `anchor_mask` is required, boolean, and is concatenated into the local
   trainable path.
2. The new output includes `anchor_admission_logits` with shape `[B,L,1]`.
3. Zero-initialized admission deltas give an anchor query a higher initial
   admission probability than a non-anchor query with identical features.
4. The admission head receives a non-zero gradient.
5. The loss requires admission logits/targets together and returns a finite
   admission term.
6. Old output/checkpoint schema without the new head is rejected by authority
   validation.

Run focused tests and confirm they fail for the missing API:

```powershell
python -m pytest -q tests/test_sr_peg.py tests/test_gcqf.py tests/test_gcqf_loss.py
```

## Step 2: Implement the trainable third stage

Files:

- `src/sr_peg.py`
- `src/gcqf.py`
- `src/gcqf_loss.py`
- `src/gcqf_training.py` (only if batch/optimizer schema requires it)

Changes:

1. Extend the local trunk input by the anchor bit and add a zero-initialized
   `anchor_delta_head`.
2. Emit `anchor_admission_logits` using the fixed `+/-log(3)` prior plus a
   bounded learned delta and utility/risk logits.
3. Thread the output through `GCQFOutput`, batch handling, checkpoint schema,
   and audit metadata.
4. Add the frozen-weight admission BCE term and explicit target validation.
5. Keep the detector frozen and keep the existing residual/global heads
   unchanged.

Run the focused tests again; they must pass.

## Step 3: Replace inference admission with capacity-aware ranking

Files:

- `src/sr_peg_routing.py`
- `tests/test_sr_peg_routing.py`

Changes:

1. Accept and validate `anchor_admission` probabilities.
2. Preserve deterministic non-tiny globals and learned-retained small
   globals exactly.
3. Reject only non-tiny local predictions and protected-global fragments.
4. Rank remaining locals by adjusted score multiplied by admission
   probability, then deduplicate and fill available slots deterministically.
5. Keep legacy learned-output invocation as an explicit compatibility error
   unless all new tensors are supplied; no silent old semantics.
6. Add capacity-fill, anchor-priority, protected-global, and
   bitwise-off tests.

## Step 4: Update targets, cache schema, and evaluator

Files:

- `src/sr_peg_targets.py`
- `src/gcqf_cache.py`
- `scripts/train_gcqf_g0.py`
- `scripts/evaluate_gcqf_g0.py`
- `scripts/calibrate_sr_peg_g0.py`
- related tests

Changes:

1. Build admission targets from frozen anchor, tiny-utility, and non-tiny-risk
   targets.
2. Bump the supervised cache schema and make old caches fail closed.
3. Persist the new tensor in checkpoints and evaluation payloads.
4. Keep calibration deterministic and forbid validation-set threshold search.
5. Record accepted-local, final-prediction, capacity-rejected, and admission
   distributions for diagnosis.

## Step 5: Run regression and deploy a new source snapshot

Local:

```powershell
python -m pytest -q tests/test_sr_peg.py tests/test_gcqf.py tests/test_gcqf_loss.py tests/test_sr_peg_routing.py tests/test_gcqf_training.py
python -m pytest -q
```

Commit the implementation only after both focused and full suites pass.
Copy the exact commit to a new server source directory under
`/home/ubuntu/`, with a new output directory under
`/home/ubuntu/`, and verify source/commit/checksum metadata before execution.

## Step 6: Seed0 diagnostic

Reuse only sealed read-only validation and train caches when their manifests
and signatures match. Generate any new schema-compatible train cache in the
new output directory, train the module for the frozen ten epochs with seed0,
calibrate on the fixed 129-image split, and evaluate the five states on the
548-image validation set.

Stop immediately if any Global-relative hard gate or safety invariant fails.
Keep the Fixed-SADED-relative deltas as nonblocking internal diagnostics,
then archive the new JSON, checksums, logs, and exact commit.

## Step 7: Formal 100-epoch entry

After the seed0 gate passes:

1. Add and test a formal training entry that uses the full 6,471/548 VisDrone
   split and the user's frozen RT-DETR-L/Ultralytics 8.4.90 protocol.
2. Run focused protocol and source-authority checks.
3. Start exactly one seed0, 100-epoch process in a new output directory.
4. Confirm PID survival, nonzero RTX 4090 utilization, GPU memory use, and
   first-batch/epoch log before reporting completion.

If the corrected seed0 gate fails, do not start formal training; stop local
patching and redesign from the successful SADED evidence.

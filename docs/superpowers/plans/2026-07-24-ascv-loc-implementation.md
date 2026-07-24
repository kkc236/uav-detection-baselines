# ASCV-Loc Implementation Plan

## 1. Pure geometry and loss

Create `src/ascv_loc.py` with:

- frozen constants and protocol validation;
- deterministic target-anchored 0.60 crop selection;
- complete-box eligibility and coordinate transforms;
- target-ID joins for independent Hungarian matches;
- scale-direction masks;
- one-way detached L1 plus GIoU localization loss;
- structured diagnostics.

Write failing unit tests first for geometry, partial-box exclusion, target-ID
join behavior, exact gradient direction, empty-pair behavior, warm-up, and
finite-loss enforcement.

## 2. RT-DETR integration

Create `src/rtdetr_ascv_loc.py` with:

- an RT-DETR model subclass that computes stock full-view detection loss;
- a training-only local forward pass;
- independent stock matcher calls;
- ASCV-Loc auxiliary loss and diagnostics;
- a trainer subclass whose `_build_train_pipeline()` constructs only the train
  loader, whose validator is a fail-closed sentinel, and whose
  `validate()`/`final_eval()` are non-reading no-ops for every stage;
- unchanged validation and inference prediction path.

Add integration tests using fakes so CPU tests do not require a full training
run.

## 3. Frozen CLI and protocol preparation

Create:

- `scripts/prepare_ascv_loc_protocol.py` for the immutable hashed 10% training
  image list and manifest;
- `scripts/train_rtdetr_ascv_loc.py` for the four frozen stages;
- `scripts/run_ascv_loc_server.sh` for status, logs, atomic stage evidence, and
  resume-safe execution.

Every development screen is paired with a stock control from the same starting
checkpoint. Training never sees val; the external SBR evaluator is invoked
once per fixed endpoint.

The CLI must reject parameter drift and must not expose tunable scientific
arguments.

## 4. Verification

Run:

1. focused ASCV-Loc tests;
2. the complete repository test suite;
3. server-side focused and complete tests on Ultralytics 8.4.90;
4. a synthetic forward/backward smoke test;
5. a short real-data smoke that does not read val.
6. a 500-batch train-only teacher-advantage mechanism adjudication.

## 5. Frozen execution

Execute the state machine in order:

`MECHANISM_500 -> paired SCREEN_6 -> paired FULL_20 -> fixed-last
SEED0_100 -> fixed-last SEED1_100 -> fixed-last SEED2_100`.

At every boundary:

- seal source commit, input manifest, logs, diagnostics, checkpoint hashes, and
  output checksums;
- ask supervisor B to validate evidence and rules;
- ask planner C for the next frozen transition;
- never inspect partial val metrics or test-dev.
- never select `best.pt`; scientific evaluation uses the predeclared final
  `last.pt`.

## 6. Handoff and public evidence

Update the private centaur handoff with server paths, hashes, commands, state,
and B/C decisions. Publish only compact, credential-free metrics and method
documentation to GitHub; exclude images, labels, checkpoints, raw caches,
absolute private paths, and credentials.

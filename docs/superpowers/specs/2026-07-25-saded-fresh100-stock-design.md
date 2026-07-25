# SADED Fresh-100 Stock Baseline Design

## Objective

Launch one fresh seed-0 Ultralytics RT-DETR-L baseline for 100 epochs on the
full 6,471-image VisDrone training split. The endpoint is the matched stock
reference for the already frozen SADED-SM route-control method.

## Scientific boundary

- The run starts from the sealed seed-0 common initial state, not from any
  10-epoch or historical 100-epoch checkpoint.
- It uses the frozen baseline contract byte-for-byte: scratch initialization,
  image size 640, batch 8, workers 8, MuSGD, fixed AMP scale 128, and the
  published augmentation and optimizer values.
- Training is train-only. It must not build a validation or test loader and
  must not inspect test-dev or partial validation metrics.
- The run has no predecessor performance gate because it is a stock baseline,
  not a T-ASCV treatment. This exception removes only an irrelevant workflow
  dependency; it does not change any scientific hyperparameter.
- The historical SADED-SM checkpoint and sealed evidence remain immutable and
  are comparison artifacts only.

## Implementation

A dedicated fail-closed stock CLI reuses `TASCVControlTrainer`, which already
enforces the matched optimizer, fixed AMP scale, train-only loader, batch
canaries, and exact 80,900-batch/10,556-step FORMAL_100 policy. A new protocol
manifest binds the source commit, environment, initial state, train-only YAML,
output endpoint, and frozen training contract.

The launcher refuses non-seed-0 runs, non-GPU-0 devices, existing targets,
test-dev paths, source drift, environment drift, data drift, initial-state
drift, and protocol drift. It writes a final runtime summary only after the
full endpoint and checkpoint pass all canaries.

## Recovery

An interruption may resume only the same endpoint under the same committed
source and manifest. A scientific or integrity failure is preserved as
INVALID; it is never repaired by changing hyperparameters or reusing a
different checkpoint.

## Post-training

After exit 0, the pipeline generates the full-view plus four-tile raw cache,
applies the frozen GT-free SADED-SM route-control rule, performs one sealed
dev-val evaluation, and adjudicates the five frozen paper gates. Test-dev
remains unopened and ablations remain out of scope.

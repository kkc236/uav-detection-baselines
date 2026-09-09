# AC-BPDD-R v1 Design Specification

## Goal

Test the highest-priority BPDD repair identified by the adversarial audit without changing the established LRS-GFDR backbone or confounding the result with additional gates or objectives.

## Mechanism

AC-BPDD currently distils adjacent cumulative decoder distributions. Because a cumulative state contains all earlier residual updates, an edge loss at layer `l` can back-propagate directly into residuals from layers before `l`. AC-BPDD-R keeps the forward loss value unchanged but detaches the cumulative history before the supervised residual. The loss therefore updates only the residual that creates the selected transition.

The first candidate changes exactly one BPDD option:

- `residual_gradient_only: true`

All other AC-BPDD choices stay at the validated G-arm settings:

- `decoded_iou_gate: false`
- `distribution_objective: kl`
- `assignment_mode: consistent`
- `weight: 0.15`
- `temperature: 0.5`
- `margin: 0.02`
- `include_dn: false`

LRS-GFDR also remains unchanged: feasible geometry is explicitly enabled and `reliability_shrinkage_alpha` remains `0.25`.

## Experiment identity

Use the internal method identity `lrs_gfdr_ac_bpdd_residual` and run name `formal-seed0-lrs_gfdr_ac_bpdd_residual-v1`. The paper-facing name is provisional `LRS-GFDR + AC-BPDD-R`; this experiment does not rename the final method.

## Frozen comparison protocol

Start from the same seed-0 initial-state artifact as the completed LRS-GFDR and AC-BPDD runs. Train from epoch 1 for 100 epochs with the existing Formal100 settings, dataset split, image size, batch size, optimizer, AMP policy, and seed. Do not resume from any trained checkpoint.

Run from a clean source-bound worktree and write an immutable launch-authority record containing source, config, dataset, initial-state, and training-setting identities. Store outputs in a new directory so no existing run is overwritten.

## Interpretation

The primary comparison is AC-BPDD-R versus the strictly paired AC-BPDD G arm. LRS-GFDR remains the secondary control. Best mAP50-95 is the headline metric; final epoch and last-10-epoch mean are stability checks. The result must be reported even if it is neutral or negative. A single seed is screening evidence, not a statistical significance claim.

## Acceptance criteria

- The candidate differs from the AC-BPDD G config only in the explicit residual-gradient option and an explanatory comment.
- Candidate and G models initialized with the same seed have byte-identical state tensors.
- A synthetic gradient test confirms unchanged forward loss and removal of direct gradients to earlier residuals.
- Dry-run authority records the intended hashes and all frozen settings.
- The remote job starts in an isolated clean worktree, uses the shared initial state, and produces finite training/runtime evidence.

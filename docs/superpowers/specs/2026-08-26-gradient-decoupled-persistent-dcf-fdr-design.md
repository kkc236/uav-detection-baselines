# Gradient-Decoupled Persistent DCF-FDR Design

Date: 2026-08-26  
Status: User-approved design

## 1. Decision

Run one fresh Formal100 seed-0 arm that preserves the exact DCF behavior used
during Epochs 1-66 of source commit
`50052b68b24d821b8905e279b312da2c06d6ee0e` for the entire 100-epoch horizon.

The method is the existing Distribution-Conditioned Feedback (DCF) adapter with
the gradient-partition correction introduced in commit `17283020`: DCF adapter
parameters belong to the private FDR gradient group and receive their own
`max_norm=10` clipping operation. The adapter remains enabled, unscaled, and
trainable from Epoch 1 through Epoch 100.

This experiment must not freeze, attenuate, remove, or export away DCF at any
point. It is a persistent DCF experiment, not a transient DCF experiment.

## 2. Exact Method Boundary

The existing DCF data path is unchanged:

1. Detach the preceding cumulative four-edge, 33-bin corner logits.
2. Apply per-edge softmax.
3. Encode each edge with the shared `Linear(33, 16) + SiLU` encoder.
4. Flatten the four encodings and project them with the shared zero-initialized
   `Linear(64, 256)` output.
5. Add the resulting feedback to the next decoder layer's regression input.

The following invariants hold for all paper epochs `e in [1, 100]`:

```text
distribution_feedback_scale = 1.0
distribution_feedback parameters require_grad = true
DCF forward path = enabled for decoder layers 2-6
checkpoint eligibility = all epochs
```

The experiment explicitly removes all transient behaviors:

- no Epoch-67 freeze;
- no Epochs 67-74 cosine withdrawal;
- no Epoch-75 exact-off path;
- no Epoch-75 best-fitness reset;
- no Clean-shaped inference export;
- no transient schedule callback.

The Clean-FDR base remains unchanged: preliminary boxes, extra DN-FDR
supervision, edge-adaptive FGL, EAW, score calibration, and all unrelated
optional components remain disabled.

## 3. Training and Gradient Semantics

The frozen VisDrone Formal100 protocol is reused without changes:

- frozen data authority and initial state;
- RT-DETR-L, 640-pixel images, batch 8, workers 8;
- seed 0, deterministic mode, MuSGD, identical LR schedule and augmentations;
- 100 epochs from the frozen Epoch-0 initial state;
- identical FDR losses and matching/post-processing behavior.

The only difference from the old persistent DCF arm is the already-tested
gradient partition:

```text
common gradient group:
  backbone + encoder + ordinary decoder parameters

private FDR gradient group:
  FDR distribution heads + preliminary-head tensors + DCF adapter
```

Each group is independently clipped to `max_norm=10`. This preserves the exact
optimization behavior observed in the successful part of the transient run
through Epoch 66 and isolates whether that benefit persists to Epoch 100 when
DCF is never withdrawn.

## 4. Approaches Considered

### Selected: dedicated persistent-gradient launcher and identity

Create a dedicated source-visible experiment identity and launcher that asserts
scale one and trainable DCF at every epoch. This is the safest option because
the run authority, output directory, logs, and paper interpretation cannot be
confused with either old persistent DCF or failed transient DCF.

### Rejected: run the transient launcher after manually removing its callback

This would be a small code diff, but the launcher name, authority record, and
schedule metadata would falsely describe a transient experiment. It is too easy
to reintroduce freezing or tail selection and is not acceptable evidence.

### Rejected: reuse the old persistent DCF launcher directly

The old launcher predates the DCF-private gradient partition. Reusing it would
silently return to the weaker common-gradient clipping behavior and would not
reproduce the first-66-epoch version requested by the user.

## 5. Components and Data Flow

### Dedicated configuration

The new YAML is structurally identical to
`rtdetr-l-transient-dcf-fdr.yaml`, but has a persistent-gradient method name.
It enables exactly one DCF adapter and keeps all four removed FDR options false.

### Dedicated launcher

The launcher builds the same frozen Formal100 settings and creates a distinct
authority record. It installs one fail-closed audit callback. The callback does
not alter training; it only verifies and records that live and EMA decoders both
have scale `1.0` and that every live DCF parameter remains trainable.

### Epoch evidence

One JSONL row per epoch records:

- paper epoch and total epochs;
- live and EMA scale, both exactly `1.0`;
- DCF trainability;
- DCF membership in the private FDR gradient group;
- absence of transient freeze/withdrawal state.

Any failed invariant raises before the affected epoch trains.

### Checkpoint protection

Ultralytics continues to own `best.pt` and `last.pt`. A detached read-only
watcher validates completed `last.pt` files and preserves full resumable copies
at selected milestones, including paper Epoch 66. The required retained set is
Epochs 25, 50, 66, 75, 90, and 100. Copies are written atomically and verified
for checkpoint epoch, optimizer, scaler, and EMA state. This avoids another
lost branch point without retaining 100 approximately 200 MB checkpoints.

## 6. Error Handling and Operational Isolation

The run launches only when all preflight gates pass:

- exact expected source commit and clean tracked worktree;
- frozen dataset and initial-state hashes;
- GPU idle and sufficient disk space;
- new output directory absent;
- no existing trainer for the same method;
- finite smoke forward/backward with DCF in the private gradient group;
- no resume argument or reused weights.

Training runs detached from SSH and writes to a new output root. Existing Clean,
old persistent DCF, and failed transient DCF results are read-only comparators
and are never overwritten. A process failure preserves all evidence and does
not automatically substitute a non-exact experiment.

## 7. Verification

Implementation tests must prove:

1. the new configuration matches the failed transient run's Epoch-1-66 model
   and loss configuration;
2. every Epoch 1-100 audit state has scale exactly `1.0` and DCF trainable;
3. no freeze, withdrawal, tail-reset, or Clean-export callback is installed;
4. DCF parameters are exclusively in the private FDR gradient group;
5. live and EMA DCF paths remain enabled;
6. the launch record uses a new method name and rejects resume;
7. the Epoch-66 checkpoint watcher accepts only a complete epoch-65
   zero-based checkpoint with optimizer, scaler, and EMA state;
8. existing FDR, DCF, and transient tests remain passing;
9. a bounded GPU smoke run has finite losses and nonzero DCF gradients.

## 8. Result Interpretation

The primary comparison is best mAP50-95 over all 100 epochs:

- Clean FDR reference: `29.696%`;
- old persistent DCF reference: `29.661%`;
- failed transient DCF is diagnostic only and is not the target method.

Verdicts:

| Verdict | Rule |
|---|---|
| Fail | best mAP50-95 `< 29.696%` |
| Technical pass | best mAP50-95 `>= 29.696%` and gain `< 0.100 pp` |
| Strong pass | best mAP50-95 `>= 29.796%` |

Precision, Recall, AP50, final-epoch metrics, and late-window means are mandatory
diagnostics. No post-hoc schedule change or checkpoint restriction may repair a
failed result.

## 9. Claim Boundary

This run tests one clean hypothesis: separating the DCF adapter from common
gradient clipping improves the optimization of a persistent FDR-internal
distribution-feedback path. It does not claim a training-only module, zero-cost
inference, cosine withdrawal, or knowledge transfer to a Clean path. Inference
retains the same small DCF adapter used during training.

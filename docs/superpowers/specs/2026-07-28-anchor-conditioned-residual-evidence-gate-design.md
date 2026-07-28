# Anchor-Conditioned Residual Evidence Gate Design

## 1. Decision

The first two trainable stages of `GCQF` remain unchanged:

1. `GeometryQueryProjector`;
2. `GlobalLocalQueryInteraction`.

The third trainable stage is corrected from an independent hard
`ScaleRiskProtectedEvidenceGate` into:

3. `AnchorConditionedResidualEvidenceGate` (`ACR-EG`).

`ACR-EG` remains part of the single `GCQF` network module. It consumes
network features, contains registered trainable layers, receives supervised
gradients, participates in the forward pass, and supports direct ablation.
The Fixed-SADED mask is an inference-time network input and prior, not the
final decision itself.

This is the only local structural correction authorized before either
starting the formal 100-epoch experiment or abandoning this architecture.

## 2. Evidence and root cause

The completed sealed seed0 diagnostic produced:

| Delta | mAP50-95 | AP-tiny-SBR | Tiny recall | AP-medium-SBR | AP-large-SBR |
|---|---:|---:|---:|---:|---:|
| Full - Global | +0.010595 | +0.013412 | +0.054338 | -0.001640 | -0.000096 |
| Full - Fixed | -0.016579 | -0.030630 | — | +0.009265 | — |

Every frozen gate passed except `Full - Fixed mAP50-95 >= 0`.

The former third stage admitted only 23,283 local predictions, while
Fixed-SADED admitted 120,326. It emitted 156,960 predictions across the
validation set versus 164,384 for Fixed-SADED, leaving 7,440 available
`max_det` slots empty. Therefore the failure is not absence of useful local
evidence: the same run improved both tiny AP and medium protection relative
to Global. The failure is that three calibrated hard thresholds reconstruct
candidate admission independently of the Fixed anchor and discard too much
of its useful tiny evidence before the 300-slot ranking stage.

This also contradicts the existing `GCQF` design invariant that the fixed
router remains authoritative around which the learned module predicts a
bounded residual.

## 3. Rejected alternatives

### 3.1 Recalibrate the three old thresholds

Rejected. It changes calibration rather than network structure, does not fix
the violated anchor invariant, and risks tuning to the 129-image calibration
split. Estimated strict-gate success is below 40%.

### 3.2 Append Fixed candidates after the learned router

Rejected as the final method. A rule-only fallback could fill empty slots,
but it is mostly post-processing, provides no trainable explanation for
which Fixed candidates are useful, and can erase the recovered medium
performance.

### 3.3 Anchor-conditioned residual admission and ranking

Selected. Fixed membership becomes an explicit feature and conservative
prior inside the trainable gate. The learned network predicts a bounded
correction to that prior, after which all safe tiny candidates compete by
capacity-aware rank instead of being destroyed by independent thresholds.
Estimated probability of passing all frozen seed0 gates is 65–70%.

## 4. Network architecture

### 4.1 Inputs

The corrected third stage receives:

- canonical local decoder query;
- global-to-local interaction context;
- geometry embedding;
- local detector score;
- Fixed-SADED `anchor_mask`;
- frozen global queries, boxes, and detector scores.

Ground truth is used only to create training targets and is never an
inference input.

### 4.2 Anchor-conditioned local trunk

For each local query, concatenate:

```text
canonical query
+ global context
+ geometry embedding
+ local base score
+ anchor membership
```

and pass it through:

```text
Linear(3D + 2, D) -> GELU -> Linear(D, D) -> LayerNorm(D)
```

with `D=256`.

The existing tiny-utility, non-tiny-risk, and bounded score-residual heads
remain. A fourth trainable local head is added:

```text
anchor_delta_head: Linear(D, 1)
```

It predicts a residual admission logit rather than an admission decision.
Its final layer is zero-initialized.

### 4.3 Conservative anchor prior

The initial admission logit is:

```text
anchor_logit =
    +log(3), anchor_mask = 1
    -log(3), anchor_mask = 0
```

which corresponds to a documented 0.75/0.25 prior and is chosen before the
new seed0 run, not from validation measurements. The network output is:

```text
admission_logit =
    anchor_logit
  + tanh(anchor_delta_head(local_hidden)) * log(3)
  + 0.5 * tiny_utility_logit
  - 0.5 * non_tiny_risk_logit
```

The bounded learned delta prevents a ten-epoch module-only diagnostic from
instantly overturning every Fixed candidate, while utility and risk retain a
direct supervised route into the admission decision. At initialization an
anchor candidate always ranks above an otherwise identical non-anchor
candidate.

### 4.4 Global protection

The existing learned global-retain head remains unchanged. Global
predictions whose effective size exceeds 16 pixels remain deterministically
protected. Small global predictions may additionally be protected by the
learned retain output. This preserves the already successful medium/large
safety behavior.

### 4.5 Forward outputs

`ACR-EG` returns:

- `tiny_utility_logits [B,1200,1]`;
- `non_tiny_risk_logits [B,1200,1]`;
- `anchor_admission_logits [B,1200,1]`;
- `global_retain_logits [B,300,1]`;
- bounded `score_residual [B,1200,1]`;
- adjusted local scores.

All new parameters are registered in `state_dict`, receive gradients, and
are included in checkpoint authority checks.

## 5. Training supervision

The frozen SR-PEG losses remain unchanged. One admission loss is added:

```text
admission_target =
    clamp(
        0.5 * anchor_mask
      + 0.5 * tiny_utility_target
      - 0.5 * non_tiny_risk_target,
        0,
        1
    )
```

and:

```text
L_admission = BCEWithLogitsLoss(
    anchor_admission_logit,
    admission_target
)
```

The admission term has weight `1.0`. It teaches the third stage to preserve
the Fixed prior unless supervised tiny utility or non-tiny risk provides
evidence to move away from it. Existing positive weighting for utility,
risk, and global retention remains frozen from the training split.

No detector weights are updated in the ten-epoch diagnosis.

## 6. Capacity-aware inference

Inference keeps the existing geometry and protection invariants but changes
local admission:

1. protect non-tiny globals and learned-retained small globals;
2. reject local predictions whose effective size exceeds 16 pixels;
3. reject local fragments that overlap a protected global by
   intersection-over-smaller at least 0.5;
4. do not apply independent utility and risk hard-rejection thresholds;
5. assign every remaining local candidate:

```text
rank_score =
    adjusted_detector_score
  * sigmoid(anchor_admission_logit)
```

6. rank unprotected globals by detector score and local candidates by
   `rank_score`, with the existing stable provenance tie-break;
7. perform the existing same-class IoU deduplication in rank order;
8. take the best candidates until all available slots are filled or the
   safe candidate pool is exhausted.

This makes candidate removal a capacity-aware competition. It avoids empty
slots caused solely by calibrated thresholds while allowing the learned
network to demote risky Fixed candidates below safe alternatives.

`Residual-Off` disables only score residuals, not anchor-conditioned
admission. `ACR-EG-Off` restores the sealed Fixed-SADED output exactly.
`GCTE-Off` restores Global RT-DETR exactly.

## 7. Error handling and invariants

The implementation must fail closed when:

- `anchor_mask` is absent, non-boolean, or shape-mismatched;
- admission tensors are non-finite or outside their declared domain;
- query provenance or selected-query indices drift;
- a protected global would be displaced;
- output exceeds 300 predictions.

Required invariants:

- protected global identity and relative order are exact;
- no local non-tiny leak;
- no local fragment overlaps a protected global above the frozen threshold;
- deterministic tie-breaking;
- output capacity is filled whenever enough safe deduplicated candidates
  exist;
- `ACR-EG-Off` is bitwise identical to Fixed-SADED.

## 8. Test-first verification

Before implementation, failing tests must cover:

1. anchor mask is part of the trainable local feature path;
2. zero-initialized admission delta gives anchor candidates higher initial
   admission probability;
3. admission loss sends non-zero gradients to the new head;
4. invalid anchor masks fail closed;
5. capacity-aware routing fills previously empty safe slots;
6. protected-global and fragment invariants remain exact;
7. stable ordering is deterministic;
8. old checkpoints fail the explicit schema/authority check rather than
   silently loading missing parameters.

Focused tests are followed by the complete regression suite before
deployment.

## 9. Experimental decision gate

Only seed0 is run. Training and calibration identities remain exactly the
same as the completed diagnostic. The sealed 548-image validation set and
all thresholds for declaring success remain unchanged.

The corrected model advances when the unchanged Global-relative gates and
network safety invariants pass:

- `Full - Global mAP50-95 >= +0.005`;
- `Full - Global AP-tiny-SBR >= +0.010`;
- `Full - Global tiny recall >= +0.020`;
- `Full - Global AP-medium-SBR >= -0.002`;
- `Full - Global AP-large-SBR >= -0.005`;
- protected global identity is exact;
- output count and fragment protections are valid;
- residual is active and not saturated.

`Full - Fixed-SADED` mAP and medium recovery remain in the evaluation JSON as
internal development diagnostics. They are not an external baseline and do
not block formal training. The paper's primary comparison is the complete
method against the original Global RT-DETR-L baseline. No validation-set
tuning, gate relaxation, seed1, or seed2 is allowed.

If the corrected seed0 run fails any hard gate, local patching stops and the
architecture is reshaped from the successful multi-view SADED evidence.
If it passes, the exact verified commit is used to start the formal
100-epoch full-train experiment on the RTX 4090 with the user-frozen
baseline protocol.

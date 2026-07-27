# SR-PEG Medium-Protection Design

## 1. Decision

GCTE-RTDETR remains one decoder-query-level network innovation named
`GCQF`. Its first two trainable stages remain unchanged:

1. `GeometryQueryProjector`;
2. `GlobalLocalQueryInteraction`.

The former third stage, `AnchorPreservedResidualFusion`, is replaced by one
trainable stage:

3. `ScaleRiskProtectedEvidenceGate` (`SR-PEG`).

`SR-PEG` is not a fourth contribution and is not a post-processing-only
rule. It is the third registered stage inside the single `GCQF` module.

## 2. Evidence and problem statement

The current-baseline Fixed-SADED anchor on the sealed VisDrone validation
set produced:

| Metric | Global | Fixed-SADED | Delta |
|---|---:|---:|---:|
| mAP50-95 | 0.198699 | 0.225872 | +0.027173 |
| AP-tiny-SBR | 0.080789 | 0.124831 | +0.044042 |
| tiny recall | 0.582079 | 0.681828 | +0.099749 |
| AP-medium-SBR | 0.257805 | 0.246901 | -0.010905 |
| AP-large-SBR | 0.154643 | 0.152511 | -0.002132 |

The multi-view evidence is therefore valuable, but the fixed
`effective_size <= 16 px` rule is not a reliable semantic tiny/non-tiny
classifier. A medium object can be localized by an underestimated prediction
whose effective size is at most 16 px. The current router may replace or
displace that global candidate inside the 300-prediction budget.

The G0 objective is to preserve enough of the observed tiny benefit while
restoring:

- `Full - Global AP-medium-SBR >= -0.002`;
- `Full - Global AP-large-SBR >= -0.005`.

## 3. Network architecture

### 3.1 Inputs

`SR-PEG` receives only tensors available during inference:

- 300 frozen global decoder queries, logits, boxes, and base scores;
- 1200 geometry-canonical local decoder queries;
- local-to-global attention context from stage 2;
- geometry embeddings and local base scores;
- fixed valid-view masks.

Ground truth is never an inference input.

### 3.2 Local shared trunk and heads

For every local query, concatenate:

```text
canonical local query
+ global attention context
+ geometry embedding
+ local base score
```

Pass this tensor through a shared two-layer MLP:

```text
Linear(3D + 1, D) -> GELU -> Linear(D, D) -> LayerNorm(D)
```

with `D=256`. Three trainable heads consume the shared representation:

- `tiny_utility_head: Linear(D, 1)`;
- `non_tiny_risk_head: Linear(D, 1)`;
- `score_residual_head: Linear(D, 1)`.

The score residual remains bounded by `tanh`, and its multiplicative scale
remains `eta=0.2`.

### 3.3 Global retain head

Global queries attend once to the canonical local queries:

```text
MultiheadAttention(D=256, heads=8, dropout=0)
```

The attended feature, global query, global box encoding, and global base
score enter:

```text
global_box_mlp: Linear(4, 64) -> GELU -> Linear(64, 64)
retain_head: Linear(2D + 64 + 1, D) -> GELU -> Linear(D, 1)
```

to produce `global_retain_logit` for each of the 300 global queries.

### 3.4 Forward outputs

`SR-PEG` returns:

- `tiny_utility_logit [B,1200,1]`;
- `non_tiny_risk_logit [B,1200,1]`;
- `score_residual [B,1200,1]`;
- `global_retain_logit [B,300,1]`;
- adjusted local scores;
- local-admission and global-retain probabilities for auditing.

All layers are registered in `state_dict`, receive gradients, appear in the
YAML module configuration, and support `+SR-PEG/-SR-PEG` ablation.

## 4. Targets and losses

All targets are generated from the fixed train10 labels and cached detector
evidence. Predicted classes are `argmax` over the frozen query logits. GT and
prediction effective sizes use the existing SBR `effective_size` function
after mapping boxes into source-image coordinates.

### 4.1 Tiny utility target

A local prediction has a positive tiny-utility target when:

- its predicted class equals the matched GT class;
- the matched GT effective size is at most 16 px;
- IoU with that GT is at least 0.5.

The soft target is the matched IoU; otherwise it is zero.

### 4.2 Non-tiny risk target

A local prediction has non-tiny risk when its intersection-over-smaller with
any GT whose effective size is greater than 16 px is at least 0.5. This
captures underestimated fragments inside small, medium, or large objects,
including class-conflicting fragments.

### 4.3 Global retain target

For a global prediction whose predicted effective size is at most 16 px, the
retain target is positive when:

- its predicted class equals a non-tiny GT class; and
- its intersection-over-smaller with that GT is at least 0.5.

Global predictions larger than 16 px remain deterministically protected and
do not depend on this learned target.

### 4.4 Frozen loss

```text
L =
    1.0 * L_quality
  + 0.1 * L_equivariance
  + 0.01 * L_residual
  + 1.0 * L_tiny_utility
  + 2.0 * L_non_tiny_risk
  + 2.0 * L_global_retain
```

The higher risk and retain weights express the success-first requirement that
medium recovery is mandatory. `L_tiny_utility`, `L_non_tiny_risk`, and
`L_global_retain` are `BCEWithLogitsLoss` terms. For each head, its
dataset-level positive weight is computed once from the 518 training records
as `clip(N_negative / max(N_positive, 1), 1, 20)` and is then frozen for the
entire seed0 run.

## 5. Inference and 300-candidate budget

The deterministic safety shell is:

1. every global prediction with effective size greater than 16 px is
   preserved bitwise;
2. an at-most-16-px global prediction is additionally protected when
   `sigmoid(global_retain_logit) >= tau_g`;
3. a local prediction is eligible only when its effective size is at most
   16 px, `sigmoid(tiny_utility_logit) >= tau_t`, and
   `sigmoid(non_tiny_risk_logit) < tau_r`;
4. a local prediction whose intersection-over-smaller with any protected
   global prediction is at least 0.5 is rejected regardless of class;
5. eligible local scores receive the bounded residual;
6. a same-class local/global pair above IoU 0.5 contributes only its
   higher-scoring member unless the global member is protected;
7. protected global predictions are emitted first; unprotected global and
   eligible local predictions then fill the remaining slots in stable
   descending-score order up to 300.

Turning `SR-PEG` off exactly restores the sealed Fixed-SADED anchor. Turning
the whole GCTE path off exactly restores Global RT-DETR.

## 6. Data split and calibration

Only seed0 is run in this G0 diagnostic.

The fixed 647-image train10 authority is split without touching validation:

- sort image identities by `SHA256("seed0:" + image_id)`;
- first 518 images: module training;
- remaining 129 images: calibration.

The validation set remains the sealed 548-image VisDrone validation set.

After 10 module-only epochs, choose `(tau_t, tau_r, tau_g)` from
`{0.4, 0.5, 0.6}^3` on the 129-image calibration split. Selection is
deterministic:

1. retain only settings satisfying calibration medium and large budgets;
2. maximize mAP50-95;
3. break ties by AP-tiny-SBR, then tiny recall, then lexicographic thresholds.

No threshold is selected on the final validation set.

The calibration medium and large deltas are measured against Global on the
same 129 calibration images, using the final validation budgets of `-0.002`
and `-0.005`.

## 7. Execution sequence

1. Upgrade the train cache schema with the three new targets.
2. Generate the fixed 647-image train10 cache once.
3. Train only `GCQF/SR-PEG` for 10 epochs with seed0; the detector remains
   frozen.
4. Calibrate the three thresholds on the fixed 129-image calibration split.
5. Evaluate the five states on the sealed validation cache:
   `Global`, `Raw-Union`, `Fixed-SADED`, `Residual-Off`, `Full-GCQF`.
   `Residual-Off` keeps utility/risk/retain gating active and disables only
   the bounded score residual.
6. Stop after seed0. Do not run seed1 or seed2.

## 8. Hard success gate

Relative to Global:

- `mAP50-95 >= +0.005`;
- `AP-tiny-SBR >= +0.010`;
- `tiny recall >= +0.020`;
- `AP-medium-SBR >= -0.002`;
- `AP-large-SBR >= -0.005`;
- protected global identity is exact;
- output count is at most 300.

To demonstrate that the learned third stage solves the observed failure:

- `Full-GCQF - Fixed-SADED AP-medium-SBR >= +0.008`;
- `Full-GCQF - Fixed-SADED mAP50-95 >= 0`.

Failure of any hard gate stops this version. No validation-set threshold
sweep, fresh100 training, seed1, or seed2 is authorized.

## 9. Expected success probability

The estimated probability of passing this seed0 G0 gate is 70-75%. The basis
is the large available margin in the Fixed-SADED evidence:

- tiny AP can lose 0.034 and still pass;
- mAP can lose 0.022 and still pass;
- medium needs to recover approximately 0.009.

The estimate applies to the experimental gate, not to eventual venue
acceptance.

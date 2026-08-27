# LRS-FGL Design

## Objective

Add one training-only refinement to Clean FDR that improves the distribution
supervision received by low-IoU normal Hungarian matches without changing the
detector, matcher, inference graph, total per-image FGL weight, or final-layer
FGL objective.

The method is named **Layerwise Reliability Shrinkage for Fine-Grained
Localization (LRS-FGL)**. The paper claim is deliberately narrow: LRS-FGL
shrinks detached matched-IoU reliability weights toward their image-and-layer
mean in shallow decoder layers, then anneals the shrinkage to zero at the final
layer.

## Baseline and observed gap

The immutable direct baseline is Clean FDR at commit
`e7b37e35892f93c306de02844aebf2b312c484eb`, using
`configs/rtdetr-l-clean-fdr.yaml`. Its seed-0 Formal100 historical best is:

| Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 88 | 0.58777 | 0.49849 | 0.49320 | 0.29696 |

For every normal Hungarian match, the current FGL coefficient is the detached
IoU `q`. Consequently, a matched query with `q = 0` receives no FDR-specific
distribution loss, even though stock L1 and GIoU still supervise its box. The
new mechanism addresses only this distribution-supervision starvation. It does
not create queries, change assignments, or directly alter classification.

## Method

For decoder layer `l` in `0..L-1`, image `b`, and its normal matched-query set
`M[b,l]`, define:

\[
q_{b,l,k}=\operatorname{sg}(\operatorname{IoU}(B_{b,l,k},G_{b,l,k})),
\]

\[
\bar q_{b,l}=\frac{1}{|M[b,l]|}\sum_{k\in M[b,l]}q_{b,l,k},
\]

\[
\alpha_l=\alpha_0\left(1-\frac{l}{L-1}\right),\qquad \alpha_0=0.25,
\]

\[
\widetilde q_{b,l,k}=(1-\alpha_l)q_{b,l,k}+\alpha_l\bar q_{b,l}.
\]

For six decoder layers, the fixed shrinkage strengths are
`[0.25, 0.20, 0.15, 0.10, 0.05, 0.00]`. The existing adjacent-bin FGL formula,
`fgl_weight=0.15`, targets, and `avg_factor` remain unchanged; only its four
repeated edge weights change from `q` to `q_tilde` for normal matched queries.

The implementation computes IoU and shrinkage weights in FP32 and detaches the
complete weight path. A box's four edges share one query reliability weight.

## Invariants

For every non-empty image/layer match group:

\[
\sum_k\widetilde q_{b,l,k}=\sum_k q_{b,l,k}.
\]

The method therefore redistributes rather than increases FGL weight. It also
satisfies:

\[
\widetilde q-\bar q=(1-\alpha_l)(q-\bar q),
\]

so reliability ordering is preserved and its within-group variance is reduced
by `(1-alpha_l)^2`.

Additional contracts:

- `alpha0=0` takes an explicit legacy short path and is tensor-identical to
  Clean FDR.
- Layer 5 takes the same legacy path, so the main/final FGL value and gradient
  are tensor-identical to Clean FDR.
- Empty groups contribute a finite differentiable zero.
- Singleton, constant-IoU, and all-zero groups remain unchanged.
- A zero-IoU query is rescued only when another match in the same image and
  layer makes `q_bar > 0`; the method does not claim universal zero-IoU rescue.
- Unmatched and denoising queries never enter the shrinkage mean.
- Decoder boxes, scores, references, corner logits, matcher calls, L1/GIoU,
  VFL, FDR targets, BPDD inputs, parameters, state dict, EMA, GFLOPs, and
  inference latency are unchanged.

## Configuration and boundaries

Create a dedicated YAML based on Clean FDR with one new loss option:

```yaml
fdr_loss:
  fgl_weight: 0.15
  supervise_pre_boxes: false
  supervise_dn_fdr: false
  edge_adaptive_fgl: false
  reliability_shrinkage_alpha: 0.25
```

The default in the shared criterion is `0.0`; existing YAML files therefore
retain exact legacy behavior. Values outside `[0, 1)` are rejected. LRS-FGL is
incompatible with edge-adaptive FGL in the first experiment so that it remains
the sole scientific variable.

The method does not add a scale term, support adaptation, dynamic reference,
distribution feedback, teacher, distillation, new matching, or inference-time
score calibration.

## Gate 0: pre-training falsification

Before launching Formal100, run a read-only diagnostic on the frozen training
diagnostic subset and inspect layers 0 through 4. All conditions must pass:

1. `q < 0.2` matched positives comprise at least 25% of matches in at least
   three shallow layers.
2. In those layers, the low-IoU group's original FGL weight share divided by
   its count share is at most 0.5.
3. Fewer than 50% of `q < 0.2` target edges are saturated at the FDR support
   boundary.
4. FP32 per-image/per-layer weight conservation error is within `2e-6`
   absolute and relative tolerance.
5. No validation/test metric is consulted and `alpha0` is not changed after
   seeing Gate 0.

If any condition fails, Formal100 is not launched.

## Verification gates

Unit and integration tests must prove:

- exact six-layer alpha schedule;
- finite `[0,1]` weights, conservation, low-q increase, and high-q decrease;
- singleton, constant, all-zero, and empty behavior;
- `alpha0=0` and layer 5 exact identity;
- no gradient through `q`, boxes, or matching; finite non-zero corner-logit
  gradient for a conditionally rescued zero-IoU match;
- no unmatched-query gradient and no extra matcher call;
- model boxes and scores are identical for all alpha values because LRS is
  loss-only;
- parameters/state dict and BPDD evidence are unchanged;
- finite AMP execution with shrinkage arithmetic performed in FP32.

During training record, per layer and scale group:

- counts and FGL-weight shares for `q<0.2`, `0.2<=q<0.5`, and `q>=0.5`;
- next-layer IoU change and the fraction crossing IoU 0.5;
- overlap between low-IoU matches and saturated target edges;
- transfer mass `0.5 * sum(abs(q_tilde-q)) / sum(q)`;
- FGL loss and private-head gradient norms.

Abort on non-finite values or private-head gradient p99 above twice the paired
Clean diagnostic reference.

## Experiment and decision rule

Only one new seed-0 Formal100 arm is authorized:

- Clean FDR + LRS-FGL, `alpha0=0.25`, 100 epochs;
- same dataset, initial state, seed, data order, augmentations, optimizer,
  learning-rate schedule, AMP, evaluator, and best-checkpoint rule as Clean FDR.

The historical Clean arm is a valid direct comparator only after the off-path
tests prove exact model/loss/gradient identity. Otherwise a fresh paired Clean
arm is required.

Strong paper-level GO requires every condition:

- best-val `mAP50-95 >= 0.29796`;
- the same checkpoint has `Recall >= 0.49849`;
- AP75 decreases by no more than 0.0005;
- Tiny AP does not decrease;
- epoch 91--100 mean mAP50-95 exceeds Clean over the same window;
- low-IoU next-layer IoU improves in at least three shallow transitions.

A result satisfying `0.29696 < mAP50-95 < 0.29796` with no Recall, AP75, or
Tiny-AP regression is a weak technical positive and may appear only as
supplementary evidence. Any `mAP50-95 <= 0.29696`, Recall decrease, repeated
`Precision up / Recall down / mAP down` pattern, or lack of low-IoU mechanism
improvement is a hard NO-GO.

The fixed `alpha0` and thresholds may not be changed after observing validation
results. Test is evaluated once using the val-selected checkpoint. A seed-0
result may be described only as a fixed-seed paired ablation, not as stable,
robust, universal, or statistically significant.

## Novelty boundary

The defensible contribution is the exact combination of:

1. detached matched-IoU reliability shrinkage inside D-FINE-style FGL;
2. image-local and decoder-layer-local mean preservation;
3. exact preservation of each group's total FGL weight;
4. depth decay to the untouched final-layer FGL objective;
5. a parameter-free, training-only implementation.

The paper must cite D-FINE/FGL, DEIM/MAL, GFLv2/DGQP, and related low-quality
matching work. It must not claim the first use of IoU weighting, low-quality
match optimization, progressive reweighting, or small-object Recall recovery.


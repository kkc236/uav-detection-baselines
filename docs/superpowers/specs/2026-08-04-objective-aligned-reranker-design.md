# Objective-Aligned Reranker for RT-DETR-L

Date: 2026-08-04

Status: approved under the user's delegated-autonomy instruction on 2026-08-04

## 1. Decision

The next scientific branch is an Objective-Aligned Reranker (OAR) for the mature
Ultralytics RT-DETR-L baseline. OAR does not begin with the proposed full CARP-FDR
module. It first isolates whether the failed quality probe was caused by its pointwise
soft-BCE objective rather than by absent predictive information. Set interaction is a
second, separately gated variable. FDR is excluded until it has independent detector
evidence.

The terminal research objective and the formal baseline protocol remain unchanged:

1. establish a deployable candidate that passes frozen internal and official-validation
   gates;
2. run the fixed 10% subset seed0 paired 30-epoch screen;
3. if Gate2 passes, start a fresh full-data seed0 100-epoch run;
4. publish every epoch, support exact resume, independently evaluate the final model,
   and report parameter, GFLOPs, and latency overhead.

No failed branch may be relabeled as successful, and no threshold may be weakened after
observing a result.

## 2. Evidence motivating the design

The frozen class-conditional quality oracle changed only final Query-by-class scores and
improved official-validation mAP from `0.24164844987309864` to
`0.3973619055936227`; AP75 improved from `0.23916375458831637` to
`0.3883675963852108`. The selected score was

```text
oracle_score[q,c] = sigmoid(stock_logit[q,c]) * quality[q,c] ** 2
```

where `quality[q,c]` is the maximum IoU between stock Query box `q` and a ground-truth
box of class `c`.

The subsequent learnable quality probe failed on the frozen 129-image internal split:

| Arm | mAP | AP75 |
|---|---:|---:|
| C0 stock | 0.2862886580 | 0.2923640748 |
| C1 probability/geometry control | 0.2834751724 | 0.2893655006 |
| Q hidden-aware pointwise probe | 0.2772183158 | 0.2874468303 |

Q was `-0.0090703422` mAP and `-0.0049172445` AP75 below the strongest control. This
rejects the existing `quality -> weighted soft-BCE -> score multiplication` realization.
It does not by itself reject a directly optimized candidate-ordering objective.

## 3. Adversarial audit of CARP-FDR

The full proposal is not authorized as the first implementation for five reasons.

1. The oracle target is class-conditional with shape `[300,10]`. A 300-query transformer
   producing 300 scalar scores cannot reproduce that target. Expanding naively to 3,000
   Query-by-class tokens materially changes attention cost.
2. The oracle gain combines class-presence suppression, background suppression,
   localization ordering, and duplicate-candidate competition. It cannot be attributed
   entirely to localization ordering without decomposition.
3. Pairwise or listwise supervision aligns the objective but cannot create information
   that is absent from logits, geometry, hidden state, and candidate context.
4. The 518 training images, not the number of correlated Query pairs, determine the
   effective sample scale.
5. FDR evidence is not part of the frozen mature baseline cache and has no passed paired
   detector screen. Combining FDR and set reranking would destroy single-variable
   attribution.

## 4. Frozen authority

- Detector: mature Ultralytics RT-DETR-L baseline checkpoint with SHA-256
  `54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`.
- Dataset SHA-256:
  `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`.
- Fixed 647-image subset SHA-256:
  `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`.
- Existing split: 518 probe-train images and 129 internal-development images with hashes
  `1E46817FFFBDBCBA0BA1675CA6142ABABBD6147394AA1D0F10B57F0ECAF7236D`
  and `FCF8749BAADBA8BDDF5870F472BDE1E937156AFBCEEFDA9F96FED21FA6BB0514`.
- Detector output: final decoder stock boxes `[B,300,4]`, class logits `[B,300,10]`,
  and hidden state `[B,300,256]`.
- Evaluation: image size 640, confidence 0.001, `max_det=300`, `NMS=False`, exact global
  flattened Query-by-class Top-300, and the existing metric implementation.
- The detector is frozen during offline probes. Boxes, logits, hidden states, matching,
  and detector parameters are unchanged.
- The official 548-image validation split remains inaccessible until an immutable
  internal passing decision exists.

The filename-prefix overlap found in the old 518/129 split is recorded as a risk, not
silently treated as independent evidence. Before training, the runner reports overlap
under both the first underscore-delimited prefix and the prefix before `_d_`. This audit
does not alter the already frozen primary split. If original dataset metadata proves
that either prefix is a source-sequence identity, a separately frozen group-disjoint
replication is required before official validation.

## 5. Stage D0: oracle-source and candidate-coverage audit

D0 is diagnostic-only and uses the existing frozen evidence. It evaluates the following
score families on internal development without training:

```text
C0:            p[q,c]
O-presence:    p[q,c] * 1[class c exists in the image]
O-query-IoU:   p[q,c] * max_i IoU(box[q], gt[i]) ** 2
O-same-class:  p[q,c] * max_{i:class[i]=c} IoU(box[q], gt[i]) ** 2
```

These oracles are not assumed to be additive. Their purpose is to determine which
information source creates the observed upper bound.

The current same-class oracle is also recomputed after restricting modifiable candidates
to the stock top-K Queries per class for

```text
K in {20, 40, 60, 100}
```

Pairs outside the pool retain their exact stock scores. Select the smallest K recovering
at least 90% of the full same-class-oracle internal mAP gain. If none reaches 90%, the
class-wise sparse OAR route is scientifically inadequate and stops before training.
The selected K and every D0 metric are frozen before OAR training.

## 6. Shared candidate representation

The trainable unit is a Query-by-class pair `(q,c)`, not a scalar Query. For each pair,
reuse the frozen 276-value Q-probe representation:

```text
stock probability and query entropy                         2
cx, cy, w, h, log(w), log(h), area, aspect                  8
candidate-class one-hot                                    10
final decoder hidden                                      256
                                                           ---
total                                                      276
```

Inputs are detached float32 tensors. The stock probability used in the final score is
never replaced inside the detector loss.

## 7. OAR-R: objective-only ranker

OAR-R is the first trainable candidate. It has no attention and no candidate interaction:

```text
Linear(276,64) -> SiLU -> Linear(64,1)
```

The final layer is initialized to exact zeros. Its raw output `a[q,c]` is converted to a
bounded residual class logit

```text
r[q,c] = 2 * tanh(a[q,c] / 2)
```

so `r` is in `[-2,2]`. Inference uses

```text
reranked_probability[q,c] = sigmoid(stock_logit[q,c] + r[q,c])
```

so epoch-zero output is exactly stock. Only the frozen top-K-per-class pool receives a
residual; every pair outside it keeps the exact stock probability.

### 7.1 Teacher order

For training only, compute detached teacher utility

```text
u[q,c] = p[q,c] * quality[q,c] ** 2
```

and student utility

```text
s[q,c] = sigmoid(stock_logit[q,c] + r[q,c])
```

Pairs with equal teacher utility within float32 equality are ties and create no ordering
loss.

### 7.2 Top-boundary RankNet loss

For each image, form the teacher Top-300 and stock Top-300 within the complete 3,000-pair
space. Training pairs consist of:

1. teacher-Top-300 candidates omitted by stock paired against stock-Top-300 candidates
   omitted by the teacher; enumerate the Cartesian product in
   `(teacher_rank, stock_rank, preferred_flat_index, other_flat_index)` order and keep
   its first 2,048 non-tied pairs;
2. deterministic adjacent pairs in teacher order within teacher Top-300;
3. pair teacher ranks 1 through 300 with teacher ranks 301 through 600 at the same
   zero-based offset when their utilities are not tied.

Discard pairs for which neither member belongs to the selected top-K-per-class pool.
The resulting per-image cap is therefore `2048 + 299 + 300 = 2647` pairs. Duplicate
index pairs are removed while preserving their first occurrence.

For a teacher-preferred pair `(i,j)`, use

```text
L_rank(i,j) = softplus(-(logit(s_i) - logit(s_j)))
```

weighted by the detached teacher-utility gap and normalized per image. Sampling order,
tie handling, pair caps, and normalization are fixed in source. No BCE, IoU regression,
detector loss, or official-validation metric enters OAR-R selection.

The old Q-BCE result is retained as historical evidence and is not rerun merely to seek
a better checkpoint.

## 8. OAR-S: separately gated set interaction

OAR-S is permitted only when OAR-R improves both internal mAP and AP75 over C0 but does
not reach the final `+0.0050` mAP Gate. If either OAR-R delta is non-positive, the
available individual candidate features have not produced a useful ranking signal and
OAR-S is not authorized by this design.

OAR-S uses the same selected top-K Query candidates independently for each class. A
shared one-layer, four-head, width-64 self-attention block processes each class set.
Input projection, attention, FFN, and output residual are shared across classes; a class
embedding preserves class identity. The output residual uses the same stock-preserving
formula and zero initialization as OAR-R. Pairs outside the pool remain untouched.

Class-wise attention costs `10*K^2` interactions rather than global attention over 3,000
pairs. OAR-S receives no FDR distribution, image feature map, P2/P3 feature, boundary
evidence, trajectory signal, GT at inference, or NMS.

OAR-S can be attributed to set interaction only if it passes the final C0 Gate and is
strictly better than OAR-R in both mAP and AP75.

## 9. Internal selection and frozen Gates

All trainable arms use seed0, float32, deterministic execution, the same 518-image order,
20 complete offline epochs, and the already pinned MuSGD settings. Checkpoints are
selected lexicographically by

```text
(mAP50-95, AP75, AP50, -epoch)
```

on the frozen 129-image internal split.

An arm passes the final internal Gate only when

```text
arm.map  - C0.map  >= 0.0050
arm.ap75 - C0.ap75 >  0
```

OAR-S additionally requires

```text
OAR-S.map  > OAR-R.map
OAR-S.ap75 > OAR-R.ap75
```

Arithmetic uses `Decimal(str(value))`. NaN, infinity, cache drift, detector mutation,
incomplete epochs, or stock reconstruction mismatch is engineering-invalid. Scientific
failure cannot be repaired by changing K, pair sampling, loss weights, model width,
epochs, split, or thresholds after seeing the result.

## 10. One-shot official validation

Only the first arm with a passing immutable internal decision is evaluated once on the
official 548-image validation split alongside C0 from the same frozen detector tensors.
C0 must reproduce the authority metrics. The candidate proceeds only when both official
conditions are strict:

```text
candidate.map  > C0.map
candidate.ap75 > C0.ap75
```

Official validation does not select checkpoints or hyperparameters. Failure freezes the
OAR branch and forbids another official pass.

## 11. Integrated 30-epoch detector screen

An official-positive offline candidate is integrated only at the last decoder layer.

- Stock RT-DETR boxes, class logits, matcher, VFL/L1/GIoU losses, encoder, decoder,
  Query count, and Top-300 semantics remain unchanged.
- OAR inputs are detached, so OAR loss cannot update detector parameters.
- Stock detector loss is computed exactly as baseline.
- OAR has its own isolated ranking loss and private optimizer parameter group.
- The OAR output affects only evaluation/inference scores.
- OAR's final residual layer is zero-initialized, so the initial model output equals
  control.

Control and OAR use the same public parameter initialization, sample order, augmentations,
checkpoint/resume rules, and the previously frozen baseline parameters: Ultralytics
8.4.90, RT-DETR-L, pretrained false, image size 640, batch 8, workers 8, MuSGD,
`lr0=0.01`, `lrf=0.01`, momentum 0.937, weight decay 0.0005, AMP true with fixed scale
128, seed0, deterministic true, query count 300, `max_det=300`, and NMS false.

The paired screen is exactly 30 epochs on the fixed 647-image subset. Every epoch writes
and uploads metrics, checkpoint hashes, protocol hashes, environment, GPU utilization,
and resume authority. The candidate must exceed the paired control in mAP and AP75 under
the already frozen Gate2 decision; thresholds are not redefined here.

## 12. Full-data 100-epoch run

Gate2 passage authorizes a fresh, from-zero, full 6,471-image seed0 control/OAR protocol.
It does not resume the screen checkpoint. The formal method runs 100 epochs with the same
public initialization and training parameters. Every epoch is published and resumable.
The final report includes best and final checkpoints, tail means, AP50, AP75, AP-small,
AP-tiny, precision, recall, parameter count, GFLOPs, end-to-end latency, environment,
dataset hashes, and an independent evaluation rerun.

## 13. Efficiency limits

The preferred target remains less than 1% parameter and GFLOPs growth and less than 3%
end-to-end latency growth. The user permits a larger cost if necessary, but any candidate
exceeding these preferred limits must earn a materially larger accuracy gain and report
the exact trade-off before full-data training.

## 14. State machine and failure policy

```text
server identity and environment audit
-> frozen cache verification or exact re-extraction
-> D0 oracle decomposition and top-K coverage
-> OAR-R 20-epoch offline probe
-> OAR-R internal decision
-> optional OAR-S 20-epoch probe only under its authorization condition
-> first passing arm one-shot official validation
-> paired fixed-subset seed0 30-epoch screen
-> Gate2
-> fresh full-data seed0 100 epochs
-> independent evaluation and overhead audit
```

Engineering failures are fixed with tests and resumed only from verified immutable
artifacts or a new run root. Scientific failures are frozen and published. The server
must not be kept busy with a branch whose prerequisite Gate has failed.

## 15. Test strategy

Tests must lock:

- exact reconstruction of C0 and epoch-zero OAR output;
- oracle decomposition formulas and class-empty behavior;
- top-K-per-class coverage, outside-pool score identity, and deterministic K selection;
- exact Query-by-class teacher utility and flattened Top-300 boundaries;
- tie exclusion, hard-pair construction, deterministic pair caps, RankNet math, and
  per-image normalization;
- detached detector evidence and zero detector gradients;
- OAR-R and OAR-S parameter isolation and zero initialization;
- class-wise attention masks and absence of cross-image leakage;
- 518/129 identities, prefix-overlap report, official-validation lockout, and one-shot
  release;
- frozen internal and official Gate boundaries;
- paired initialization, MuSGD/AMP authority, 30-epoch publication/resume, fresh
  full-data launch, and independent final evaluation;
- parameter, GFLOPs, and latency measurement without hidden preprocessing costs.

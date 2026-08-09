# PFCR v1: Protected Frequency Candidate Rescue

**Date:** 2026-08-09  
**Status:** approved for offline learnability execution  
**Role:** candidate design for the third thesis contribution beside FDR

## 1. Objective

PFCR tests whether the complementary candidates proven by the frozen FDR/FrequencyCM
oracle can be selected by a deployable learned gate without modifying the mature FDR
predictions. The immediate deliverable is an offline learnability decision using the two
completed epoch-100 checkpoints. A passing result authorizes a later shared-network YAML
module and paired Screen30; it is not itself the final efficient detector.

The completed authority is:

- FDR stock mAP: `0.289659749097641`;
- FrequencyCM stock mAP: `0.28617480114898347`;
- union candidate-oracle gain over the better individual oracle: `+0.0558781448012044`;
- tiny/small one-to-one recall@0.50 gain: `+0.05736667964188402`;
- oracle decision: `green`.

These numbers prove candidate complementarity but use ground truth. PFCR must predict the
rescue decision without ground truth.

## 2. Non-negotiable isolation

1. The FDR checkpoint, boxes, logits, ordering, and stock Top-300 are detached and frozen.
2. PFCR never changes an FDR box or FDR class logit.
3. FrequencyCM is a side candidate source only. It never writes features into the FDR
   backbone, encoder, decoder, or FDR head in this probe.
4. Inference uses no labels, target counts, target scales, IoU-to-target values, or oracle
   assignments.
5. `NMS=False`, `max_det=300`, the ten-class mapping, image preprocessing, and the project
   independent evaluator remain unchanged.
6. Turning rescue slots to zero must reproduce FDR stock bit-for-bit.
7. The completed FrequencyCM-v1 negative result and complementarity oracle are immutable.

## 3. Data separation

The two frozen detectors are run once on all 6,471 VisDrone training images. The paired
cache stores raw final-decoder `[300,4]` boxes and `[300,10]` logits for both models plus
training targets and geometry. Cache authority contains checkpoint, dataset, evaluator,
source, schema, and payload SHA-256 values.

Training images are split deterministically from their normalized image identifiers:

```text
sha256(image_id) interpreted as an integer modulo 5
0       -> internal development
1..4    -> gate training
```

No official validation image participates in gate fitting, epoch selection, rescue-slot
selection, feature normalization, or early stopping. The existing 548-image paired val
cache is opened only after the internal Gate passes. Because the earlier oracle already
used official val for design selection, this result remains design-selection evidence;
the final thesis claim still requires a fresh formal run or independent evidence.

## 4. Candidate representation

FDR stock is reconstructed from all 3,000 query-class pairs using the exact Ultralytics
flattened Top-300 implementation. Each of the 3,000 FrequencyCM query-class pairs is a
potential rescue candidate.

For each FrequencyCM pair `(query, class)`, PFCR finds the FDR query maximizing
`box_IoU * FDR_class_probability`, with stable query-index tie breaking. It then builds a
35-value detached feature vector:

- FrequencyCM class logit/probability, query maximum probability, top-two margin,
  normalized entropy, and normalized flattened rank: 6;
- FrequencyCM `cx,cy,w,h,log(area),log(aspect)`: 6;
- matched-FDR class logit/probability, query maximum probability, top-two margin,
  normalized entropy, and normalized flattened rank: 6;
- cross-model box IoU, center deltas, log width/height ratios, class-score difference,
  and query-maximum-score difference: 7;
- one-hot class identity: 10.

All divisions and logarithms use fixed numerical guards. Non-finite evidence is an
engineering failure, not silently replaced.

## 5. Gate and score path

The pointwise gate is deliberately small:

```text
35 -> Linear(64) -> SiLU -> Linear(32) -> SiLU -> Linear(1)
```

The final layer is zero initialized. Its output is a bounded FrequencyCM-only logit
residual:

```text
r = 2 * tanh(raw / 2)
adjusted_cm_logit = detached_cm_logit + r
```

FDR logits remain byte-identical. Gate parameters are the only optimized parameters.

## 6. Protected merge

For a rescue budget `R`, PFCR preserves the first `300-R` FDR stock candidates exactly.
The remaining `R` positions are selected from:

- the original FDR stock tail of length `R`; and
- all adjusted FrequencyCM query-class candidates.

Selection uses descending score with deterministic source/query/class tie breaking. No
NMS or IoU suppression is introduced. Training one-to-one utility makes duplicate
FrequencyCM candidates negatives rather than relying on post-hoc suppression.

The frozen internal rescue-budget grid is `{15, 30, 60}`. It changes only how many low
rank FDR positions may be contested; the gate is trained once. The chosen budget is the
smallest budget within `0.0002` mAP of the best internal-development result, with ties
broken by AP75, AP50, then smaller budget.

## 7. Training teacher and loss

Ground truth is used only while fitting the gate. On each training image, the union of
FDR and FrequencyCM candidates receives deterministic same-class one-to-one assignment
that maximizes total IoU. Assigned candidates receive

```text
teacher_utility = candidate_probability * matched_IoU^2
```

and unassigned duplicates/background candidates receive zero. Only pairs that can change
the protected Top-300 boundary are trained. The loss is:

```text
L = weighted_boundary_RankNet + 0.25 * one_to_one_quality_BCE
```

Teacher values, detector tensors, and features are detached. The gate trains for 20
epochs with fixed seed 0 and deterministic algorithms. Optimizer and learning rate are
frozen in the implementation plan; they are probe-only choices and do not alter the
formal detector protocol. Every epoch writes a create-only checkpoint and metrics row.

## 8. Controls and decisions

Internal development evaluates:

- `C0`: exact FDR stock;
- `C1`: protected union using unmodified FrequencyCM stock scores;
- `PFCR`: protected union using learned FrequencyCM residuals.

The internal Gate passes only when the selected PFCR checkpoint satisfies all conditions:

```text
PFCR.mAP - max(C0.mAP, C1.mAP) >= 0.0020
PFCR.AP75 > max(C0.AP75, C1.AP75)
PFCR.AP50 > C0.AP50
PFCR tiny/small recall@0.50 >= C0 tiny/small recall@0.50
```

If the pointwise gate improves both mAP and AP75 over both controls but misses only the
`+0.0020` buffer, one pre-authorized contextual variant may add a single 64-wide,
four-head self-attention layer over the 300 FrequencyCM queries. Otherwise the branch is
frozen as `scientific_failed`; no validation tuning is allowed.

After an internal pass, official val is evaluated once. Detector integration is eligible
only when:

```text
PFCR.mAP > FDR.mAP
PFCR.AP75 >= FDR.AP75
```

AP50, precision, recall, AP-tiny, AP-small, per-class AP, parameters, and offline gate
latency are reported but do not replace these conditions.

## 9. Failure handling and evidence

Engineering failures are repaired test-first and resume from verified immutable state.
Scientific failure never changes the frozen official threshold or reopens val. Reports
include cache manifests, split hashes, feature schema, every checkpoint hash, internal
selection, official one-shot decision, evaluator identity, and SHA256SUMS. GitHub Git
transport failure must not stop computation; evidence is queued locally and published by
the verified Release API path.

## 10. Interpretation boundary

A passing offline PFCR result proves that complementary FrequencyCM candidates are
learnably selectable. It does not yet prove a lightweight shared-network detector,
because this probe consumes two frozen detector outputs. The next authorized stage is a
YAML-pluggable shared-backbone side head that reproduces the learned rescue behavior,
followed by paired Screen30 and, only after passing, a fresh formal100 run.

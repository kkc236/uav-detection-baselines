# BQP-DETR: Fixed-Budget Boundary Query Promotion

Date: 2026-07-23

## 1. Goal

BQP-DETR targets one failure mode in stock RT-DETR-L on VisDrone: under the fixed
300-query budget, high-scoring complex-background candidates can remain inside
Top-300 while localization-valid tiny-object candidates remain immediately below
the cutoff and never reach the decoder.

The method introduces one training-only pairwise ranking loss on the existing
encoder classification head. It does not add queries, a P2 branch, a new prediction
head, a new decoder path, or inference-time computation.

The first deliverable is not training. It is a zero-training feasibility audit,
G0, that determines whether the required boundary pairs exist often enough to
justify implementing and training BQP.

## 2. Evidence and lessons incorporated

The design is constrained by the recorded failures:

- Direct P2 query injection produced negative net replacement value:
  `V_replace=-231` for A1 and `V_replace=-499` for A2.
- QG-P2 reduced replacement damage but failed its target mechanism:
  AP-tiny and tiny recall were zero, quality-IoU Spearman was `-0.114`, and about
  99.6% of injected queries were not matched to the intended tiny targets.
- A1-no-injection contained a global-gradient-clipping confound and therefore
  cannot be treated as evidence for P2 auxiliary supervision.
- CPU copies of roughly 33 million parameters at every optimizer step caused a
  major engineering slowdown. BQP diagnostics must remain GPU-vectorized and
  must not copy model parameters per batch or step.
- Short-run aggregate mAP can be positive while the claimed mechanism is
  negative. Every BQP stage therefore has separate mechanism and efficacy gates.

Consequently, BQP operates only on stock P3/P4/P5 encoder candidates and the
existing encoder score head. All P2 injection, background subtraction, learned
quality heads, dynamic query counts, fixed tiny quotas, and decoder changes are
out of scope.

## 3. Research hypothesis

Let the stock encoder rank all candidates by its original selection score, with
rank 1 being highest. For a tiny ground-truth object missed by stock Top-300, BQP
asks whether:

1. a same-class, localization-valid stock candidate exists in ranks 301-420; and
2. a clear hard-background stock candidate exists in ranks 251-300.

If such pairs are frequent, the existing encoder score head may be trained to
move the missed tiny candidate above the incorrectly retained background
candidate without changing the number or origin of decoder queries.

This is a boundary-ranking hypothesis, not a general claim that more candidates
or higher-resolution features always improve tiny-object detection.

## 4. Frozen scope and definitions

### 4.1 Base detector

- Detector: stock Ultralytics RT-DETR-L.
- Dataset: VisDrone with 10 detection classes.
- Input size: 640.
- Final query budget: exactly 300.
- Candidate sources: stock P3/P4/P5 encoder candidates only.
- Inference path: original encoder scores, original Top-300, original decoder.
- G0 and later comparisons must record repository commit, checkpoint SHA256,
  dataset signatures, software fingerprint, GPU model, and configuration.

### 4.2 Tiny ground truth

After the exact letterbox transformation used by the detector, a ground-truth box
is tiny when its normalized area is strictly less than:

```text
(32 / 640)^2 = 0.0025
```

Boxes marked ignored by VisDrone are not detection ground truth. They are retained
only to prevent ignored regions from being mined as background negatives.

### 4.3 NWD localization similarity

For two normalized `xywh` boxes, use the existing NWD implementation with distance
constant:

```text
C = 12.8 / 640 = 0.02
```

A candidate covers a ground truth when:

- its evaluated class equals the ground-truth class; and
- NWD similarity is at least `0.50`.

NWD is used only to construct and audit training pairs. It is not added to
Hungarian matching, the main box loss, validation AP, or inference ranking.

### 4.4 Stock ranking

For every encoder candidate `i`, let:

```text
s_i = max_c sigmoid(z_i,c)
c_i = argmax_c z_i,c
```

The candidate order must exactly reproduce the stock RT-DETR encoder Top-K
selection, including its implementation's deterministic tie behavior. G0 is
invalid if the intercepted Top-300 indices differ from those passed to the stock
decoder.

Ranks are one-based in reports:

- selected boundary insiders: ranks 251-300;
- missed-candidate search band: ranks 301-420.

The rank bands are frozen for G0 and the first BQP screen. They must not be changed
after seeing G0 or training results.

## 5. G0 zero-training feasibility audit

### 5.1 Data and checkpoint

Primary G0 uses the already frozen 10% VisDrone training subset used by the D2
screening protocol. It must not use the validation split for method selection.
The subset image list and hash must be copied into the G0 manifest.

G0 requires a mature stock-only RT-DETR-L checkpoint. Selection priority is:

1. the SHA256-verified matched 100-epoch RT-DETR-L baseline checkpoint from the
   existing evidence/release assets;
2. another SHA256-verified 100-epoch stock RT-DETR-L checkpoint with the same data,
   input size, and class mapping, marked exploratory if its hardware or training
   protocol differs.

An immature 10-epoch screening checkpoint cannot pass G0. If no mature stock
checkpoint is available, G0 stops with `G0_BLOCKED_NO_MATURE_CHECKPOINT`.

The validation split remains untouched during G0. If the 10% subset passes, the
full 6471-image training split may be run once as a confirmation using the same
frozen thresholds.

### 5.2 Positive candidate mining

For each tiny ground-truth box `g`:

1. Determine whether any stock Top-300 candidate covers `g`.
2. If yes, `g` is already covered and is excluded from the missed-tiny denominator.
3. If no, search ranks 301-420 for candidates whose evaluated class equals the
   class of `g` and whose NWD similarity to `g` is at least 0.50.
4. If multiple candidates qualify, choose the candidate with maximum NWD;
   break an exact NWD tie by higher stock score, then lower original index.

The selected candidate is `i+`. A missed tiny ground truth without `i+` is
unpromotable under the frozen search band.

### 5.3 Negative candidate mining

Search ranks 251-300 for a candidate that is a clear hard-background insider:

- its maximum NWD similarity to every non-ignored ground-truth box, regardless of
  class, is below `0.10`; and
- its intersection-over-area with every VisDrone ignored region is below `0.50`.

Choose the qualifying candidate with the highest stock score; break a tie by lower
original index. The selected candidate is `i-`.

Within one image, a candidate may be assigned to only one pair. Pair construction
is greedy in descending score of `i+`, so the resulting oracle replacement count
cannot be inflated by reusing one candidate for many targets.

### 5.4 G0 outputs

G0 writes one compact JSON summary, one immutable manifest, and one per-image
JSONL record. It does not save encoder feature tensors or model copies.

Required aggregate fields:

- number of images and all/tiny ground truths;
- number and rate of tiny ground truths already covered by Top-300;
- missed-tiny count `M`;
- missed tiny objects with a qualifying `i+`;
- missed tiny objects with both `i+` and `i-`, denoted `P`;
- promotable ratio `P/M`;
- `i+` rank histogram and median rank;
- number of eligible hard-background insiders;
- class-wise and tiny-area-quartile breakdowns;
- unique oracle swaps, gained tiny coverage, lost existing coverage, and net
  replacement value;
- exact stock Top-300 reproduction status;
- runtime, peak GPU memory, and peak process RAM.

For auditability, the per-image record stores only image identifier, ground-truth
identifier/class/box, selected candidate indices/ranks/scores/boxes, NWD values,
and exclusion reason. It must not contain credentials or full feature tensors.

### 5.5 G0 gate

G0 passes only if all conditions hold:

1. stock Top-300 indices are reproduced exactly for every audited batch;
2. all recorded tensors and statistics are finite;
3. `M >= 200`;
4. `P >= 60`;
5. `P / M >= 0.30`;
6. unique oracle swaps have strictly positive net tiny-coverage replacement value;
7. no ignored-region candidate is used as `i-`.

If conditions 3-5 fail, the fixed 301-420 boundary does not contain enough
recoverable tiny candidates and BQP is stopped. The bands and thresholds are not
expanded in response to failure.

If only condition 6 fails, the proposed one-for-one boundary promotion is unsafe
under the fixed budget and BQP is stopped.

G0 is a feasibility gate, not evidence that AP will improve.

## 6. BQP training mechanism after G0

For each valid pair `(i+, i-)`, use the existing encoder class logits:

- `z+ = z_i+,c(g)`, the positive candidate logit for the tiny target's class;
- `z- = max_c z_i-,c`, the negative insider's stock selection logit.

The only new objective is:

```text
L_BQP = mean softplus(m + z- - z+)
```

with frozen first-screen values:

```text
margin m = 0.20
loss weight lambda_BQP = 0.02
```

The total training loss is:

```text
L = L_stock + lambda_BQP * L_BQP
```

No-pair batches contribute an exact zero BQP loss and continue normally.

The BQP branch receives detached encoder candidate features and box coordinates.
Its gradient is allowed to update only the existing encoder score head parameters.
It must not update the backbone, neck, encoder features, box head, decoder,
Hungarian matcher, denoising path, or any added head.

To avoid repeating the A1-no-injection confound, BQP and stock gradients must be
audited separately before aggregation. Any clipping policy must prove that adding
a zero or isolated BQP branch cannot rescale the stock gradient. A failed isolation
check invalidates the run.

## 7. Execution stages after G0

### 7.1 E0 isolation preflight

Before any epoch-level screen:

- confirm stock and BQP models have identical initial stock forward outputs;
- confirm BQP-off and `lambda_BQP=0` produce identical gradients and one-step
  updates to stock control;
- confirm non-score-head parameters receive exactly zero BQP-only gradient;
- confirm the stock Top-300 query count remains 300;
- confirm AMP scale/skip behavior and gradient clipping are identical in the
  zero-BQP control;
- confirm eval/export outputs are bitwise equal when BQP is disabled or removed.

Any failure stops the experiment.

### 7.2 G1 paired mature-checkpoint screen

Starting from the same mature stock checkpoint:

- control arm: stock fine-tuning;
- treatment arm: stock fine-tuning plus BQP;
- 10-15 epochs;
- seeds 0, 1, and 2;
- identical data order, augmentation, optimizer, AMP, and validation protocol.

The screen passes only if:

- at least 2/3 seeds improve stock Top-300 tiny coverage;
- at least 2/3 seeds improve AP-tiny;
- mean AP-tiny improves by at least 0.5 absolute percentage points;
- mean total mAP50-95 changes by no less than -0.1 absolute percentage points;
- no seed collapses numerically or loses more than 0.3 total mAP;
- the observed rank movement is concentrated in the audited boundary rather than
  being explained by a broad score explosion.

G1 uses no weight or margin search. Failure does not trigger tuning.

### 7.3 Formal training

Fresh scratch 100-epoch training is permitted only after G0, E0, and G1 pass.
Formal control and BQP runs use the frozen matched protocol and seeds 0, 1, and 2.
Results are reported as mean and standard deviation.

The minimum publishable target is:

- significant improvement in stock Top-300 tiny coverage and AP-tiny;
- total mAP50-95 approximately preserved or improved;
- exactly 300 stock queries;
- no inference-time parameters, FLOPs, or latency added after exporting away the
  training-only loss path.

## 8. Runtime and storage budget

Historical evidence on the same detector/GPU reports 548-image VisDrone validation
in about 5.1 seconds of steady-state iteration time. G0 performs additional
candidate interception, vectorized NWD matrices, pairing, and audit writes.

Expected wall time after the G0 script is implemented and deployed:

- frozen 10% training subset: 2-5 minutes warm, 5-10 minutes including cold start,
  cache creation, and self-checks;
- full 6471-image training split confirmation: 15-30 minutes;
- output storage: below 100 MB unless optional debug examples are explicitly
  enabled.

G0 must use batched GPU tensor operations. It must not copy model parameters to
CPU per batch. The expanded 100 GB server capacity is sufficient; GPU throughput
and data loading, not RAM capacity, determine runtime.

## 9. Failure handling

- Missing or hash-mismatched dataset/checkpoint: stop before inference.
- Stock Top-300 mismatch: stop and fix instrumentation; do not interpret results.
- CUDA OOM: reduce G0 analysis batch size only. Candidate definitions and results
  must be batch-size invariant and verified on a fixed sample.
- Non-finite tensor: record the image/batch identifier and stop.
- Insufficient pairs or negative oracle replacement: archive G0 as a valid negative
  result and abandon BQP without threshold changes.
- Server interruption: G0 may restart from the beginning because it is short and
  has no optimizer state.

## 10. Paper positioning

BQP combines three established ideas at the level of motivation:

- RT-DETR: encoder confidence ranking and fixed Top-K query initialization;
- NWD: stable localization similarity for tiny boxes;
- EASE-DETR-style query competition: detection queries can compete and suppress
  useful alternatives.

The claimed contribution is narrower:

> Under a fixed Top-K cutoff, mine a missed tiny stock candidate immediately
> outside the budget and a clear background stock candidate immediately inside
> the budget, then supervise the original encoder score head with one pairwise
> boundary-promotion loss.

DQ-DETR, Dome-DETR, NRQO, RT-DETRv3, Focus-DETR, DDQ, and SQ-DETR remain related
work unless a later verified implementation explicitly imports one of their
modules. They must not be described as components of BQP.

Before submission, the exact boundary-pair construction must receive a fresh
novelty search against current DETR query-selection and ranking literature. G0 or
positive experimental results do not by themselves establish novelty.

## 11. Explicit non-goals

BQP does not:

- guarantee a 70% probability of AP improvement;
- add P2 candidates or a P2 auxiliary branch;
- alter Hungarian matching or the main NWD/IoU loss;
- learn a new quality head;
- reserve a tiny-object query quota;
- change the number of decoder queries;
- rerank candidates with NWD at inference;
- change decoder supervision or box refinement;
- search thresholds after observing results.

The defensible probability statement is conditional: if G0, E0, and the frozen
three-seed G1 gates pass, the engineering confidence for entering a formal C-class
conference experiment is materially higher than before G0. It is not a statistical
guarantee of acceptance or metric improvement.

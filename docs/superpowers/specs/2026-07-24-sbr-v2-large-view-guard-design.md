# SBR-V2 Large-View Guard Design

**Status:** proposed post-hoc revision after the frozen SBR G0-A failure  
**Date:** 2026-07-24  
**Scope:** zero-training, offline diagnosis and one predeclared fusion revision  

## 1. Decision Context

The original SBR G0-A remains permanently recorded as `SBR_G0A_FAIL`.
Independent adjudication verified the following `C-A` deltas on the 548-image
VisDrone validation split:

- `AP-tiny-SBR`: `+0.0400347711`
- `mAP50-95`: `+0.0424115098`
- `tiny_recall`: `+0.1001918253`
- `AP75`: `+0.0432520654`
- `AP-large-SBR`: `-0.0317937425`

The large-object gate required at least `-0.005`, so the failed arm must
recover at least `0.0267937425` AP-large without losing the other four gates.
The original gate, evidence, and hashes are immutable. SBR-V2 is a new,
validation-inspired hypothesis and must never be described as part of the
original preregistered G0.

Existing diagnostic arms show:

- Arm A large AP: `0.145847`
- Arm B large AP: `0.006618`
- Arm C large AP: `0.114053`
- Arm D large AP: `0.114053`
- Arm F large AP: `0.001195`

This supports, but does not prove, a scale-conflict hypothesis: local views are
useful for tiny objects but cannot represent many large objects completely;
IoS clustering may absorb partial local boxes into full-view boxes and move
the fused coordinates.

## 2. Research Question and Falsifiable Hypothesis

The only V2 hypothesis is:

> Most large-object A-to-C true-positive losses are caused by a mixed
> full-view/local-view cluster changing a valid full-view box into a fused box
> that no longer matches the same target.

The hypothesis is eligible for implementation only when the frozen offline
audit attributes at least 60% of unique large-object `A TP -> C FN` events at
IoU `0.75` to that mechanism. A denominator of zero fails the hypothesis.
The 10-threshold pooled result is reported separately as a repeated-measures
diagnostic and is not interpreted as a set of independent samples.

No tile ratio, IoS, confidence, max-det, query count, fusion weight, or size
boundary may be searched. No ground-truth property may enter inference-time
routing.

## 3. Frozen Offline Causal Audit

### 3.1 Inputs

The audit consumes only the artifact URIs resolved from an immutable input
manifest. The production manifest currently resolves to:

- `/mnt/uav/evidence/sbr-g0a-51ee6c44/g0_manifest.json`
- `raw_views.jsonl.gz`
- `arm_predictions.jsonl.gz`
- `g0_metrics.json`, `g0_deltas.json`, and checksums
- the exact VisDrone validation labels and ignore labels identified by the
  recorded dataset signature

Before analysis, all evidence checksums and the checkpoint, dataset, protocol,
and source hashes must verify. Hard-coded server paths are forbidden in the
audit implementation. The audit must not run inference or modify the original
evidence directory.

### 3.2 Large targets and matching

A ground-truth target is large when

`sqrt(width * height) * min(640 / W, 640 / H, 1) > 96`.

This is the original frozen SBR large bin. Matching is class-aware and uses the
same deterministic score/source/query order, confidence `0.001`, ignore
neutralization, and final `max_det=300` as the original evaluator.

The primary audit population uses IoU `0.75`. Each `(image, large GT index)` is
one unique event. A secondary repeated-measures table is pooled over IoU
thresholds `0.50:0.05:0.95`, where each
`(image, large GT index, IoU threshold)` is one event. An `A TP -> C FN` event
exists when Arm A matches the target and Arm C does not at the same threshold.

An Arm-A full raw detection maps to Arm C by the immutable key
`(image_id, class_id, source_order=0, query_index)`. Its score,
`network_xyxy`, `view_xyxy`, and `global_xyxy` must be byte-equivalent after
canonical float serialization; absence, collision, or disagreement is an
artifact-integrity failure. The Arm-C raw-record index then maps to exactly one
stored cluster-member list. No nearest-box or IoU-based mapping is allowed.
`image_id` is the exact UTF-8 relative POSIX path stored in the input manifest;
the audit performs no case, Unicode, or separator normalization.

### 3.3 Exclusive failure attribution

All matching and counterfactual calculations use float64 without rounding.
Image width and height come from the checksum-verified dataset record in the
input manifest. Prediction ordering is exactly
`(-score, source_order, query_index, original_index)`.

Each `A TP -> C FN` event is assigned to the first applicable category:

1. **Mixed-cluster localization loss:** the matching Arm-A full-view raw
   detection exists in the corresponding C mixed cluster; the standard fused
   box fails the target; replacing only that cluster's coordinates with the
   highest-score full member and rerunning the complete deterministic
   one-to-one matching for the image makes this target a TP. All other
   predictions and scores are held fixed.
2. **Final-300 truncation:** a matching fused cluster exists before the final
   cap but is absent after applying the same
   `(-score, source_order, query_index, original_index)` ordering and
   `max_det=300`.
3. **Matching competition:** a candidate can match the target but deterministic
   one-to-one matching assigns it elsewhere.
4. **Class or candidate loss:** no same-class candidate can match the target.
5. **Other:** none of the above.

The audit reports counts and fractions for every category at AP75, at every
other IoU threshold, and pooled. It also reports the unique-large-GT macro
share, full-to-fused center shift, width ratio, height ratio, area ratio,
cluster source composition, and final-cap rank. The canonical per-event
attribution table is hashed. Its schema has an explicit version string, and
the canonical schema JSON is hashed separately.

The implementation gate is:

`AP75_unique_mixed_cluster_localization_events / AP75_unique_A_TP_to_C_FN_events >= 0.60`.

The audit additionally computes a recoverable upper bound by replacing every
qualifying V2 cluster with its full anchor while keeping scores, ordering,
matching, and the final cap fixed. If the resulting AP-large cannot reach
`A AP-large - 0.005`, the route stops before train-fold inference. The 60%
mechanism gate does not imply this required `0.0267937425` recovery; both
conditions must independently pass.

This audit is descriptive/post-hoc evidence and is not G0-B or G0-C.

## 4. The Only Allowed SBR-V2 Rule

### 4.1 Large-View Guard

SBR-V2 reuses byte-identical Arm-C raw records and byte-identical Greedy NMM
cluster membership.

For each cluster:

1. Identify full-view members by `source_order == 0`.
2. A cluster is mixed only when it contains at least one full-view member and
   at least one local-view member.
3. Select the highest-score full-view member using the existing deterministic
   `(-score, source_order, query_index, original_index)` order.
4. Compute that predicted full box's effective size using the Arm-A gain:

   `sqrt(predicted_width * predicted_height) * min(640/W, 640/H, 1)`.

5. If the cluster is mixed and the effective size is strictly greater than
   `96`, output coordinates exactly equal to that full-view anchor.
6. Otherwise use the original score-weighted standard fusion.

The cluster output score remains the maximum member score. Class, cluster
membership, seed ordering, final ordering, and `max_det=300` remain unchanged.
There is no special final-300 priority for large boxes. Singletons are
byte-for-byte identities. Labels and ignore regions are unavailable to this
rule.

### 4.2 Invariants

For every image, Arm C and V2 must have:

- identical raw-view bytes and raw hash;
- identical cluster membership bytes and cluster hash;
- identical cluster count;
- identical output scores and classes before and after the final cap;
- identical final selected cluster identities and count;
- identical original-index tie order;
- changed coordinates only for qualifying mixed clusters;
- 100% singleton preservation.

Any invariant violation invalidates the run.

## 5. Development and Confirmation Protocol

### 5.1 Frozen train-development folds

Before any V2 train-fold metric is visible, create three disjoint deterministic
approximately 10% engineering-screen folds from VisDrone train. For each
relative image path:

`bucket = int(SHA256(relative_posix_path)[0:8], 16) mod 10`.

Folds use buckets `0`, `1`, and `2`. The manifests, image counts, content
signatures, and code commit are written before inference.

Each fold runs only A, C, and V2 from one shared full-plus-four-local raw cache.
The checkpoint and all original scientific constants remain frozen.
Because the checkpoint was trained on the complete VisDrone train split, these
folds are in-sample engineering screens. They do not measure generalization,
constitute statistical repetitions, or support a paper performance claim.

### 5.2 Fold gate

A fold passes only when all original V2-A gates hold:

- `AP-tiny-SBR >= +0.010`
- `mAP50-95 >= +0.003`
- `tiny_recall >= +0.020`
- `AP75 >= -0.002`
- `AP-large-SBR >= -0.005`

At least two of three folds and the pooled union of all three folds must pass.
The remaining fold, if any, must improve AP-large relative to Arm C on that
fold and must not regress AP-tiny relative to Arm A. In addition:

- V2 loses no more than `0.005` AP-tiny relative to Arm C in the pooled union;
- all invariants in Section 4.2 hold on every fold.

Every fold report includes image count, large-GT count, raw prediction count,
A/C/V2 metrics, deltas, and the exact pass or failure reason. The 2-of-3 rule
is only an engineering stop-loss and is not described as statistical
significance.

Failure stops the SBR positive-result route. No second V2 rule is allowed.

### 5.3 One-shot post-hoc validation replay

Only after the train-development gate passes may V2 be applied once to the
existing 548-image validation raw cache. No inference rerun is needed. The
same five original V2-A gates and all invariants apply.

If validation fails any item, SBR-V2 is archived as a negative result and no
third repair is attempted. This replay is not independent confirmation because
the original validation failure motivated V2. If it passes, the result is
eligible to become an engineering candidate for innovation point 1. A
genuinely held-out test or a second detector plus second aerial dataset is
mandatory before any generalization or conference-level efficacy claim.

## 6. Evidence and Independent Adjudication

The audit and every fold/confirmation produce:

- immutable input manifests and checksums;
- per-event large-loss attribution;
- cluster and coordinate-drift summaries;
- A/C/V2 metrics and deltas;
- raw/cluster/selection invariant reports;
- runtime and peak-memory records;
- a primary gate artifact;
- a separate independent adjudication artifact.

The main evaluator and independent adjudicator must agree on the gate. Any
disagreement is a software failure, not a scientific result. The adjudicator
runs in a separate process, does not import the primary evaluator, and records
its own source commit, source-tree hash, script hash, environment, input
manifest hash, and output hash.

## 7. Stop Rules and Claim Boundary

Stop immediately when:

- the offline mechanism share is below 60%;
- evidence integrity or an invariant fails;
- fewer than two train folds pass;
- the pooled union fails any original V2-A gate or the pooled tiny-retention
  requirement;
- one-shot validation fails.

Current allowed claim:

> Original SBR improved tiny and aggregate accuracy but failed its preregistered
> large-object protection gate.

Only after all V2 stages and a genuinely held-out replication pass may the
paper claim:

> A validation-inspired, training-free Large-View Guard resolves the observed
> cross-scale fusion conflict while preserving the original small-object gain.

The method must always be reported as five forward views and up to 1,500 query
opportunities per image, not as zero-overhead or fixed-total-budget inference.

## Amendment — 2026-07-24: Sealed G0 Coordinate Semantics

This dated amendment preserves the original proposal above as an audit trail
and corrects its description of the frozen G0 evaluator. It supersedes only
statements that describe Arm C as being evaluated on the score-weighted fused
coordinates.

The sealed G0 implementation performed class-aware Greedy clustering, but its
metric-row projection retained `prediction.global_xyxy`. For a multi-member
cluster this field is the highest-ranked seed detection's global coordinate;
the separately computed score-weighted `prediction.box` was retained in the
frozen arm-prediction artifact but was not the coordinate consumed by the G0
metric evaluator. The observed mechanism is therefore named **local-seed
coordinate displacement**, not weighted-fusion coordinate drift.

The corrected causal comparison is:

1. Reconstruct the exact frozen Greedy clusters and verify the weighted
   `prediction.box` against the immutable arm-prediction artifact.
2. Reproduce Arm C metrics and all attribution matching from every selected or
   pre-cap cluster's sealed seed `global_xyxy`.
3. Define V2 from that same complete Arm C seed-coordinate baseline.
4. For each eligible mixed cluster only, replace the evaluated coordinate with
   the highest-ranked full-view member's `global_xyxy`. Eligibility remains
   exact: a full member, a local member, and full-anchor effective size strictly
   greater than 96 pixels.
5. Preserve score, class, seed provenance, ordering, cluster membership,
   pre-cap population, and final `max_det=300`. If the sealed seed coordinate
   already equals the selected full anchor, the cluster is not counted as a
   recovered local-seed displacement event.
6. In a per-event counterfactual, copy the complete final-300 Arm C
   seed-coordinate baseline and override only the target eligible cluster.

The two audit attempts preceding this amendment stopped at internal consistency
checks before any audit metric or gate was published or independently
adjudicated. They are software-diagnostic attempts, not scientific results.
This correction changes neither data nor model outputs and is not parameter
tuning: the five original gates, the `0.60` mechanism-share threshold, IoS,
confidence, size boundary, and `max_det` remain unchanged.

# SBR Score-Only Causal Oracle Design

**Status:** approved for implementation  
**Date:** 2026-07-24  
**Scope:** one frozen, development-only feasibility screen on the existing
VisDrone validation evidence  

## 1. Decision and scientific boundary

The SBR-V2 Large-View Guard is permanently stopped. Its independently
adjudicated result did not recover the original large-object gate. The next
and only immediate question is narrower:

> Can decreasing selected local-view candidate scores, without changing the
> detector, boxes, classes, query identities, or full-view scores, make the
> original five SBR gates pass?

This oracle answers only whether that score-control mechanism has enough
best-case capacity to justify building a learnable causal cross-view
calibration head. It is not a deployable method, a validation result, a paper
result, or an upper bound over arbitrary score-calibration policies.

The data boundary is frozen:

- use the existing 548-image VisDrone `val` split only as a development
  feasibility screen;
- do not continue or enlarge duplicate, scene, sequence, or dataset-correlation
  auditing during this phase;
- do not read or evaluate VisDrone `test-dev` during this phase; it remains
  locked for one same-domain confirmation after a complete method is frozen;
- do not make a second dataset a prerequisite for this oracle;
- do not describe `test-dev` as scene-independent, and do not make a
  cross-dataset generalization claim from this oracle.

If this oracle stops, the score-calibration route stops. The next route will
receive a separate design rather than changing this oracle after seeing its
result.

## 2. Immutable starting point

The oracle consumes the checksum-verified G0-A evidence and the exact dataset
records identified by that evidence. It never writes into the original
evidence directory.

The starting arms are:

- Arm A: the full-view baseline;
- Arm C: one full view plus four 60% overlapping local views, using the frozen
  class-aware Greedy IoS clustering and the sealed seed-coordinate metric
  projection.

The trusted `C-A` development deltas are:

| Gate | Frozen delta | Required delta |
| --- | ---: | ---: |
| `AP-tiny-SBR` | `+0.0400347711` | `>= +0.010` |
| `mAP50-95` | `+0.0424115098` | `>= +0.003` |
| `tiny_recall` | `+0.1001918253` | `>= +0.020` |
| `AP75` | `+0.0432520654` | `>= -0.002` |
| `AP-large-SBR` | `-0.0317937425` | `>= -0.005` |

All original constants remain fixed: confidence `0.001`, per-view and final
`max_det=300`, IoS `0.5`, image sizes, tile geometry, class-aware matching,
ignore neutralization, effective-size bins, and IoU thresholds
`0.50:0.05:0.95`.

## 3. Exact intervention boundary

The raw cache is the output of detector inference after the frozen per-view
confidence filter and per-view `max_det=300`. A future calibrator would sit
after this retained-record boundary and before cross-view fusion. Therefore
the oracle does not rerun the detector and cannot recover candidates already
discarded inside a view.

For each image, the applicable pipeline is:

1. load the byte-identical retained Arm-C raw records;
2. reconstruct a stock set of probe clusters with the original scores;
3. select oracle interventions from those frozen probe clusters;
4. decrease only selected local-view raw scores;
5. apply the frozen `score >= 0.001` filter to the overlaid scores;
6. rerun the complete class-aware Greedy IoS clustering and standard
   score-weighted fusion from all still-active retained raw records;
7. project every fused prediction through the sealed seed
   `global_xyxy` coordinate semantics;
8. apply the frozen evaluator confidence/order/final-300 logic and compute the
   original metrics.

Probe clusters are used only to define intervention groups. They are not
recomputed during selection, and their membership is never forced onto the
counterfactual fusion run.

## 4. Frozen candidate groups

A stock probe cluster is eligible when all of the following hold:

1. it contains at least one full-view member (`source_order == 0`);
2. it contains at least one local-view member (`source_order > 0`);
3. its highest-ranked seed under
   `(-score, source_order, query_index, original_index)` is local-view;
4. at least one local member has a score strictly greater than the
   highest-ranked full member's score.

The highest-ranked full member is the cluster's **full anchor**. The local
members whose original scores are strictly greater than the full-anchor score
form one **aggressor group**. Stock clusters are disjoint, so a raw candidate
belongs to at most one group.

No ground-truth size, target identity, IoU to a target, sequence name, image
name, manually inspected scene relation, or test-set information participates
in candidate eligibility. There is no predicted-size threshold in the oracle.

## 5. Frozen score counterfactual

For one aggressor group, every member score is replaced by:

`nextafter(full_anchor_score, -infinity)`.

This is the largest representable float64 value strictly below the full-anchor
score. All aggressors in the group receive that same value; the existing
source/query/original-index keys preserve deterministic ordering after the
full anchor.

The intervention:

- may only decrease scores;
- may only touch local-view aggressors in an eligible group;
- never changes a full-view score;
- never changes any box, class, query index, source, transform, tile bounds,
  image identifier, or retained-record population;
- never adds, deletes, or copies a raw record;
- does not directly choose a fused box or cluster.

The retained overlay table is immutable in population. A changed score that
falls below `0.001` is nevertheless excluded from the active fusion set by the
frozen post-overlay filter. Changed scores are allowed to alter the active
set, seed order, cluster membership, fused coordinates and scores, final
ranking, and the final selected population. Those are causal downstream
effects and must be recomputed rather than held fixed.

## 6. Predeclared oracle selection rule

Each eligible group is first intervened on alone. The resulting image is
evaluated against the stock Arm-C image using the frozen one-to-one matcher.

A group is **safe-beneficial** only when:

- at every one of the ten IoU thresholds, TP count does not decrease for
  overall, tiny, or large targets; and
- the sum of large-target TP changes over the ten thresholds is strictly
  positive.

Small/medium TP counts, every FP count, and AP are recorded but do not enter
this per-group label. The overall-TP constraint prevents an unbounded exchange
of small/medium matches, while the unchanged original five-gate joint
adjudication remains the only dataset-level performance decision. A group that
fails either condition is not selected.

After every group has been labeled independently, all safe-beneficial groups
are applied simultaneously in one joint pass. That joint pass is the only
oracle output used by the GO/STOP gate.

The following are prohibited:

- iterative relabeling after the joint result;
- greedy addition or removal of groups;
- subset, beam, random, Bayesian, or threshold search;
- changing the safety definition;
- using dataset-level AP feedback to revise group membership;
- trying alternative demotion values;
- inspecting `test-dev` or a second dataset to rescue the decision.

Interactions between independently safe groups may make the joint result
worse. If that happens, the route stops; no subset search is permitted.

## 7. GO, STOP, and invalid states

The primary output status is exactly one of:

- `SBR_SCORE_ORACLE_GO`;
- `SBR_SCORE_ORACLE_STOP`;
- `SBR_SCORE_ORACLE_INVALID`.

`GO` requires the joint oracle output versus Arm A to pass all five original
gates without tolerance changes:

- `AP-tiny-SBR >= +0.010`;
- `mAP50-95 >= +0.003`;
- `tiny_recall >= +0.020`;
- `AP75 >= -0.002`;
- `AP-large-SBR >= -0.005`.

It also requires every integrity invariant in Section 8 and agreement from the
independent adjudicator. Zero eligible groups or zero selected
safe-beneficial groups yields `STOP`, not `INVALID`.

`STOP` is a valid negative result whenever inputs and invariants are valid but
the joint output misses any gate. `INVALID` is reserved for corrupted inputs,
baseline reproduction failure, non-finite arithmetic, invariant failure,
unexpected CLI options, or primary/adjudicator disagreement. An invalid run
may be repaired only at the software level and rerun with the same frozen
scientific rule.

## 8. Fail-closed invariants

Before oracle labeling, the implementation must reproduce the sealed Arm-A and
Arm-C metrics and hashes. The run then proves:

- the input manifest, evidence files, dataset signature, checkpoint hash,
  protocol hash, and source provenance verify;
- Arm-A and stock Arm-C reconstruction match the immutable evidence;
- the retained raw-record count and identity set are unchanged, while any
  post-overlay active-set exclusion is exactly explained by `score < 0.001`;
- every modified record is local and belongs to a recorded eligible aggressor
  group;
- every modified score equals the predeclared float64 predecessor of its full
  anchor and is lower than its original score;
- every unselected score and every full-view score is bit-identical;
- all non-score fields are bit-identical for every raw record;
- the stock/no-selection path is bit-identical to Arm C;
- both the single-group and joint paths reapply the post-overlay confidence
  filter and rebuild standard fusion rather than editing stored fused
  predictions;
- all scores, coordinates, metrics, and deltas are finite;
- all ten IoU thresholds and all four size bins are evaluated;
- no test-dev or external-dataset artifact appears in the input manifest.

Any invariant failure prevents a scientific decision.

## 9. Evidence package and independent adjudication

The primary run writes a new immutable evidence directory containing:

- resolved input manifest and checksums;
- frozen rule/schema document and hashes;
- stock probe-cluster and eligible-group tables;
- one row per single-group intervention with before/after TP and FP counts at
  every threshold and size bin;
- selected-group table;
- joint counterfactual raw-score patch table;
- Arm A, Arm C, and joint-oracle metrics;
- exact deltas and five gate booleans;
- invariant report;
- candidate coverage by image, class, source, and VisDrone sequence token;
- runtime and peak-memory report;
- primary status and checksums.

Coverage and sequence concentration are diagnostics, not gates. They prevent
overclaiming but cannot turn a `STOP` into `GO` or a `GO` into `STOP`.

A separate adjudicator process must not import the primary oracle or metric
module. From canonical evidence, it independently verifies checksums,
intervention eligibility, score patches, per-group labels, joint metrics,
invariants, and the five gates. It writes its own source/script hashes and
exactly one matching decision. Disagreement yields `INVALID`.

## 10. Test and execution requirements

Implementation follows test-driven development. Unit and adversarial fixtures
must prove:

- exact eligible/ineligible group boundaries, including score ties;
- float64 predecessor demotion and full-view bypass;
- immutable non-score fields and record population;
- score reordering really rebuilds seed-only non-transitive clusters;
- single-group safety checks cover all thresholds and bins;
- one selected group can interact with another only in the final joint pass;
- no-op reconstruction is bit-identical;
- every original gate boundary is inclusive at the frozen value;
- malformed, missing, non-finite, or mismatched evidence fails closed;
- the CLI exposes only operational paths and rejects scientific overrides;
- the independent adjudicator reaches the same decision without importing the
  primary implementation.

The real oracle runs once on the existing immutable validation cache after the
tests and a deterministic smoke fixture pass. No GPU inference, train-fold
screen, test-dev evaluation, second-dataset evaluation, or 100-epoch training
belongs to this oracle phase.

## 11. Result-dependent next step

If the result is `GO`, the next separate design will specify a learnable,
GT-free causal cross-view score calibrator and its train/development protocol.
The oracle itself cannot be promoted into inference or reported as method
performance.

If the result is `STOP`, score calibration is permanently abandoned and the
next separate design will consider a training-time cross-view consistency
route. In either case, the formal 100-epoch paper run begins only after the
resulting deployable method and its confirmation protocol are separately
frozen.

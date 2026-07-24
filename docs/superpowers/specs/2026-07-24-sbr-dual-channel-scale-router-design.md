# SBR Dual-Channel Scale Router Design

**Status:** superseded before execution by `2026-07-24-sp-ppaf-design.md`; never evaluated
**Date:** 2026-07-24
**Scope:** validation feasibility screening only

## 1. Decision Context

The sealed SBR comparison established that Arm C preserves strong tiny-object
gains but fails the original large-object gate:

- `AP-tiny-SBR`: `+0.0400347711`
- `mAP50-95`: `+0.0424115098`
- `tiny_recall`: `+0.1001918253`
- `AP75`: `+0.0432520654`
- `AP-large-SBR`: `-0.0317937425`

The subsequent score-only oracle and coordinate-only Large-View Guard both
stopped. The score oracle changed the joint result negligibly. The coordinate
guard improved AP-large over Arm C by only `0.0001151185`, leaving an
AP-large delta of `-0.0316786240` relative to Arm A.

The remaining post-processing hypothesis is stronger than either stopped
route: use two independent candidate channels and make the final detection
budget scale-asymmetric.

## 2. Research Question

Can a single legal prediction set preserve:

- the full-view Arm A candidates needed for large objects; and
- the full-plus-tile Arm C candidates needed for tiny and medium objects?

The hypothesis is falsified if one frozen cache replay fails any original
five-item gate. No threshold, quota, score, or overlap rule may be changed
after the result is visible.

## 3. Frozen Algorithm

### 3.1 Inputs

The replay uses the same checksum-verified 548-image validation cache that
produced the sealed Arm A and Arm C results. It performs no detector inference
and reads no held-out test split.

- The large channel may use only Arm A predictions that survived Arm A's
  original confidence threshold and final Top-300 selection.
- The small/medium channel reconstructs Arm C's deterministic pre-cap cluster
  candidates from the same raw cache.
- All coordinates, scores, classes, source orders, query indices, and cluster
  provenance remain byte-equivalent to their source channel.

### 3.2 Scale decision

For an image of width `W` and height `H`, define the predicted effective size
of box `(x1, y1, x2, y2)` in float64 as:

```text
gain = min(640 / W, 640 / H, 1)
effective_size = sqrt((x2 - x1) * (y2 - y1)) * gain
```

The route is frozen at the original SBR size boundary:

```text
effective_size > 96  -> large
effective_size <= 96 -> tiny / small / medium
```

No ground-truth property is available to this decision.

### 3.3 Large channel

Select every final Arm A prediction whose Arm A full-view box has
`effective_size > 96`. Preserve its original:

- `global_xyxy`;
- score;
- class;
- `source_order=0`;
- query index; and
- Arm A final order.

These candidates are protected members of the final 300-detection budget.
Candidates outside Arm A's original final Top-300 are ineligible.

### 3.4 Tiny/medium channel

From the Arm C pre-cap cluster candidates, keep candidates whose sealed seed
`global_xyxy` has `effective_size <= 96`.

For each selected Arm A large prediction, find the Arm C cluster containing
the exact raw full-view identity:

```text
(image_id, class_id, source_order=0, query_index)
```

Remove that one Arm C cluster before filling the remaining budget. No geometric
IoU or IoS deduplication is added. Local-only fragments without the exact
provenance remain unchanged.

### 3.5 Final budget

1. Retain all selected Arm A large candidates in Arm A order.
2. Fill the remaining slots from eligible Arm C pre-cap candidates in Arm C's
   original deterministic order.
3. Cap the unified prediction set at 300.
4. If Arm A contains more than 300 predicted-large candidates, retain the first
   300 in Arm A's original order.
5. Evaluate this one unified prediction set for every metric. Metric-specific
   switching between Arm A and Arm C is forbidden.

The selection budget is scale-asymmetric, but candidate scores remain
unchanged. Evaluation uses the original deterministic metric implementation.

## 4. Required Invariants

The replay is invalid unless all of the following hold on all 548 images:

- every selected Arm A large candidate exists bit-exactly in sealed Arm A;
- every selected Arm C candidate exists bit-exactly in reconstructed Arm C;
- no Arm A candidate outside its original final Top-300 is used;
- scale routing uses float64 and the single strict `96` boundary;
- only exact raw-full provenance removes an Arm C cluster;
- selected output count is at most 300;
- boxes, scores, classes, and query identities are not rewritten;
- one unified prediction set is used for all metrics;
- Arm A and Arm C baseline metrics reproduce their sealed values.

Any disagreement is a software failure rather than a scientific result.

## 5. Five-Item Gate

Relative to Arm A, the unified route must satisfy all conditions:

| Metric | Required delta |
|---|---:|
| `AP-tiny-SBR` | `>= +0.010` |
| `mAP50-95` | `>= +0.003` |
| `tiny_recall` | `>= +0.020` |
| `AP75` | `>= -0.002` |
| `AP-large-SBR` | `>= -0.005` |

This is one feasibility replay, not a parameter search. If any item fails, the
dual-channel post-processing route permanently stops and work moves to the
predeclared training-time cross-view consistency route.

## 6. Known Risks

- A true large object can be underestimated to `effective_size <= 96`.
- A medium object can be overestimated into the large channel.
- Local truncated fragments can remain as high-scoring false positives.
- Protected low-quality Arm A large predictions can displace useful Arm C
  tiny predictions.
- Exact-provenance removal cannot remove unrelated local-only duplicates.
- The route must recover `0.0266786240` AP-large beyond the coordinate guard,
  which is approximately 84% of the remaining original large-object deficit.

Because the replay requires no inference or training and adds candidate
recovery plus budget protection not tested by the stopped guard, it is worth
one run. The pre-run estimated probability of passing all five gates is
`30%` to `50%`, with a center estimate of `40%`.

## 7. Claim Boundary

This protocol and its validation replay are internal feasibility screening.
They are not paper efficacy evidence. No test-dev data, second dataset, or
expanded data-correlation audit is a prerequisite for this screening. A
successful replay only freezes a candidate method for later controlled
experiments.

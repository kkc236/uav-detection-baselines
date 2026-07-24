# SP-PPAF Design

**Status:** approved and frozen for one zero-inference cache replay
**Date:** 2026-07-24
**Scope:** innovation-1 feasibility screening on the sealed 548-image cache

## 1. Method

SP-PPAF means **Scale-Partitioned Prefix-Preserved Additive Fusion**,
or **尺度分区的前缀保护式增量融合**.

The method forms one legal prediction set from two frozen sources:

```text
Arm A predicted-large final detections
+
Arm C tile-supported non-large pre-cap candidates
=
one final prediction set with at most 300 detections
```

Arm A large detections are an immutable high-priority prefix. Arm C detections
are a strictly lower-score additive tail. The router uses prediction metadata
only; ground truth is unavailable until the prediction set has been sealed.

This design supersedes the unexecuted dual-channel scale-router design. The
older document remains in Git history and is explicitly marked superseded.

## 2. Decision Context

The sealed Arm C result passes four original gates but fails large AP:

| Metric | Arm A | Arm C | C minus A |
|---|---:|---:|---:|
| `AP-tiny-SBR` | 0.0710571443 | 0.1110919154 | +0.0400347711 |
| `mAP50-95` | 0.1806213966 | 0.2230329063 | +0.0424115098 |
| `tiny_recall` | 0.5537479711 | 0.6539397964 | +0.1001918253 |
| `AP75` | 0.1666655849 | 0.2099176503 | +0.0432520654 |
| `AP-large-SBR` | 0.1458467938 | 0.1140530513 | -0.0317937425 |

Score-only and coordinate-only repairs are permanently stopped. Literal
all-Arm-A prefixing is retained only as a predeclared fallback: 504 of 548
images already contain 300 Arm A predictions, leaving only 578 total slots.
The tiny-recall gate requires at least 543 new true positives, so literal
All-A would require 93.94% of all fillers to be new tiny true positives.

## 3. Frozen Inputs

The replay consumes:

- sealed Arm A final predictions after `conf=0.001` and `max_det=300`;
- deterministic Arm C pre-cap clusters reconstructed from the sealed raw cache;
- exact box, score, class, source, query, raw-member, image-size, and manifest
  metadata;
- the unchanged SBR evaluator.

It performs no detector inference and no training. The routing stage accepts no
ground-truth boxes, labels, ignore boxes, matches, or oracle events. Evaluation
is a later process over already-sealed prediction rows.

## 4. Frozen Constants

```text
conf = 0.001
max_det = 300
large_effective_size = 96.0
fragment_ios = 0.5
a_floor = 0.01706760562956333
c_ceiling = 0.008533802814781666
```

`a_floor` is the frozen expected global minimum score among all sealed Arm A
final predictions. Before routing, the implementation must recompute the real
cache minimum and require exact float64 equality with `a_floor`; a mismatch is
invalid. No constant may be recomputed from visible method metrics.

## 5. Scale Semantics

The implementation must call the existing `src.sbr_v2_audit.effective_size`
function:

```text
gain = min(640 / width, 640 / height, 1)
effective_size = sqrt(box_width * box_height) * gain
```

The boundary is strict:

```text
effective_size > 96  -> predicted large
effective_size <= 96 -> predicted non-large
```

## 6. Arm A Prefix

For primary arms P1, P2, and P3:

```text
A_prefix = sealed Arm A final predictions with effective_size > 96
```

Every retained Arm A prediction preserves its box, score, class, source, query,
and relative Arm A order exactly. Predictions outside the sealed Arm A final
Top-300 are ineligible.

For the fallback arm:

```text
All-A prefix = every sealed Arm A final prediction
```

All-A uses the same P3 tail filters: exact-provenance removal is evaluated
against every All-A prefix member, while fragment suppression is evaluated
against the predicted-large subset of that prefix.

## 7. Arm C Tail

A pre-cap Arm C cluster is initially eligible when:

1. at least one raw member is tile-view (`source_order > 0`);
2. the sealed seed `global_xyxy` has effective size at most 96;
3. the cluster score is finite and originally at least `conf`;
4. its complete raw provenance is valid.

The seed `global_xyxy` is used only for the frozen scale test. The emitted C
candidate preserves the existing pre-cap cluster `box` exactly; SP-PPAF changes
only its score band and never replaces that fused box with seed coordinates.

P2 removes any C cluster containing the exact raw full-view identity of a
selected A-prefix detection:

```text
(image_id, class_id, source_order=0, query_index)
```

P3 additionally removes a tile-only C cluster when its candidate box and a
same-class selected A-large box have class-aware IoS at least 0.5. P3 is the
primary method. P1 and P2 are mechanism ablations and cannot replace P3 after
metrics become visible.

## 8. Strict Score Band

The original multiplicative draft is forbidden because it can move fillers
below the evaluator confidence threshold.

Define:

```text
low = nextafter(conf, +infinity)
high = nextafter(c_ceiling, -infinity)
```

For original C score `s`:

```text
mapped(s) = low + (high - low) * (s - conf) / (1 - conf)
```

The map must be float64, finite, strictly monotone over distinct input scores,
and remain inside `(conf, c_ceiling)`. Equal original scores retain the frozen
source/query/original-index tie order. Arm A scores are never changed.

Before routing is accepted, all distinct real-cache eligible C scores must be
mapped and checked as a set. Distinct inputs must remain strictly ordered with
no float64 collision; equal inputs must remain stable under the frozen
source/query/original-index tie order. Any collision or order change is invalid.

## 9. Capacity and Output

For each arm and image:

```text
remaining = 300 - len(prefix)
output = prefix + first remaining eligible C candidates
```

C candidates are selected in the original deterministic Arm C pre-cap order.
The output is then passed unchanged to the existing evaluator, which will
reapply its frozen score/source/query/original-index order.

No joint NMS, NMM, coordinate fusion, class edit, score search, quota search, or
metric-specific output is allowed.

## 10. Predeclared Arms

| Arm | Definition | Role |
|---|---|---|
| A | sealed full-view Arm A | baseline |
| All-A | full Arm A prefix plus fillers in its actual free slots | fallback |
| P1 | A-large prefix plus scale/source-eligible C tail | scale ablation |
| P2 | P1 plus exact-provenance cluster removal | dedup ablation |
| P3 | P2 plus frozen IoS fragment suppression | primary SP-PPAF |

All five outputs are produced during the same cache pass and sealed before any
dataset label, ignore annotation, evaluator match, or method metric is loaded.
Routing and evaluation run as separate processes. The routing process cannot
import the dataset loader or evaluator.

## 11. Invariants

The run is invalid unless:

- Arm A and Arm C baselines reproduce their sealed metrics;
- every prefix member is byte-identical to sealed Arm A;
- every C tail member maps to exactly one reconstructed pre-cap cluster;
- P1/P2/P3 preserve every selected A-large prediction;
- All-A preserves every sealed Arm A final prediction;
- every C mapped score is strictly above `conf` and below `c_ceiling`;
- the real-cache Arm A minimum equals the frozen `a_floor` exactly;
- distinct real-cache C scores have no mapped-score collision and preserve
  strict order; equal-score ties preserve their frozen order;
- no output exceeds 300 predictions;
- P2 removes only exact-provenance clusters;
- P3 uses only class-aware IoS 0.5 beyond P2;
- routing code has no ground-truth or evaluator-match input;
- one output per arm is used for all metrics.

## 12. Gate and Decision

Relative to Arm A, a successful arm must simultaneously satisfy:

| Metric | Required delta |
|---|---:|
| `AP-tiny-SBR` | `>= +0.010` |
| `mAP50-95` | `>= +0.003` |
| `tiny_recall` | `>= +0.020` |
| `AP75` | `>= -0.002` |
| `AP-large-SBR` | `>= -0.005` |

Decision order:

1. P3 passes all gates: emit `SP_PPAF_PASS` and freeze P3.
2. P3 fails, but the simultaneously generated All-A passes all gates: emit
   `SP_PPAF_FALLBACK_PASS` and freeze All-A.
3. Otherwise emit `SP_PPAF_STOP` and permanently close post-processing repair.

No result-visible adjustment is allowed. Test-dev remains unread. A successful
validation replay is a feasibility result, not final paper evidence.

## 13. Evidence Contract

The one replay writes two checksum-separated closures.

Routing closure, created without loading ground truth:

```text
route_manifest.json
predictions.jsonl.gz
coverage.json
route_invariants.json
checksums.sha256
```

Evaluation closure, created by a later process after verifying the routing
checksums:

```text
evaluation_manifest.json
metrics.json
deltas.json
evaluation_invariants.json
primary_gate.json
checksums.sha256
```

This is an internal feasibility replay, so it does not require a standalone
paper-grade adjudicator. B and C perform read-only review of both checksum
closures, the frozen constants, the route/evaluation process boundary, the
invariants, and the final decision.

Coverage includes per-arm/per-image prefix size, remaining capacity, candidate
counts, scale rejects, provenance rejects, fragment rejects, appended count,
and output count. It contains no ground-truth matching information from the
routing stage.

## 14. Stop Rules

Stop immediately on malformed provenance, nonfinite data, score-band violation,
baseline reproduction failure, output overrun, source-tree mutation, checksum
failure, route/evaluation boundary violation, or B/C discovery of an invalid
decision.

If P3 and All-A both fail the five gates, do not modify 96, IoS, the score band,
maxDet, quotas, or classes. The next route is the predeclared training-time
asymmetric cross-view consistency method.

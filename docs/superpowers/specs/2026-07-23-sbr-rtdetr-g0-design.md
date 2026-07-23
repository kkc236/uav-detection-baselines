# SBR-RTDETR Zero-Training G0 Design

## Objective

Test whether a mature stock RT-DETR-L checkpoint can improve tiny-object
detection on the complete 548-image VisDrone validation split by replicating
the detector's fixed 300-query opportunity across one full-image view and four
overlapping local views. The G0 stage changes only input views and output
fusion. It does not train, change the checkpoint, patch the decoder, add
queries, or modify Ultralytics model code.

The accepted method is:

```text
full-resolution image
  -> full view + four fixed overlapping tiles
  -> stock RT-DETR-L independently on every view
  -> restore every detection to full-image pixels
  -> class-aware greedy IoS clustering
  -> score-weighted class-aware Greedy NMM or singleton-preserving
     border-reliable fusion
```

Each view has its own stock 300-query forward. The complete method therefore
uses five forwards and 1,500 query opportunities. It must never be described as
fixed-total-budget or zero-compute-overhead inference.

## Selected Approach

Three approaches were considered:

1. Call SAHI end to end. This supplies a recognized baseline but its generic
   slice generator uses overlap relative to each tile, while SBR freezes overlap
   relative to the original image. It also makes exact provenance, per-view
   query accounting, and SP-BRF auditing harder.
2. Implement a small SBR-only geometry, fusion, metric, and evidence pipeline
   around the unchanged Ultralytics predictor. This is selected because every
   scientific degree of freedom is explicit and testable.
3. Patch the RT-DETR decoder or reuse BQP Top-420 capture hooks. This is rejected
   because it changes the query path and violates the zero-training stock-model
   contract.

The implementation may reuse general engineering patterns such as SHA256
manifests, atomic JSON writes, and non-empty-output rejection. It must not
import `bqp_capture`, `bqp_g0`, or `bqp_g0_validator`.

## Frozen Geometry

For an image of integer width `W` and height `H`:

```text
tile_w = ceil(0.60 * W)
tile_h = ceil(0.60 * H)
x origins = [0, W - tile_w]
y origins = [0, H - tile_h]
```

The ordered local views are top-left, top-right, bottom-left, bottom-right.
Every boundary is stored as half-open integer pixels `[left, top, right,
bottom)`. This produces an overlap close to `0.20W` and `0.20H`; manifests store
the exact integer overlap for every image.

Arm F uses four tiles only and partitions each axis at `floor(size/2)`. The
right and bottom tiles receive any odd remainder. It has no full-image view.

All views use the same explicit Ultralytics 8.4.90 transform:
`LetterBox(new_shape=(imgsz,imgsz), auto=False, scale_fill=False,
scaleup=False, center=True, padding_value=114)`. The runner applies this
transform before calling the predictor on an already-square image. The
predictor's own scale-fill preprocessing is therefore a no-op, and raw model
coordinates are in the square network-input pixel frame. Raw coordinates are
restored in this order:

```text
if a decoder emits normalized coordinates, multiply by network width/height
-> remove letterbox padding -> divide by x/y gain -> add tile offset
-> clip to [0, W] x [0, H]
```

The runner records `network_xyxy`, `view_xyxy`, and `global_xyxy`. It never
applies both the predictor's source-view scaling and the explicit inverse
transform. A record consumed from `Results.boxes.xyxy` is explicitly marked as
network-input pixels before inverse mapping.

## Frozen Clustering and Fusion

Every view independently retains at most 300 stock detections at
`conf=0.001`. Source order is frozen as full=`0`, top-left=`1`, top-right=`2`,
bottom-left=`3`, bottom-right=`4`; Arm F uses the same local order without a
full view. Combined predictions are deterministically sorted by descending
score, then source order, then original query/detection index.

Clustering is class aware. For two boxes:

```text
IoS = intersection_area / min(area_1, area_2)
```

A higher-scored seed directly absorbs still-unassigned same-class boxes with
`IoS > 0.5` (strictly greater is intentional; an exact `IoS == 0.5` pair does
not match). Matching is seed-only and non-transitive, matching greedy NMM
rather than transitive NMM. The manifest stores comparator `ios_strict_gt`,
and the boundary case is independently tested. Cluster membership is
calculated once and is identical for the standard and SP-BRF arms.

The standard SBR arm uses score-only coordinate fusion so that SP-BRF changes
exactly one variable:

- singleton: unchanged;
- multi-member weight: `w_i=score_i`;
- multi-member cluster box:
  `box_hat=sum(w_i*box_i)/sum(w_i)`;
- score: maximum member score;
- class: seed class.

After fusion, predictions are stably sorted and truncated to `max_det=300`.
Official SAHI Greedy NMM's enclosing-union merge is a separate external
baseline for the later paper comparison; it is not silently substituted for
the frozen SBR standard arm.

For SP-BRF, full-view reliability is `r_i=1`. A local prediction considers only
artificial internal tile edges. Edges coincident with the real image boundary
are not penalized. For local box `[x1,y1,x2,y2]` in tile coordinates and exact
horizontal/vertical overlaps `o_x,o_y`:

```text
left internal edge:   r_left   = clip(x1 / (o_x / 2), 0, 1)
right internal edge:  r_right  = clip((tile_w - x2) / (o_x / 2), 0, 1)
top internal edge:    r_top    = clip(y1 / (o_y / 2), 0, 1)
bottom internal edge: r_bottom = clip((tile_h - y2) / (o_y / 2), 0, 1)
r_i = min(all applicable internal-edge reliabilities)
```

If no artificial edge applies, `r_i=1`.

SP-BRF is frozen as:

```text
singleton -> preserve box, score, and class bit for bit
multi-member weight_i = score_i * (1 + r_i)
box_hat = sum(weight_i * box_i) / sum(weight_i)
score_hat = max(score_i)
class_hat = seed class
```

It never multiplies the final score by reliability, never changes cluster
membership, and never merges across classes.

## Arms and Shared Inference

| Arm | Views | Fusion |
|---|---|---|
| A | full image at 640 | none |
| B | four 0.60 overlapping tiles at 640 | standard |
| C | full 640 plus the same four overlapping tiles | standard |
| D | exactly the C raw views and clusters | SP-BRF |
| E | full image at 1088 | none |
| F | four non-overlapping half-image tiles at 640 | standard |

The runner performs each unique raw view once and caches each detection with
`network_xyxy`, tile-local `view_xyxy`, and full-image `global_xyxy`, together
with tile bounds, transform metadata, source order, and query index. C and D
must consume byte-identical raw-view records and cluster membership. Labels may
only enter evaluation and mechanism accounting after predictions are frozen;
they cannot affect slicing, confidence, clustering, or fusion.

## Frozen Evaluation

All arms use the same checkpoint, image list, labels, ignored labels, class
mapping, evaluator, and postprocessing limits. A prediction whose intersection
over its own area with a VisDrone ignore region is `>=0.50` is neutral: it is
removed from both TP and FP accounting. Predictions not neutralized remain
eligible for normal class-aware matching.

- IoU grid: `0.50:0.05:0.95`.
- Prediction threshold: `conf=0.001`.
- Final maximum detections: 300 per image.
- Size is assigned from each GT's effective dimensions after the Arm-A 640
  letterbox gain, so every arm uses the same GT bin.
- Tiny: effective square-root area `<=16 px`.
- Small: `(16,32] px`.
- Medium: `(32,96] px`.
- Large: `>96 px`.
- `AP-tiny`: mean AP over the IoU grid for the tiny bin.
- Tiny recall gate: micro recall at IoU 0.50, `conf>=0.001`,
  `max_det=300`, after class-aware one-to-one matching.
- AP75 protection: `AP75(C)-AP75(A) >= -0.2` percentage points.

For every size-bin AP, GT outside the bin is ignored. At each AP IoU threshold
`t` in `0.50:0.05:0.95`, a prediction matching an out-of-bin same-class GT at
IoU `>=t` is also ignored for that bin; it is not counted as a false positive.
Other predictions remain false positives. This is the pre-registered
COCO-style area-range behavior, implemented under the explicit `*-SBR` names
above.

The G0-A validity gate requires 548/548 identical images, finite/legal boxes,
complete A--D artifacts, checkpoint/data/source hashes, Arm-A parity with the
stock Ultralytics validator, per-view and final 300 limits, and agreement with
the independent adjudicator. E and F are required diagnostic ablations and
must be complete and finite, but their metrics do not alter the C-A
effectiveness gate.

G0-A passes only if all hold for `C-A`, in absolute percentage points:

- `AP-tiny >= +1.0`;
- `mAP50-95 >= +0.3`;
- tiny recall at IoU 0.50 `>= +2.0`;
- `AP75 >= -0.2`;
- `AP-large >= -0.5`.

Failure emits `SBR_G0A_FAIL` and stops the slicing route without parameter
search.

Only after G0-A passes, G0-B compares D with C. The runner refuses G0-B/C
unless it reads a matching `g0_gate.json` whose status is exactly
`SBR_G0A_PASS` and whose source, checkpoint, dataset, and protocol hashes equal
the requested run. A seam band is the union of strips of width `o_x/2` and
`o_y/2` centered on each artificial tile seam.
A prediction is seam-near when its global box intersects a seam band. An
internal-boundary false positive is seam-near, unmatched to a non-ignore GT at
IoU `>=0.50` by the same one-to-one matching order as evaluation, and
`conf>=0.001`. A fused prediction is a truncated false positive when any
cluster member's view-local box intersects an artificial tile edge. Duplicate
detections are the sum over fused clusters of `max(|C_k|-1,0)`. Boundary
targets are non-ignore GT boxes intersecting a seam band; their recall uses the
same class-aware IoU `>=0.50` matching and `max_det=300`.

The D-C gate is:

- overall AP delta at least `-0.1`;
- AP-tiny delta at least `-0.1`;
- internal-boundary FP or duplicate detections decrease by at least 15%;
- singleton preservation exactly 100%;
- boundary-target recall delta at least `-0.2`;
- AP-large delta at least `-0.2`.

The word “or” is inclusive: either reduction is sufficient, and both reductions are
reported. For each count `C` is the Arm-C count and `D` is Arm-D; reduction is
`1-D/C` and is valid only when `C>0`. When `C=0`, `D=0` is reported
`not_applicable` and never claimed as an improvement; `D>0` fails the gate.
G0-C records target area amplification, per-view Top-300 tiny coverage (number
of tiny GTs with at least one same-class raw candidate at IoU `>=0.50` divided
by tiny GT count), candidate multiplicity (unique raw candidates across views
satisfying that same class/IoU rule divided by tiny GT count), seam
predictions/FPs, duplicate counts, singleton preservation, view/fusion time,
peak GPU memory, and throughput. These explain results but cannot replace the
AP/recall gates.

SBR size-bin AP fields are pre-registered custom metrics based on Arm-A's
effective pixel radius and must be labeled `AP-tiny-SBR`, `AP-small-SBR`,
`AP-medium-SBR`, and `AP-large-SBR`; they are not interchangeable with
standard COCO AP-small/medium/large.

## Evidence and Failure Handling

Every run goes to a new empty directory and atomically writes:

- `g0_manifest.json`;
- compressed raw-view and arm-prediction JSONL;
- `g0_metrics.json`, `g0_deltas.json`, and `g0_gate.json`;
- `runtime.json`;
- `independent_adjudication.json`;
- `checksums.sha256`;
- a concise `README.md`.

The manifest records source branch/commit/dirty state, checkpoint bytes/hash,
the 548-image content signature, environment versions, exact per-image tile
bounds, all frozen constants, command, and artifact hashes.

Any hash mismatch, dirty tracked worktree, version drift, non-finite value,
illegal coordinate, query/detection-limit violation, image-set mismatch,
Arm-A parity failure, or independent-recomputation mismatch fails closed.
No G0 result may change tile geometry, confidence, max detections, IoS,
clustering, fusion, size bins, or gates after metrics become visible.

## Verification Strategy

Development follows test-driven development:

1. pure geometry and inverse-letterbox tests;
2. deterministic class-aware clustering and standard-fusion tests;
3. SP-BRF reliability, singleton, and hand-calculated fusion tests;
4. synthetic evaluator and Arm-A parity tests;
5. fail-closed CLI/artifact tests;
6. an 8--16 image S0 smoke run whose research metrics are not interpreted;
7. full 548-image G0-A once, followed by an independent adjudication.

S0 and G0 do not touch trainers, optimizers, loss code, model YAML, or the
checkpoint.

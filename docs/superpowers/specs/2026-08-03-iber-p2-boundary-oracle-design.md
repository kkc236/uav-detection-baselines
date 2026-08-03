# IBER P2 Boundary Oracle Design

Date: 2026-08-03

## Decision

The next scientific candidate is a diagnostic-only P2 boundary oracle. It does not
change RT-DETR, does not start the 30-epoch screen, and does not weaken any existing
Gate-1 threshold. Its sole purpose is to test whether stride-4 shallow features contain
enough held-out boundary-direction information to justify implementing a P2 branch.

## Frozen authority

- Detector: the existing frozen Ultralytics RT-DETR-L baseline checkpoint.
- Data: the fixed 647-image hashed training subset and the complete 548-image validation split.
- Seed: private seed 10000; deterministic sample ordering and deterministic optimizer state.
- Feature source: model layer 1 (`HGBlock`, 128 channels, 160x160 for a 640x640 input).
- Matching: reuse the stock last-decoder-layer Hungarian indices; never rematch refined boxes.
- Validation: train labels are never used to select validation examples, epochs, or thresholds.

## Evidence extraction

For every matched stock query, sample each of the four stock-box edges from P2 with
`align_corners=False` and border padding. Use five fixed tangent positions and seven
fixed normal offsets at `[-12, -8, -4, 0, 4, 8, 12]` input pixels. Tangent samples are
averaged after sampling, yielding a `4 x 7 x 128` profile per matched object. Store only
detached profiles, detached decoder context, stock geometry, target edges, and area
bucket metadata in an immutable, hashed cache.

The direction label and validity mask reuse the existing Gate-1 normalized
`sign(target_edge - stock_edge)` implementation exactly; no new deadband is introduced.
Tiny and small buckets use the same target-area definition as the existing Gate-1
implementation.

## Oracle

Train a deliberately small edge-wise classifier on the fixed training split:

- P2 profile encoder: LayerNorm(896), Linear(896,128), SiLU.
- Context encoder: detached decoder hidden state plus stock geometry to 64 dimensions.
- Edge identity: an 8-dimensional embedding.
- Classifier: Linear(200,128), SiLU, Linear(128,1).

The report includes a per-bucket/per-edge majority baseline, context-only, P2-only, and
P2-plus-context held-out direction accuracy so context cannot disguise the absence of
useful P2 evidence. Model selection is frozen to the final epoch; no best-validation
checkpoint is permitted.

## Predeclared decision

P2 is viable only if the final held-out validation report is finite and both conditions
hold:

- tiny direction accuracy >= 0.624866;
- small direction accuracy >= 0.634066.

These are the current B0 Gate reference values plus the already frozen three percentage
point margin. Passing this diagnostic authorizes a minimal P2 boundary branch and a new
four-arm Gate-1 run; it does not imply AP improvement. Failure stops further
boundary-only architectural work under the current Gate and is published as negative
evidence. The threshold, split, epoch choice, and labels must not be changed after the run.

## Isolation and failure handling

- No detector parameter receives a gradient.
- No P2 tensor is retained after its detached matched profiles are written.
- Cache and report paths are immutable and include source, baseline, dataset, subset,
  runtime, and schema hashes.
- Engineering failures may be fixed and rerun under a new source commit and run root.
- Scientific failure is not repaired by changing the decision threshold or evaluating
  the training set.

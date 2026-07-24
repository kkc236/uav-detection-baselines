# SADED: Scale-Aware Dual Expert Detector Design

Date: 2026-07-25

Status: written specification for user review after authoritative
`ASCV_LOC_STOP`

## 1. Objective

The innovation is a complementary detector system rather than a replacement
model:

- a sealed stock RT-DETR-L baseline expert owns non-tiny predictions and all
  large-object coordinates;
- an independently trained tiny expert owns tiny predictions;
- a fixed prediction-scale router produces one final prediction set for
  ordinary COCO-style evaluation.

The frozen name is **SADED: Scale-Aware Dual Expert Detector**.

The paper claim is limited to scale-specialized expert collaboration. It does
not claim that the tiny expert is a general full-scale replacement for the
baseline.

## 2. Relation to the ASCV-Loc STOP

The authoritative ASCV-Loc run remains a valid scientific stop under its
registered all-scale mechanism gate:

- batches `401..500`, tiny: `16,173` pairs in `100/100` batches, teacher
  advantage `+0.317368`, win rate `0.725407`;
- batches `401..500`, non-tiny: `6,836` pairs in `100/100` batches, teacher
  advantage `-0.244247`, win rate `0.342891`;
- emitted and independently reconstructed decision: `ASCV_LOC_STOP`.

This result is not relabeled as a pass and does not authorize the old
`SCREEN_10`. It motivates a new hypothesis: local-view supervision is useful
for a tiny specialist, while the stock baseline should retain ownership of
non-tiny objects.

No ASCV-Loc checkpoint from the failed route may be resumed, used as
initialization, or reported as a successful endpoint.

Because this hypothesis was selected after inspecting seed-0 M500 on the fixed
647-image subset, that result is exploratory evidence only. The confirmatory
T-ASCV mechanism run uses a fresh seed-1 initial state under the new protocol.

## 3. Independent experts

### 3.1 Baseline expert

The baseline expert is the sealed Ultralytics RT-DETR-L matched baseline:

- Ultralytics `8.4.90`;
- scratch training, `pretrained=False`;
- VisDrone train/val protocol and hashes already frozen in the parent
  attestations;
- seed-specific common initial state;
- `imgsz=640`, `batch=8`, `workers=8`, one RTX 4090;
- fixed MuSGD, augmentation, query-count, AMP, and endpoint contracts.

Its checkpoint is immutable during tiny-expert training. It has independent
parameters, BatchNorm buffers, optimizer state, and inference execution.

### 3.2 Tiny expert

The tiny expert is a new method and protocol, provisionally named
**T-ASCV: Tiny-Only Asymmetric Cross-View Localization**.

It starts fresh from the same seed-specific common initial state as its matched
stock control. Its stock full-view detection loss remains unchanged. The only
new auxiliary term is:

- eligibility: complete target with full-view effective size `s <= 16`;
- teacher: matched local-view prediction, detached;
- student: matched full-view prediction;
- localization term: the frozen ASCV L1 plus GIoU term;
- the existing crop-v2, target identity, partial-object ignore,
  activation-checkpointing, BN-preservation, weight `0.1`, and three-epoch
  warm-up contracts remain unchanged.

There is no non-tiny auxiliary direction. This is a new single-purpose method,
not a threshold edit to the stopped ASCV-Loc experiment.

The failed ASCV-Loc evidence may motivate the new design but may not serve as
the new method's formal mechanism pass.

## 4. Inference views

The baseline expert runs on the full image only.

The tiny expert runs the frozen SBR local-view inference path: the full image
plus four fixed local tiles. All predictions are mapped to original-image
coordinates before routing.

Neither expert uses ground truth, target identity, training crop identity, or
validation annotations during inference.

## 5. Frozen scale-aware router

For a predicted box in original-image coordinates, define:

`gain = min(640 / image_width, 640 / image_height, 1)`.

`s_pred = sqrt(width_px * height_px) * gain`.

The implementation must call the existing frozen
`src.sbr_v2_audit.effective_size` function. This is the 640-network-frame
effective size used by the existing SBR metrics and the training mechanism;
raw original-image pixel area is forbidden.

For every routed prediction, `Detection.box` is the actual post-fusion box
that the final JSON emits. Scale classification, route invariants, matching,
fragment checks, and metric evaluation all consume that actual box.
`global_xyxy` remains authenticated source provenance only; for a fused
multi-view cluster it may describe the seed member and must not substitute
for the emitted box.

The router uses an analytic, pre-registered soft expert weight:

`tau = 16`

`k = ln(9) / 8`

`alpha(s) = sigmoid(k * (tau - s))`.

Thus a network-frame size of 8 gives local weight `0.9`, 16 gives `0.5`, and
24 gives `0.1`. `tau` is the frozen SBR tiny boundary, and `k` is derived from
the stated 8/24-pixel transition anchors. Neither value may be optimized on
val.

The soft weight operates only inside the tiny conflict-resolution path. Every
baseline prediction with `s_pred > 16`, including small, medium, and large
objects, is a protected baseline prediction. This safety envelope prevents the
known local-expert non-tiny weakness from leaking into the final system.

The exact router order is:

1. run each view with `conf=0.001`, per-view `max_det=300`, `NMS=False`;
2. map all tiny-expert tile detections to original-image coordinates;
3. run the frozen standard SBR full-plus-four-tile fusion for the tiny expert;
4. compute `s_pred` on each fused local-expert cluster output;
5. construct `B_non_tiny` from the baseline full-view final predictions with
   `s_pred > 16` and mark every member immutable;
6. form same-class cross-expert candidate pairs with IoU strictly greater than
   `0.5`, sort by descending IoU and then baseline order, local source ID,
   query ID, and original detection index, and greedily take a deterministic
   one-to-one matching;
7. retain every unmatched baseline prediction;
8. for a matched pair whose baseline box has `s_pred > 16`, retain the baseline
   prediction unchanged and discard the local member;
9. for a matched pair whose baseline box has `s_pred <= 16`, first require the
   local box to also have `s_pred <= 16`; if it does not, retain the baseline
   prediction and discard the local member; otherwise compute
   `alpha(s_pred)`, preserve the local box and class, and set
   `score=(1-alpha)*baseline_score+alpha*local_score`;
10. retain an unmatched local prediction only when its own `s_pred <= 16`, its
    provenance is complete, and it is not suppressed by the frozen
    `fragment_ios=0.5` rule against `B_non_tiny`;
11. reserve `B_non_tiny` in its original Arm-A relative order;
12. fill the remaining final `max_det=300` slots with all other retained
    predictions in descending final score and the frozen tie-break order.

Unmatched tiny-expert predictions with `s_pred > 16` are discarded before the
final merge. Baseline predictions in `B_non_tiny` are immutable: the router
cannot change their class, score, coordinates, query identity, relative order,
or source identity.

The frozen SBR fusion order, source priority, query identity, and cluster
ordering are reused unchanged. Local provenance contains image ID, view/tile
ID, query ID, mapped coordinates, and original detection index. Missing or
inconsistent provenance rejects the local prediction; it is never repaired by
lowering a score. Any remaining exact score tie is broken by source identifier,
query index, and original detection index, all ascending. No filesystem
enumeration order or hash-map iteration order may affect the output.

The soft weight changes only the score of a matched tiny pair. Coordinate
ownership is hard: the accepted local box is used for a matched tiny pair, and
every protected baseline box is used unchanged. No validation-fitted score
calibration or tuned score band is permitted.

The evaluator receives one ordinary prediction JSON. Metrics are never selected
from different experts after evaluation.

The router module has no annotation or dataset API. It writes and seals the
prediction JSON before the evaluator process is launched. Only the separate
evaluator may read ground truth.

## 6. Control and attribution

Three systems are distinguished:

- **Arm A:** `B(stage, seed)` with full-view inference;
- **Route control:** full-view non-tiny from `B(stage, seed)` plus five-view
  tiny from the same `B(stage, seed)`, using the SADED router;
- **Route treatment:** full-view non-tiny from `B(stage, seed)` plus five-view
  tiny from the independent `T(stage, seed)`, using the same router.

For every screen or formal stage and seed, the manifest binds exactly one stock
endpoint `B(stage, seed)` and one fresh T-ASCV endpoint `T(stage, seed)`.
`B(stage, seed)` supplies Arm A, the protected baseline expert, and the
route-control tiny expert. It is frozen while `T(stage, seed)` trains. The two
experts use distinct model instances, parameters, buffers, optimizer state,
run directories, and checkpoint hashes.

The formal five gates compare route treatment against Arm A:

- AP-tiny-SBR `>= +0.010`;
- mAP50-95 `>= +0.003`;
- tiny recall `>= +0.020`;
- AP75 `>= -0.002`;
- AP-large-SBR `>= -0.005`.

For seed 0, route treatment relative to route control must satisfy:

- mAP50-95 `> 0`;
- AP-tiny-SBR `>= 0`;
- tiny recall `>= 0`;
- AP75 `>= -0.002`;
- AP-large-SBR `>= -0.005`.

For the three-seed result, at least two seeds must have positive mAP50-95
difference, the mean mAP50-95 difference must be positive, at least two seeds
must have non-negative AP-tiny and tiny-recall differences, and the mean
differences must satisfy the same four secondary guards above.

These route-control comparisons separate tiny-expert training benefit from the
deterministic router itself.

All reported values come from the same unified prediction set per system.

## 7. Reuse and rerun boundary

Reusable immutable inputs:

- dataset and 647-image subset attestations;
- seed-specific common initial states;
- frozen baseline checkpoints and metrics only through a pre-registered
  control allowlist;
- frozen SBR raw caches for a router-integrity replay;
- external evaluator and metric definitions.

Must be rerun:

- T-ASCV paired preflight under a new commit and manifest;
- a fresh seed-1 T-ASCV mechanism run;
- every T-ASCV treatment training endpoint;
- any stock control endpoint that cannot be closed against the new manifest.

Never reusable as a scientific endpoint:

- ASCV-Loc M500 `last.pt` or `best.pt`;
- old preflight gates as authorization for T-ASCV;
- any INVALID or partial run.

Before any T-ASCV treatment starts, the new manifest records the only permitted
`B(stage, seed)` path and SHA256 for every planned stage and seed. A historical
stock endpoint is admitted only if its immutable provenance exactly matches
the required seed, common initial state, dataset/subset, batch canary, batch,
optimizer, schedule, source/upstream closure, fixed endpoint, evaluator, and
raw-prediction hashes. The resolver must find exactly one match without reading
metrics. Zero or multiple matches require a fresh stock-control run. Rebinding
records provenance; it never changes it, and no endpoint may be selected after
viewing performance.

## 8. Execution state machine

### R0: zero-inference router integrity

Use the already sealed baseline full-view and multi-view raw caches only.
Instantiate the SADED router with the stock baseline acting as both experts.
This does not estimate T-ASCV benefit. It verifies:

- no non-tiny baseline box is modified;
- no tiny-expert non-tiny box leaks into the final set;
- the predicted-size boundary is applied after coordinate remapping;
- fragment suppression is deterministic;
- `max_det=300` never removes a protected baseline non-tiny box;
- a single final prediction JSON is produced;
- cache and output checksums close.

R0 must emit a route-control prediction JSON, per-image protected counts,
remaining tiny-slot counts, and their min/median/max distribution. It then
performs one pre-registered development-val safety evaluation. Before any
training, route control must satisfy the Arm-A AP75 and AP-large guards
(`>= -0.002` and `>= -0.005`, respectively) and have at least one remaining
tiny slot in the aggregate. A deterministic/invariant failure is repaired as
software; a safety-gate failure is a router-design stop.

### PREFLIGHT_1

Run matched stock control and T-ASCV for one real batch from the same sealed
seed-0 initial state. Require identical batch canaries, batch/workers `8/8`,
MuSGD, AMP scale `128`, successful backward recomputation, no validation/test
loader, and source/checkpoint closure.

### TINY_MECHANISM_500

Run fresh seed-1 T-ASCV for exactly 500 successful batches and 106 optimizer
attempts on the frozen 647-image subset. Do not read val.

The pre-registered scientific window is batches `401..500`. Require:

- at least 100 tiny pairs;
- tiny pairs in at least 80 batches;
- pair-weighted tiny teacher advantage greater than zero;
- tiny teacher win rate strictly greater than `0.5`;
- no non-tiny auxiliary pairs or loss contribution;
- unchanged inference parameter/output schema;
- all runtime and evidence invariants.

The old ASCV-Loc result is not counted toward this gate.

### SCREEN_10

Use a staged cost-saving screen:

1. bind or run the unique `B(SCREEN_10, 0)` and train
   `T(SCREEN_10, 0)` to the fixed 10-epoch endpoint;
2. evaluate each endpoint once on development val;
3. stop unless treatment-minus-route-control satisfies seed-0 mAP `>0`,
   AP-tiny `>=0`, tiny recall `>=0`, AP75 `>=-0.002`, and
   AP-large `>=-0.005`;
4. only after seed 0 is positive, run seeds 1 and 2;
5. require the exact three-seed route-control attribution rules in Section 6.

Historical controls may replace a rerun only after exact attestation and
checkpoint/prediction rebinding.

### FORMAL_100

After screen GO:

1. run seed-0 T-ASCV fresh to epoch 100;
2. freeze epoch-100 `last.pt`; never select `best.pt`;
3. generate Arm A, route-control, and route-treatment unified predictions;
4. require seed 0 to pass all five gates and have positive treatment-minus-
   route-control mAP;
5. only then run seeds 1 and 2;
6. reapply the five gates to the three-seed mean and require at least two
   positive route-treatment-minus-control mAP seeds with positive mean, plus
   all secondary attribution guards in Section 6.

After the method, endpoints, router, thresholds, and paper tables are frozen,
generate and seal exactly nine test-dev prediction JSON files before any
test-dev annotation or result is opened: Arm A, route control, and route
treatment for each of seeds 0, 1, and 2. One adjudication batch then evaluates
that fixed set. The unique primary estimate is the arithmetic mean across the
three per-seed route-treatment-minus-Arm-A metric deltas, and it must pass the
same five gates. The three per-seed route-treatment-minus-route-control deltas
and their arithmetic mean are mandatory attribution outputs and must pass the
Section 6 attribution rules.

Do not ensemble checkpoints or predictions across seeds. Do not retune, return
to val, replace a JSON, or submit a second test-dev batch after opening the
result. Failure of the primary five-gate mean or the attribution mean is a
scientific stop. A second dataset is not an oracle prerequisite.

Only the final three-seed result plus the one frozen test-dev confirmation is
paper-ready.

## 9. Failure interpretation

- The stopped ASCV-Loc route remains stopped.
- T-ASCV mechanism failure stops T-ASCV; it is not repaired by changing the
  size boundary, loss weight, crop ratio, or tail gate after observing data.
- Router-integrity failure is debugged as software because R0 has no
  performance-selection role.
- Screen or formal performance failure is a scientific stop.
- `test-dev` remains unread until the method, router, and formal protocol are
  frozen. A second dataset is not an oracle prerequisite.

## 10. Main risk

The mechanism evidence gives high confidence that local predictions can teach
tiny full-view localization, but loss-space teacher advantage may not become
the required AP-tiny, recall, and mAP improvements. The staged seed-0 screen is
therefore the first performance stop-loss point. The independent baseline
expert and hard ownership rule are specifically intended to cap AP75 and large
regression risk.

The deployed system loads two RT-DETR-L checkpoints and executes six forwards
per image: one full-view baseline forward plus one full and four tile forwards
for the tiny expert, exposing at most 1,800 query opportunities before routing.
The paper must report end-to-end latency, throughput, peak VRAM, model storage,
and detector-forward count. It must not describe SADED as a low-overhead
method.

## 11. Paper positioning and ablations

SADED is one innovation point with three inseparable components:

1. the independent global/local dual-expert framework;
2. the tiny-only asymmetric cross-view consistency expert;
3. the scale-aware prediction router with a protected non-tiny envelope.

The paper does not present these as three unrelated innovations. The mandatory
ablation rows are:

- stock global baseline;
- local expert alone;
- route control with the stock model acting as both experts;
- complete SADED with the T-ASCV local expert.

The intended causal story is tested rather than assumed: local-expert
specialization should improve tiny behavior, route control measures the router
alone, and complete SADED must preserve the baseline-owned non-tiny results
while retaining a route-treatment gain.

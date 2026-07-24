# ASCV-Loc Design

Date: 2026-07-24

Status: frozen implementation target after authoritative `SP_PPAF_STOP`

## Objective

ASCV-Loc is the training-time fallback for SBR innovation 1. It is intended to
retain the small-object benefit of full-view plus local-view inference without
using post-processing rules that trade high-IoU localization quality for large
object recovery.

The method adds no inference-time parameters and does not change the frozen
SBR-C inference pipeline.

## Evidence motivating the route

The sealed SP-PPAF replay passed the tiny AP, tiny recall, and large AP gates,
but failed:

- `mAP50-95`: `-0.582477 pp` versus Arm A;
- `AP75`: `-2.289700 pp` versus Arm A.

P1, P2, and P3 were nearly identical. Candidate supply and hard scale
protection were therefore sufficient, while the shared non-large
ranking/localization path remained unsuitable. The frozen result does not
identify score-band and box-quality effects separately, so neither is tuned
after the result.

## Training views

Each ordinary training sample remains the full-view sample used by the stock
RT-DETR detection loss. During training only, one square local view is derived
from the already transformed full-view tensor:

- side ratio: exactly `0.60`;
- resize: bilinear back to the full tensor size;
- tile selection: deterministic target-anchored crop-v2 selection using the
  dataset-relative canonical image identity; absolute server prefixes,
  flattened batch offsets, worker count, and rank are absent from the hash;
- one local view per full view;
- full-view detection loss: unchanged;
- local-view detection loss: absent.

The local view is an auxiliary teacher/student view only. Because it has no
detection loss, an object cut by the tile boundary cannot be learned as local
background.

## Matched baseline contract

Control and treatment use the existing sealed VisDrone baseline protocol:

- scratch `RT-DETR-L`, `pretrained=False`;
- dataset SHA256
  `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`;
- the exact 647-image subset with SHA256
  `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`;
- the exact seed-specific common initial-state artifact;
- `imgsz=640`, `batch=8`, `workers=8`, single RTX 4090;
- `MuSGD`, `lr0=0.01`, `lrf=0.01`, `momentum=0.937`,
  `weight_decay=0.0005`, `nbs=64`;
- CUDA AMP with fixed scale `128` and disabled scale growth;
- the frozen augmentation settings of the matched baseline.

The local branch uses activation checkpointing, including its backward
recomputation, so the scientific batch remains eight on 24GB. OOM auto-reduce,
AMP-scale drift, optimizer drift, or initial-state drift is `INVALID`.

The D2 list is reused verbatim: raw file SHA256
`4BDEE4F03CC903422ADBBF4BD3511027628000DB578DEFC07DFE6E45F1E7CB60`,
semantic path-list SHA256
`52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`,
and count `647`. ASCV never selects a replacement subset.

The full dataset signature `FD92...BDAB` is a sealed parent attestation before
validation. Its canonical algorithm includes val files, so training-side
preflight verifies the pinned parent-attestation checksum and does not recompute
the dataset signature. Full recomputation is allowed only inside the authorized
external A/C evaluation process; test-dev remains forbidden.

## Crop-v2 canonicalization

The frozen crop protocol is `ascv-loc/crop-v2`. The input is exactly
`640x640`, and every crop is exactly `384x384`.

`canonical_image_id` is the resolved image path relative to the resolved
dataset root, serialized as POSIX text. Paths outside the root fail closed.
The base digest is:

`SHA256(b"ascv-loc/crop-v2\0image\0" + canonical_image_id)`.

For each image, target ordinals preserve their order in that image's transformed
target sequence. The first ordinal is selected by the first unsigned big-endian
64-bit digest word modulo the target count, then ordinals are tried circularly.
The origin digest uses the original ordinal:

`SHA256(b"ascv-loc/crop-v2\0origin\0" + id + b"\0" + ascii(ordinal))`.

Legal origins fully contain the anchor box. If no target fits, fallback x/y use
the second and third 64-bit words of the base digest modulo `257`. Complete-box
eligibility uses a fixed `1e-6` pixel tolerance.

## Target identity and partial-object rule

Every full-view target receives an identity equal to its index in the flattened
training batch. A target is eligible in the local view only if its complete
box lies inside the crop. A boundary-intersecting target is excluded from:

- the local Hungarian target set;
- the ASCV pair set; and
- every local auxiliary contribution.

This is the frozen partial-target ignore rule.

## Matching and directional localization

The final regular decoder layer is matched independently in the two views with
the stock RT-DETR Hungarian matcher. Denoising queries are excluded.

Matches are joined by target identity. Let `s` be the full-view target
effective size in pixels:

`s = sqrt(width_px * height_px)`.

The only scale boundary is the frozen SBR tiny boundary:

- `s <= 16`: the mapped local prediction is detached and teaches the full
  prediction;
- `s > 16`: the full prediction is detached, transformed into local
  coordinates, and teaches the local prediction.

Exactly one branch is detached for every pair. No classification, confidence,
EMA-teacher, feature, adapter, or query-ranking loss is permitted.

## Auxiliary loss

For paired normalized `xywh` boxes:

`L_ASCV = mean(L1(student, teacher) + 1 - GIoU(student, teacher))`.

The training objective is:

`L = L_RT-DETR(full) + 0.1 * w(epoch) * L_ASCV`,

where the frozen warm-up is:

`w(epoch) = min((epoch + 1) / 3, 1)`.

The model architecture, parameter count, and inference output are unchanged.
The local forward runs all BatchNorm layers with frozen running statistics
while keeping affine and upstream gradients live. This prevents the detached
teacher forward from changing the shared inference model through buffers.

## Runtime invariants

The implementation must prove:

1. inference and validation do not construct tiles or consume GT identities;
2. the stock detection criterion sees the full batch only;
3. local partial targets never enter local matching;
4. matched pairs share the exact same full target identity;
5. exactly one prediction in each pair is detached;
6. the auxiliary contribution is finite;
7. lambda, boundary, tile ratio, and warm-up equal the frozen values;
8. no training stage constructs a val/test loader or invokes an internal
   validator;
9. screen/full/seed checkpoints are evaluated once, after training, by the
   external frozen SBR evaluator;
10. `test-dev` is never read during feasibility work.

## Frozen execution state machine

1. `PREFLIGHT_1`
   - seed-0 control and ASCV each execute one real optimizer step;
   - exact batch `8`, `imgsz=640`, workers `8`, MuSGD, and fixed AMP scale
     `128`;
   - ASCV activation checkpointing must complete forward and backward
     recomputation below 24 GiB;
   - OOM, batch reduction, scale drift, source drift, or any val/test read is
     `INVALID`.
2. `MECHANISM_500`
   - hashed 10% training subset;
   - fresh scratch seed-0 initial state shared with the stock baseline;
   - 500 training batches;
   - no val read;
   - require finite loss, non-zero shared pairs, both scale directions, and
     unchanged inference parameter/output contract;
   - require exactly `500` successful batches and `106` optimizer attempts;
   - the full 500 batches establish validity and coverage;
   - the scientific direction gate is frozen to batches `401..500`: each
     direction requires at least 100 pairs, presence in at least 80 batches,
     pair-weighted mean teacher advantage greater than zero, and strict
     pairwise win rate greater than `0.5`.
3. `SCREEN_10`
   - paired stock control and ASCV treatment for seeds `0`, `1`, and `2`, each
     restarted from the same seed-specific scratch common state;
   - same hashed 10% subset;
   - ten epochs, matching the authoritative D2 screening protocol;
   - no internal val; one external SBR val evaluation at the end;
   - every run requires `810` successful batches and `145` optimizer attempts;
   - define `dC=T_C-Control_C`, `dA=T_A-Control_A`, and
     `DID=dC-dA`;
   - mAP requires at least two positive `dC` seeds, positive mean `dC`, at
     least two positive `DID` seeds, and positive mean `DID`;
   - every seed requires `T_C_mAP >= 0.8*Control_C_mAP`;
   - mean `dC` for AP-tiny-SBR, tiny recall, AP75, and AP-large-SBR must each
     be non-negative.
4. `SEED0_100`
   - fresh scratch 100-epoch seed-0 run;
   - epoch 100 `last.pt` is fixed before evaluation; `best.pt` is never used;
   - require the fixed checkpoint to pass all five gates.
   - also require treatment `C-control C` mAP greater than zero and mAP DID
     greater than zero.
5. `SEED1_100` and `SEED2_100`
   - run only after seed 0 passes;
   - fresh scratch runs with the same fixed epoch-100 rule;
   - used for the paper-ready three-seed result.

Any scientific gate failure yields `ASCV_LOC_STOP`. Only software or evidence
failures yield `INVALID` and may be repaired. There is no lambda, boundary,
tile-ratio, direction, quota, or threshold search on val.

## Paired control and attribution

Every screen and formal seed has a stock RT-DETR control started from the same
sealed scratch common state, using the same image manifest, seed, batch,
optimizer, schedule, and fixed endpoint. The control omits only ASCV-Loc.
Each fixed endpoint is evaluated once with both SBR Arm A and Arm C.

The evidence reports:

- treatment `C - A` for the frozen five gates;
- treatment `C - control C` to isolate training-time benefit;
- treatment `A - control A` to expose full-view drift; and
- treatment/control absolute metrics.

The 10-epoch development gate does not use the formal percentage-point
thresholds because scratch D2 metrics are near the numerical floor. It uses the
three-seed paired sign, mean, DID, no-collapse, and secondary non-regression
rules above.

The seed-100 treatment `C-A` gate remains fixed:

- AP-tiny-SBR `>= +0.010`;
- mAP50-95 `>= +0.003`;
- tiny recall `>= +0.020`;
- AP75 `>= -0.002`;
- AP-large-SBR `>= -0.005`.

Seed 0 must also have positive `dC` mAP and positive mAP DID before seeds 1/2.
Paper-ready evidence reapplies the five gates to the three-seed mean and
requires at least two positive `dC` mAP seeds with positive mean plus at least
two positive DID mAP seeds with positive mean.

## Data policy

VisDrone val is a development screen. The sealed test-dev split remains unread
until the innovation and training protocol are frozen. A second dataset is not
an oracle prerequisite.

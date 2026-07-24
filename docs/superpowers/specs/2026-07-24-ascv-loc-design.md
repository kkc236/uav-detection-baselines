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
- tile selection: deterministic target-anchored selection with
  `SHA256(protocol | im_file)` and a per-image annotation ordinal; flattened
  batch offsets, worker count, and rank are absent from the hash;
- one local view per full view;
- full-view detection loss: unchanged;
- local-view detection loss: absent.

The local view is an auxiliary teacher/student view only. Because it has no
detection loss, an object cut by the tile boundary cannot be learned as local
background.

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

1. `MECHANISM_500`
   - hashed 10% training subset;
   - fresh load from the frozen mature checkpoint;
   - 500 training batches;
   - no val read;
   - require finite loss, non-zero shared pairs, both scale directions, and
     unchanged inference parameter/output contract;
   - before any val read, require the mapped local teacher to beat the full
     prediction on tiny pairs and the full teacher to beat the mapped local
     prediction on non-tiny pairs, both by positive mean advantage and
     win-rate greater than `0.5` across exactly 500 successful batches.
2. `SCREEN_6`
   - restart from the same frozen mature checkpoint;
   - same hashed 10% subset;
   - six epochs;
   - no internal val; one external SBR val evaluation at the end;
   - require tiny AP, mAP, tiny recall, and AP75 to retain their original gate
     directions, and require large AP recovery of at least `1.5 pp` from Arm C
     (`delta versus Arm A >= -1.5 pp`).
3. `FULL_20`
   - fresh full-train run, not resumed from the screen;
   - no internal val; one external SBR val decision;
   - require all original five gates.
4. `SEED0_100`
   - fresh scratch 100-epoch seed-0 run;
   - epoch 100 `last.pt` is fixed before evaluation; `best.pt` is never used;
   - require the fixed checkpoint to pass all five gates.
5. `SEED1_100` and `SEED2_100`
   - run only after seed 0 passes;
   - fresh scratch runs with the same fixed epoch-100 rule;
   - used for the paper-ready three-seed result.

Any scientific gate failure yields `ASCV_LOC_STOP`. Only software or evidence
failures yield `INVALID` and may be repaired. There is no lambda, boundary,
tile-ratio, direction, quota, or threshold search on val.

## Paired control and attribution

`SCREEN_6` and `FULL_20` each have a stock RT-DETR control started from the
same frozen mature checkpoint, using the same image manifest, seed, batch,
optimizer, schedule, and number of epochs. The control omits only ASCV-Loc.
Each fixed endpoint is evaluated once with both SBR Arm A and Arm C.

The evidence reports:

- treatment `C - A` for the frozen five gates;
- treatment `C - control C` to isolate training-time benefit;
- treatment `A - control A` to expose full-view drift; and
- treatment/control absolute metrics.

The 6-epoch development gate uses the original first four thresholds and
`AP-large C-A >= -1.5 pp`. The 20-epoch and seed-100 gates use the original
five thresholds, including `AP-large C-A >= -0.5 pp`. A passing internal
`C-A` result is not reported as an absolute improvement if treatment C is
worse than control C; attribution requires the paired deltas.

## Data policy

VisDrone val is a development screen. The sealed test-dev split remains unread
until the innovation and training protocol are frozen. A second dataset is not
an oracle prerequisite.

# Ultralytics RT-DETR-L FDR-only Migration Design

## 1. Status and scientific claim

This document freezes an **official-mechanism-faithful FDR-only migration** from
D-FINE into Ultralytics RT-DETR-L. It does **not** reproduce the complete D-FINE
detector, training recipe, or reported D-FINE accuracy.

The permitted claim is:

> We migrate the Fine-grained Distribution Refinement (FDR) box-regression
> mechanism and its Fine-Grained Localization (FGL) supervision into the frozen
> Ultralytics RT-DETR-L baseline while retaining the baseline classification,
> query-selection, encoder, optimization, evaluation, and deployment contracts.

The forbidden claim is:

> This is an exact or complete reproduction of D-FINE.

## 2. Immutable upstream authority

The official D-FINE source is pinned before implementation:

```text
repository: https://github.com/Peterande/D-FINE
commit: 7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6
```

Only these commit-pinned files define the migrated FDR mechanics:

- [decoder and Integral](https://github.com/Peterande/D-FINE/blob/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/src/zoo/dfine/dfine_decoder.py)
- [weighting, distance2bbox, bbox2distance](https://github.com/Peterande/D-FINE/blob/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/src/zoo/dfine/dfine_utils.py)
- [FGL criterion](https://github.com/Peterande/D-FINE/blob/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/src/zoo/dfine/dfine_criterion.py)
- [official reg/loss constants](https://github.com/Peterande/D-FINE/blob/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/configs/dfine/include/dfine_hgnetv2.yml)
- [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/6cf58a87e3097e7d1f9be3e8693a93de-Paper-Conference.pdf)

Before any implementation commit, the repository must contain an immutable
authority record with the exact 40-character commit, source URLs, file SHA256
values, license notice, and vendored test-only reference formulas. A moving
branch such as `master` is never an authority.

## 3. Frozen experimental baseline

The control and FDR-only arms use the same frozen protocol:

```text
model: Ultralytics RT-DETR-L
ultralytics: 8.4.90
GPU: NVIDIA GeForce RTX 4090 24GB
driver: 550.142
python: 3.10.12
torch: 2.5.1+cu121
torchvision: 0.20.1+cu121
CUDA: 12.1

dataset: identical VisDrone train/val
train images: 6471
val images: 548
classes: 10
dataset SHA256: FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB
fixed 10% subset: 647 images
subset SHA256: 52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0

initialization: pretrained=False, scratch
screen: fixed 10% subset, seed0, 30 epochs
formal: full data, seed0, 100 epochs
imgsz: 640
batch: 8
workers: 8
device: 0
AMP: true, fixed scale 128
deterministic: true
cache: false

optimizer: MuSGD
lr0/lrf: 0.01/0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0
nbs: 64
cos_lr: false

queries: 300
max_det: 300
NMS: false

mosaic: 1.0
close_mosaic: 10
mixup: 0.0
scale/translate: 0.5/0.1
degrees/shear/perspective: 0/0/0
flipud/fliplr: 0/0.5
hsv_h/hsv_s/hsv_v: 0.015/0.7/0.4
cutmix/copy_paste: 0/0
```

Control and FDR-only must additionally share the same public parameter state,
sample order, augmentation random stream, validation preprocessing, class map,
checkpoint rules, and metric implementation. FDR-private parameters use a
separately recorded deterministic seed and must not consume the public RNG
stream.

## 4. Exact migration boundary

### 4.1 Included

- `reg_max = 32`, hence 33 bins per edge.
- `reg_scale = 4.0` and the official fixed `up = 0.5` authority.
- The official non-uniform weighting function and Integral operation.
- A traditional four-coordinate `pre_bbox_head` at decoder layer 0.
- Six per-layer distribution heads, each producing `4 * 33 = 132` logits.
- Cumulative residual distribution logits across decoder layers.
- Official `distance2bbox` and `bbox2distance` transformations.
- FGL adjacent-bin interpolation targets.
- Detached matched-IoU weighting of FGL.
- FGL weight `0.15`, with stock VFL/L1/GIoU weights retained as `1/5/2`.
- Normal-query, auxiliary-layer, and denoising execution, using the corresponding
  stock Ultralytics matching indices for each prediction group.

### 4.2 Explicitly excluded

- Decoupled Distillation Focal loss (DDF).
- Global Optimal Localization Self-Distillation (GO-LSD).
- Teacher/student targets, teacher logits, or teacher distributions.
- Localization Quality Estimation (LQE) or any change to classification logits.
- Target gating or confidence-based target filtering.
- Cross-layer matching-union logic introduced for GO-LSD.
- D-FINE backbone, Hybrid Encoder variants, query-selection variants, decoder
  gateway, altered attention point schedules, wider decoder layers, EMA, AdamW,
  Objects365 pretraining, and D-FINE data/training schedules.
- P2/P3, boundary, trajectory, LPR, quality reranking, NWD, or other prior
  experimental modules.

Any excluded symbol appearing in the FDR-only model, loss keys, state dict, or
runtime report is an engineering failure.

## 5. FDR architecture

### 5.1 Stock components retained

The RT-DETR-L backbone, feature projections, encoder, uncertainty-minimal query
selection, query count, decoder attention/FFN layers, classification heads,
denoising construction, and postprocessing remain stock Ultralytics 8.4.90.
Only the decoder box representation and its additional FGL supervision change.

### 5.2 Preliminary box

At layer 0, decoder hidden state `h_0` produces a traditional preliminary box:

```text
B_pre = sigmoid(pre_bbox_head(h_0) + inverse_sigmoid(R_0))
```

`pre_bbox_head` is a four-coordinate MLP with the stock layer-0 bbox-head
architecture. Its public-compatible tensors are initialized from the same
seed0 public state as the control's first bbox head. `B_pre.detach()` is the
fixed reference box for the distribution decoder.

### 5.3 Distribution residuals

For six decoder layers and batch size `B`:

```text
delta_logits_l: [B, 300, 132]
corner_logits_l: [B, 300, 4, 33]
```

The update is cumulative:

```text
Z_0 = Head_0(h_0)
Z_l = Z_(l-1) + Head_l(h_l + stop_gradient(h_(l-1))), l > 0
P_l = softmax(Z_l, dim=bin)
```

The `h_l + stop_gradient(h_(l-1))` head input follows the pinned official
decoder mechanism; it changes only the FDR head input, not the stock decoder
layer itself. `Z_(l-1)` remains connected for residual distribution learning,
while the box passed as the next decoder reference is detached exactly as in
the stock iterative-reference boundary.

### 5.4 Non-uniform Integral and box decoding

For each edge, the Integral computes:

```text
d_l = sum_n softmax(Z_l)[n] * W(n)
```

`W(n)` is copied mechanically from the pinned official `weighting_function`;
it is not re-derived, approximated, or tuned. The four offsets are decoded
relative to `B_pre` with the pinned official `distance2bbox` operation and
`reg_scale=4`.
The resulting `B_l` is the decoder-layer box output and `stop_gradient(B_l)` is
the next stock decoder reference.

The weighting vector is symmetric around its exact zero bin. The neutral
distribution contract is defined by the exact official zero-offset target
encoding, not by an approximate Gaussian or hand-selected center index.
Encoding then decoding a zero offset must return `B_pre` within the frozen
float32 tolerance, and an exact representable center-bin target must be exact.

### 5.5 Inference contract

The final decoder layer supplies `B_5`. Classification logits are untouched.
Ultralytics' flattened Query-by-class Top-300 selection is retained exactly;
boxes, scores, classes, ordering, `max_det=300`, and `NMS=False` interfaces do
not change. No FGL, reference tensor, or distribution tensor may enter the
deployed prediction schema.

## 6. FGL supervision and stock loss preservation

For each stock-matched prediction and GT box:

1. Use the detached preliminary reference and official `bbox2distance` to map
   the continuous four-edge target to the two adjacent non-uniform bins.
2. Compute left/right interpolation weights from the exact official weighting
   values.
3. Apply weighted cross entropy to the two adjacent bins.
4. Multiply each four-edge term by detached IoU between that layer's decoded
   prediction and its matched GT.
5. Normalize by the same positive-box authority used by the stock criterion.

The total loss is:

```text
L = L_VFL + 5 * L_L1 + 2 * L_GIoU + 0.15 * L_FGL
```

Stock VFL, L1, GIoU, matching costs, matching cardinality, auxiliary-loss
structure, and DN behavior are retained. FGL uses each stock prediction group's
existing one-to-one assignment. It must not request a second matcher call and
must not construct a cross-layer union.

The `FGL weight = 0` isolation gate means: for identical supplied predictions,
targets, and recorded stock assignments, the extended criterion's VFL/L1/GIoU
keys and total are exactly equal to the stock criterion. It does not claim that
a structurally different FDR model produces the same boxes as a stock model.

## 7. Initialization and paired authority

- All unchanged tensors load byte-for-byte from one seed0 public initial state.
- The traditional `pre_bbox_head` receives the control layer-0 bbox-head state.
- The six 132-output distribution heads are private and deterministically
  initialized without advancing the public RNG stream.
- Distribution-head final weights and biases start at zero, yielding a neutral,
  symmetric initial distribution residual.
- The protocol manifest records public/private key sets, tensor SHA256 values,
  optimizer group membership, upstream commit, source commit, dataset hashes,
  and environment versions.
- Missing, duplicated, or reclassified parameters are fatal before training.

## 8. Mandatory pre-30-epoch Gates

All gates are pass/fail. A failed gate blocks the 30-epoch screen; thresholds
must not be weakened after observing results.

### Gate F0: pinned-source golden parity

- Vendor only the required official formulas from commit
  `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6` for tests.
- CPU float64/float32 golden vectors must match the official weighting,
  Integral, `distance2bbox`, `bbox2distance`, and FGL adjacent-bin targets.
- Test endpoints, zero, both sides of zero, random in-range values, clipping,
  and out-of-range target saturation.

### Gate F1: neutral and loss isolation

- Neutral/zero-offset encode-decode returns the preliminary box.
- Zero cumulative residual preserves the previous cumulative distribution.
- `FGL weight = 0` gives exact stock VFL/L1/GIoU and total loss.
- Classification logits, matcher inputs/costs, Top-300, and NMS behavior remain
  stock exact for identical supplied tensors.

### Gate F2: shapes and edge cases

- Six distribution outputs have exact shape `[6, B, 300, 132]`.
- Decoded boxes are `[6, B, 300, 4]`; scores remain `[6, B, 300, 10]`.
- Normal queries, DN queries, empty GT, mixed empty/non-empty batches, boundary
  clipping, and auxiliary layers complete with finite forward/backward values.
- AMP with fixed scale 128 produces finite expected gradients and no skipped step.

### Gate F3: RTX 4090 one-step integration

- One real VisDrone batch on `cuda:0`, `imgsz=640`, `batch=8` completes forward,
  backward, MuSGD step, validation postprocess, and checkpoint round-trip.
- Every expected public/FDR-private parameter has finite gradients; excluded
  components and unexpected trainable parameters are absent.
- Public initialization and data-order hashes equal the baseline authority.

### Gate F4: representation oracle

Using frozen baseline references and matched GT only, encode GT edge offsets and
decode them through the FDR representation without training. Publish:

- reconstruction L1 and max error;
- per-edge and aggregate bin saturation rates;
- invalid/non-finite counts;
- width/height and tiny/small-object stratification.

Gate F4 passes only when all reconstructed values are finite, the implementation
matches the official reference, reconstruction error is within the predeclared
numeric tolerance, and saturation is reported rather than hidden. This oracle
tests representation correctness and coverage; it is not an AP result.

## 9. Training progression and unchanged detector Gate2

Only F0-F4 passing authorizes one fixed 10% subset, seed0, 30-epoch FDR-only
screen. The paired control is the existing seed0 baseline only if every frozen
authority hash matches; otherwise a new matched control is required.

The existing detector Gate2 thresholds are immutable and are not restated with
new values here. The evaluator must load them from the already frozen authority.
No result may advance by substituting a new threshold, another seed, a different
epoch cutoff, or a favorable checkpoint selected on official validation.

If the 30-epoch comparison passes Gate2, launch full-data seed0 100 epochs from
the formal seed0 initial state, not from the screen checkpoint. If it fails,
freeze FDR-only as `scientific_failed`; do not add excluded D-FINE components as
an unregistered rescue.

## 10. Immutable evidence, resume, and publication

Each run is a new immutable directory keyed by source commit, protocol digest,
variant, stage, and seed. Reports and manifests are create-only and use canonical
JSON. Every artifact records its SHA256 and byte length in an externally bound
manifest; symlinks and mutable in-place replacement are rejected.

Every completed epoch must publish, with retry and remote verification:

- metrics and loss components, including FGL;
- `last.pt` and scheduled epoch checkpoint digest;
- optimizer/AMP evidence and gradient summaries;
- public/private model and optimizer authority;
- environment, source, upstream, dataset, subset, sample-order, and augmentation
  digests;
- GPU memory, timing, parameter count, GFLOPs, and latency evidence when due;
- state-machine stage and next permitted action.

Resume is permitted only from the latest fully verified epoch whose checkpoint,
metrics, manifest, protocol, source, and optimizer state all match. A partial
epoch is discarded; a complete but unpublished epoch is verified and published
before further training. Formal 100-epoch training never resumes from a screen
checkpoint.

## 11. Success and failure semantics

- **Engineering failure:** source mismatch, golden mismatch, non-finite value,
  wrong shape, altered stock contract, missing publication, corrupt checkpoint,
  or failed resume authority. Repair with TDD and restart from a new immutable
  run identity when evidence integrity changed.
- **Scientific failure:** all engineering gates pass, but the unchanged detector
  Gate2 fails. Freeze and publish the result without threshold changes.
- **Screen success:** F0-F4 and the unchanged 30-epoch Gate2 pass.
- **Final success:** full-data seed0 100 epochs complete, every epoch is remotely
  verified, independent evaluation completes under the frozen authority, and
  the final comparison report is published.

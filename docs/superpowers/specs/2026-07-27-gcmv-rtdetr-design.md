# GCMV-RTDETR Network Design

> **Historical design:** Paper-facing PVC/GRCA/QCVR names in this document were
> superseded by the approved PLEC/GGLF/PEG freeze in
> `2026-07-27-gcmv-rtdetr-frozen-design.md`. Use the frozen document for all
> new implementation and paper wording.

Date: 2026-07-27

Status: proposed for written-spec review; implementation is not authorized until
the user approves this document.

## 1. Decision

The next network experiment will be:

**GCMV-RTDETR: Geometry-Canonical Multi-View RT-DETR**

Chinese name:

**几何规范化多视图 RT-DETR**

The method contains exactly three trainable structural modules:

1. **PVC — Phase-Preserving View Canonicalizer**  
   相位保持视图规范化模块。
2. **GRCA — Geometry-Restricted Reciprocal Correspondence Attention**  
   几何约束双向对应注意力模块。
3. **QCVR — Query-Conserving View-Scale Router**  
   查询守恒视图尺度路由模块。

All three modules are `torch.nn.Module` structures with trainable parameters,
forward computation, gradient flow, checkpoint state, model-configuration
entries, and independent ablations. Fixed cropping, coordinate transforms,
auxiliary supervision, and evaluation scripts support the modules but are not
claimed as innovation modules.

The architecture deliberately keeps the following RT-DETR-L components
unchanged:

- P4 and P5 supplied to the decoder;
- encoder score and box heads;
- uncertainty-minimal Top-300 query selection;
- total query count, exactly 300;
- decoder architecture and detection heads;
- Hungarian matching and the stock detection losses;
- inference `max_det=300` and NMS-disabled behavior.

Only the P3 memory supplied immediately before `RTDETRDecoder` is replaced by a
guarded, geometry-aligned global/local memory.

## 2. Evidence boundary

### 2.1 Positive evidence

The authoritative SADED-SM fresh-100 seed-0 result uses one RT-DETR-L checkpoint
with a full view and four overlapping high-resolution local views:

- global network input: 640;
- local network input: 1088;
- local tile ratio: 0.60;
- views: `full`, `TL`, `TR`, `BL`, `BR`;
- mAP50-95 delta: +1.5402 percentage points;
- AP-tiny-SBR delta: +1.8680 percentage points;
- tiny recall delta: +16.7220 percentage points;
- AP-large-SBR delta: -0.0428 percentage points.

This proves only that the local high-resolution observations contain useful tiny
object evidence and that retaining the full-view path protects larger objects.
It does not prove that PVC, GRCA, or QCVR will improve a trained network.

### 2.2 Negative evidence that constrains the design

- BTD-SE, P2 fusion, and P2 residual variants failed matched short and long
  screens. GCMV must not generate a supposed detail residual from the same
  already-downsampled global feature.
- VSF-RMR learned a scale-correlated field but degraded detection. GCMV must not
  route evidence among P3, P4, and P5 or modify all three levels.
- IOQC-SA reduced recall, so GCMV must not add query-competition penalties.
- BQP found only 446 safe pairs among 6,005 missed tiny targets, or 7.43%.
  GCMV must not promote candidates around the stock Top-300 boundary.
- T-ASCV stopped. GCMV must use local evidence at inference; it must not train a
  full-view student and remove the local evidence path at deployment.

## 3. Scientific problem chain

The method addresses one sequential problem rather than three unrelated
problems:

1. A 640-pixel full view loses sub-cell evidence for tiny targets. Actual
   high-resolution local observations are required.
2. Local observations use a different magnification, lose scene context, overlap
   each other, and contain crop-boundary fragments. Their features cannot be
   directly added to the full-view feature.
3. Even after correspondence interaction, local evidence must not globally
   overwrite the stock P3 representation. Only scale-appropriate and reliable
   local corrections may enter the single memory seen by stock query selection.

The corresponding module chain is:

```text
source-resolution image
    |
    +-- full view -> 640 -> shared RT-DETR-L -> global P3/P4/P5
    |
    +-- TL/TR/BL/BR, 60% overlap -> 1088
            -> shared RT-DETR-L backbone + hybrid encoder -> four local P3 maps
                    |
                    v
                  PVC
        local maps -> canonical local P3
                    |
                    v
                  GRCA
        global context <-> corresponding local detail
                    |
                    v
                  QCVR
        guarded single P3 + untouched P4/P5
                    |
                    v
        stock Top-300 + stock decoder + stock head
```

There is one model and one set of shared detector weights. The five view passes
are real computation and must not be described as a single forward, lightweight,
or zero-overhead inference.

## 4. Input and geometry contract

### 4.1 Source-resolution requirement

Local crops must be generated from the source-resolution training image after
shared geometric augmentation and before either branch is letterboxed. Cropping
an already-resized 640 tensor and enlarging it to 1088 is invalid because it
cannot recover pixels removed by the global resize.

The paired data path must therefore retain:

- the augmented source-resolution RGB tensor;
- the original image width and height after augmentation;
- transformed boxes and ignored regions in that source coordinate frame;
- the exact full-view letterbox transform;
- four exact local crop rectangles;
- the exact letterbox transform for every local crop.

The stock control must use the identical source image, geometric and photometric
augmentations, labels, ordering, optimizer, and effective batch. It discards the
four local tensors after dataset construction.

### 4.2 Frozen view construction

For a source image with width `W` and height `H`:

```text
tile_w = ceil(0.60 * W)
tile_h = ceil(0.60 * H)

TL = [0,          0,          tile_w, tile_h]
TR = [W-tile_w,   0,          W,      tile_h]
BL = [0,          H-tile_h,   tile_w, H]
BR = [W-tile_w,   H-tile_h,   W,      H]
```

Rectangles are half-open `[left, top, right, bottom)`, and the implementation
must reuse the established `overlapping_tiles()` helper rather than duplicate
these equations.

The full view is letterboxed to 640. Each tile is independently letterboxed to
1088. Padding value is 114. The transforms are centered and deterministic.

### 4.3 Augmentation boundary

The first implementation uses paired, correspondence-preserving augmentations:

- shared horizontal flip;
- shared HSV/color augmentation;
- shared resize and translation represented by an explicit affine transform;
- no mosaic, mixup, cutmix, or copy-paste.

Mosaic is excluded because a 640-only mosaic would destroy the source-resolution
evidence contract. A future high-resolution paired mosaic is outside this
design. The stock control for every GCMV comparison must use the same
augmentation boundary.

### 4.4 Feature contract

Let:

```text
G3: [B, 256, H3, W3]
G4: [B, 256, H4, W4]
G5: [B, 256, H5, W5]
Lv: [B, 256, Hv, Wv] for v in TL/TR/BL/BR
```

`G3/G4/G5` are the normal hybrid-encoder outputs immediately before
`RTDETRDecoder`. Each `Lv` is the local branch's corresponding semantic P3, not
P2 and not a shallow stem feature.

Spatial sizes are derived from the actual letterboxed tensors and detector
stride. No code may assume that local-to-global magnification is an integer.

## 5. Module A: PVC

### 5.1 Responsibility

PVC maps the four local semantic P3 maps into a single local representation on
the global P3 lattice without first averaging away within-cell phase.

PVC does not:

- read the global P3 feature;
- predict objectness, boxes, or query scores;
- select crops;
- fuse P3/P4/P5;
- modify RT-DETR queries.

### 5.2 Inputs and outputs

Inputs:

- four local P3 tensors;
- exact crop rectangles;
- per-view letterbox transforms;
- the global P3 spatial shape;
- per-view valid masks and normalized distance-to-crop-boundary maps.

Outputs:

```text
C3:          [B, 256, H3, W3]
valid_count: [B, 1,   H3, W3]
edge_prior:  [B, 1,   H3, W3]
```

`C3` is a canonical local feature. `valid_count` records how many views
contributed to each position. `edge_prior` is high only when the contributing
samples are far from crop boundaries. Masks and geometry are buffers, not
trainable innovations.

### 5.3 Trainable structure

For every global P3 cell center, the known crop-to-global transforms identify a
fixed `3 x 3` phase sample pattern in each covering local feature. Sampling uses
`grid_sample` with explicit normalized coordinates and `align_corners=False`.

Each sample is augmented with:

- a learned 256-dimensional view embedding;
- an MLP embedding of the continuous sub-cell offset;
- an MLP embedding of local magnification and valid-edge distance.

The nine phase samples per view are concatenated in a fixed order. A grouped
`1x1` projection followed by a depthwise `3x3` convolution and a pointwise
projection compresses them to 256 channels. Overlapping views are combined by a
masked softmax weight head.

Trainable parameters:

- phase-offset MLP;
- view embeddings;
- magnification/edge MLP;
- grouped and depthwise projections;
- overlap weight head;
- output normalization.

All mask-empty locations output exact zeros.

### 5.4 PVC ablations

Required comparisons:

1. bilinear canonicalization plus uniform overlap averaging;
2. phase sampling without phase/view embeddings;
3. PVC without learned overlap weights;
4. full PVC.

PVC is successful only if its downstream use improves tiny evidence over the
bilinear reference. Merely reconstructing a valid tensor is not a paper result.

## 6. Module B: GRCA

### 6.1 Responsibility

GRCA transfers full-view scene context to the local representation, then returns
a local-detail correction candidate at the same global positions.

The known crop geometry defines the attention center. Learned offsets are bounded
to a small neighborhood and cannot search the entire image.

GRCA does not directly overwrite `G3`. It returns a candidate correction for
QCVR to accept or reject.

### 6.2 Inputs and outputs

Inputs:

```text
G3, C3, valid_count, edge_prior
```

Outputs:

```text
C3_ctx: [B, 256, H3, W3]
D3:     [B, 256, H3, W3]
agreement: [B, 1, H3, W3]
```

### 6.3 Trainable structure

Local-context pass:

```text
C3_ctx = C3 + Wo_l(Attn(Q_l(C3), K_g(G3), V_g(G3), offsets_l))
```

Detail-return pass:

```text
D3 = Wo_g(Attn(Q_g(G3), K_l(C3_ctx), V_l(C3_ctx), offsets_g))
```

Both passes use four heads and four samples per head. Offset logits pass through
`tanh` and are multiplied by a frozen maximum radius of two global P3 cells.
Invalid local samples are masked before softmax.

The agreement map is:

```text
agreement = sigmoid(
    MLP([normalize(G3), normalize(C3_ctx),
         abs(normalize(G3) - normalize(C3_ctx)),
         edge_prior, log1p(valid_count)])
)
```

Trainable parameters:

- independent Q/K/V projections in both directions;
- two bounded-offset heads;
- two attention output projections;
- agreement MLP;
- normalization layers.

The projections producing `C3_ctx` and `D3` use standard finite initialization.
They do not affect the stock path at initialization because QCVR's channel guard
is exactly zero. A nonzero `D3` is required so the first detection backward can
move that guard away from zero.

### 6.4 GRCA ablations

Required comparisons:

1. direct canonical feature without interaction;
2. local-to-global only;
3. unrestricted global cross-attention with no geometry bound;
4. full reciprocal GRCA.

The unrestricted reference is necessary to prove that exact correspondence, not
generic attention capacity, is responsible for any gain.

## 7. Module C: QCVR

### 7.1 Responsibility

QCVR decides where the GRCA correction may modify global P3. It produces exactly
one P3 memory. Stock query selection never sees separate global and local pools.

This is view-scale routing at the same physical position. It is not routing
among P3/P4/P5 and is not query routing.

### 7.2 Inputs and output

Inputs:

```text
G3, D3, agreement, edge_prior, valid_count
```

Output:

```text
M3: [B, 256, H3, W3]
```

P4 and P5 bypass QCVR unchanged.

### 7.3 Trainable structure

The scale posterior predicts whether a location benefits from local
magnification:

```text
S = sigmoid(scale_head([G3, D3]))
```

The reliability posterior predicts whether the local evidence is safe:

```text
R = sigmoid(reliability_head(
    [G3, D3, agreement, edge_prior, log1p(valid_count)]
))
```

The guarded output is:

```text
M3 = G3 + tanh(gamma) * S * R * project(D3)
```

`gamma` has shape `[1, 256, 1, 1]` and is initialized to zero. `project` is a
depthwise `3x3` plus pointwise `1x1` block whose final normalization scale is
initialized to one. The stock path is therefore exactly preserved at
initialization.

Trainable parameters:

- scale posterior head;
- reliability posterior head;
- correction projection;
- bounded channel scale `gamma`.

### 7.4 Sparse auxiliary supervision

Auxiliary supervision supports QCVR but is not itself a contribution.

Scale targets are defined only at GT centers:

- positive when the effective global-view GT size is at most 16 pixels;
- negative when it is greater than 16 pixels;
- every non-center location is ignored.

Reliability targets are defined only for sampled locations:

- positive for a tiny GT completely contained in at least one contributing tile
  with its center away from the crop boundary;
- negative for a GT fragment that intersects a tile but is not completely
  contained, and for balanced hard-background samples;
- all other locations are ignored.

Positive and negative samples are balanced per batch. This avoids the historical
BTD-SE failure where dense background supervision dominated tiny targets.

The auxiliary weights are frozen for the first screen:

```text
lambda_scale = 0.05
lambda_reliability = 0.05
```

No weight search is permitted after reading the first screening result.

### 7.5 QCVR ablations

Required comparisons:

1. fixed scalar local addition;
2. scale posterior only;
3. reliability posterior only;
4. scale and reliability without zero-initialized channel guard;
5. full QCVR.

## 8. Gradient and initialization contract

The complete GCMV-off path must be numerically identical to stock RT-DETR-L.

With GCMV enabled at initialization:

- P4 and P5 are bitwise identical to stock;
- `M3` is bitwise identical to `G3` because `gamma=0`;
- stock Top-300 indices, decoder inputs, and predictions are bitwise identical;
- the first detection backward gives finite nonzero gradient to `gamma`;
- auxiliary supervision gives finite nonzero gradients to the PVC, GRCA, scale,
  and reliability parameters;
- after a simulated optimizer update moves `gamma`, detection gradients reach
  the local projections and correspondence attention;
- local and global view passes share model weights but must not update BatchNorm
  running buffers five times per training step.

BatchNorm uses the existing global-view update once. Local passes run with frozen
running statistics while affine parameters remain trainable. Tests must prove
that the local passes do not mutate running means or variances.

## 9. Model and configuration boundary

The implementation will use focused repository-owned files:

```text
src/gcmv_geometry.py      fixed differentiable coordinate construction
src/gcmv_pvc.py           PVC only
src/gcmv_grca.py          GRCA only
src/gcmv_qcvr.py          QCVR only
src/gcmv_data.py          paired source-resolution dataset/view bundle
src/rtdetr_gcmv.py        RT-DETR integration and auxiliary loss consumption
configs/rtdetr-l-gcmv.yaml
scripts/train_rtdetr_gcmv.py
scripts/audit_gcmv_g0.py
```

The configuration must expose:

```yaml
gcmv:
  enabled: true
  global_imgsz: 640
  local_imgsz: 1088
  tile_ratio: 0.60
  views: [TL, TR, BL, BR]
  p3_channels: 256
  phase_grid: 3
  attention_heads: 4
  samples_per_head: 4
  max_offset_cells: 2.0
  lambda_scale: 0.05
  lambda_reliability: 0.05
```

These values are frozen for G0 and the first screens.

No `site-packages` file may be edited. Ultralytics `8.4.90` is wrapped through
repository-owned model/trainer classes.

## 10. Compute and memory contract

The method intentionally pays for actual local evidence. The implementation must
measure rather than estimate:

- parameters;
- trainable parameters;
- GFLOPs for full plus four local feature passes;
- peak allocated and reserved CUDA memory;
- images per second and end-to-end latency;
- local view generation time;
- PVC, GRCA, and QCVR latency separately.

Local views are packed when memory permits and processed sequentially with
activation checkpointing otherwise. The scientific result must be invariant to
the packing choice.

The formal effective batch remains eight. Micro-batch size and gradient
accumulation may change only if both stock control and GCMV use the same effective
batch, optimizer step count, sample order, and learning-rate schedule. If the
model cannot complete one finite forward/backward step within a 24 GiB budget
under these rules, GCMV stops before training.

## 11. Evaluation and ablation design

### 11.1 Main progression

```text
stock RT-DETR-L
naive multi-view canonicalization and addition
+ PVC
+ PVC + GRCA
+ PVC + GRCA + QCVR
```

The naive multi-view row is a reference, not a claimed module.

### 11.2 Deletion ablations

The full model must also report:

```text
Full - PVC   (replace PVC with bilinear/uniform canonicalization)
Full - GRCA  (replace GRCA with direct projected local feature)
Full - QCVR  (replace QCVR with frozen scalar addition)
```

### 11.3 Mechanism diagnostics

Required diagnostics:

- tiny GT coverage before and after the P3 memory score head;
- target/background phase-feature separability;
- attention displacement relative to known correspondence;
- GRCA agreement on complete targets, fragments, and background;
- scale and reliability posterior distributions;
- QCVR effective residual norm;
- crop-boundary false positives;
- AP-tiny-SBR, tiny recall, AP-small/medium/large-SBR, AP75, and mAP50-95;
- stock Top-300 reproduction when GCMV is off.

Auxiliary losses or correlated gates cannot substitute for detection gains.

## 12. Fail-closed development gates

### 12.1 G0: zero/near-zero training feasibility

All conditions must pass:

1. GCMV-off reproduces stock outputs and Top-300 exactly.
2. Enabled zero initialization reproduces stock P3/P4/P5 and predictions exactly.
3. PVC geometry agrees with the established tile and letterbox mapping on fixed
   synthetic and real examples.
4. Four local-view predictions cover at least 15% of tiny GT missed by the
   full-view Top-300 on the frozen 647-image training subset, using the
   preregistered same-class IoU >= 0.50 definition.
5. PVC, GRCA, and QCVR all receive finite nonzero gradients under their intended
   detection or auxiliary paths.
6. Local passes do not mutate BatchNorm running buffers.
7. One effective-batch-eight forward/backward step is feasible on 24 GiB under
   the paired-control rules.

Failure of condition 4 means the local evidence source is insufficient under
this frozen construction. Thresholds, K, tile ratio, and IoU may not be changed
in response.

### 12.2 Three-epoch directional screen

Compared with the paired stock control at the fixed endpoint:

- mAP50-95 delta >= +0.002;
- AP-tiny-SBR delta >= 0;
- tiny recall delta >= +0.010;
- AP75 delta >= -0.005;
- AP-large-SBR delta >= -0.005;
- all tensors finite;
- no view, attention head, or gate collapse.

A `0.000x` mAP improvement is noise and does not pass.

### 12.3 Ten-epoch 10% screen

Compared with the paired stock control:

- mAP50-95 delta >= +0.003;
- AP-tiny-SBR delta >= +0.010;
- tiny recall delta >= +0.020;
- AP75 delta >= -0.002;
- AP-large-SBR delta >= -0.005;
- mean mAP delta over epochs 8-10 is positive;
- endpoint mAP is no more than 0.002 below the method's best screen epoch;
- every module's deletion damages its target metric by at least 0.002.

Only a complete pass authorizes a formal 100-epoch, seeds 0/1/2 experiment.
Failure does not authorize tuning thresholds or renaming a failed block as a
contribution.

## 13. Closest-work boundary

The defensible claim is narrow:

> GCMV uses a fixed exhaustive full-plus-four-view input, exact crop-to-global
> correspondence, phase-preserving canonicalization, bounded reciprocal
> interaction, and query-conserving pre-selection memory fusion.

Required distinctions:

- **SADED-SM:** independently decodes five views and routes final predictions;
  GCMV learns cross-view representations and emits one prediction set from one
  pre-query memory.
- **QueryDet/ESOD:** use coarse object seeking to trigger sparse high-resolution
  computation; GCMV selects no patch and uses fixed exhaustive coverage.
- **DQ-DETR/Dome-DETR:** alter query number, density, or initialization; GCMV
  leaves Top-300 and decoder unchanged.
- **UHR-DETR:** uses sparse high-resolution coverage and a global-local decoder;
  GCMV collapses fixed views before unchanged query selection and decoding.
- **SET/LGI-DETR:** enhance frequency or cross-level features within one view;
  GCMV exchanges actual full/crop observations of the same scene.
- **VSF-RMR:** routes among FPN levels of one view; GCMV changes only P3 and
  routes between magnifications of one canonical location.

The paper must not claim first global-local detection, first tiling, first
multi-view DETR, first dynamic query, real-time operation, lightweight
operation, uniform all-scale gains, or that SADED proves the new modules.

## 14. Stop conditions

The project stops this design rather than patching it when any of these occurs:

- source-resolution paired views cannot be supplied without changing labels or
  losing the matched-control contract;
- G0 local coverage is below 15%;
- the method cannot fit the 24 GiB/effective-batch contract;
- PVC reduces to bilinear/uniform behavior;
- GRCA offsets consistently saturate at their bound or ignore correspondence;
- QCVR drives its residual to zero or reproduces the BTD-SE background-dominant
  failure;
- the ten-epoch gate fails.

There is no automatic fallback to P2 fusion, scale-field routing, query
promotion, extra queries, dynamic cropping, or teacher-only distillation. A
failed gate requires a new evidence review and a new approved design.

## 15. Completion criteria for local implementation

The local implementation is complete only when:

- the three modules, paired data contract, integration wrapper, configuration,
  and G0 audit exist in repository-owned files;
- all unit and integration tests pass;
- GCMV-off and zero-initialized GCMV reproduce stock outputs;
- a CPU synthetic forward/backward proves finite gradients;
- a local CUDA smoke test is run if a compatible GPU is available;
- configuration loading, checkpoint round-trip, EMA registration, and export-off
  behavior are tested;
- no remote server is started and no formal performance claim is made.

Server training begins only after the user separately authorizes it.

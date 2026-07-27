# GCMV-RTDETR Frozen Design

**Status:** Approved and frozen on 2026-07-27.

**Method:** GCMV-RTDETR — Geometry-Canonical Multi-View Evidence Injection
for RT-DETR.

**Chinese name:** 几何规范化多视图证据注入 RT-DETR.

This specification supersedes the paper-facing PVC/GRCA/QCVR names in
`2026-07-27-gcmv-rtdetr-design.md`. The original document remains historical
design evidence. The frozen module names are PLEC, GGLF, and PEG.

## 1. Scientific claim

Global resizing removes observable evidence from tiny UAV targets. Real
source-resolution local views recover some of that evidence, but their feature
coordinates, sampling phase, context, overlap, and crop-boundary reliability
differ from the full view.

GCMV-RTDETR addresses one sequential question:

> How can a detector inject real local high-resolution evidence into a global
> representation while preserving RT-DETR's original set-prediction mechanism?

The method is one framework with three trainable and independently ablatable
network stages:

1. **PLEC — Phase-Preserving Local Evidence Canonicalizer**
2. **GGLF — Geometry-Constrained Global-Local Fusion**
3. **PEG — Protected Evidence Gate**

The paper may present three contributions, but they must be explained as one
causal chain rather than three unrelated enhancement blocks.

## 2. Frozen detector boundary

The full view is letterboxed to `640 x 640`. Four source-resolution TL/TR/BL/BR
views use the established 60%-overlap construction and are independently
letterboxed to `1088 x 1088`.

All five passes share detector weights. They run through the RT-DETR backbone
and Hybrid Encoder. The four local passes stop after producing their semantic
P3 maps; they do not execute Top-300 selection, the decoder, detection heads,
or post-processing.

The frozen data flow is:

```text
source-resolution image
    |
    +-- full view -> 640 -> shared Backbone + Hybrid Encoder
    |                         -> G3, G4, G5
    |
    +-- four 60%-overlap crops -> 1088
              -> shared Backbone + Hybrid Encoder
              -> L3_TL, L3_TR, L3_BL, L3_BR
                              |
                              v
                            PLEC
                              |
                              v
                            L3c
                              |
                    G3 ------ GGLF
                              |
                              v
                         correction Δ3
                              |
                    G3 ------- PEG
                              |
                              v
                         one memory M3
                              |
                     M3, unchanged G4, G5
                              |
                              v
                 stock Top-300 + stock Decoder + stock heads
```

Fusion occurs **after the Hybrid Encoder and immediately before
`RTDETRDecoder`**. This position is frozen.

GCMV must not change:

- query count;
- Top-300 selection logic;
- decoder layers or decoder attention;
- Hungarian matching;
- box/classification heads;
- P4 or P5;
- inference post-processing.

Only the single P3 memory supplied to stock query selection may change.

## 3. Source-resolution and geometry contract

Local crops are produced from the augmented source-resolution image before the
global 640 letterbox. Cropping an already resized 640 tensor and enlarging it is
invalid.

The first implementation retains the previously frozen construction:

```text
tile_w = ceil(0.60 * source_width)
tile_h = ceil(0.60 * source_height)
views  = [TL, TR, BL, BR]
```

Rectangles are half-open. Code reuses `src.sbr_geometry.overlapping_tiles()`.
Every batch carries exact full-view and local-view letterbox transforms.

No implementation may assume:

- an integer local/global magnification;
- a fixed source aspect ratio;
- a fixed P3 height or width;
- that letterbox padding is zero;
- that a crop-boundary sample is reliable.

## 4. Module 1: PLEC

### 4.1 Name and responsibility

**PLEC — Phase-Preserving Local Evidence Canonicalizer**

Chinese: **相位保持局部证据规范化模块**.

PLEC transforms four local semantic P3 maps into one canonical local evidence
map on the full-view P3 lattice. It solves coordinate, magnification, sampling
phase, overlap, and boundary-validity mismatch.

PLEC is not a resize operation. The fixed geometry provides correspondence;
the trainable network decides how to encode the corresponding multi-phase local
evidence.

PLEC does not:

- read global `G3`;
- produce a global/local correction;
- decide how much evidence enters `G3`;
- predict queries, boxes, objectness, or classes;
- use P4/P5;
- choose crops.

These exclusions keep PLEC separable from GGLF and PEG.

### 4.2 Inputs and outputs

Inputs:

```text
four local semantic features:
    L3_v: [B, 256, Hl, Wl], v in [TL, TR, BL, BR]

frozen metadata:
    source shape
    crop rectangles
    global and local letterbox transforms
    global P3 shape
```

Outputs:

```text
L3c:             [B, 256, Hg, Wg]
valid_count:     [B,   1, Hg, Wg]
edge_prior:      [B,   1, Hg, Wg]
overlap_weights: [B,   4,  1, Hg, Wg]  # diagnostic output
```

All locations with no valid local evidence output exact zeros.

### 4.3 Exact geometry

For each full-view P3 cell, PLEC evaluates a fixed row-major phase table:

```text
(-1/3,-1/3), (0,-1/3), (+1/3,-1/3),
(-1/3,   0), (0,   0), (+1/3,   0),
(-1/3,+1/3), (0,+1/3), (+1/3,+1/3)
```

Each phase point follows this exact transform chain:

```text
global P3 phase point
  -> global network pixel
  -> inverse global letterbox
  -> source-image coordinate
  -> local crop coordinate
  -> local letterbox pixel
  -> local P3 coordinate
  -> align_corners=False normalized grid
```

The geometry builder also emits:

- per-phase validity;
- continuous subcell offset;
- non-quantized x/y magnification;
- normalized distance to the local crop boundary.

Geometry tensors are buffers and receive no gradient. This fixed mapping is
necessary infrastructure, not the claimed trainable contribution.

### 4.4 Trainable structure

Each valid local phase sample is enriched by addition:

```text
sampled local feature
  + learned TL/TR/BL/BR view embedding
  + MLP(continuous x/y subcell phase)
  + MLP(log2 x/y magnification, crop-edge distance)
```

The enriched phases are remasked, arranged channel-major, and compressed:

```text
9 phase values per channel
  -> grouped 1x1 phase reduction, groups=256
  -> SiLU
  -> depthwise 3x3 spatial mixing
  -> SiLU
  -> pointwise 1x1 projection
```

The four canonical view candidates are fused with a masked learned overlap
softmax. Invalid views receive exactly zero weight. Output normalization is
channel-only LayerNorm at each spatial position; empty positions are remasked
after normalization.

Trainable PLEC families:

- four 256-dimensional view embeddings;
- phase-offset MLP;
- magnification/boundary MLP;
- grouped phase reducer;
- depthwise spatial mixer;
- pointwise projection;
- overlap-weight head;
- output normalization.

The frozen parameter budget is at most 200,000 trainable parameters at 256
channels.

### 4.5 PLEC initialization and gradient contract

PLEC itself need not reproduce an identity mapping because it has no direct
global path. In the full model PEG's zero-initialized residual guard makes the
enabled model reproduce stock `G3` at initialization.

Local P3 tensors and every enabled PLEC parameter family must receive finite
nonzero gradients under the intended training path. Geometry tensors must never
receive gradients.

### 4.6 PLEC references and ablations

Required PLEC comparisons:

1. center-phase bilinear sampling plus uniform overlap averaging;
2. nine-phase sampling without phase/view embeddings;
3. PLEC with uniform overlap weights;
4. full PLEC.

The canonicalization mechanism is supported only if the full PLEC improves
downstream tiny detection evidence over the same bilinear reference. Producing
a geometrically valid tensor is not an accuracy result.

Required PLEC diagnostics:

- feature correspondence error under known transforms;
- target/background phase-feature separability;
- overlap weights by view and crop region;
- crop-boundary false-positive rate;
- tiny targets covered by local features but missed by global Top-300;
- gradient norms for every PLEC family.

## 5. Module 2: GGLF

**GGLF — Geometry-Constrained Global-Local Fusion**

Chinese: **几何约束全局—局部融合模块**.

Inputs:

```text
G3, L3c, valid_count, edge_prior
```

Output:

```text
Δ3: [B, 256, Hg, Wg]
correspondence_confidence: [B, 1, Hg, Wg]
```

GGLF exchanges global context and local detail only within a bounded window
centered by the exact crop correspondence. It must not perform unrestricted
global search. The trainable interaction may use sparse attention internally,
but the paper's claim is the deterministic geometric constraint, not attention
itself.

GGLF does not overwrite `G3`; it produces a correction candidate for PEG.

Detailed GGLF implementation remains a separate post-PLEC design/plan cycle.

## 6. Module 3: PEG

**PEG — Protected Evidence Gate**

Chinese: **保护式证据门控模块**.

Inputs:

```text
G3, Δ3, correspondence_confidence, edge_prior, valid_count
```

Output:

```text
M3 = G3 + gamma * g(x,y) * Δ3
```

The learned gate `g(x,y)` is driven by:

- a tiny-benefit/scale posterior;
- global/local correspondence confidence;
- crop-boundary reliability and valid-view coverage.

`gamma` is a zero-initialized channel guard. At initialization:

```text
M3 == G3
```

exactly. P4/P5, Top-300, and the decoder remain stock.

PEG is evidence gating, not query routing and not a mixture of experts.

Detailed PEG implementation remains a separate post-GGLF design/plan cycle.

## 7. Stage-wise experimental adapter

PLEC and GGLF must be independently testable before PEG exists. All pre-PEG
detection experiments therefore use the same non-contribution reference
adapter:

```text
M3_ref = G3 + gamma_ref * Project(Δ_ref)
```

where:

- `Project` is one shared `1x1` projection used unchanged in every pre-PEG row;
- `gamma_ref` is one trainable scalar initialized to zero;
- the adapter has no spatial gate, scale posterior, reliability predictor, or
  boundary reasoning;
- the adapter is reported as experimental plumbing, not a contribution.

For the PLEC row, `Δ_ref = L3c`. For the GGLF row, `Δ_ref = Δ3`.

This common adapter prevents the PLEC ablation from having no detection path and
prevents a PEG-like mechanism from being introduced early.

## 8. Frozen progressive ablation

Main progression:

| Row | Local views | PLEC | GGLF | PEG | Purpose |
| --- | --- | --- | --- | --- | --- |
| Stock RT-DETR-L | no | no | no | no | baseline |
| Naive local | yes | bilinear/uniform | no | reference adapter | prove local evidence source |
| PLEC | yes | yes | no | reference adapter | prove canonicalization |
| PLEC + GGLF | yes | yes | yes | reference adapter | prove constrained interaction |
| Full GCMV | yes | yes | yes | yes | prove protected injection |

Deletion ablations:

- replace PLEC with bilinear/uniform canonicalization;
- replace GGLF with direct projected `L3c`;
- replace PEG with the common scalar reference adapter.

No row may modify query selection, decoder, matcher, P4, or P5.

## 9. Efficiency and reporting boundary

GCMV intentionally pays for real local evidence. It must not be described as
lightweight, zero-overhead, a single forward, or one complete detector pass.

Accurate wording:

> shared-weight multi-view feature enhancement with local passes terminating
> before query selection and decoding.

Required reporting:

- model and trainable parameters;
- total GFLOPs for one full and four local feature passes;
- peak allocated/reserved CUDA memory;
- end-to-end batch-1 latency and throughput;
- view generation, PLEC, GGLF, and PEG latency separately;
- packed versus sequential local execution.

## 10. Closest-method boundary

- **SAHI:** sliced input and output-level detection fusion. GCMV performs
  trainable feature-level canonical injection before one stock decoder.
- **Deformable DETR:** learns sparse sampling offsets around reference points.
  GCMV's cross-view center and admissible region come from exact crop geometry.
- **QueryDet:** predicts coarse sparse locations to trigger high-resolution
  computation. GCMV does not add a query map or sparse detection head.
- **UHR-DETR:** uses sparse high-resolution coverage and a global-local decoder.
  GCMV uses fixed same-image crop correspondence, creates one pre-query P3
  memory, and leaves the RT-DETR decoder unchanged.

The novelty claim must be:

> phase-preserving geometry canonicalization, correspondence-constrained
> pre-query interaction, and protected residual evidence injection.

It must not be:

> another attention module for small objects.

## 11. Local-first development order

The execution order is frozen:

1. exact PLEC geometry and parameter-free bilinear reference;
2. trainable PLEC and local structural/gradient verification;
3. server-side PLEC screen after the user starts the server;
4. separate GGLF design and local implementation;
5. separate PEG design and local implementation;
6. full GCMV server experiments.

The current local phase is limited to steps 1 and 2: step 1 is complete and
step 2 is in progress. It must not start a server, dataset training, GGLF, or
PEG.

## 12. Fail-closed PLEC gate

PLEC may advance to a server screen only if:

1. exact geometry matches established crop/letterbox helpers;
2. non-integer magnification is preserved;
3. phase order and `align_corners=False` are tested;
4. invalid locations and overlap weights are exact zeros;
5. overlap weights sum to one only over valid views;
6. all enabled PLEC parameter families receive finite nonzero gradients;
7. trainable parameters do not exceed 200,000;
8. focused and full local regression suites introduce no failure;
9. no RT-DETR, GGLF, PEG, SADED post-processing, or query logic is imported by
   the standalone PLEC module.

Passing this gate proves structure and trainability only. It does not prove an
AP improvement or publication-level novelty.

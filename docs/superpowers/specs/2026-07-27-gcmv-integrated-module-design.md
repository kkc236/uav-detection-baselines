# GCMV Integrated Network Module Design

**Status:** Frozen for direct end-to-end seed0 screening

**Paper role:** One network innovation point, not three parallel contributions

**Name:** GCMV-EI — Geometry-Canonical Multi-View Evidence Injection Module

## 1. Innovation boundary

GCMV-EI is one trainable network module inserted after RT-DETR-L's Hybrid
Encoder and before its unchanged query selection and decoder. PLEC, GGLF, and
PEG are internal stages of this one module:

```text
four local P3
  -> PLEC geometry canonicalization
  -> GGLF bounded global/local interaction
  -> PEG protected residual injection
  -> one enhanced global P3
```

The paper-level innovation is:

> an integrated geometry-canonical tiny-evidence injection module that aligns
> source-resolution local evidence, exchanges it with global context only
> around deterministic correspondences, and injects it through a protected
> residual path.

PLEC, GGLF, and PEG remain named internal mechanisms for explanation and
ablation, but this seed0 experiment validates the complete GCMV-EI module only.

## 2. Frozen placement and untouched baseline

The module receives semantic P3 after Hybrid Encoder layer 21:

```text
G3, local P3, exact crop/augmentation geometry
  -> GCMV-EI
  -> M3
  -> stock Top-300 query selection
  -> stock RT-DETR decoder
```

P4, P5, query count, query selection, decoder, matcher, loss, and detection head
remain stock. The global backbone and Hybrid Encoder receive gradients only
through the global image. Four local passes share current weights, terminate at
P3, preserve BatchNorm buffers, and are detached before GCMV-EI.

## 3. Internal stage A: PLEC

The existing verified 122,497-parameter PLEC core is retained unchanged. It
maps four source-resolution local P3 maps onto the global P3 lattice using
exact crop, resize, augmentation, phase, overlap, and boundary geometry:

```text
(L3_TL, L3_TR, L3_BL, L3_BR, T)
  -> (L3c, valid_count, edge_prior, overlap_weights)
```

PLEC owns trainable view/phase/scale embeddings, grouped phase reduction,
depthwise/pointwise encoding, learned overlap fusion, and channel
normalization.

## 4. Internal stage B: GGLF

GGLF uses deterministic same-lattice correspondence after PLEC. It does not
search the whole feature map.

Inputs:

```text
G3:          [B, 256, H, W]
L3c:         [B, 256, H, W]
valid_count: [B,   1, H, W]
edge_prior:  [B,   1, H, W]
```

The trainable interaction projects global queries and local keys/values to 64
channels. For each global cell, keys and values are unfolded only from its
fixed 3-by-3 canonical neighborhood. Attention logits include the local
query/key similarity and a reliability prior from valid-view coverage and crop
edge distance. Invalid locations are excluded before softmax.

```text
Q = Conv1x1(Norm(G3))
K,V = Conv1x1(Norm(L3c))
A_p = softmax_p(Q · K_p / 8 + log R_p), p in fixed 3x3 window
C = sum_p A_p V_p
```

The correction candidate is:

```text
Delta3 = Conv1x1([G3, L3c, C])
```

and correspondence confidence is predicted from query/center-key agreement,
attention concentration, valid coverage, and edge reliability. Both outputs
are masked to exact zero where no local evidence is valid.

The 3-by-3 window is fixed for the first screen. It is the smallest window that
allows a one-cell discretization tolerance while keeping interaction strictly
local and memory bounded.

## 5. Internal stage C: PEG

PEG never replaces or multiplicatively suppresses the global representation.
It predicts a conservative spatial gate from:

- global/local correction magnitude;
- a learned local-detail benefit posterior;
- GGLF correspondence confidence;
- PLEC edge reliability;
- normalized valid-view coverage.

```text
b = sigmoid(BenefitHead([G3, abs(Delta3)]))
g = sigmoid(GateHead([b, confidence, edge_prior, valid_count/4]))
M3 = G3 + gamma * g * Delta3
```

`gamma` is a 256-channel parameter initialized to exact zero. The gate's final
bias is initialized to `-2`, so the residual opens conservatively after the
channel guard starts learning. At initialization:

```text
M3 == G3
```

bit for bit. The global identity path is never gated.

## 6. Gradient and initialization contract

- Local pixels, local P3 tensors, and the local backbone path own no autograd
  graph.
- Global detector weights train only through the stock global path plus the
  downstream effect of `M3`.
- PLEC, GGLF, and PEG are trained by the unchanged detection loss.
- Geometry owns no gradient.
- In an audit with `gamma=1`, every enabled PLEC, GGLF, and PEG parameter
  family must receive a finite nonzero gradient.
- With the frozen zero initialization, only the PEG channel guard is required
  to receive a nonzero gradient on the first backward pass; upstream module
  gradients open after the guard's first optimizer update.

## 7. Success-first experimental contract

The earlier PLEC-only reference adapter is deleted from the formal path. The
first scientific screen compares:

```text
stock-equivalent control
vs
complete PLEC + GGLF + PEG GCMV-EI
```

Both arms use the same GCMV-capable model initialization, global data path,
seed0 scratch stock state, fixed 647-image list, frozen augmentation, MuSGD,
batch 8, fixed AMP scale 128, and 10 epochs. The control bypasses the complete
module.

The screen advances only if:

- delta mAP50-95 is strictly positive;
- delta AP-tiny is nonnegative;
- delta tiny recall is nonnegative;
- delta AP75 is at least `-0.002`;
- delta AP-large is at least `-0.005`.

If batch 8 exceeds 23 GiB reserved memory, automatic batch reduction is
forbidden. The implementation must first reduce activation memory without
changing mathematical outputs or the frozen batch.

## 8. Required diagnostics

The end-to-end run records:

- total and per-stage trainable parameters;
- PLEC valid coverage and overlap statistics;
- GGLF attention entropy and correspondence confidence;
- PEG benefit posterior, gate, and channel-guard distributions;
- local-path detachment and BatchNorm preservation;
- stage-wise gradient norms in the audit;
- peak allocated/reserved CUDA memory;
- full detection metrics and scale-stratified deltas.

## 9. Not part of this innovation

GCMV-EI is not:

- a mixture of experts or router;
- a separate non-tiny expert;
- output-box fusion or SAHI post-processing;
- dynamic crop selection;
- query routing or query enhancement;
- a decoder modification;
- a new loss branch.

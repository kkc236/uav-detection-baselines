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

The paper-level innovation is an integrated geometry-canonical tiny-evidence
injection module that aligns source-resolution local evidence, exchanges it
with global context only around deterministic correspondences, and injects it
through a protected residual path.

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

P4, P5, query count, query selection, decoder, matcher, and detection head
remain stock. The original detection objective is retained and three
module-internal auxiliary terms supervise tiny demand, evidence opening, and
non-tiny protection. The global backbone and Hybrid Encoder receive gradients
only through the global image. Four local passes share current weights,
terminate at P3, preserve BatchNorm buffers, and are detached before GCMV-EI.

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

The interaction projects global queries and local keys/values to 64 channels
split across four attention heads. For each global cell, keys and values are
unfolded only from its fixed 3-by-3 canonical neighborhood. Attention logits
combine global/local query-key similarity, a learned relative position bias,
and a reliability prior from valid-view coverage and crop-edge distance.
Invalid locations are excluded before softmax.

```text
Q = Conv1x1(Norm(G3))
K,V = Conv1x1(Norm(L3c))
A_hp = softmax_p(Q_h dot K_hp / 4 + r_hp + log R_p)
C = Concat_h(sum_p A_hp V_hp)
```

The attended local evidence is compared explicitly with the projected global
feature through a difference descriptor:

```text
D = [C, Gp, C - Gp, abs(C - Gp)]
E3 = EvidenceEncoder(D)
t = sigmoid(TinyHead(E3))
```

Correspondence confidence is the detached product of semantic agreement,
attention concentration, valid coverage, and PLEC confidence. Evidence,
tiny-demand, and correspondence maps are masked to exact zero where no local
evidence is valid.

The 3-by-3 window is fixed for the first screen. It is the smallest window that
allows one-cell discretization tolerance while keeping interaction strictly
local and memory bounded.

## 5. Internal stage C: PEG

PEG never replaces or multiplicatively suppresses the global representation.
It predicts a spatial evidence gate from reduced global/evidence features,
their absolute difference, the GGLF tiny-demand and correspondence maps, PLEC
confidence, and edge reliability:

```text
g_hat = sigmoid(GateHead([Gr, Er, abs(Gr-Er), t, c, p, e]))
r = coverage * cubert(p * c * e)
g = g_hat * r
gamma = tanh(rho)
M3 = G3 + gamma * g * Project(E3)
```

`rho` is one trainable scalar initialized to exact zero. The gate's final
weight and bias are also zero initialized, so `g_hat=0.5` before reliability
masking. At initialization:

```text
M3 == G3
```

bit for bit. The global identity path is never gated.

## 6. Gradient and supervision contract

- Local pixels, local P3 tensors, and the local backbone path own no autograd
  graph.
- Global detector weights train only through the stock global path plus the
  downstream effect of `M3`.
- Geometry owns no gradient.
- In an audit with `rho` opened, every enabled PLEC, GGLF, and PEG parameter
  family must receive a finite nonzero gradient.
- With zero initialization, detection loss first updates `rho`, while the
  auxiliary objectives provide direct supervision to the tiny and gate paths.
  The residual identity remains exact.

The frozen auxiliary objective is:

```text
L = L_det + 0.25 L_tiny + 0.02 L_gate + 0.01 L_protect
```

`L_tiny` is focal BCE against a Gaussian heatmap for targets whose effective
input size is at most 16 pixels. `L_gate` opens evidence only where tiny demand
and local coverage coexist. `L_protect` penalizes injection over non-tiny
ground-truth regions. These terms train internal maps; they do not add an
inference-time prediction branch.

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
- GGLF attention entropy, tiny demand, and correspondence confidence;
- PEG raw/final gates and scalar `rho/gamma`;
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
- a separate detector loss or inference-time prediction branch.

## 10. First-screen implementation boundary

The supplied freeze also describes a higher-risk PLEC variant with learned
sampling offsets and FiLM modulation. The first matched screen deliberately
retains the already verified nine-phase PLEC implementation. This isolates the
value of the complete geometry-canonical evidence chain without simultaneously
changing canonicalization, interaction, gating, and supervision. Learned
offset/FiLM PLEC remains a later internal ablation only if the integrated
screen advances.

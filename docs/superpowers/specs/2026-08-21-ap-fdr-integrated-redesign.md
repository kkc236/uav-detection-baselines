# ACE-FDR: Integrated Distribution Refinement Design

Date: 2026-08-21

## Objective

Design ACE-FDR (Anchor-Consistent Edge-adaptive Fine-grained Distribution
Refinement) as one indivisible localization method rather than a collection
of detachable submodules.  The final method uses the native RT-DETR decoder
reference as its stable distribution anchor, refines only cleanly matched normal
queries, and allocates FGL gradients according to the learning state of each
box edge.  The design removes the learned preliminary-reference path and all
additional DN-side FDR losses while retaining stock RT-DETR denoising training.

The experiment answers one method-level question: does ACE-FDR
outperform the registered original AP-FDR under the same Formal100 seed-0
protocol?  BPDD and FIA are outside this first experiment.

## Unified Method Contract

ACE-FDR has a single data flow:

```text
native layer reference
  -> cumulative four-edge distributions
  -> adjacent-bin target probability for each matched edge
  -> reliability/difficulty-balanced FGL
  -> refined boxes
```

The three constraints below jointly define the method and are not presented as
three paper modules:

1. `preliminary_box: false`: the native layer reference is the only distribution
   anchor.  No learned preliminary box or preliminary-box L1/GIoU loss is used.
2. `supervise_dn_fdr: false`: stock DN classification/L1/GIoU remain enabled,
   but perturbed DN queries receive no additional FGL or preliminary-box loss.
3. normal-query FGL uses edge-adaptive weights.  This is the standard
   supervision rule of ACE-FDR, not a separately named module.

The fixed 33-bin representation, non-uniform Integral, cumulative refinement,
stock Hungarian matching, classification path, decoder blocks, post-processing,
and evaluator contract remain unchanged.

## Edge-Adaptive FGL

For a cleanly matched query `q` and edge `e`, let the adjacent target bins be
`k` and `k+1`, with interpolation weights `lambda_left` and `lambda_right`.
The detached probability mass assigned to the continuous target is

```text
p_target(q,e) = lambda_left * p(q,e,k)
              + lambda_right * p(q,e,k+1)
```

and its detached edge difficulty is

```text
d(q,e) = 1 - p_target(q,e).
```

Difficulty is normalized over the four edges of the same matched box so that
the method redistributes, rather than globally increases, localization weight:

```text
m(q,e) = clip(d(q,e) / max(mean_edge(d(q,*)), 1e-6), 0.5, 2.0).
```

The final FGL weight is

```text
w(q,e) = stop_gradient(IoU(q) * m(q,e)).
```

This retains the inherited box-level IoU reliability signal while distinguishing
the heterogeneous learning state of four boundaries.  At near-uniform initial
predictions, the four difficulties are nearly equal and `m(q,e)` approaches 1,
so optimization starts close to the current FGL behavior.  No new trainable
parameter, inference branch, FLOP, or checkpoint tensor is introduced.

## Configuration

Add `configs/rtdetr-l-ace-fdr.yaml`, byte-derived from the registered full FDR
configuration with exactly these semantic changes:

- `preliminary_box: false`;
- `supervise_dn_fdr: false`;
- `edge_adaptive_fgl: true`.

The edge modulation constants are frozen in the implementation at epsilon
`1e-6` and clip interval `[0.5, 2.0]`; they are not exposed as tuning knobs in
the first experiment.  The historical `configs/rtdetr-l-fdr.yaml` remains
unchanged and reproducible.

## Compatibility and Safety

- The loss accepts the same corner logits, boxes, assignments, and targets.
- `edge_adaptive_fgl: false` must be numerically identical to the current FGL.
- Empty matches must return a finite differentiable zero.
- Detached modulation must not create a second gradient path through its own
  probability or IoU calculations.
- BPDD continues to receive the same six-layer `[B,Q,4*(reg_max+1)]`
  distributions; FIA remains unaware of this loss change.
- Existing checkpoints load strictly because ACE-FDR adds no parameters.

## Test-First Requirements

Before production code is changed, tests must fail for the missing behavior and
then pass after the minimal implementation.  Tests cover:

1. identical edges produce the original repeated IoU weights;
2. a lower target-bin probability receives a larger edge weight within the same
   matched box;
3. weights are finite, detached, and bounded by the frozen modulation interval;
4. zero matches return a finite differentiable zero;
5. the declarative configuration selects native reference, disables added DN
   FDR, and enables edge-adaptive FGL;
6. model construction wires the new loss option without changing output shapes;
7. stock DN losses remain present while all four added DN-FDR loss keys remain
   absent;
8. BPDD/FIA interfaces remain shape-compatible.

## Training and Evaluation

The first deployment is one Formal100 seed-0 run from the same frozen initial
state used by the existing AP-FDR authority:

- VisDrone2019 train/val;
- RT-DETR-L, input 640, batch 8, workers 8;
- 100 epochs, MuSGD, AMP and deterministic mode enabled;
- no pretrained weights and no cache;
- best checkpoint selected only by validation mAP50-95.

After completion, the same best checkpoint is evaluated separately on val and
test with `imgsz=640`, `batch=8`, `workers=8`, `conf=0.001`, `max_det=300`, and
`nms/cache/half/rect=false`.  Report Precision, Recall, AP50, AP75 and
mAP50-95.  Server-side evidence must bind the source commit, config hash,
initial-state hash, dataset identity, settings, checkpoint hashes, log, and
results CSV before publication.

## Decision and Claim Boundary

The primary comparison is ACE-FDR versus the registered original AP-FDR.  A
positive result supports ACE-FDR as a whole; the paper
does not attribute the gain to three detachable components.  The earlier
preliminary-reference and DN-FDR runs remain development evidence showing why
those paths were rejected, not modules in the final method.

Adopt ACE-FDR only if best-val mAP50-95 is above the original AP-FDR and
AP75 does not regress.  If it does not pass, retain the existing AP-FDR and do
not report ACE-FDR as successful.  The paper must continue to credit
D-FINE for the discrete boundary representation, non-uniform Integral and base
adjacent-bin FGL formulation; the original claim is the unified reference-stable,
clean-query, edge-adaptive organization used by ACE-FDR.

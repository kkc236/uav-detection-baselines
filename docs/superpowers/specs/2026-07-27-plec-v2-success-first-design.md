# PLEC-v2 Success-First Design

**Status:** Frozen for implementation and the first formal screen

**Scope:** PLEC only; GGLF and PEG are excluded

**Decision:** Reuse the verified PLEC geometry/network core and repair its
training, data, and experimental boundaries

## 1. Objective

The first experiment answers one question only:

> Can a trainable, geometry-canonical local P3 representation improve
> RT-DETR-L detection when the stock global training path is protected from
> local-view gradients?

PLEC-v2 remains the first network innovation module:

**PLEC — Phase-Preserving Local Evidence Canonicalizer**

It consumes four local semantic P3 tensors and produces one canonical local
evidence tensor `L3c` on the global P3 lattice. It has trainable embeddings,
MLPs, grouped/depthwise/pointwise convolutions, an overlap head, and output
normalization. It participates in forward propagation, receives gradients, and
can be enabled or removed as one structural ablation.

## 2. Evidence behind the revision

The earlier 10-epoch engineering run cannot be used as paper evidence because
it used a Fresh100 checkpoint, batch 4, workers 4, warmup 1, `nbs=8`, and
disabled augmentation.

Its paired diagnostic is still useful for root-cause isolation:

- method mAP50-95: `0.087407`;
- control mAP50-95: `0.133848`;
- method AP-tiny: `0.014876`;
- control AP-tiny: `0.043123`;
- disabling the endpoint PLEC injection changed method mAP by only about
  `-0.000233`.

Therefore, almost all degradation was already present in the shared detector
weights. The old implementation let four local-view detection-loss paths update
the shared backbone and Hybrid Encoder. PLEC-v2 removes that coupling.

## 3. Frozen architecture

```text
global 640 image
  -> stock RT-DETR backbone + Hybrid Encoder
  -> G3 -------------------------------------------> G3*
                                                       |
four 1088 local views                                  v
  -> shared RT-DETR backbone + Hybrid Encoder       stock decoder
  -> stop_gradient(local P3)
  -> PLEC
  -> L3c
  -> reference 1x1 projection
  -> zero-initialized scalar residual ----------------^
```

The screen adapter is:

```text
G3* = G3 + gamma_ref * Conv1x1(L3c)
gamma_ref(0) = 0
```

The adapter is experimental infrastructure, not a paper contribution. It exists
only because standalone PLEC otherwise has no path to the detection loss.

PLEC-v2 does not modify P4, P5, query selection, decoder layers, loss terms,
query count, or detection head.

## 4. Gradient ownership

The success-first boundary is mandatory:

```text
Lv = stop_gradient(E_local(xv))
L3c = PLEC(L1, L2, L3, L4; geometry)
```

- Global detection loss updates the stock global backbone, Hybrid Encoder, and
  decoder exactly once through the global path.
- Local forward passes share current detector weights but run without autograd
  and do not update BatchNorm running buffers.
- Local P3 tensors and local pixels receive no gradient.
- PLEC and the reference adapter receive finite gradients from the detection
  loss after the residual scalar opens.
- Geometry tensors remain detached buffers.

This deliberately sacrifices local-backbone fine-tuning to protect the baseline
representation during the first screen.

## 5. Data and geometry contract

The global arm must use the stock Ultralytics 8.4.90 RT-DETR training image and
label path. PLEC metadata may observe that path but may not alter its random
draw order or output.

For the 10-epoch screen, `close_mosaic=10` closes mosaic at epoch 0 in
Ultralytics 8.4.90. The remaining scale, translation, HSV, and horizontal flip
augmentations stay enabled.

Each sample records the exact raw-source-to-global-network transform:

```text
raw source
  -> stock long-side resize
  -> RandomPerspective matrix (scale + translation in this protocol)
  -> horizontal/vertical flip decision
  -> global 640 network frame
```

PLEC geometry inverts this composed transform before mapping a global P3 phase
point into each raw-source local crop. Local views retain the source-resolution
evidence and receive the same photometric HSV transform as the global image.
Unknown, missing, singular, perspective, rotation, or shear provenance is a
hard error for this screen.

The four local views remain fixed, exhaustive 60%-width by 60%-height crops in
TL/TR/BL/BR order, each centered-letterboxed to 1088. Crop selection is not
learned and is not a contribution.

Mosaic support is outside this 10-epoch screen. Formal 100-epoch training may
not start until piecewise mosaic provenance is implemented and tested; mosaic
must never be silently disabled in the 100-epoch protocol.

## 6. Formal seed0 screen

The authoritative screen settings are:

- Ultralytics `8.4.90`;
- VisDrone train/val, 10 classes;
- fixed 647-image 10% training list and full 548-image validation set;
- exact seed0 scratch initial-state artifact, shared by method and control;
- 10 epochs, `imgsz=640`, batch 8, workers 8, one RTX 4090;
- deterministic seed 0, cache disabled;
- MuSGD, `lr0=0.01`, `lrf=0.01`, momentum `0.937`,
  weight decay `0.0005`;
- warmup 3 epochs, `nbs=64`, cosine LR disabled;
- AMP enabled with fixed initial scale 128 and no scale growth;
- query count 300, max detections 300, NMS disabled;
- the exact frozen augmentation values supplied by the user;
- `fraction=1.0` because the YAML already names the fixed 647-image list.

Method and control must use the same source commit, data list, seed initial
state, batch order, augmentation configuration, optimizer configuration, and
evaluator. The control disables PLEC injection but otherwise uses the same
global data path.

## 7. Fail-closed preflight

Training may start only after all checks pass:

1. the global control data path matches the stock RT-DETR path for identical
   RNG state;
2. `gamma_ref=0` gives exact tensor equality with stock model output;
3. local forward passes do not change BatchNorm buffers;
4. no local pixel, local P3, backbone, or Hybrid Encoder gradient is created by
   the local path;
5. every enabled PLEC family and both adapter parameters receive finite nonzero
   gradients in an audit with `gamma_ref=1`;
6. transform round trips and known identity/scale/translate/flip cases satisfy
   the geometry tolerance;
7. all invalid PLEC samples are masked and overlap weights normalize over valid
   views;
8. one real CUDA forward/backward succeeds at batch 8 under fixed AMP;
9. peak reserved memory stays below 23 GiB;
10. protocol paths, counts, hashes, environment, initial state, and expected
    optimizer-attempt count are validated.

No automatic batch-size reduction is allowed. Batch-size drift invalidates the
run.

## 8. Screen decision

The first completed pair is seed0 PLEC-v2 versus its matched control.

PLEC advances only if:

- `delta mAP50-95 > 0`;
- `delta AP-tiny >= 0`;
- `delta tiny recall >= 0`;
- `delta AP75 >= -0.002`;
- `delta AP-large >= -0.005`.

If mAP or AP-tiny is negative, PLEC stops for root-cause analysis; GGLF and PEG
do not start. A positive seed0 result is screening evidence only and must later
be repeated on seeds 1 and 2 before a paper claim.

## 9. Rejected alternatives

### Rewrite PLEC from scratch

Rejected because the current geometry and trainable layer families already pass
local structural tests. A rewrite adds implementation risk without addressing
the observed failure.

### Train the local backbone jointly

Rejected because the paired fusion-off diagnostic attributes the earlier
degradation to the coupled training trajectory.

### Add a non-tiny branch

Rejected because the stock global path already supplies the protected
all-scale representation and an extra branch would create a fourth mechanism.

### Add GGLF or PEG now

Rejected because a combined run could not identify whether canonicalization
works. They remain separate later modules.

### Change the formal augmentation or batch size

Rejected because it would break comparability with the frozen baseline.

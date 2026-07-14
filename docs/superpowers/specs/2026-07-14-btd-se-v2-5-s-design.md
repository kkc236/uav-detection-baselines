# BTD-SE V2.5-S Design

Date: 2026-07-14

Status: Frozen for implementation

## Objective

BTD-SE addresses weak small-object responses caused by complex UAV backgrounds. Its contribution is a validity-aware local background reference followed by semantic confirmation of the resulting target-background residual. The module changes only the highest-resolution RT-DETR-L top-down fusion path and does not modify the decoder, matching, queries, or inference post-processing.

## Integration

Ultralytics RT-DETR-L produces an upsampled 256-channel Y4 feature at layer 18 and a 256-channel projected P3 feature at layer 19. The module consumes their concatenation immediately before the original P3/Y4 RepC3 fusion:

```text
Y4 upsample G (256) ----+
                        +--> BTD-SE([G, P]) --> [G, enhanced P] --> RepC3
input_proj.0 P (256) ---+
```

The implementation returns 512 channels so the remaining RT-DETR graph is unchanged.

## Forward Equations

For `P, G` in `R^(B x 256 x H x W)`:

```text
W_b = sigmoid(h_b(concat(P, G)))
Ring(U) = 81 AvgPool9(U) - 25 AvgPool5(U)
N = Ring(W_b * P)
Z = clamp_min(Ring(W_b), 0)
B_ref = N / (Z + tau), tau = 1
R = P - B_ref
Q_r = q_r(R), Q_g = q_g(G), q_r/q_g: 256 -> 32
C = cosine(Q_r, Q_g) along the channel dimension at each location
S = sigmoid(h_s(concat(R, C)))
P_out = P + gamma * S * R
```

`h_b` and `h_s` are 1x1 convolutions. `gamma` has shape `1 x 256 x 1 x 1` and is initialized to zero. Pooling uses stride 1, padding 4/2, and `count_include_pad=True`. Out-of-image reliability is therefore zero and the denominator normalizes incomplete neighborhoods.

## Supervision

The total loss is:

```text
L = L_det + lambda_b L_b + lambda_sal L_sal
```

Initial implementation values are `lambda_b = 0.1` and `lambda_sal = 0.1`; these are training options rather than architectural constants.

`L_b` is class-balanced focal BCE on `W_b`. Ordinary background is 1, object boxes and VisDrone ignored regions are 0. The positive-class weight is `alpha_b = 0.25` and the focal exponent is `eta_b = 2`.

`L_sal` is focal BCE on `S` using the maximum of anisotropic Gaussian targets. Box coordinates are mapped to the P3 grid. The standard deviations are `max(1, w_p3 / 8)` and `max(1, h_p3 / 8)`. Ignored regions are excluded from this loss. The initial saliency settings are `alpha_sal = 0.25` and `eta_sal = 2`.

VisDrone ignored boxes are preserved in sidecar files, appended to the training instances with class `-1`, transformed by the same Ultralytics geometric pipeline, excluded from RT-DETR detection targets, and used only by BTD-SE supervision.

## Ablations

1. RT-DETR-L baseline.
2. Static ring residual with `W_b = 1`.
3. Validity-aware aggregation with `W_b` and `L_b`.
4. Full BTD-SE with semantic confirmation, `S`, and `L_sal`.

All comparisons use scratch training, VisDrone, image size 640, seed 0, the same augmentations, and the same evaluation settings. The existing baseline is mAP50-95 0.221.

## Acceptance

Implementation is accepted for full training only after tests confirm ring equivalence, finite border behavior, zero-initialized identity output, target-map correctness, ignore propagation, finite auxiliary losses, and nonzero gradients. A one-epoch local smoke run must complete before the 100-epoch experiment.

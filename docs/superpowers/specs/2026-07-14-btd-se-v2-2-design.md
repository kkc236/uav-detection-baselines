# BTD-SE V2.2 Design

Date: 2026-07-14

Status: Proposed final design for user review

## 1. Objective

BTD-SE addresses weak small-object responses and false activations caused by roads, roofs, shadows, and vegetation in UAV imagery. The module is designed for the existing scratch-trained RT-DETR-L baseline on VisDrone.

The design follows one causal chain:

1. reconstruct a low-frequency scene background base;
2. separate local target residuals from that base;
3. confirm residuals using higher-level semantics;
4. force target saliency to outrank confusing local background.

The module must remain lightweight, train stably from scratch, and expose each claim through a direct ablation.

## 2. Name and Scope

Chinese name: 背景感知目标-背景解耦显著性增强模块

English name: Background-aware Target-Background Decoupling and Saliency Enhancement Module (BTD-SE)

The term "decoupling" is used instead of "disentanglement" because the method performs explicit residual separation and does not claim statistically independent latent factors.

BTD-SE only modifies the highest-resolution P3 path. It does not change the RT-DETR decoder, Hungarian matching, query count, or inference post-processing.

## 3. Integration Point

In the Ultralytics RT-DETR-L configuration, backbone C3 has 512 channels and is projected to a 256-channel P3 feature by `input_proj.0`. The top-down Y4 feature is also 256 channels and is already upsampled to the P3 spatial resolution.

BTD-SE is inserted after `input_proj.0` and before P3 is concatenated with the upsampled Y4 feature:

```text
Backbone C3 (512 channels)
    -> input_proj.0
P3 projection P (256 channels) -----> BTD-SE -----> enhanced P3
                                           ^
                                           |
                              upsampled Y4 semantic feature
                                           |
enhanced P3 + upsampled Y4 -> Concat -> RepC3 -> remaining Hybrid Encoder
```

This placement reuses RT-DETR's existing projections and top-down semantic feature, avoiding a duplicate C4 upsampling path.

## 4. Module Architecture

Let `P` be the projected P3 feature with shape `B x 256 x H x W`. Let `Y` be the upsampled Y4 feature with the same spatial resolution. The internal width is fixed to `d = 128` for the first implementation.

### 4.1 Channel compression

```text
X = phi_x(P)
```

`phi_x` is a 1x1 projection from 256 to 128 channels.

### 4.2 Local detail extraction

```text
D = X + phi_d(DWConv3x3(X))
```

The residual connection preserves the original localization information. A single 3x3 depthwise branch is used to keep the module focused and lightweight. The design does not claim a fixed compute reduction until measured FLOPs are available.

### 4.3 Low-frequency background-base reconstruction

```text
B7  = AvgPool7x7(X)
B15 = AvgPool15x15(X)
B   = phi_b(Concat(B7, B15))
```

Both pooling operations use stride 1 and same padding. `phi_b` is a 1x1 projection back to 128 channels.

`B` is described as an explicit approximation of the low-frequency background base, not as a complete background representation. High-frequency clutter such as lane markings and roof edges is handled by semantic confirmation and hard-background supervision.

### 4.4 Target residual

```text
R = D - B
R_hat = phi_r(R)
```

`phi_r` is a 1x1 linear projection without an additional standalone normalization layer. It aligns the residual distribution before semantic interaction while preserving the signed residual operation.

### 4.5 Semantic confirmation and saliency prediction

```text
G = phi_g(Y)
I = Concat(R_hat, G, R_hat elementwise-multiply G)
S = Sigmoid(phi_s(I))
```

`phi_g` projects Y4 from 256 to 128 channels. `phi_s` outputs a single-channel spatial saliency map `S` with shape `B x 1 x H x W`. The map is broadcast across channels when modulating `R_hat`.

The elementwise interaction distinguishes the module from ordinary concatenation: a local residual receives a high score only when it agrees with the higher-level semantic feature.

### 4.6 Residual output

```text
F_out = P + psi(S elementwise-multiply R_hat)
```

`psi` is a 1x1 projection from 128 back to 256 channels. Its weights and bias are initialized to zero. Therefore, the initial network is functionally equivalent to the original P3 path, while the saliency branch still receives direct auxiliary supervision from the first iteration.

The main version does not subtract the background feature a second time. The residual `R = D - B` already performs background-base removal. Bidirectional suppression may be evaluated later as an ablation, but it is not part of BTD-SE V2.2.

## 5. Supervision

The same single-channel saliency map `S` is used by both auxiliary losses. No external segmentation annotation or separate post-processing branch is required.

### 5.1 Scale-adaptive saliency target

For a box with width `w` and height `h` in input-image pixels, construct an anisotropic Gaussian on the P3 grid:

```text
sigma_x = clip(0.25 * w / 8, 1, 4)
sigma_y = clip(0.25 * h / 8, 1, 4)
```

Overlapping targets are merged by the pixelwise maximum. Gaussian focal loss supervises `S`:

```text
L_sal = GaussianFocalLoss(S, M)
```

No additional inverse-area object weight is used. Each object contributes a unit center peak, and focal modulation controls the foreground-background imbalance.

### 5.2 Local hard-background ranking

For each ground-truth object `i`, compute its positive saliency as the Gaussian-weighted average:

```text
S_pos_i = Sum(M_i * S) / (Sum(M_i) + epsilon)
```

Hard-negative candidates are sampled from a local annulus around the object. The candidate area is the box expanded by a factor of 2.0, excluding all ground-truth boxes expanded by a factor of 1.2. The highest responses in this region are the confusing local background:

```text
S_hard_i = Mean(TopK(S in local-background-region, K=32))
L_rank = Mean_i ReLU(margin - S_pos_i + S_hard_i)
```

The default margin is `0.2`. If fewer than 32 valid cells are available, all valid cells are used. Objects without valid background cells do not contribute to `L_rank` for that image.

VisDrone ignored regions must not be treated as hard negatives. The current YOLO conversion discards ignored annotations, so the implementation will preserve ignore boxes in an auxiliary annotation field and apply the same geometric transforms used for the image and detection boxes. The local hard-negative mask excludes these transformed ignore boxes. Global Top-K background mining is outside the V2.2 scope.

### 5.3 Total loss

```text
L_total = L_det + lambda_sal * L_sal + lambda_rank * L_rank
```

Initial settings are `lambda_sal = 1.0` and `lambda_rank = 0.1`. The ranking coefficient is linearly warmed from 0 to 0.1 during the first five epochs to avoid unstable hard-negative selection before the saliency map becomes meaningful.

## 6. Training and Inference Behavior

Training starts from the existing RT-DETR-L YAML architecture without pretrained weights, matching the baseline protocol. Ablations use the same dataset split, image size, epoch count, optimizer, augmentation, fixed seed, and evaluation settings. The final baseline and full model are then repeated with three seeds to estimate run-to-run variation.

BTD-SE adds feature computation during both training and inference. The saliency and ranking losses are training-only and add no inference-time post-processing. Parameters, GFLOPs, peak memory, and latency must be measured rather than estimated from kernel sizes.

## 7. Ablation Plan

Run the following experiments under the same training protocol:

| Experiment | Background residual | Semantic interaction | Saliency loss | Ranking loss |
| --- | --- | --- | --- | --- |
| A. RT-DETR-L baseline | No | No | No | No |
| B. Background residual | Yes | No; use an all-one gate | No | No |
| C. Semantic confirmation | Yes | Yes | No | No |
| D. Saliency supervision | Yes | Yes | Yes | No |
| E. Full BTD-SE V2.2 | Yes | Yes | Yes | Yes |

Primary metrics are mAP50-95, mAP50, precision, recall, AP-small, and AR-small. Efficiency metrics are parameter count, GFLOPs, peak GPU memory, and single-image latency at batch size 1.

Mechanism evidence should include saliency visualizations and the average gap between target saliency and local hard-background saliency. This diagnostic is supplementary and does not replace standard detection metrics.

## 8. Novelty Boundary

BTD-SE does not claim that shallow-detail fusion, saliency prediction, or feature pooling is individually new. Its contribution is the problem-specific combination of:

1. an explicit low-frequency background-base reconstruction on the RT-DETR P3 path;
2. residual target extraction followed by higher-level semantic confirmation;
3. a local hard-background ranking objective tied to the same saliency map.

Unlike token-filtering DETR methods, BTD-SE does not remove encoder tokens or modify query selection. Unlike scale-disentanglement methods, it separates target residuals from a scene-background base rather than scale-related from scale-invariant factors.

## 9. Acceptance Criteria

The design is retained only if the full model:

1. improves mAP50-95 and AP-small over the existing scratch RT-DETR-L baseline in repeated runs;
2. shows that the full module outperforms the background-residual-only and no-ranking variants;
3. reduces target-versus-hard-background saliency confusion in visual and diagnostic analysis;
4. reports measured compute and latency overhead without obscuring an excessive efficiency cost.

The current baseline mAP50-95 is 0.221. A gain is evidence only after repeated experiments; the design does not assume a guaranteed improvement.

## 10. Reference Points

- RT-DETR: https://arxiv.org/abs/2304.08069
- Ultralytics RT-DETR-L architecture: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml
- Salience DETR: https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html
- Focus-DETR: https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Less_is_More_Focus_Attention_for_Efficient_DETR_ICCV_2023_paper.html
- Small Object Detection by DETR via Information Augmentation and Adaptive Feature Fusion: https://arxiv.org/abs/2401.08017

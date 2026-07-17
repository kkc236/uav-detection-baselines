# VSF-RMR Final Design

Date: 2026-07-17

Status: Frozen for implementation

## 1. Objective

创新点3解决无人机飞行高度、拍摄角度和场景透视变化导致的目标表观尺度不稳定问题。本文提出视角尺度场引导的残差式多尺度路由模块：

**View-Scale-Field-Guided Residual Multi-Scale Routing (VSF-RMR)**

VSF-RMR不直接回归真实飞行高度或相机姿态。它从图像特征中估计由高度、视角、焦距和场景布局共同形成的表观尺度状态，并据此在P3、P4和P5之间进行有序残差路由。

当前实验仅验证独立的 `RT-DETR-L + VSF-RMR`，不包含BTD-SE或IOQC-SA。飞行元数据只保留扩展接口，不进入VisDrone主实验、参数量、FLOPs或核心贡献。

## 2. Experimental Boundary

- Baseline: stock Ultralytics RT-DETR-L.
- Dataset: the existing VisDrone train/validation splits.
- Initialization: scratch training with `pretrained=False`.
- Schedule: 100 epochs, image size 640, seed 0.
- Isolation: no BTD-SE modules, no IOQC-SA probe or loss.
- Decoder: unchanged RTDETRDecoder, Hungarian matching, queries, losses, and post-processing.
- Performance limits: inference latency below 5%, parameters below 3%, GFLOPs below 5%, and training time below 10% relative to the matched baseline.
- Fairness: the baseline and VSF-RMR run use identical augmentation, optimizer, learning-rate, batch, hardware, and evaluation settings.

Because VSF-RMR contains an image-level scale state, the official comparison disables augmentations that mix unrelated camera states:

```yaml
imgsz: 640
epochs: 100
pretrained: false
seed: 0
mosaic: 0.0
mixup: 0.0
scale: 0.5
perspective: 0.0
```

The existing Mosaic-enabled RT-DETR result remains an engineering reference only. A new stock RT-DETR-L baseline must be trained with the configuration above before the final comparison.

## 3. Integration

The original `rtdetr-l.yaml` remains unchanged. Its Hybrid Encoder sends feature layers 21, 24, and 27 directly to the RTDETRDecoder as P3, P4, and P5. VSF-RMR is inserted programmatically between these stages:

```text
Backbone and Hybrid Encoder
           |
       F3, F4, F5
           |
        VSF-RMR
           |
   F3_hat, F4_hat, F5_hat
           |
    stock RTDETRDecoder
```

The custom detection model owns the module as a normal registered child:

```python
self.vsf_rmr = VSFRMR(
    channels=256,
    hidden_channels=32,
)
```

VSF-RMR is not registered as a YAML layer. Ultralytics' parser does not provide a generic channel-inference rule for a custom layer that consumes and returns a list of pyramid features.

`VSFRMRRTDETRDetectionModel.predict()` preserves the stock encoder loop and changes only the final decoder call:

```python
head = self.model[-1]
features = [y[index] if index != -1 else x for index in head.f]
features = self.vsf_rmr(features)
predictions = head(features, batch)
```

`VSFRMRRTDETRTrainer.get_model()` must instantiate `VSFRMRRTDETRDetectionModel`; otherwise the official trainer would silently construct the stock RT-DETR model.

VSF-RMR validates its input contract before routing:

```text
C3 = C4 = C5 = 256
H3 = 2 H4 = 4 H5
W3 = 2 W4 = 4 W5.
```

Any mismatch raises an explicit error. The implementation must not silently resize an irregular pyramid because the conditional cancellation equations rely on exact `2x` and `4x` alignment.

## 4. Shared Scale Space

For Hybrid Encoder outputs

```text
F3: B x 256 x H   x W
F4: B x 256 x H/2 x W/2
F5: B x 256 x H/4 x W/4,
```

each level has an independent GroupNorm and all levels share one `1x1` projection from 256 to 32 channels:

```text
U_l = SiLU(phi(N_l(F_l))),  l in {3, 4, 5}.
```

The independent normalizers handle level-specific feature statistics. The shared projection supplies a common coordinate assumption for cross-level addition and subtraction. It does not claim mathematically identical level distributions.

P4 and P5 are aligned to P3 with nearest-neighbor upsampling:

```text
U3_bar = U3
U4_bar = NNUp2(U4)
U5_bar = NNUp4(U5).
```

Nearest-neighbor upsampling preserves block-constant values so aligned features can be exactly recovered by matching average pooling when the routing selection is spatially constant inside a pooling block.

## 5. Global-Local View Scale Field

### 5.1 Global visual bias

The global descriptor and scalar bias are

```text
z_vis = Concat(GAP(U3), GAP(U4), GAP(U5)) in R^(B x 96)
b_vis = h_vis(z_vis) in R^(B x 1),
```

where `h_vis` is:

```text
Linear 96 -> 32
SiLU
Linear 32 -> 1
```

The final weight is initialized to zero and the final bias to `-0.1`, producing an initial global scale value of approximately `2 sigmoid(-0.1) = 0.95`.

### 5.2 Local scale residual

The aligned low-dimensional features are concatenated:

```text
U_cat = Concat(U3_bar, U4_bar, U5_bar) in R^(B x 96 x H x W).
```

The local head is:

```text
3x3 depthwise convolution, 96 channels
GroupNorm
SiLU
1x1 convolution, 96 -> 1
```

It predicts `R_v = h_local(U_cat)`. The final `1x1` weight uses `Normal(mean=0, variance=1e-6)`, corresponding to `std=1e-3` in PyTorch, and zero bias. This creates a small spatial variation around the initial global value and avoids placing all pixels exactly on the `V=1` routing knot.

### 5.3 Continuous field

With no metadata, the scale logit and scale field are

```text
A_v(x, y) = b_vis + R_v(x, y)
V(x, y) = 2 sigmoid(A_v(x, y)),  V in (0, 2).
```

`V` is an apparent-scale coordinate:

```text
V -> 0: prefer P3
V -> 1: prefer P4
V -> 2: prefer P5.
```

## 6. Ordered Scale Routing

A single continuous coordinate is analytically converted into adjacent-level weights:

```text
alpha3 = relu(1 - V)
alpha5 = relu(V - 1)
alpha4 = 1 - alpha3 - alpha5.
```

These weights are nonnegative and sum to one. The route permits continuous P3-P4 or P4-P5 transitions but never directly mixes P3 and P5 while skipping P4.

The common routed feature at P3 resolution is

```text
M = alpha3 * U3_bar + alpha4 * U4_bar + alpha5 * U5_bar.
```

The novelty claim is limited to the combination of a GT-supervised global-local apparent-scale field, scalar-derived adjacent ordered routing, and conditional self-cancelling residual transfer. Dynamic scale selection or adaptive pyramid fusion alone is not claimed as novel.

## 7. Residual Cross-Scale Transfer

The residuals at each target level are

```text
Delta3 = M - U3
Delta4 = AvgPool2(M) - U4
Delta5 = AvgPool4(M) - U5.
```

One shared restoration projection `psi: 32 -> 256` maps all residuals back to the decoder channel width:

```text
T_l = psi(Delta_l).
```

Each level retains an independent zero-initialized channel scale:

```text
F_l_hat = F_l + gamma_l * T_l
gamma_l in R^(1 x 256 x 1 x 1), initialized to zero.
```

Consequently, the initial module is an exact identity mapping for all three decoder inputs.

Self-cancellation is conditional rather than universal. For example, `Delta4=0` only when a corresponding `2x2` P3 region performs spatially consistent pure P4 selection. P5 requires consistent pure P5 selection over the corresponding `4x4` region.

## 8. Future Flight-Metadata Extension

The first implementation contains no metadata flags, tensors, adapter class, parameters, or tests. This keeps the VisDrone model and paper contribution strictly visual.

The architecture remains extensible at the scale-logit boundary. A future adapter may encode altitude, pitch, and focal length as `[log(h/h_ref), sin(theta), cos(theta), log(f/f_ref)]`, predict a global intercept and vertical slope, and add `b0 + by * y_norm` to `A_v` before the sigmoid. This documented injection point is sufficient for compatibility; it is not implemented or evaluated in the present work.

## 9. Apparent-Scale Supervision

For each augmented normalized `xywh` GT box in image `b`, use the actual batch tensor shape `(H_img, W_img)` rather than the nominal configuration value:

```text
w_b,g = w_norm,b,g * W_img
h_b,g = h_norm,b,g * H_img
r_b,g = sqrt(w_b,g * h_b,g).
```

Target generation runs without gradients and in FP32. Invalid, ignored, or nonpositive boxes do not produce scale targets.

The continuous target is

```text
v_b,g = clip(log2(r_b,g / 8), 0.05, 1.95).
```

The reference value 8 is the P3 stride. The fixed margins prevent labels from falling at the sigmoid and routing extremes. The mapping provides transitions around 8, 16, and 32 pixels. A pre-design audit of the local training labels at 640 input measured a median equivalent size of approximately 11.09 pixels and a 90th percentile of approximately 31.20 pixels. With the rejected 32-pixel reference, 90.5% of training targets would have collapsed to the P3 endpoint.

### 9.1 Image-balanced local loss

Sample the predicted field bilinearly at each normalized GT center. For center `c=(cx, cy)` in `[0, 1]`, use `grid=(2cx-1, 2cy-1)` with `align_corners=False`:

```text
v_hat_b,g = Sample(V_b, c_b,g).
```

For valid images `B+ = {b | N_b > 0}`:

```text
L_local = mean over b in B+ (
    mean over g in image b SmoothL1(v_hat_b,g, v_b,g)
).
```

`SmoothL1` uses `beta=1.0` and unreduced per-target values before the two-stage averaging. The per-image reduction prevents dense images from dominating merely because they contain more objects.

### 9.2 Image-balanced global loss

For each valid image:

```text
v_bar_b = mean_g(v_b,g)
v_global_hat_b = 2 sigmoid(b_vis,b)
L_global = mean over b in B+ SmoothL1(v_global_hat_b, v_bar_b).
```

The auxiliary loss is

```text
L_VSF = L_local + L_global.
```

If a batch has no valid GT, return the graph-connected FP32 zero

```python
loss_vsf = scale_field.float().sum() * 0.0
```

rather than a newly allocated scalar.

## 10. Total Loss and Precision

The standalone objective is

```text
L = L_det + lambda_v * L_VSF
lambda_v = 0.1.
```

VSF supervision is active from epoch 1 with no warm-up. Target generation, bilinear sampling, logarithms, SmoothL1 with `beta=1.0`, and reductions execute with autocast disabled in FP32. The feature network, VSF-RMR forward, and stock detection path remain AMP-capable. `scale_field`, both auxiliary losses, detection loss, and total loss are checked independently with `torch.isfinite` before optimizer update.

At the first optimizer step, `L_VSF` trains the scale field while zero `gamma_l` blocks detection gradients from entering the residual route. Detection loss can update `gamma_l` because restoration outputs are nonzero. After `gamma_l` leaves zero, detection gradients reach the restoration projection, shared scale space, and routing path. Initializing `V` near 0.95 avoids beginning exactly at the `V=1` ReLU knot; it does not bypass the initial gate.

## 11. Auxiliary Cache Contract

VSF-RMR returns only `[F3_hat, F4_hat, F5_hat]`. It never returns an auxiliary dictionary to the decoder.

At the beginning of every forward, `_aux_cache` is cleared. Only training forwards cache `scale_field` and `global_scale`. The loss immediately retrieves and clears them through `pop_aux()`.

Inference and validation do not create a cache. When Ultralytics requests loss items from evaluation predictions, the custom loss follows the stock detection-loss path and skips both `L_VSF` and `pop_aux()`. Routing weights, routed features, projected features, and metadata are not cached. A missing cache during an actual training loss is a hard error rather than silently reusing stale tensors.

## 12. Custom Model and Trainer

`VSFRMRRTDETRDetectionModel` overrides:

- `__init__`: create VSF-RMR and store `lambda_v`.
- `predict`: preserve the stock encoder loop and its `profile`, `visualize`, `augment`, and `embed` behavior, collect `head.f`, route the three features, and call the stock decoder.
- `loss`: compute the stock detection loss; in training, pop the VSF cache, compute FP32 `L_VSF`, verify finiteness, and return the combined loss; in evaluation, skip the auxiliary branch and return stock loss items.

The first implementation may retain the stock three displayed loss names and store `last_vsf_loss = loss_vsf.detach()` for diagnostics. The saved run arguments must include `lambda_v`, the module width, initialization, and augmentation boundary.

`VSFRMRRTDETRTrainer` overrides `get_model()` to construct `VSFRMRRTDETRDetectionModel`, load optional resume weights, and preserve the standard RT-DETR validator. Resume-time runtime overrides must preserve the configured optimizer, learning rate, momentum, project, name, AMP state, and lambda value. Tests must also prove that VSF-RMR parameters enter the optimizer, EMA, checkpoint, and resumed model state.

No file under the installed `ultralytics` package is edited.

## 13. Training Algorithm

For each batch:

1. Apply the official non-Mosaic augmentation pipeline and produce normalized transformed `xywh` GT boxes.
2. Run the stock backbone and Hybrid Encoder under the configured AMP state.
3. Validate the three-level shape contract, run VSF-RMR, and cache only the scale field and global scale.
4. Pass the routed feature list into the unchanged RT-DETR decoder and detection criterion.
5. Pop the current VSF cache exactly once.
6. Under disabled autocast, convert valid normalized boxes to FP32 pixel sizes with the actual batch `H_img` and `W_img`.
7. Build clipped log-scale targets, transform centers to the fixed `grid_sample` convention, and group targets by image.
8. Compute unreduced `beta=1.0` SmoothL1 values, then image-balanced local and global reductions.
9. Check the field, local loss, global loss, detection loss, and total loss for finiteness.
10. Form `L_det + 0.1 L_VSF`, backpropagate, and let the stock trainer perform scaling, clipping, optimizer, EMA, and scheduler steps.
11. At epoch end, record only `L_local`, `L_global`, scale-field mean/std/range, three derived route-weight means, GT-center scale correlation, three `gamma_l` norms, AMP state, batch, and peak CUDA memory. Full maps and routed features are never written to disk.

The module adds no extra image forward, teacher model, matching pass, or inference post-processing.

## 14. Ablations and Evaluation

Experiments are staged to avoid spending full-run compute before the mechanism is viable.

Stage 1 runs only:

1. Stock RT-DETR-L under the matched no-Mosaic configuration.
2. Full VSF-RMR.

Only after the full module beats the matched baseline and passes the mechanism diagnostics does Stage 2 run the explanatory ablations:

1. Global-only scale routing without the local branch.
2. Global-local scale routing without `L_VSF`.
3. Full scale field and ordered routing without residual subtraction, using direct routed addition.

Report:

- precision, recall, mAP50, and mAP50-95;
- AP by object-size bins, including `AP_tiny` for equivalent size below 16 pixels and `AP_small` for equivalent size below 32 pixels at 640 input;
- parameters, GFLOPs, peak memory, single-image latency, batch latency, and FPS;
- performance under deterministic image-scale and perspective stress sets;
- correlation between predicted and GT apparent scale at object centers;
- routing-weight distributions versus GT size.

The image-scale and perspective stress sets are fixed before examining model results. Scale variants use centered factors `0.75` and `1.25`, followed by padding or center cropping back to `640x640`. Perspective variants use vertical homography coefficients `-5e-4` and `+5e-4`. Boxes receive the identical transform, are clipped to the image, and are retained only when at least 50% of their transformed area remains visible and both sides are at least one pixel. The deterministic script saves the source image ID, transform, retained boxes, and checksum in a manifest. Synthetic perturbations measure controlled robustness; they do not prove recovery of physical flight pose.

Latency is measured on the same GPU, software stack, precision, image set, and power mode for both models. Report batch 1 and training-batch timing after 50 warm-up iterations and 200 measured iterations with CUDA synchronization, including mean, P50, and P95 latency.

The seed-0 method is scientifically successful when the 100-epoch VSF-RMR run improves matched-baseline mAP50-95 by at least 0.5 percentage points, does not reduce `AP_tiny` or `AP_small`, degrades less under the fixed stress sets, and remains within all resource limits. If the mAP50-95 gain is positive but below 0.5 points, repeat baseline and full VSF-RMR with seed 1 and require a positive two-seed mean gain before treating the method as effective. Passing software tests alone proves implementation correctness, not method effectiveness. The paper must not attribute a gain to viewpoint adaptation unless the scale-field diagnostics and robustness tests support the mechanism.

## 15. TDD and Acceptance

Implementation is eligible for a full 100-epoch run only after tests prove:

1. Valid pyramid shapes pass and any channel or `2x/4x` mismatch fails explicitly.
2. Zero `gamma_l` gives exact identity outputs.
3. Scale targets use actual batch dimensions, fixed grid coordinates, `align_corners=False`, FP32, and `beta=1.0` SmoothL1.
4. `L_VSF` gives nonzero first-step gradients to global/local scale heads.
5. Detection loss gives nonzero first-step gradients to `gamma_l` but zero first-step gradients through the gated residual route.
6. After a simulated optimizer step moves `gamma_l`, detection gradients reach restoration and routing parameters.
7. Initial scale-field mean is near 0.95 and not identically 1; ordered weights are nonnegative, sum to one, and never mix P3 and P5 directly.
8. Conditional P3/P4/P5 self-cancellation holds for block-consistent pure selections.
9. Images with different object counts but equal mean target error contribute equally to `L_local`; empty-GT loss is finite FP32 zero and graph-connected.
10. Training cache exists after forward, clears after `pop_aux()`, and is absent during evaluation; evaluation loss does not attempt to pop it.
11. The stock YAML remains unchanged, the custom trainer constructs the custom model, and all router parameters appear in the optimizer and EMA.
12. The custom prediction path preserves official profiling, visualization, embedding, validation, and inference behavior.
13. Decoder input is a three-tensor list and training, validation, prediction, checkpoint save, and checkpoint resume all work without losing VSF configuration or state.
14. A real local forward/backward smoke test has finite loss and gradients.
15. A one-to-three-epoch diagnostic run shows positive scale correlation, non-collapsed routing, and moving `gamma_l` before full training.

Measured acceptance limits are:

- inference latency increase at most 5%;
- parameter increase at most 3%;
- GFLOPs increase at most 5%;
- training time increase at most 10%.

The theoretical main-branch estimate is approximately 0.288 GFLOPs at 640 input. Parameters, interpolation cost, activation cost, latency, and memory must be measured from the implemented model rather than accepted from the estimate.

## 16. References for Method Boundary

- Ultralytics RT-DETR-L configuration: <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml>
- Ultralytics RT-DETR trainer: <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py>
- ASFF: <https://arxiv.org/abs/1911.09516>
- Dynamic Head: <https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Dynamic_Head_Unifying_Object_Detection_Heads_With_Attentions_CVPR_2021_paper.html>
- POD scale-sensitive network: <https://arxiv.org/abs/1909.02225>
- Dynamic Scale Routing: <https://arxiv.org/abs/2210.13821>

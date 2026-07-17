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
    use_metadata=False,
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

## 8. Optional Flight Metadata

The main experiment sets `use_metadata=False`, so no metadata module is instantiated. A future metadata adapter may encode altitude `h`, pitch `theta`, and focal length `f` as

```text
m = [log(h / h_ref), sin(theta), cos(theta), log(f / f_ref)].
```

The adapter outputs an intercept and vertical slope `(b0, by)` and forms

```text
B_meta(x, y) = b0 + by * y_norm,  y_norm in [-1, 1]
A_v = b_vis + R_v + delta_meta * B_meta.
```

The final metadata-head weight and bias are initialized to zero. No separate zero-initialized metadata scale is used, avoiding a dead double-zero gradient path. `metadata_valid` must be explicitly reshaped to `B x 1 x 1 x 1` before multiplication.

This adapter is an extension interface, not a validated contribution of the current VisDrone experiment.

## 9. Apparent-Scale Supervision

For each augmented GT box in image `b`, convert the normalized width and height to input-image pixels and define

```text
r_b,g = sqrt(w_b,g * h_b,g).
```

The continuous target is

```text
v_b,g = clip(log2(r_b,g / 8), 0.05, 1.95).
```

The reference value 8 is the P3 stride. The fixed margins prevent labels from falling at the sigmoid and routing extremes. The mapping provides transitions around 8, 16, and 32 pixels, which is compatible with the observed VisDrone scale distribution and avoids the collapse caused by the earlier 32-pixel reference.

### 9.1 Image-balanced local loss

Sample the predicted field bilinearly at each normalized GT center:

```text
v_hat_b,g = Sample(V_b, c_b,g).
```

For valid images `B+ = {b | N_b > 0}`:

```text
L_local = mean over b in B+ (
    mean over g in image b SmoothL1(v_hat_b,g, v_b,g)
).
```

The per-image reduction prevents dense images from dominating merely because they contain more objects.

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

VSF supervision is active from epoch 1 with no warm-up. Target generation, bilinear sampling, logarithms, SmoothL1, and reductions execute with autocast disabled in FP32. The feature network, VSF-RMR forward, and stock detection path remain AMP-capable.

At the first optimizer step:

1. `L_VSF` trains the global and local scale-field path.
2. Zero `gamma_l` blocks detection gradients from entering the residual route.
3. Detection loss can update `gamma_l` because restoration outputs are nonzero.
4. After `gamma_l` leaves zero, detection gradients reach the restoration projection, shared scale space, and routing path.

Initializing `V` near 0.95 prevents later ordered-routing gradients from beginning exactly at the `V=1` ReLU knot; it does not bypass the initial `gamma_l` gate.

## 11. Auxiliary Cache Contract

VSF-RMR returns only `[F3_hat, F4_hat, F5_hat]`. It never returns an auxiliary dictionary to the decoder.

At the beginning of every forward, `_aux_cache` is cleared. Only training forwards cache `scale_field` and `global_scale`. The loss immediately retrieves and clears them through `pop_aux()`.

Inference and validation do not create a cache. When Ultralytics requests loss items from evaluation predictions, the custom loss follows the stock detection-loss path and skips both `L_VSF` and `pop_aux()`. Routing weights, routed features, projected features, and metadata are not cached. A missing cache during an actual training loss is a hard error rather than silently reusing stale tensors.

## 12. Custom Model and Trainer

`VSFRMRRTDETRDetectionModel` overrides:

- `__init__`: create VSF-RMR and store `lambda_v`.
- `predict`: preserve the stock encoder loop, collect `head.f`, route the three features, and call the stock decoder.
- `loss`: compute the stock detection loss; in training, pop the VSF cache, compute FP32 `L_VSF`, verify finiteness, and return the combined loss; in evaluation, skip the auxiliary branch and return stock loss items.

The first implementation may retain the stock three displayed loss names and store `last_vsf_loss = loss_vsf.detach()` for diagnostics. The saved run arguments must include `lambda_v`, the module width, initialization, and augmentation boundary.

`VSFRMRRTDETRTrainer` overrides `get_model()` to construct `VSFRMRRTDETRDetectionModel`, load optional resume weights, and preserve the standard RT-DETR validator. Resume-time runtime overrides must preserve the configured optimizer, learning rate, momentum, project, name, AMP state, and lambda value.

No file under the installed `ultralytics` package is edited.

## 13. Training Algorithm

For each batch:

1. Apply the official non-Mosaic augmentation pipeline and produce normalized transformed GT boxes.
2. Run the stock backbone and Hybrid Encoder.
3. Run VSF-RMR on the three decoder features and cache the two supervision tensors.
4. Run the unchanged RT-DETR decoder and detection criterion.
5. Pop the current VSF cache.
6. Convert the transformed boxes to pixel sizes using the actual batch image height and width.
7. Build clipped log-scale targets and group them by image.
8. Compute image-balanced local and global FP32 losses.
9. Form `L_det + 0.1 L_VSF` and reject any non-finite component before optimizer update.
10. Record `L_VSF`, local/global components, scale-field mean/std/range, correlation at GT centers, three route-weight means, `gamma_l` norms, AMP state, batch, and peak CUDA memory.

The module adds no extra image forward, teacher model, matching pass, or inference post-processing.

## 14. Ablations and Evaluation

The minimum ablation sequence is:

1. Stock RT-DETR-L under the matched no-Mosaic configuration.
2. Global-only scale routing without the local branch.
3. Global-local scale routing without `L_VSF`.
4. Full scale field and ordered routing without residual subtraction, using direct routed addition.
5. Full VSF-RMR.

Report:

- precision, recall, mAP50, and mAP50-95;
- AP by object-size bins, including a VisDrone-relevant small-object breakdown;
- parameters, GFLOPs, peak memory, single-image latency, batch latency, and FPS;
- performance under deterministic image-scale and perspective stress sets;
- correlation between predicted and GT apparent scale at object centers;
- routing-weight distributions versus GT size.

The paper must not attribute a gain to viewpoint adaptation unless the scale-field diagnostics and robustness tests support the mechanism.

## 15. TDD and Acceptance

Implementation is eligible for a full 100-epoch run only after tests prove:

1. Output shapes equal the three decoder input shapes.
2. Zero `gamma_l` gives exact identity outputs.
3. `L_VSF` gives nonzero first-step gradients to global/local scale heads.
4. Detection loss gives nonzero first-step gradients to `gamma_l` but zero first-step gradients through the gated residual route.
5. After a simulated optimizer step moves `gamma_l`, detection gradients reach restoration and routing parameters.
6. Initial scale-field mean is near 0.95 and not identically 1.
7. Ordered weights are nonnegative, sum to one, and never mix P3 and P5 directly.
8. Conditional P3/P4/P5 self-cancellation holds for block-consistent pure selections.
9. Images with different object counts but equal mean target error contribute equally to `L_local`.
10. Empty-GT loss is finite FP32 zero and remains graph-connected.
11. Training cache exists after forward, clears after `pop_aux()`, and is absent during evaluation; evaluation loss does not attempt to pop it.
12. Metadata validity broadcasts as `B x 1 x 1 x 1`; a zero-output metadata head can receive a first-step gradient when enabled.
13. The stock YAML remains unchanged and the custom trainer constructs the custom model.
14. Decoder input is a three-tensor list and training, validation, prediction, checkpoint save, and checkpoint resume all work.
15. A real local forward/backward smoke test has finite loss and gradients.
16. A one-to-three-epoch diagnostic run shows positive scale correlation, non-collapsed routing, and moving `gamma_l` before full training.

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

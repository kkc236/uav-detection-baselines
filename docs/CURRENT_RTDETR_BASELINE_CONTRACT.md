# Current RT-DETR-L Baseline Contract

Extracted and verified on 2026-07-25. This document separates the historical
100-epoch reference from the current strict paired control. Innovation 1, 2,
and 3 must use the strict paired control for formal comparisons.

## 1. Baseline identity

| Item | Frozen value |
| --- | --- |
| Detector | Ultralytics RT-DETR-L |
| Model config | stock `rtdetr-l.yaml` |
| Classes | 10 VisDrone detection classes |
| Backbone | HGNetv2, P3/8 to P5/32 outputs |
| Hybrid encoder output | three 256-channel feature levels |
| Decoder input levels | P3, P4, P5 |
| Decoder query budget | 300 |
| `max_det` | 300 |
| NMS | disabled |
| Initialization | `pretrained=False`, scratch |

The stock graph ends with:

```text
HGNetv2 backbone
  -> P3/8, P4/16, P5/32
  -> AIFI + FPN/PAN hybrid encoder
  -> 3 x 256-channel features
  -> RTDETRDecoder
  -> exactly 300 stock queries
```

The model YAML source SHA-256 is
`85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83`.

## 2. Frozen source and environment

Current strict paired experiments use:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| Driver | 550.142 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| CUDA runtime | 12.1 |
| Ultralytics | 8.4.90 |
| Current validation commit | `9d9d799404caea426f768b75b471691aa253238d` |

Frozen Ultralytics source hashes:

| Source | SHA-256 |
| --- | --- |
| `head.py` | `5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F` |
| `tasks.py` | `B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92` |
| `rtdetr-l.yaml` | `85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83` |

## 3. Dataset contract

| Item | Frozen value |
| --- | --- |
| Dataset | VisDrone train/val |
| Train images / labels | 6471 / 6471 |
| Validation images / labels | 548 / 548 |
| Validation instances | 38759 |
| Dataset semantic file count | 14038 |
| Dataset semantic SHA-256 | `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB` |
| Category mapping SHA-256 | `1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6` |
| Full VisDrone YAML SHA-256 | `CF4946E0A34BA2168D29B5411E8321E1ABE3A5B282BFDD02C8BB3FFEBA09BB9F` |

Class mapping:

```text
0 pedestrian
1 people
2 bicycle
3 car
4 van
5 truck
6 tricycle
7 awning-tricycle
8 bus
9 motor
```

The fixed 10% screening subset contains 647 train images:

| Item | SHA-256 |
| --- | --- |
| Semantic subset signature | `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0` |
| Subset list file | `4BDEE4F03CC903422ADBBF4BD3511027628000DB578DEFC07DFE6E45F1E7CB60` |

The subset is hash-selected once. It must never be randomly resampled.

Current strict initial-state authority:

| Seed | Initial-state file SHA-256 | Common-state fingerprint |
| ---: | --- | --- |
| 0 | `C1D93F83EE8BB90CC8A41B313B446E68E91945E53C7CCB597D5434FC3580304A` | `0B968046FDC89BE5A31581C81F7335A9742BC422503428113637B1CC829F0FA0` |
| 1 | `E6C986F53C4FB7076BA52948E959FFBE71F16EE4762E16D4553827C3A46EC465` | `A73D3A57F5DCF3F62FA4B30329C32204E3A74BC57AA4FFEE577873D14F0A3D65` |
| 2 | `EBA9851A3BAF98DE77228443702F944058C5581A72C00ADA1C452B83BCB598C4` | `1CCA2D745106F949268B3978722A415439623376D01C1D188B1450C6230AF1B2` |

Innovation-private initial-state values may differ by method, but the
common-state fingerprint for the selected seed must match this authority.

## 4. Common training contract

The following values must be identical in the control and method arm:

| Item | Value |
| --- | --- |
| Epochs | 100 formal; 10 for current E1 screening |
| Image size | 640 |
| Batch | fixed 8 |
| Workers | 8 |
| AMP | true |
| Seeds | 0, then 1 and 2 |
| Deterministic | true |
| Optimizer | MuSGD |
| Effective `lr0` | 0.01 |
| `lrf` | 0.01 |
| Effective momentum | 0.937 in the current strict launcher |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| Warmup momentum | 0.8 |
| Warmup bias LR | 0.0 |
| NBS | 64 |
| Cosine LR | false |
| Cache | false |
| Query count | 300 |
| `max_det` | 300 |
| NMS | false |
| Resume | only a complete checkpoint with optimizer, EMA, scaler, and update state |

Augmentation contract:

| Item | Value |
| --- | --- |
| mosaic | 1.0 |
| close_mosaic | 10 |
| mixup | 0.0 |
| scale | 0.5 |
| translate | 0.1 |
| degrees | 0.0 |
| shear | 0.0 |
| perspective | 0.0 |
| flipud | 0.0 |
| fliplr | 0.5 |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |
| cutmix | 0.0 |
| copy_paste | 0.0 |

### Optimizer warning

The historical stock Ultralytics run was configured with `optimizer=auto`,
`lr0=0.01`, and `momentum=0.937`, but its runtime log proves that stock
Ultralytics auto-selection used:

```text
MuSGD(lr=0.01, momentum=0.9)
```

The current strict paired trainer resolves `auto` to MuSGD without discarding
the frozen values, and its runtime log proves:

```text
MuSGD(lr=0.01, momentum=0.937)
```

Therefore the historical 100-epoch result is a reference result, not the
formal paired A0 for the current protocol. Innovation 1, 2, and 3 must all use
the current strict optimizer resolver, or all arms and their controls must be
rerun under another explicitly frozen optimizer contract.

### AMP contract

Current Innovation-1 screening uses a fixed AMP scale of 128 and growth
interval `2147483647` in both control and method arms. A skip, non-finite
gradient, scale change, or unequal optimizer-attempt count invalidates the
pair. If this numerical contract is retained for formal experiments, it must
also be used by Innovation 2 and 3 and by every corresponding control.

## 5. Historical 100-epoch reference

Recovered artifact:

```text
/mnt/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt
SHA-256:
54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B
```

Best and final are both epoch 100:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.51131 | 0.43493 | 0.41451 | 0.24170 |

Limitations of this artifact:

- training environment was Python 3.10.20 on an RTX 4090, not the current
  Python 3.10.12 environment;
- runtime MuSGD momentum was 0.9;
- the checkpoint contains no Git commit identity;
- it is stripped for inference: `epoch=-1`, with null optimizer, scaler, and
  update state;
- it cannot provide a strict same-initialization, same-random-sequence control
  for a new method run.

It is valid for historical reference and evaluation, but not as the sole
formal control for new Innovation-1/2/3 runs.

## 6. Current strict paired control evidence

The valid AMP128 10-epoch E1 comparison used the fixed 10% subset and three
paired seeds. Final control evidence:

| Seed | mAP50-95 | AP-tiny | Recall-tiny | Stock Top-300 tiny coverage | Normalized best rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00008 | 0.0 | 0.0 | 0.1448247 | 0.4045363 |
| 1 | 0.00005 | 0.0 | 0.00004 | 0.1213690 | 0.4099677 |
| 2 | 0.00001 | 0.0 | 0.00006 | 0.1042725 | 0.4538995 |

These low values are expected for scratch training on only 647 images for 10
epochs. They must not be compared numerically with the full-data 100-epoch
reference.

Innovation 1 TSGR failed the frozen three-seed effectiveness gate:

```text
classification: TSGR_E1_FAIL
final wins: 1/3
mean final mAP50-95 delta: -0.00001333
mean stock-coverage delta: -0.0173894
```

It must not enter a 100-epoch run in its current form.

## 7. Stock model cost reference

Measured on the current RTX 4090 with 640 input, FP16 inference, 50 warmups,
200 alternating measurements:

| Item | Stock baseline |
| --- | ---: |
| Unfused checkpoint parameters | 32,826,626 |
| Fused inference parameters | 32,004,290 |
| GFLOPs | 103.4730 |
| Batch-1 mean latency | 17.8381 ms |
| Batch-1 P95 latency | 18.2475 ms |
| Batch-8 mean latency | 22.3587 ms |
| Batch-8 P95 latency | 22.5665 ms |
| Batch-1 inference peak memory | 118.49 MiB |
| Batch-8 inference peak memory | 442.05 MiB |
| Paired 10-epoch training peak memory | 9.84 GiB |

Every innovation must report the same measurements and the percentage increase
over a control measured in the same alternating benchmark process.

## 8. Required correspondence for Innovation 1, 2, and 3

| Innovation | Current status | Required action before a formal comparison |
| --- | --- | --- |
| 1: TSGR / training-only P2 | Training conditions are strictly paired; current three-seed E1 failed | Do not start 100 epochs. Redesign only after attribution evidence, then rerun its paired gate |
| 2: IOQC-SA | Current launcher defaults to AdamW, `lr0=0.000714`, momentum 0.9, batch 1/adaptive, workers 2 | Replace those defaults with the common fixed contract and add a same-initial-state stock control |
| 3: VSF-RMR | Current launcher has the same optimizer/batch/worker drift and also forces `mosaic=0` | Use `mosaic=1.0`; if the method fundamentally requires `mosaic=0`, run and report a separate no-Mosaic stock control |

For every innovation and seed, the method and its control must share:

- the exact common initial-state fingerprint;
- the same GPU and software environment;
- the same dataset and subset hashes;
- the same sample order and augmentation RNG sequence;
- the same optimizer, scheduler, AMP, and optimizer-attempt sequence;
- the same validation preprocessing and metric code;
- the same checkpoint and resume policy;
- the same training subset at the same screening stage.

Only innovation-module structure, new innovation parameters, auxiliary loss and
its weight, and the module's pre-registered activation stage may differ.

## 9. Formal comparison rule

The historical `mAP50-95=0.24170` is the reference target. The publishable
delta for each innovation must come from its own strict paired stock control,
not by subtracting from that historical number.

A shared main table is valid only after all three innovations are rerun under
the same current common contract. Until then, IOQC-SA and VSF-RMR results are
standalone own-control experiments and must not be mixed with the current
Innovation-1 control.

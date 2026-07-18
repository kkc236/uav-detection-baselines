# RT-DETR UAV Detection Experiment Control Protocol

Audit date: 2026-07-19

This protocol applies to the matched RT-DETR-L baseline and the three standalone innovations: BTD-SE, IOQC-SA, and VSF-RMR. Results may be placed in the same comparison table only when they satisfy every common control below.

## Frozen common controls

| Item | Frozen value |
| --- | --- |
| Dataset | VisDrone train/val |
| Train images | 6471 |
| Validation images | 548 |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 8, fixed for all epochs |
| Workers | 8 |
| Initialization | Scratch, `pretrained=False` |
| Optimizer | `auto` |
| Initial learning rate | 0.01 |
| Final LR factor | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| Warmup momentum | 0.8 |
| Initial warmup bias LR | 0.1 |
| Nominal batch size | 64 |
| AMP | Enabled for the detector path |
| Seed | 0 |
| Deterministic mode | True |
| Cache | False |
| Close mosaic | Final 10 epochs |
| Mosaic / mixup | 1.0 / 0.0 |
| Scale / translate | 0.5 / 0.1 |
| Perspective / shear / degrees | 0 / 0 / 0 |
| Horizontal / vertical flip | 0.5 / 0.0 |
| HSV H/S/V | 0.015 / 0.7 / 0.4 |
| Multi-scale training | Disabled |
| Max detections | 300 |
| NMS | Disabled |
| Checkpoint period | Every epoch |

`warmup_bias_lr` may appear as `0.0` in an `args.yaml` written after a late resume. This is an Ultralytics resume artifact after the three warmup epochs have already completed; the scratch run starts with `0.1`.

## Verified environment

All three active innovation runs were audited with the same environment:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| CUDA runtime | 12.1 |
| Ultralytics | 8.4.90 |
| cuDNN | 9.1 |

Dataset signatures are identical on all three servers:

| Split | Label SHA256 | Image name/size manifest SHA256 |
| --- | --- | --- |
| Train | `e81b7a146f419368cb9df47b0a5e902af3bbfb9780008f194feff52aa6d1516d` | `c2f7625ece24a149b0872f55e29a362133a7675122a7b68e3e76bf346bd2665e` |
| Validation | `80da57a1491545e2b538dfb7a08771793521b09511ff218c9dfe887c06d69e2e` | `6d02333ffe627d4a1822eac086d2776a6901b8176ca26e2d9253cef9df491f55` |

## Allowed method-specific differences

Only the proposed module, its required forward hooks or feature graph, its auxiliary targets, and its auxiliary loss weights may differ. Project paths, run names, checkpoint paths, and resume status are operational metadata rather than optimization variables.

Numerically sensitive auxiliary losses may explicitly compute in FP32 while the common detector path remains AMP-enabled. This is part of numerical stability, not a change to the detector precision protocol, but it must remain fixed for the whole final run.

## Active runs

| Innovation | Valid run name | Batch ladder | AMP |
| --- | --- | --- | --- |
| BTD-SE | `scratch-rtdetr-l-btdse-100ep-4090` | Fixed 8 | True |
| IOQC-SA | `scratch-rtdetr-l-ioqc-sa-btdse-matched-100ep` | `[8]` | True |
| VSF-RMR | `scratch-rtdetr-l-vsf-rmr-100ep` | `[8]` | True |

The legacy IOQC-SA run named `scratch-rtdetr-l-ioqc-sa-100ep` used AdamW, a different learning rate, AMP off, and adaptive batch sizes. It is excluded from every paper comparison.

## Failure and resume rules

OOM, NaN, or Inf must stop the run without changing batch size, AMP, optimizer, learning rate, or augmentation. Resume is allowed only from a complete epoch checkpoint containing model/EMA, optimizer, scheduler, and epoch state. A partially processed epoch is discarded.

BTD-SE changed its auxiliary focal-loss arithmetic to FP32 before resuming epoch 57 to eliminate an FP16 saturation NaN. The objective is unchanged, and the saved epoch-56 checkpoint is finite. For the strictest final-paper claim, obtain one clean BTD-SE replication that uses the stabilized implementation from epoch 1; the current run remains valid as engineering and ablation evidence but should not be the only statistical replicate.

## Paper reporting rule

The main table must compare the matched baseline and standalone innovations under this protocol. Report at least the seed-0 result for every ablation. Before submission, run multiple seeds for the matched baseline and final combined method and report mean and standard deviation when compute permits.

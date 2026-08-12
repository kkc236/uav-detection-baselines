# FDR + BPDD + IRA Formal100 Validation Design

Date: 2026-08-12

## 1. Objective

Implement an RT-DETR-L-compatible IRA feature module and train a single `FDR + BPDD + IRA` seed-0 arm for 100 epochs under the frozen FDR/BPDD protocol. Validate on the frozen 548-image VisDrone validation split after every epoch and perform an independent final validation of the epoch-100 EMA checkpoint.

This run must establish both engineering compatibility and preliminary scientific performance. It must not inherit any trained FDR, BPDD, or IRA checkpoint.

## 2. Frozen authorities

- Base model: Ultralytics RT-DETR-L, Ultralytics 8.4.90.
- Runtime: Python 3.10.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, CUDA 12.1, one RTX 4090.
- Initialization: `pretrained=false`; load the frozen scratch FDR initial state with SHA256 `51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D` for every shared and FDR-private tensor. Initialize only IRA-private tensors from a deterministic private seed.
- Dataset: 6,471 training images, 548 validation images, ten classes, SHA256 `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`.
- Experiment seed: 0.
- Input/training: image size 640, batch 8, workers 8, device 0, cache disabled, deterministic enabled, AMP enabled with fixed scale 128.
- Detector: 300 queries, `NMS=false`, `max_det=300`.
- Optimizer: MuSGD, `lr0=0.01`, `lrf=0.01`, momentum 0.937, weight decay 0.0005, warmup 3 epochs, warmup momentum 0.8, warmup bias LR 0.0, `nbs=64`, cosine LR disabled.
- Augmentation: mosaic 1.0, close mosaic 10, mixup 0.0, scale 0.5, translate 0.1, degrees/shear/perspective 0.0, vertical flip 0.0, horizontal flip 0.5, HSV 0.015/0.7/0.4, cutmix/copy-paste 0.0.

## 3. Architecture

IRA is adapted from the earlier RT-DETR-X implementation to the current RT-DETR-L neck width of 256 channels. It is an independent declarative YAML layer placed after the P3 `RepC3` output and before the FDR decoder consumes P3:

```text
P3 RepC3 -> IRA(256) ---------+
P4 RepC3 ---------------------+-> FDRRTDETRDecoder
P5 RepC3 ---------------------+
                                      +
                              BPDD training loss
```

The responsibilities remain isolated:

- IRA changes only the highest-resolution P3 feature representation.
- FDR remains the only decoder box representation and retains its six cumulative 132-logit heads, preliminary box, Integral conversion, FGL, and preliminary-box supervision.
- BPDD remains a parameter-free training-only objective. It adds no parameter, decoder branch, post-processing, or inference-time cost.

## 4. IRA contract

IRA contains a 256-channel local-detail projection, two depth-wise dual-residual refinement blocks, joint spatial/channel attention, and an outer residual gate:

```text
y = x + alpha * R(x)
```

`alpha` is initialized to zero. This preserves the exact FDR graph at initialization and lets the optimizer introduce IRA evidence only when useful. The module must be separately addressable in YAML and removable without editing FDR or BPDD code.

IRA-private parameters use a deterministic private seed. Existing shared/FDR tensors are loaded bit-exactly from the frozen scratch initial state; unexpected or silently skipped shared tensors are forbidden.

## 5. Preflight

Formal100 may start only after all checks pass:

1. Static YAML/parser check: exactly one IRA layer on P3 and unchanged FDR/BPDD options.
2. Initial-state check: every common/FDR tensor equals the frozen initial-state authority; only IRA-private tensors are newly initialized.
3. Real VisDrone batch-8 CUDA forward/backward with finite stock, FGL, preliminary-box, and BPDD losses.
4. Finite common, FDR-private, and IRA-private gradient groups; fixed AMP scale remains 128 and no optimizer step is skipped.
5. BPDD activity statistics exist and use the final normal-query assignment only.
6. Evaluation-mode output is finite and keeps the frozen 300-query/no-NMS contract.
7. Combined checkpoint save/reload reproduces the same state and output.

Engineering failures are fixed with regression tests before deployment. Preflight data comes from the training split; validation labels are not used for implementation tuning.

## 6. Formal100 execution

- Start fresh from the frozen initial state; do not inherit the completed BPDD100 checkpoint or any Screen checkpoint.
- Train exactly 100 epochs on the full 6,471-image training split.
- Run the ordinary frozen validator on all 548 validation images after every epoch.
- Save a create-only epoch checkpoint, metrics record, gradient evidence, loss evidence, IRA gate/activity evidence, and SHA256 manifest after every epoch.
- Upload each completed epoch to GitHub. Network publication may lag and retry, but it must never pause GPU training or alter local evidence.
- Resume only from a hash-verified `last.pt` plus its immutable run identity. Never overwrite completed epoch evidence.
- Do not early-stop, lower thresholds, or change hyperparameters based on intermediate validation results.

## 7. Final validation and comparison

After epoch 100:

1. Independently reload the epoch-100 EMA checkpoint into the combined graph.
2. Evaluate once over the frozen 548-image validation split with the same preprocessing, class mapping, query count, `NMS=false`, and `max_det=300`.
3. Report Precision, Recall, F1, AP50, AP75, mAP50-95, tiny/small/medium/large diagnostics, and ten-class AP/AP50/AP75.
4. Audit parameter count, GFLOPs, FP16 median/P95 latency, throughput, peak memory, and checkpoint hashes.
5. Compare against the existing strict FDR100 and FDR+BPDD100 authorities as preliminary cross-run references. Because this task runs only the combined arm, it must not be described as a new strict paired Formal100 comparison.

Any positive difference over the existing BPDD100 authority is reported as preliminary evidence. A final paper claim requires the later unified ablation rerun requested by the user.

## 8. Completion criteria

The task completes only when:

- preflight passes;
- all 100 epochs and all 100 validation passes complete with finite evidence;
- epoch-100 EMA independent validation completes;
- efficiency and SHA256 audits complete;
- every epoch checkpoint/manifest and the final report are uploaded or have a verified publication ledger proving successful upload;
- the final report clearly separates engineering compatibility, preliminary gain, and claims requiring a future paired rerun.

## 9. Non-goals

- No duplicate FDR or BPDD control arm in this run.
- No inheritance from trained BPDD100 weights.
- No test-set access in the current scope.
- No changes to frozen FDR/BPDD mathematics or training hyperparameters.
- No claim of IRA benefit before the 100-epoch result exists.

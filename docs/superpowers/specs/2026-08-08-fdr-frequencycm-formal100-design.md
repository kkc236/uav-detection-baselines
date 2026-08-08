# FDR + FrequencyCM Formal100 Design

## 1. Objective

Train one full-data, seed-0, 100-epoch Ultralytics RT-DETR-L experiment that adds exactly one FrequencyCM feature module to the proven FDR model while preserving the strict FDR control protocol everywhere else.

The experiment is an incremental comparison against the completed strict FDR arm, not a replacement stock-control run and not a fine-tuning run from the epoch-100 FDR checkpoint.

## 2. Frozen authorities

- Base model and training semantics: FDR authority commit `d97e1eb7f98414752a1c1f38287697db3f2a0679`.
- FDR mechanism authority: D-FINE commit `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`.
- Runtime: Ultralytics 8.4.90, Python 3.10.12, PyTorch 2.5.1+cu121, CUDA 12.1, one RTX 4090.
- Data: the same VisDrone train/val tree and frozen dataset signatures used by the formal FDR/control arms.
- Shared initialization: the same formal FDR `initial-state.pt`; all parameter names shared with FDR must restore byte-identical tensors.
- Training: scratch semantics, full 6471-image train split, 548-image validation split, seed 0, 100 epochs, image size 640, batch 8, workers 8, deterministic true, cache false.
- Optimizer and augmentation: unchanged MuSGD and the complete frozen `FROZEN_SETTINGS` from the FDR authority.

## 3. Architecture

The existing FDR feature inputs are P3, P4, and final fused P5 at model indices `[21, 24, 27]`. Insert one `FrequencyCM(256, 256)` after index 27 and route the decoder from `[21, 24, 28]`.

```text
P3 index 21 -----------------------------+
P4 index 24 --------------------------+  |
P5 index 27 -> FrequencyCM index 28 --+--+--> FDRRTDETRDecoder
```

No FrequencyCM instance is added to P3, P4, the backbone, AIFI, Query selection, classification heads, matcher, FDR distribution heads, or loss functions.

The stock `configs/rtdetr-l-fdr.yaml` remains unchanged. The experiment uses a separate `configs/rtdetr-l-fdr-frequencycm.yaml`, making the new feature module a declarative and independently removable unit.

## 4. FrequencyCM contract

FrequencyCM contains two residual branches:

1. A frequency branch that normalizes P5, applies `rfft2`, learns a transformation of the magnitude, recombines the original phase, and applies `irfft2`.
2. A spatial branch using depth-wise convolution, SimpleGate-style multiplication, channel scaling, and projection.

Both branches are multiplied by per-channel residual scales `gamma` and `beta`, initialized to zero. With equal input/output channels, the module must therefore be an exact identity at initialization.

Because the P5 map for 640-pixel input is 20 x 20 and CUDA half-precision FFT is not generally valid for non-power-of-two transformed dimensions, the FFT/phase reconstruction path runs explicitly in FP32. Its real-valued output is cast back to the input dtype before residual fusion. The surrounding detector remains under the frozen AMP protocol.

The implementation must not claim explicit high-pass/low-pass decomposition or input-conditioned weather adaptation: the code transforms the full FFT magnitude and uses learned per-channel residual scales.

## 5. Initialization and optimizer isolation

- Shared FDR tensors are restored from the existing formal initial state with exact name/shape/value checks.
- FrequencyCM-private tensors use a fixed private seed distinct from FDR's private seed 10000.
- `gamma` and `beta` are zero.
- All FrequencyCM trainable tensors are included in MuSGD without changing shared parameter groups, LR, momentum, weight decay, scheduler, AMP scale, or gradient clipping semantics.
- A functional equivalence test must show that the initialized FDR+FrequencyCM prediction equals initialized FDR within the frozen numerical tolerance.

## 6. Tests and preflight gates

Implementation follows test-first development. The formal run cannot start until all gates pass:

1. Parser/export gate: FrequencyCM is importable, registered, and built only by the new YAML.
2. Shape gate: input and output shapes match for representative P5 sizes, including 20 x 20.
3. Identity gate: zero-initialized module returns the input exactly for equal channels.
4. Gradient gate: the first backward step produces finite `gamma`/`beta` gradients; later branch gradients become reachable after nonzero residual scales.
5. Authority gate: every shared FDR initial tensor matches the frozen initial state; only FrequencyCM-private keys are new.
6. CUDA gate: one real VisDrone batch at 640, batch 8 completes forward, loss, fixed-scale AMP backward, MuSGD step, EMA update, validation inference, checkpoint save, and resume step with finite values.
7. Protocol gate: a dry run proves all frozen settings equal the strict FDR formal arm.

ONNX/TensorRT export is outside the training gate. Native `torch.fft` export limitations must be reported in the final overhead/deployment audit.

## 7. Formal run and recovery

- Start a fresh immutable run; never inherit the screen or epoch-100 FDR checkpoint.
- Save append-only evidence for every completed epoch.
- Keep `last.pt`, `best.pt`, current epoch evidence, run arguments, source identity, and SHA256 manifests locally.
- Upload each completed epoch checkpoint and manifest to a dedicated GitHub Release before rotating the corresponding numbered local checkpoint.
- Rotation may remove only a numbered checkpoint whose remote asset identity, byte size, and SHA256 have been verified. `last.pt`, `best.pt`, evidence, manifests, logs, formal FDR/control artifacts, source, protocols, and data are never rotated.
- Resume only from a complete `last.pt`; an interrupted partial epoch is discarded.
- Training must continue when GitHub is temporarily unavailable. Assets remain queued locally and rotation pauses until publication succeeds.

## 8. Disk preparation

The server data volume has insufficient free space for 100 simultaneous numbered checkpoints. Before deployment, reclaim space only from generated caches belonging to scientifically frozen IBER/ITBER/P2-oracle experiments under `/data/uav/cache`.

The cleanup must resolve and verify every absolute target below `/data/uav/cache`; it must not touch datasets, source trees, protocols, formal results, run evidence, or checkpoints. The removed cache shards are reproducible but not recoverable from trash.

## 9. Evaluation

After epoch 100, use one frozen evaluator and identical validation preprocessing/class mapping to compare:

- strict stock RT-DETR-L control;
- strict FDR;
- FDR + FrequencyCM.

Report Precision, Recall, F1, AP50, AP75, mAP50-95, AP-tiny/small/medium/large, ten-class AP/AP50/AP75, tail-three metrics, training stability, parameters, GFLOPs, latency, FPS, peak VRAM, checkpoint size, and all relevant SHA256 values.

No best-epoch cherry-picking, threshold changes, evaluator changes, or post-hoc exclusion of negative metrics is allowed.

## 10. Acceptance boundary

The run is an experiment, not a guaranteed improvement. Engineering completion means 100 epochs, independent evaluation, overhead audit, complete evidence, and remote publication. Scientific success is reported only from the true signed deltas against strict FDR; a negative result remains valid evidence and is not converted into a pass by changing thresholds after training.

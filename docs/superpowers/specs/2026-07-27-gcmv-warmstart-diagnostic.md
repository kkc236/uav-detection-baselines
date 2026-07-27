# GCMV-EI Mature-Baseline Diagnostic

**Status:** Frozen for one seed0 diagnostic run

**Purpose:** Determine whether GCMV-EI contributes useful tiny-object evidence
once the underlying RT-DETR-L detector is already converged.

## Common initialization

Both arms load detector weights from:

```text
matched-baseline-best-epoch-0100.pt
SHA256 54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B
```

The checkpoint is a weights-only epoch-100 RTX 4090 baseline. Neither arm
restores its historical optimizer, scaler, or scheduler. Both create fresh
matched fine-tuning optimizers.

## Calibration

Before paired fine-tuning, Method receives one full-data calibration epoch:

- detector parameters and PEG `rho` are frozen;
- `gamma` remains exactly zero;
- only PLEC, GGLF, and the spatial PEG gate are optimized;
- only the frozen tiny/gate/protect auxiliary objective contributes gradients;
- the resulting module-only state is saved as a small calibrated artifact.

This is module initialization, not part of the reported ten matched detector
fine-tuning epochs.

## Paired ten-epoch fine-tuning

Control and Method both restart deterministic seed0 data/augmentation state and
fine-tune for ten epochs on all 6,471 VisDrone training images.

Common detector settings:

- batch 8, workers 8, image size 640;
- MuSGD, momentum 0.937, weight decay 0.0005;
- constant detector learning rate `1e-4`;
- AMP with fixed scale 128;
- no pretrained or historical optimizer state;
- identical augmentation and validation data.

Method-only settings:

- load the calibrated module state;
- initialize `gamma=0.02` via `rho=atanh(0.02)`;
- module learning rate `1e-3`;
- scalar `rho` has no weight decay.

The detector learning rate and all detector-side training choices remain
identical in both arms.

## Three-state evaluation

The final validation evaluates:

1. Control;
2. Method-On, with GCMV-EI active;
3. Method-Off, using the same Method checkpoint with GCMV-EI bypassed.

Required deltas:

```text
total effect   = Method-On  - Control
direct effect  = Method-On  - Method-Off
training drift = Method-Off - Control
```

## Diagnostic advance gate

The architecture advances only if:

- Method-On improves AP-tiny and tiny recall over Control;
- Method-On does not reduce mAP50-95;
- AP-medium delta is at least `-0.002`;
- AP-large delta is at least `-0.005`;
- Method-On beats Method-Off on AP-tiny;
- trained gamma is finite and materially open;
- the final spatial gate is non-degenerate.

This diagnostic is not a substitute for the final full scratch-trained
100-epoch paper comparison.

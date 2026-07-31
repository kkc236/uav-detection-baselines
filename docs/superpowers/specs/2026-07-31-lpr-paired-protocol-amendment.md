# LPR strict paired-protocol amendment

Date: 2026-07-31

This amendment supersedes the historical-baseline screening constants in the
original LPR design. No LPR result may be compared formally with the historical
100-epoch checkpoint alone.

## Frozen comparison

Every screen is a fresh paired experiment with two arms:

- `control`: stock Ultralytics RT-DETR-L;
- `lpr`: the output-isolated LPR decoder from the original design.

For seeds 0, 1, and 2, both arms load the same frozen common state. LPR-private
parameters are seed-specific and their construction must not advance the global
RNG. The arms use the same fixed 647-image hash-selected subset, sample order,
augmentation RNG sequence, validator, category map, checkpoint rules, GPU, and
software environment.

The fixed subset semantic SHA-256 is
`52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`.
The full train/val semantic SHA-256 is
`FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`.

## Frozen runtime

- Ultralytics 8.4.90, RT-DETR-L, 300 queries, 10 classes;
- RTX 4090 24 GB, driver 550.142;
- Python 3.10.12;
- PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA runtime 12.1;
- batch 8, workers 8, device 0, image size 640;
- deterministic true, cache false, pretrained false;
- AMP true with initial scale 128 and growth interval `2**31 - 1`;
- any AMP skip, scale change, or non-finite optimizer evidence invalidates the arm and its pair;
- optimizer is explicitly MuSGD with lr0 0.01, lrf 0.01, momentum 0.937,
  weight decay 0.0005, warmup epochs 3.0, warmup momentum 0.8,
  warmup bias LR 0.0, nbs 64, and cosine LR false;
- mosaic 1.0, close mosaic 10, mixup 0.0, scale 0.5, translate 0.1,
  degrees/shear/perspective/flipud 0.0, fliplr 0.5, HSV 0.015/0.7/0.4,
  cutmix 0.0, and copy-paste 0.0;
- max detections 300 and NMS false.

## Screening stage

Screening is fresh 10-epoch training on the frozen 647-image subset for all
three paired seeds. Run order is frozen to reduce order bias:

- seed 0: control, then LPR;
- seed 1: LPR, then control;
- seed 2: control, then LPR.

The LPR screen passes only if all artifact/runtime invariants pass and:

1. final mAP50-95 wins for at least two of three seeds and mean delta is positive;
2. epoch 8-10 mean mAP50-95 wins for at least two seeds and mean delta is positive;
3. no LPR tail mean is below 80% of its paired control;
4. mean validation L1 or mean validation GIoU improves across the three seeds;
5. every LPR arm has finite, nonzero gate and gradient evidence.

These criteria are frozen before any paired LPR result is visible.

## Formal stage

Passing screening does not resume a subset checkpoint into full-data training.
Formal runs start fresh on all 6471 images with the corresponding frozen
seed state and a 100-epoch scheduler. Seed 0 runs as a paired control/LPR arm
first; seeds 1 and 2 start only after the seed-0 formal gate is accepted.

The historical mAP50-95 0.24170 remains a reference target, but publishable
deltas come from the strict paired control.

## Current launch blockers

The newly provisioned server currently reports driver 595.84 and Python 3.12.3.
Its current VisDrone semantic SHA-256 is also different from the frozen full
dataset hash, although the 647-image subset name hash matches. Formal launches
must remain blocked until all three values are made exact or the user explicitly
freezes a new protocol and reruns every control under it.

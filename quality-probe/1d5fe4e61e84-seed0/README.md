# RT-DETR Learnable Quality Probe Result

- Status: `scientific_failed` at the internal Gate.
- Official validation: not opened.
- Source commit: `1d5fe4e61e8484c9364d4f0fce09ac1e009e6c81`.
- Split: fixed 518 probe-train / frozen 129 internal-dev.
- Training: C1 and Q each completed all 20 deterministic MuSGD epochs.

Internal metrics:

| Arm | Selected epoch | mAP | AP75 |
|---|---:|---:|---:|
| C0 stock | — | 0.2862886580 | 0.2923640748 |
| C1 control | 1 | 0.2834751724 | 0.2893655006 |
| Q hidden-aware | 13 | 0.2772183158 | 0.2874468303 |

Q versus the strongest control was `-0.0090703422` mAP and `-0.0049172445`
AP75. The fixed internal Gate required at least `+0.0050` mAP and strictly positive
AP75 against both C0 and C1, so the probe was frozen without accessing the official
548-image validation split.

The directory includes all 40 checkpoint/sidecar pairs, cache authorities and manifests,
hook-neutrality evidence, the immutable selection and decision reports, and the run log.
The authorized next branch is the pre-audited Ultralytics FDR-only migration; thresholds
must not be weakened and C1 must not be promoted.

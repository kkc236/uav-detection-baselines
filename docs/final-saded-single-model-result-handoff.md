# Innovation 1 Final Handoff: SADED-SM

Date: 2026-07-25

Status: `SADED_SINGLE_SEED_GO`

## Outcome

Innovation 1 now has a five-gate-positive seed-0 development-validation main
experiment.

The final method is **SADED-SM: Single-Model Scale-Aware Multi-View Routing**.
One 100-epoch RT-DETR-L checkpoint performs one full-view and four local-tile
forwards. A fixed predicted-scale router protects non-tiny full-view
detections and admits local-view information only through the tiny path.

The failed learned T-ASCV specialist is not part of the final method. Its
authoritative `TASCV_STOP` remains preserved as a negative experiment.

## Main table

| Metric | RT-DETR-L Arm A | SADED-SM | Delta | Gate | Result |
|---|---:|---:|---:|---:|:---:|
| AP-tiny-SBR | 7.1057% | 11.0251% | **+3.9194 pp** | >= +1.0 pp | PASS |
| mAP50-95 | 18.0621% | 20.6470% | **+2.5849 pp** | >= +0.3 pp | PASS |
| Tiny recall | 55.3748% | 65.5526% | **+10.1778 pp** | >= +2.0 pp | PASS |
| AP75 | 16.6666% | 18.6902% | **+2.0237 pp** | >= -0.2 pp | PASS |
| AP-large-SBR | 14.5847% | 14.3938% | **-0.1909 pp** | >= -0.5 pp | PASS |

All calculations use the exact unrounded values in
`docs/evidence/saded_single_model_final/formal_adjudication.json`.

## What was reused

The detector checkpoint is the user's existing matched seed-0 100-epoch
baseline, not a 10-epoch continuation:

- checkpoint SHA-256:
  `54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b`;
- endpoint metadata records epoch 100;
- scratch initialization (`pretrained=False`);
- seed 0, image size 640, batch 8, workers 8;
- deterministic AMP inference/training contract;
- 300 queries, `max_det=300`, NMS disabled.

No duplicate 100-epoch run was launched because the final router has no
trainable parameters. Reusing the immutable official baseline removes
unnecessary run variance and is faster than retraining an identical control.

## Frozen method

For each image:

1. obtain Arm-A predictions from the full image;
2. obtain a second candidate set by standard fusion of the same checkpoint's
   full image plus four fixed local tiles;
3. mark every full-view prediction with network-frame effective size greater
   than 16 pixels as protected;
4. deterministically match same-class full/local candidates at IoU greater
   than 0.5;
5. keep protected non-tiny predictions byte-for-byte;
6. route only emitted-size-at-most-16 local candidates through the tiny path;
7. suppress local fragments overlapping protected same-class predictions at
   IoS at least 0.5;
8. fill the remaining Top-300 capacity using the fixed score/source/query/index
   order.

The evaluator receives one unified prediction JSON. Metrics are never selected
from different systems after evaluation.

## Integrity closure

- successor source commit:
  `7f1f1e11f0c0c6d373e6172a7511ee645b4421cd`;
- full server suite: `829 passed`;
- formal checksum closure:
  `731a74f9be5ff53f589cfa8cbd500b29c2fa1678830d410ae76245ff1cf55e29`;
- formal adjudication:
  `2d46754db4166a10e64c6ec17576679bc2ddb20ad2bd8884e111fe488c00d278`;
- external root anchor:
  `9beb847c2d11db3ab67fd08c1a9c8990f4811674e2ca4864d9d8fb96f15eef02`;
- route prediction JSON:
  `4c8e4998f0cbdbbc5963fecbf05ac4dc26d56db6b95d71a076fd129a66aa740e`;
- all source, route, evaluation, checkpoint, parent-STOP, and recorded-delta
  invariants passed.

## Paper-ready claim

Recommended wording:

> We introduce a single-model scale-aware multi-view routing strategy for UAV
> tiny-object detection. The method preserves full-view non-tiny predictions
> while using local views only for the tiny-object path. On the seed-0
> VisDrone development validation experiment, it improves mAP50-95 by 2.58
> percentage points, AP-tiny by 3.92 points, tiny recall by 10.18 points, and
> AP75 by 2.02 points, while limiting AP-large change to -0.19 points.

Do not describe this result as:

- a three-seed mean;
- an independent test-set confirmation;
- evidence that T-ASCV passed;
- a zero-cost method;
- a five-model ensemble.

It is one checkpoint evaluated through five views. The router is
training-free, but inference uses five detector forwards per image and this
cost must be reported.

## Remaining paper work

The main innovation result is complete. Later paper preparation should add:

- end-to-end latency, throughput, peak VRAM, and storage;
- the already deferred ablation rows;
- multiple seeds if the submission standard requires variance estimates;
- one untouched test-dev confirmation only after the paper method is frozen.

These items do not change the current seed-0 five-gate decision.

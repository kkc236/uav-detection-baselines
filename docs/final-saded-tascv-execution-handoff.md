# final-SADED / T-ASCV authoritative execution handoff

## Objective

Innovation Point 1 is a single scale-aware dual-expert detector:

- stock RT-DETR-L owns every predicted non-tiny object;
- T-ASCV is an independent tiny specialist trained with asymmetric
  cross-view consistency;
- the SADED router protects the baseline non-tiny prefix and admits only
  tiny local candidates;
- every reported metric is computed from one unified prediction set.

No metric is selected from different models after evaluation.

## Frozen baseline and environment

- Ultralytics RT-DETR-L, Ultralytics `8.4.90`
- Python `3.10.12`
- PyTorch `2.5.1+cu121`, Torchvision `0.20.1+cu121`, CUDA `12.1`
- NVIDIA GeForce RTX 4090 24 GB, driver `550.142`
- VisDrone train/val: 6471 / 548 images, 10 classes
- dataset SHA256:
  `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`
- fixed 647-image subset SHA256:
  `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`
- from-scratch initialization, batch 8, workers 8, image size 640,
  AMP scale 128, deterministic seeds 0/1/2
- MuSGD, `lr0=0.01`, `lrf=0.01`, momentum `0.937`,
  weight decay `0.0005`
- 300 queries, `max_det=300`, NMS disabled
- the complete augmentation and optimizer contract is checksum-bound in
  the generated protocol manifest.

## Frozen method

For each stage and seed the system produces:

1. `A`: baseline full-view predictions;
2. `route_control`: baseline full-view non-tiny plus baseline five-view tiny;
3. `route_treatment`: baseline full-view non-tiny plus T-ASCV five-view tiny.

The router is GT-free. Predicted effective size above 16 px is baseline-owned.
The complete baseline non-tiny prefix is preserved byte-for-byte and cannot be
displaced by the 300-result capacity. Local candidates above 16 px are
rejected. Fragment suppression, matching, score interpolation and tie-breaking
are fixed.

## Scientific gates

Screen seed 0 uses only treatment-minus-route-control attribution:

- mAP50-95 `> 0`
- AP-tiny-SBR `>= 0`
- tiny recall `>= 0`
- AP75 `>= -0.002`
- AP-large-SBR `>= -0.005`

The three-seed screen requires at least two positive mAP seeds, positive mean
mAP, at least two non-negative AP-tiny and recall seeds, and the frozen mean
secondary guards.

Formal primary gates compare route treatment against Arm A:

- AP-tiny-SBR `>= +0.010`
- mAP50-95 `>= +0.003`
- tiny recall `>= +0.020`
- AP75 `>= -0.002`
- AP-large-SBR `>= -0.005`

Formal seed 0 must also have positive treatment-minus-route-control mAP.
The paper result reapplies the five gates to the three-seed mean and the
three-seed attribution rules.

## Authoritative state machine

```text
R0_GO
  -> paired one-batch preflight
  -> fresh 500-batch mechanism gate
  -> paired 10-epoch seed0 attribution gate
  -> paired 10-epoch seeds1/2
  -> three-seed screen gate
  -> paired 100-epoch seed0 primary+attribution gate
  -> paired 100-epoch seeds1/2
  -> three-seed formal gate
  -> exactly nine sealed confirmation predictions
  -> one protocol-scoped O_EXCL GT-open claim
  -> one nine-system confirmation evaluation
  -> no-GT terminal replay
```

Any code change after the authoritative preflight invalidates the active
protocol and requires restarting at preflight.

## Completed integration evidence

The earlier `e2864659` run is retained as integration-only evidence:

- paired preflight: `TASCV_PREFLIGHT_GO`
- fresh mechanism500: `TASCV_MECHANISM_GO`
- mechanism tail: 16,154 tiny pairs, advantage mean
  `0.3384951758764014`, win rate `0.748730964467005`
- screen seed0 baseline: 10/10 epochs, 810 batches, 145 optimizer attempts
- screen seed0 T-ASCV: 10/10 epochs, 810 batches, 145 optimizer attempts,
  810 local forwards and 810 BN-preserved batches

These screen endpoints are never formal performance evidence because the
downstream cache/router/evaluator implementation was not yet the final commit.
They may only exercise the GT-free integration closure.

## Implemented evidence closure

- endpoint summaries and `last.pt` are rehashed and predecessor gates replayed;
- training and evaluation model-source files are bridged explicitly;
- five executed views are sealed independently, including zero-detection views;
- raw views are replayed to reproduce endpoint predictions exactly;
- paired initial states, batch canaries and optimizer/runtime facts must match;
- route outputs are independently recomputed from both caches;
- GT imports occur only after cache, route, source, checksum and snapshot closure;
- a standalone adjudicator repeats the metric evaluation for development val;
- screen/formal gates and all successor authorizations are independently replayable;
- the confirmation evaluator creates one fixed protocol-level claim before any
  confirmation annotation import/read; the claim is never removed, even on error.

## Next execution

1. Commit and push the final source.
2. Run the full test suite in the frozen server environment.
3. Use the integration-only screen endpoints for GT-free cache/route smoke tests.
4. Generate a new all-`RUN_FRESH` protocol from the final clean commit.
5. Execute the authoritative state machine without source changes.

No development-val metric may be inspected before the corresponding sealed
evaluation and independent adjudication complete. The confirmation split
remains unopened until formal three-seed GO.

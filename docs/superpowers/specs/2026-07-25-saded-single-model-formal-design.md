# SADED Single-Model Formal Pivot Design

Date: 2026-07-25

Status: frozen after authoritative T-ASCV screen `STOP`

## 1. Decision

The T-ASCV screen is terminal for the learned tiny-expert route. Its sealed
decision remains `TASCV_STOP`; it is not changed, retried, or used to authorize
T-ASCV training at 100 epochs.

Innovation 1 pivots to the earlier, independently sealed SADED `route_control`
method:

**Single-Model Scale-Aware Dual-View Routing**

The same RT-DETR-L checkpoint supplies both:

- full-view global predictions; and
- full plus four local-view predictions.

The existing SADED router protects predicted non-tiny full-view detections and
uses local views only in the tiny path. There is no learned specialist,
additional training loss, second checkpoint, or parameter search.

## 2. Why no new 100-epoch training is required

The bound checkpoint is already the user's matched seed-0 100-epoch baseline:

- path:
  `/mnt/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt`;
- SHA-256:
  `54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b`;
- Ultralytics `8.4.90`;
- `epochs=100`, `pretrained=False`, seed `0`;
- `imgsz=640`, batch `8`, workers `8`;
- deterministic `True`, AMP `True`;
- `lr0=0.01`, `lrf=0.01`, momentum `0.937`,
  weight decay `0.0005`;
- warm-up `3.0`, warm-up momentum `0.8`, warm-up bias LR `0.0`;
- `nbs=64`, cosine LR disabled;
- mosaic `1.0`, close mosaic `10`, mixup `0.0`, scale `0.5`,
  translate `0.1`, degrees/shear/perspective `0.0`, flipud `0.0`,
  fliplr `0.5`, HSV `0.015/0.7/0.4`, cutmix/copy-paste `0.0`;
- 300 queries, `max_det=300`, NMS disabled.

CTAF/SADED route-control is training-free. Retraining the same stock checkpoint
after observing the result would add run variance and cost without adding a
method parameter. The existing immutable 100-epoch endpoint is therefore the
formal trained endpoint requested by the user. This is checkpoint reuse, not
continuation from a 10-epoch screen.

The paper must state that the detector was trained for 100 epochs and the
scale-aware self-ensemble router was applied without further training.

## 3. Parent evidence

### 3.1 Required stopped experiment

The successor binds the complete T-ASCV seed-0 screen adjudication:

- protocol source:
  `dbf84670a89c74eb287978d5d70bb557625ef630`;
- protocol SHA-256:
  `13d0e3ef66bfa2d35bb6037640888f7ac97993f2e43c090c6ec261a9701c25e3`;
- decision: exactly `TASCV_STOP`;
- gate SHA-256:
  `e039a9111e22badee2884fd25f25dd9b38805da29f678ffb5476abeed7652e40`;
- adjudication anchor SHA-256:
  `8341638a812483d7f9bfdaa713b3615f7fbf0a5da9128606f9ccb2b7abe247dd`.

The successor must reject `GO`, `INVALID`, a missing gate, or any digest drift.

### 3.2 Earlier R0 route

The pivot is allowed only because this route predates the stopped T-ASCV
performance result and is already sealed:

- route source commit:
  `ada48a1f09e468138e70eaa4b20cd426de6157da`;
- original model/inference source:
  `51ee6c446ffd967c12481894a9ac1cf00cad2105`;
- input manifest SHA-256:
  `aa85a80d2f43bc0a72d6a083657aa2fe539746bb79f8cabbef71516dc014cbff`;
- route anchor SHA-256:
  `e3c3a391496774412c60c921bf2db11cdbc2de908a562e5ad173123f36fb077c`;
- route checksums SHA-256:
  `6a5a4430dc53d4b196364ea5022ef88fcb3b5d165053db808db54689f7bf74fe`;
- route prediction SHA-256:
  `4c8e4998f0cbdbbc5963fecbf05ac4dc26d56db6b95d71a076fd129a66aa740e`;
- evaluation checksums SHA-256:
  `7a9598773b7c4b32ffe0d1658f785d4131146438015cbd3a32a2c946cb1efc69`;
- evaluation metrics SHA-256:
  `1708f636d60d16090b69e691a2d4d28ba16af202ae9821ed00bf97c31f45905e`;
- route and evaluation invariants: passed;
- R0 safety decision: `R0_GO`.

The earlier `43360575` coordinate-contract run remains invalid history and is
never used.

## 4. Frozen router

The implementation is reused byte-for-byte from the sealed R0 source. It must
not be rewritten in the successor.

1. Arm A is the checkpoint's full-view prediction set.
2. The local input is the same checkpoint's standard fused full plus four
   local views.
3. Baseline predictions with emitted effective size greater than 16 are the
   immutable protected prefix.
4. Same-class full/local pairs use deterministic one-to-one matching with IoU
   strictly greater than `0.5`.
5. A matched baseline tiny prediction uses the local box only when the local
   emitted box is also tiny; its score uses the frozen analytic logistic
   blend.
6. Unmatched local non-tiny predictions are rejected.
7. Unmatched local candidates fragmented against a protected same-class
   prediction at IoS at least `0.5` are rejected.
8. All remaining tiny candidates use the frozen
   `(-score, source_order, query_index, original_index)` order.
9. Fill `300 - protected_count` slots; no protected prediction is evicted.

One unified prediction JSON is used for every metric.

## 5. Formal five-item decision

The independent successor adjudicator authenticates the stopped experiment,
the input manifest, the route closure, the evaluation closure, source
identity, checkpoint identity, image count, and every invariant before reading
metrics.

It then independently computes `route_control - A` and requires:

- `AP-tiny-SBR >= +0.010`;
- `mAP50-95 >= +0.003`;
- `tiny_recall >= +0.020`;
- `AP75 >= -0.002`;
- `AP-large-SBR >= -0.005`.

Exact unrounded values are used. A missing, Boolean, or non-finite primary
metric is invalid. Non-primary diagnostic metrics may remain in the sealed
metric record. The output decision is exactly one of:

- `SADED_SINGLE_SEED_GO`;
- `SADED_SINGLE_SEED_STOP`;
- `INVALID`.

No fallback arm or changed threshold exists.

## 6. Expected sealed result to reproduce

The already sealed R0 evaluator recorded:

| Metric | Arm A | route-control | Delta |
|---|---:|---:|---:|
| AP-tiny-SBR | 0.0710571443 | 0.1102511667 | +0.0391940224 |
| mAP50-95 | 0.1806213966 | 0.2064703073 | +0.0258489108 |
| tiny recall | 0.5537479711 | 0.6555260440 | +0.1017780729 |
| AP75 | 0.1666655849 | 0.1869023302 | +0.0202367453 |
| AP-large-SBR | 0.1458467938 | 0.1439375721 | -0.0019092217 |

These values are not copied into a passing gate. The adjudicator must reproduce
them from the authenticated evaluation artifact and verify its independent
checksum closure.

## 7. Evidence and reporting boundary

The machine-generated formal checksum closure contains:

- successor manifest;
- authenticated parent bindings;
- recomputed exact deltas and five Boolean gates;
- decision;
- checksums;
- one external root anchor outside the closed directory.

The GitHub delivery commit separately contains the B/C review record and a
paper-facing result summary. Those human-facing post-adjudication documents
bind the formal root anchor but are not recursively inserted into the
machine-generated closure they describe.

This is a single-seed development-val result. Seeds 1/2, ablations, and
test-dev are out of scope. The paper may call it the seed-0 main result, but
must not claim multi-seed stability or test-set confirmation.

## 8. Failure handling

- Hash, schema, source, checkpoint, evidence, or invariant failure is
  `INVALID` and is debugged without altering scientific rules.
- A valid metric failure is `SADED_SINGLE_SEED_STOP`.
- No output from the stopped T-ASCV treatment is used by the final method.
- Existing valid and invalid evidence is preserved read-only.

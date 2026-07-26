# Innovation Point 1: SADED-SM Fresh100 Seed-0 Results

## Authoritative decision

- Decision: `SADED_SINGLE_SEED_GO`
- Pipeline status: `PIPELINE_GO`
- Frozen protocol: `final-saded-fresh-eval-f9d01f65`
- Post-processing source commit: `f9d01f6510e95e02454688654115a6b0d3f9ad33`
- Training source commit: `c5c353744f0d07366350389bf8d6c5fe0f62b8f8`
- Seed: `0`
- Detector training: fresh 100 epochs
- Validation set: fixed VisDrone val, 548 images
- Test-dev: not accessed

The baseline arm `A` and the scale-routed arm `route_control` were produced from
the same fixed 100-epoch endpoint. The comparison therefore isolates the
training-free five-view SADED-SM routing and fusion logic.

## Five frozen gates

All values below are absolute percentages; deltas are percentage points.

| Gate | Arm A | SADED-SM | Delta | Frozen requirement | Result |
|---|---:|---:|---:|---:|---|
| mAP50-95 | 7.256641% | 8.796812% | +1.540171 pp | at least +0.300000 pp | PASS |
| AP75 | 5.750867% | 6.375656% | +0.624789 pp | at least -0.200000 pp | PASS |
| AP-tiny-SBR | 0.981150% | 2.849149% | +1.867999 pp | at least +1.000000 pp | PASS |
| Tiny Recall | 22.251734% | 38.973735% | +16.722001 pp | at least +2.000000 pp | PASS |
| AP-large-SBR | 14.639262% | 14.596481% | -0.042782 pp | at least -0.500000 pp | PASS |

Additional headline metric:

| Metric | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| AP50 | 16.668525% | 21.474425% | +4.805900 pp |

## Scale-wise diagnostic results

SBR scale metrics use the evaluator's effective GT edge length,
`sqrt(effective width * effective height)`, after applying the frozen view
gain. The bins are tiny `<=16`, small `(16,32]`, medium `(32,96]`, and large
`>96` pixels. These custom SBR bins are not the official COCO scale bins.

| SBR metric | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| AP-tiny-SBR | 0.981150% | 2.849149% | +1.867999 pp |
| AP-small-SBR | 5.917290% | 4.951659% | -0.965630 pp |
| AP-medium-SBR | 16.937079% | 16.087555% | -0.849524 pp |
| AP-large-SBR | 14.639262% | 14.596481% | -0.042782 pp |

The result supports a scale-specialist claim, not a claim of uniform
improvement at every scale. In this seed-0 VisDrone development-validation
run, the tiny and overall metrics increase, AP-large-SBR changes by
-0.042782 pp, and the small and medium partitions decline.

## Detection-accounting diagnostic

The sealed evaluator's aggregate counts changed as follows:

| Count | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| Predictions | 164,283 | 164,400 | +117 |
| True positives | 14,811 | 19,376 | +4,565 |
| False positives | 141,835 | 136,326 | -5,509 |
| False negatives | 23,948 | 19,383 | -4,565 |
| Neutralized | 7,637 | 8,698 | +1,061 |

At IoU 0.50, the tiny-partition TP count independently rises by 4,533. The
scale diagnostics are non-additive with the evaluator's aggregate TP count, so
this value is not presented as an exact decomposition or a causal ablation.

## Routing-capacity diagnostic

- Images: 548
- Accepted local candidates: 112,348 total; median 218 per image
- Protected baseline predictions: 68,305 total; median 112.5 per image
- Capacity-rejected local candidates: 59,792 total; median 109 per image
- Remaining tiny slots: 96,095 total; median 187.5 per image
- All route invariants: PASS
- GT fields/modules in router: absent
- Source/cache snapshots: unchanged

## Integrity closure

The protocol, cache, route, sealed evaluation and standalone adjudication
closures were independently revalidated after the terminal decision.

- Runtime evidence files, excluding this explanatory report: 47
- Runtime evidence size, excluding this explanatory report: 107,421,342 bytes
- Adjudication root-anchor SHA256:
  `D8AE2EACD52C26BCC40E01671CD96C49BD9FBEC241F666DD405E732F49437FFA`
- Adjudication checksums SHA256:
  `BC526022D8D8595AFCBE01DCFCF9EEE3C56FBE579DCAF602C8093863AAFC5AAE`
- Training checkpoint SHA256:
  `515674348D0FF542663FE6FB4317240FC167A71EA31FACC1DEFE6A7E91B521F8`

The earlier `75c8f85d` post-processing attempt remains preserved as
`PIPELINE_INVALID`; none of its partial cache, route or evaluation artifacts
were reused.

## Paper-safe conclusion

Under the approved single-seed, same-domain dev-val feasibility protocol,
SADED-SM passes all five frozen gates and can be frozen as Innovation Point 1.
The defensible claim is:

> On the seed-0 VisDrone development-validation run, scale-decoupled
> multi-view routing improves the tiny and overall metrics, while
> AP-large-SBR is approximately maintained with a change of -0.0428
> percentage points.

This result is not yet a multi-seed, test-dev, ablation, or cross-dataset
generalization claim. Those confirmations can be performed later without
changing the frozen Innovation Point 1 endpoint or routing rules.

# Complete Fresh100 Seed-0 Metric Summary

All accuracy and recall values are percentages. Deltas are percentage points.
These values come from the sealed 548-image VisDrone validation evaluation of
the fresh seed-0 100-epoch checkpoint.

## Overall metrics

| Metric | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| mAP50-95 | 7.256641% | 8.796812% | +1.540171 pp |
| AP50 | 16.668525% | 21.474425% | +4.805900 pp |
| AP75 | 5.750867% | 6.375656% | +0.624789 pp |

## Scale AP50-95

| Scale | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| Tiny | 0.981150% | 2.849149% | +1.867999 pp |
| Small | 5.917290% | 4.951659% | -0.965630 pp |
| Medium | 16.937079% | 16.087555% | -0.849524 pp |
| Large | 14.639262% | 14.596481% | -0.042782 pp |

## Scale AP50

| Scale | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| Tiny | 3.556424% | 9.842975% | +6.286551 pp |
| Small | 16.899690% | 14.312865% | -2.586824 pp |
| Medium | 30.748243% | 29.097066% | -1.651177 pp |
| Large | 19.441598% | 19.366524% | -0.075074 pp |

## Scale AP75

| Scale | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| Tiny | 0.318031% | 0.934908% | +0.616877 pp |
| Small | 3.088574% | 2.548715% | -0.539858 pp |
| Medium | 17.354687% | 16.449362% | -0.905325 pp |
| Large | 17.428858% | 17.382315% | -0.046543 pp |

## Recall and detection accounting

| Metric | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| Tiny recall | 22.251734% | 38.973735% | +16.722001 pp |
| Predictions | 164,283 | 164,400 | +117 |
| True positives | 14,811 | 19,376 | +4,565 |
| False positives | 141,835 | 136,326 | -5,509 |
| False negatives | 23,948 | 19,383 | -4,565 |
| Neutralized predictions | 7,637 | 8,698 | +1,061 |

The aggregate TP, FP, FN and neutralized counts correspond to the evaluator's
IoU 0.50 accounting. Complete per-scale counts for IoU 0.50 through 0.95 are
available in `PER_IOU_COUNTS.csv`.

## Frozen five-gate decision

| Gate | Observed delta | Requirement | Margin | Result |
|---|---:|---:|---:|:---:|
| mAP50-95 | +1.540171 pp | >= +0.300000 pp | +1.240171 pp | PASS |
| AP75 | +0.624789 pp | >= -0.200000 pp | +0.824789 pp | PASS |
| AP-tiny-SBR | +1.867999 pp | >= +1.000000 pp | +0.867999 pp | PASS |
| Tiny recall | +16.722001 pp | >= +2.000000 pp | +14.722001 pp | PASS |
| AP-large-SBR | -0.042782 pp | >= -0.500000 pp | +0.457218 pp | PASS |

## Routing-capacity summary

| Metric | Total | Minimum/image | Median/image | Maximum/image |
|---|---:|---:|---:|---:|
| Accepted local candidates | 112,348 | 15 | 218.0 | 300 |
| Capacity-rejected candidates | 59,792 | 6 | 109.0 | 205 |
| Protected baseline predictions | 68,305 | 0 | 112.5 | 300 |
| Remaining tiny slots | 96,095 | 0 | 187.5 | 300 |

All routing invariants passed for all 548 images. The router contained no GT
fields or GT module and preserved the source/cache snapshots.

## Machine-readable files

- `ALL_METRICS.csv`: all scalar metrics and aggregate counts.
- `PER_IOU_COUNTS.csv`: both arms, four scales and ten IoU thresholds.
- `ROUTING_SUMMARY.csv`: compact route-capacity statistics.
- `evaluation/metrics.json`: authoritative full evaluator output.
- `adjudication/adjudication.json`: absolute metrics, deltas, gates and
  decision.

The evaluator also records `AP-tiny`, `AP-small`, `AP-medium` and `AP-large`
aliases. They are byte-for-byte numerically identical to the corresponding
`*-SBR` entries shown above, so they are not duplicated in the tables.
Class-wise AP was not emitted by this sealed evaluator and therefore is not
invented or inferred here.

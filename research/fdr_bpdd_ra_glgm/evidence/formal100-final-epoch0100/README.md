# FDR + BPDD + RA-GLGM v1.1 Formal100 final report

## Outcome

The D arm completed 100/100 epochs on VisDrone train 6471 / official val 548.
Engineering integrity passed: epochs and queue are contiguous 1-100, all audited numeric
values are finite, AMP skipped steps are zero, and FDR/RA gradients plus BPDD loss are
positive in every epoch.

The preregistered scientific endpoint did not pass. Epoch100 mAP50-95 is
`0.28555` and the epoch96-100 mean is `0.286062`. The best
online epoch is 89 at `0.28733`, but best
epoch is supplemental only and cannot replace the fixed endpoint.

## D metrics

| Endpoint | P | R | F1 | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Epoch100 | 0.57379 | 0.48786 | 0.52735 | 0.47932 | 0.28881 | 0.28555 |
| Epoch96-100 mean | 0.576484 | 0.487578 | 0.528317 | 0.480248 | 0.289102 | 0.286062 |
| Best online epoch89 | 0.58249 | 0.48487 | 0.52922 | 0.48057 | 0.29138 | 0.28733 |

## A/B/C/D endpoint reference

| Arm | Model | Endpoint kind | P | R | F1 | AP50 | AP75 | mAP50-95 | Tail5 mAP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | FDR | online epoch100 | 0.56778 | 0.49350 | 0.52804 | 0.48480 | 0.29273 | 0.28971 | 0.289874 |
| B | FDR+BPDD | locked independent final EMA | 0.57063 | 0.49446 | 0.52983 | 0.48641 | 0.29810 | 0.29226 | not recorded |
| C | FDR+RA-GLGM v1.1 | online epoch100 | 0.56796 | 0.48326 | 0.52220 | 0.47882 | 0.28966 | 0.28674 | 0.286670 |
| D | FDR+BPDD+RA-GLGM v1.1 | online epoch100 | 0.57379 | 0.48786 | 0.52735 | 0.47932 | 0.28881 | 0.28555 | 0.286062 |

D epoch100 mAP deltas are `-0.4160 pp`
vs A, `-0.6708 pp` vs B, and
`-0.1190 pp` vs C.

## Interpretation boundary

A/B and C/D do not share one initialization authority. B's published per-epoch receipts
contain checkpoint identity but no per-epoch metrics, so its tail-five value is deliberately
left unreported. The table is a controlled historical endpoint reference, not a fresh
four-arm paired result. It cannot establish synergy, bitwise reproducibility, or statistical
significance. A synergy claim requires fresh matched A/B/C/D arms, multiple seeds, and the
interaction contrast `D + A > B + C`.

The framework uses zero-based raw epoch indices. Therefore `epoch99.pt` is the milestone
saved after completed epoch 100. It is byte-identical to `committed.pt` at SHA-256
`F066B23B2B6679BD7FFBD95CA151859A947F38F658F7C03B057367D3A101F986`. The framework subsequently stripped `last.pt`; that file is an inference
copy and is not the recovery checkpoint bound by the epoch100 receipt. No checkpoint is
included in this publication.

`grid_sampler_2d_backward_cuda` produced a deterministic-algorithm warning once per epoch,
so protocol determinism does not imply bitwise-identical CUDA reruns.

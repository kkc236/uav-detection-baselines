# Innovation 1 Fresh100 Result: SADED-SM

Date: 2026-07-26

Status: `SADED_SINGLE_SEED_GO`

The later fresh-from-scratch seed-0 100-epoch endpoint has completed its
authoritative single-endpoint SADED-SM evaluation. All five frozen gates pass:

| Metric | Arm A | SADED-SM | Delta |
|---|---:|---:|---:|
| mAP50-95 | 7.2566% | 8.7968% | +1.5402 pp |
| AP75 | 5.7509% | 6.3757% | +0.6248 pp |
| AP-tiny-SBR | 0.9812% | 2.8491% | +1.8680 pp |
| Tiny recall | 22.2517% | 38.9737% | +16.7220 pp |
| AP-large-SBR | 14.6393% | 14.5965% | -0.0428 pp |

The full paper-facing evidence package is in
[`docs/evidence/final-saded-fresh100-seed0-go`](evidence/final-saded-fresh100-seed0-go/README.md).

This is a single-seed VisDrone development-validation result. AP-small-SBR
declines by 0.9656 pp and AP-medium-SBR declines by 0.8495 pp; therefore the
result supports a tiny-specialist and large-preservation claim rather than an
all-scale improvement claim.

The immutable endpoint identifiers are:

- training commit:
  `c5c353744f0d07366350389bf8d6c5fe0f62b8f8`;
- post-processing commit:
  `f9d01f6510e95e02454688654115a6b0d3f9ad33`;
- checkpoint SHA256:
  `515674348D0FF542663FE6FB4317240FC167A71EA31FACC1DEFE6A7E91B521F8`;
- adjudication root-anchor SHA256:
  `D8AE2EACD52C26BCC40E01671CD96C49BD9FBEC241F666DD405E732F49437FFA`.

Earlier SADED evidence based on a different endpoint remains preserved and
must not be mixed with this fresh100 result.

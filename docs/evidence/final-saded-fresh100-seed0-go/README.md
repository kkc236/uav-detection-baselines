# Final SADED Fresh100 Seed-0 GO Evidence

This directory is the paper-facing evidence package for the authoritative
fresh-from-scratch 100-epoch seed-0 SADED-SM run.

## Decision

- Pipeline: `PIPELINE_GO`
- Adjudication: `SADED_SINGLE_SEED_GO`
- Five frozen gates: all passed
- Failures: none
- Test-dev: not accessed
- Post-processing source:
  `f9d01f6510e95e02454688654115a6b0d3f9ad33`
- Training source:
  `c5c353744f0d07366350389bf8d6c5fe0f62b8f8`
- Fixed checkpoint SHA256:
  `515674348D0FF542663FE6FB4317240FC167A71EA31FACC1DEFE6A7E91B521F8`

The baseline arm and SADED-SM arm use the same fixed detector endpoint. The
comparison isolates the training-free five-view scale router and fusion logic.

## Main result

| Metric | Arm A | SADED-SM | Delta | Frozen gate | Result |
|---|---:|---:|---:|---:|:---:|
| mAP50-95 | 7.256641% | 8.796812% | +1.540171 pp | >= +0.300000 pp | PASS |
| AP75 | 5.750867% | 6.375656% | +0.624789 pp | >= -0.200000 pp | PASS |
| AP-tiny-SBR | 0.981150% | 2.849149% | +1.867999 pp | >= +1.000000 pp | PASS |
| Tiny recall | 22.251734% | 38.973735% | +16.722001 pp | >= +2.000000 pp | PASS |
| AP-large-SBR | 14.639262% | 14.596481% | -0.042782 pp | >= -0.500000 pp | PASS |

AP50 changes from 16.668525% to 21.474425%, a gain of 4.805900
percentage points.

## Important limitation

This is not a uniform all-scale improvement:

- AP-small-SBR changes by -0.965630 pp.
- AP-medium-SBR changes by -0.849524 pp.

The paper-safe interpretation is that SADED-SM improves tiny-object
sensitivity and overall accuracy while approximately preserving large-object
performance in this run. This package supports a sealed seed-0 VisDrone
development-validation claim only. It does not establish multi-seed
stability, test-dev performance, cross-dataset generalization, statistical
significance, or SOTA.

## Package contents

- `RESULTS_AND_ANALYSIS.md`: human-readable result and interpretation.
- `PAPER_MAIN_RESULTS.csv`: paper-table values in a compact form.
- `ALL_METRICS.md` and `ALL_METRICS.csv`: all available scalar metrics,
  aggregate detection counts and five-gate margins.
- `PER_IOU_COUNTS.csv`: TP, FP, FN and neutralized counts for both arms, four
  scales and every IoU threshold from 0.50 to 0.95.
- `ROUTING_SUMMARY.csv`: compact route-capacity statistics.
- `figures/`: three SVG figures for the current result, scale trade-off and
  frozen-gate margins.
- `RELEASE_MANIFEST.json`: machine-readable release identity and hashes.
- `protocol/`: frozen image, dataset, endpoint and source authority.
- `training/`: training arguments, epoch results, protocol and endpoint
  summary. Model weights are indexed but intentionally not committed.
- `cache/`: cache metadata, invariants and original checksum closure.
- `route/`: route metadata, invariants and per-image capacity diagnostics.
- `evaluation/`: absolute metrics, deltas, invariants and sealed claim.
- `adjudication/`: five-gate decision, bindings and checksum closure.
- `logs/`: terminal stage statuses and exit codes.
- `runtime/`: the exact serial post-processing driver.
- `full_evidence_file_index.tsv`: SHA256 and size for the 47 runtime-evidence
  files in the local 107 MB closure; explanatory reports are excluded.
- `training_endpoint_file_index.tsv`: SHA256 and size for the locally archived
  training endpoint files.
- `PACKAGE_CHECKSUMS.sha256` and `PACKAGE_ANCHOR.json`: checksum closure for
  the GitHub-facing package, including the derived tables and SVG figures.

Related repository report:

- [`../../final-saded-fresh100-experiment-progress-report.md`](../../final-saded-fresh100-experiment-progress-report.md):
  Chinese paper-oriented experiment progress, research questions, main table,
  figure analysis and remaining experiments. It is stored outside this
  evidence-package checksum closure.

## Large-file policy

The three large inference blobs and the two checkpoints are not committed to
GitHub. Their exact sizes and SHA256 values remain in the two file indexes.
The original cache/route `checksums.sha256` files are retained, so a holder of
the archived blobs can reconstruct and verify the complete closure.

The 12,148,595-byte full training log is also kept out of GitHub; its SHA256
is recorded in `training_endpoint_file_index.tsv` and `RELEASE_MANIFEST.json`.

Paths in `full_evidence_file_index.tsv` use the original local archive layout.
This public package reorganizes anchors and the runtime driver for display.
Only `saded-fresh100-seed0-last.pt` is the evaluated paper endpoint;
`saded-fresh100-seed0-best.pt` is indexed for archival recovery only and
contributes to no reported metric.

`training_endpoint_file_index.tsv` is a raw inventory of the local archive.
Its `saded-postprocess-f9d01f65.bundle` entry is a locally regenerated Git
transport bundle with file SHA256 `DFF8587B...`, pointing to commit
`f9d01f65`. It is not the sealed protocol's `15E785AA...` canonical 46-file
source-closure digest. The `75c8f85d.bundle` entry belongs to the preserved
historical INVALID attempt.

The full local archive is:

`artifacts/final-saded-fresh100-c5c35374/final-saded-fresh-eval-f9d01f65`

The adjudication root-anchor SHA256 is:

`D8AE2EACD52C26BCC40E01671CD96C49BD9FBEC241F666DD405E732F49437FFA`

## Relationship to earlier evidence

Earlier immutable SADED evidence in this repository uses a different
checkpoint and protocol. It is intentionally preserved. Results from the two
endpoints must not be mixed. This directory is the authoritative package for
the later fresh100 checkpoint identified above.

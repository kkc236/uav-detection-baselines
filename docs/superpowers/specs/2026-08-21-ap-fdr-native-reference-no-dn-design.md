# AP-FDR Native-Reference + No-DN Combined Formal100 Design

## Objective

Run one source-bound seed-0 Formal100 experiment that combines the two already
validated AP-FDR changes:

1. use the native RT-DETR decoder reference instead of the preliminary box as
   the shared distribution-decoding reference;
2. retain stock RT-DETR denoising losses while disabling the four additional
   DN-side FDR losses.

The experiment tests their interaction. It does not introduce a third method,
change BPDD/FIA, or claim that the single-factor gains are additive.

## Frozen Method Contract

The combined YAML is byte-derived from `configs/rtdetr-l-fdr.yaml` and changes
exactly two leaves:

- `head[-1][3][2].preliminary_box`: `true` to `false`;
- `fdr_loss.supervise_dn_fdr`: `true` to `false`.

All other graph and loss fields remain identical. In particular, cumulative
six-layer distribution refinement, 33 bins per side, `reg_scale=4.0`,
`up=0.5`, normal-query FGL, the native RT-DETR classification/L1/GIoU losses,
and native RT-DETR DN losses remain enabled.

## Frozen Training Contract

- dataset: VisDrone2019 train/val from the existing server authority;
- model: RT-DETR-L;
- initial state: the same frozen FDR initial-state artifact used by both
  completed single-factor ablations;
- seed: 0;
- epochs: 100;
- input: 640;
- batch: 8;
- workers: 8;
- optimizer and all remaining settings: `FROZEN_SETTINGS` from
  `scripts/train_rtdetr_fdr.py`;
- run identity: `formal-seed0-ap-fdr-no-preliminary-no-dn`;
- no automatic retry after a training failure;
- server remains powered on.

## Evidence and Evaluation

Training authority binds the committed source, combined YAML SHA-256, frozen
initial-state SHA-256, dataset signature, and complete settings. Completion
requires a continuous 100-row `results.csv`, `best.pt`, and `last.pt`.

The best checkpoint is selected only by maximum validation mAP50-95. After
training, the same `best.pt` is evaluated on val and test with the existing
frozen evaluator contract: `imgsz=640`, `batch=8`, `workers=8`, `conf=0.001`,
`max_det=300`, and `nms/cache/half/rect=false`. P, R, AP50, AP75, and
mAP50-95 are recorded separately for each split.

## Decision Rule

- If combined best-val mAP50-95 is at least `0.29666`, adopt the combined
  configuration as the leading FDR candidate.
- If it is below `0.29666`, retain the already validated no-preliminary-only
  configuration as the leading candidate.
- No paper table is updated automatically. Results enter the manuscript only
  after evidence review.

## Publication Scope

After successful training and evaluation, upload only this combined run to a
separate prerelease in the private repository
`kkc236/icassp2027-fdr-bpdd-fia-material`. The evidence package includes the
committed config, source identity, authority record, `args.yaml`, `results.csv`,
`best.pt`, `last.pt`, training log, exact val/test outputs, hashes, and a
publication manifest. Neither earlier single-factor archive is republished.

## Failure Behavior

Incomplete training, missing artifacts, non-continuous epochs, failed exact
evaluation, a non-private publication target, or asset verification failure
stops publication and writes a local failure record. It does not restart
training, alter formal paper tables, shut down the server, or upload partial
evidence.

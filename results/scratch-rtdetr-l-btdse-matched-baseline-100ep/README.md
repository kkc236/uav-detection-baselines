# Matched RT-DETR-L Baseline Result Archive

This archive contains the completed 100-epoch result artifacts for the fixed-protocol RT-DETR-L VisDrone baseline. Model weights are intentionally excluded from this compact result archive.

## Result continuity

- Epochs 1-11 come from the previously protected GitHub `training-results` record.
- Epochs 12-100 come from the resumed run supplied in `scratch-rtdetr-l-btdse-matched-baseline-100ep.zip`.
- The two segments are non-overlapping and form a complete 1-100 epoch sequence.
- Resume-stage elapsed times were offset by the epoch-11 cumulative time so the merged `time` column remains monotonic.
- `args.yaml` preserves the epoch-1 scratch protocol. `args-resume-epoch12.yaml` records the operational paths and overrides used when continuing from epoch 11.

## Best result

- Best epoch: 100
- Precision: 0.51131
- Recall: 0.43493
- mAP50: 0.41451
- mAP50-95: 0.24170

## Integrity

`RESULT_MANIFEST.json` records hashes for the merged CSV, arguments, source archive, and available checkpoints. The original source archive SHA256 is:

`856bbf6cbc6229586aaf258a48b32d8f9ea3f9427bba00991c6f57b265cee34c`

The complete training log is stored separately in the GitHub Release as `matched_baseline_training.log`.

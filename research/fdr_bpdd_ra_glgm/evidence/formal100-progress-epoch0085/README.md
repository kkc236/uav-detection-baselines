# FDR + BPDD + RA-GLGM v1.1: epoch 1-85 snapshot

This directory is a checkpoint-free interim snapshot of the single-arm Formal100 run.

- Data: VisDrone train 6471 / official val 548
- Schedule: 100 epochs; this snapshot contains exactly completed epochs 1-85
- Model: RT-DETR-L + FDR + BPDD + RA-GLGM v1.1
- Seed: 0
- Input/batch: 640 / 8
- Source commit: `5926ac7502ab355b8e50efc4f7af94a16b532de0`
- Epoch 85 online mAP50-95: `0.28634`
- Best online mAP50-95 through epoch 85: `0.28669` at epoch 83

Use `metrics-epoch001-0085.csv` for direct comparison and
`combo-epochs-001-085.jsonl` for complete per-epoch diagnostics. F1 is derived as
`2 * precision * recall / (precision + recall)`.

These are online validation metrics, not a locked epoch100 re-evaluation. The run is
cross-authority relative to historical FDR/BPDD evidence and cannot establish module
synergy or statistical significance by itself. No `.pt` checkpoint is included.

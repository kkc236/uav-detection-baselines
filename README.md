# UAV Object Detection Reproduction

This workspace reproduces baseline detectors for UAV aerial object detection.

Scientific problems:

1. Background interference: small objects have weak responses in complex aerial backgrounds.
2. Dense objects: vehicles and pedestrians are densely distributed with unclear adjacent boundaries.
3. Scale variation: flight altitude and viewpoint changes create unstable object scales.

Baselines:

- YOLO from scratch: `python scripts/train_yolo.py --epochs 1 --imgsz 640 --batch 4 --name smoke-yolo-scratch`
- RT-DETR from scratch: `python scripts/train_rtdetr.py --epochs 1 --imgsz 640 --batch 1 --name smoke-rtdetr-scratch`

Adaptive RTX 5090 resume:

- Start from a scratch `last.pt` with optimizer state: `python scripts/train_rtdetr_adaptive.py --checkpoint /absolute/path/to/last.pt --batch 16`
- Batch levels are `10, 12, 14, 16, 18`. Stable epochs promote by two; OOM events demote by two and true-resume from the last completed epoch.
- Live state: `cat logs/adaptive_rtdetr_status.json`
- Training log: `tail -f logs/adaptive_rtdetr.log`
- The server launcher publishes metrics plus `best.pt` and `last.pt`, verifies the GitHub release, and only then powers off.

The scripts use `VisDrone.yaml`. Ultralytics downloads and converts VisDrone automatically on first use.
Training is configured with YAML architecture files and `pretrained=False`, so pretrained `.pt` weights are not loaded.

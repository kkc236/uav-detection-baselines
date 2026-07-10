# UAV Object Detection Reproduction

This workspace reproduces baseline detectors for UAV aerial object detection.

Scientific problems:

1. Background interference: small objects have weak responses in complex aerial backgrounds.
2. Dense objects: vehicles and pedestrians are densely distributed with unclear adjacent boundaries.
3. Scale variation: flight altitude and viewpoint changes create unstable object scales.

Baselines:

- YOLO from scratch: `python scripts/train_yolo.py --epochs 1 --imgsz 640 --batch 4 --name smoke-yolo-scratch`
- RT-DETR from scratch: `python scripts/train_rtdetr.py --epochs 1 --imgsz 640 --batch 1 --name smoke-rtdetr-scratch`

The scripts use `VisDrone.yaml`. Ultralytics downloads and converts VisDrone automatically on first use.
Training is configured with YAML architecture files and `pretrained=False`, so pretrained `.pt` weights are not loaded.

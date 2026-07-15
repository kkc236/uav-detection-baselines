# UAV Object Detection Reproduction

This workspace reproduces baseline detectors for UAV aerial object detection.

Scientific problems:

1. Background interference: small objects have weak responses in complex aerial backgrounds.
2. Dense objects: vehicles and pedestrians are densely distributed with unclear adjacent boundaries.
3. Scale variation: flight altitude and viewpoint changes create unstable object scales.

Baselines:

- YOLO from scratch: `python scripts/train_yolo.py --epochs 1 --imgsz 640 --batch 4 --name smoke-yolo-scratch`
- RT-DETR from scratch: `python scripts/train_rtdetr.py --epochs 1 --imgsz 640 --batch 1 --name smoke-rtdetr-scratch`
- BTD-SE smoke on the local RTX 4070: `powershell -ExecutionPolicy Bypass -File scripts/run_btdse_local.ps1 -Smoke`
- BTD-SE full scratch run: `powershell -ExecutionPolicy Bypass -File scripts/run_btdse_local.ps1 -Epochs 100 -Batch 1`
- BTD-SE RTX 4090 protected run: see `docs/RTX4090_SERVER_GUIDE.md`
- Standalone IOQC-SA protected run on RTX 3090/4090/5090: see `docs/IOQC_SA_SERVER_GUIDE.md`

BTD-SE uses `configs/rtdetr-l-btdse.yaml`, preserves VisDrone ignored boxes in `labels_ignore`, and adds background-reliability and saliency focal losses. The local runtime used for reproducibility is `C:\uav_env\Scripts\python.exe` with Ultralytics 8.4.90 and PyTorch 2.5.1+cu121.

Local BTD-SE recovery:

- `last.pt` is updated every completed epoch and an independent `epochN.pt` snapshot is also retained.
- `run_btdse_local.ps1` validates checkpoints after an abnormal exit, resumes from `last.pt`, and falls back to the newest readable `epochN.pt` if needed.
- Training output: `Get-Content logs\btdse_local_latest.log -Wait -Tail 20`
- Recovery events: `Get-Content logs\btdse_local_supervisor.log -Wait -Tail 20`
- Checkpoints: `runs\btdse\scratch-rtdetr-l-btdse-100ep\weights`

RTX 4090 server protection uses `scripts/setup_btdse_4090.sh` and `scripts/run_btdse_4090.sh`. Heavy checkpoints are uploaded as rolling GitHub Release assets, while metrics and SHA256 manifests are committed to the `training-results` branch.

Standalone IOQC-SA keeps the stock `rtdetr-l.yaml` inference graph and adds only a training-time FP32 P3 sampling objective. `scripts/setup_ioqc_sa_server.sh` detects the GPU generation, and `scripts/run_ioqc_sa_server.sh` adapts batch size, falls back from AMP after a non-finite loss, validates checkpoints, and publishes isolated IOQC-SA Release assets. The complete clean-server and migration workflow is documented in `docs/IOQC_SA_SERVER_GUIDE.md`.

Adaptive RTX 5090 resume:

- Start from a scratch `last.pt` with optimizer state: `python scripts/train_rtdetr_adaptive.py --checkpoint /absolute/path/to/last.pt --batch 16`
- Batch levels are `10, 12, 14, 16, 18, 20`. Stable epochs promote by two; OOM events demote by two and true-resume from the last completed epoch.
- Live state: `cat logs/adaptive_rtdetr_status.json`
- Training log: `tail -f logs/adaptive_rtdetr.log`
- The server launcher publishes metrics plus `best.pt` and `last.pt`, verifies the GitHub release, and only then powers off.

The scripts use `VisDrone.yaml`. Ultralytics downloads and converts VisDrone automatically on first use.
Training is configured with YAML architecture files and `pretrained=False`, so pretrained `.pt` weights are not loaded.

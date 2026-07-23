# SBR-RTDETR G0 server guide

The runner is fail-closed and exposes operational settings only. Scientific
constants (640/1088 letterbox sizes, confidence 0.001, max detections 300,
IoS 0.5, tile geometry, SP-BRF and gates) are frozen in source.

## Install and smoke run

```powershell
python -m pip install -r requirements.txt
python scripts/run_sbr_g0.py s0 `
  --checkpoint C:\models\rtdetr.pt `
  --checkpoint-sha256 <sha256> `
  --data C:\datasets\VisDrone.yaml --split val `
  --smoke-manifest smoke.json --output evidence\s0 --device 0 --workers 0
```

`smoke.json` is a deterministic list of 8–16 validation image paths. S0
produces evidence and never emits a research pass.

## Full G0-A and gated follow-ups

```powershell
python scripts/run_sbr_g0.py g0-a --checkpoint C:\models\rtdetr.pt `
  --checkpoint-sha256 <sha256> --data C:\datasets\VisDrone.yaml `
  --split val --output evidence\g0-a --device 0 --workers 0

python scripts/run_sbr_g0.py g0-b --checkpoint C:\models\rtdetr.pt `
  --checkpoint-sha256 <sha256> --data C:\datasets\VisDrone.yaml `
  --split val --gate evidence\g0-a\g0_gate.json --output evidence\g0-b
```

G0-A requires all 548 images and writes deterministic JSON/JSONL/GZip
artifacts, SHA-256 checksums, runtime metadata, and an
`independent_adjudication.json` placeholder with status `NOT_RUN`. G0-B/C
refuse to run unless the prior gate status is exactly `SBR_G0A_PASS` and all
source, checkpoint, dataset, and protocol hashes match.

Never add tile, overlap, confidence, IoS, max-det, size-bin, fusion-weight,
or gate-threshold flags. Any unknown option is rejected by `argparse`.

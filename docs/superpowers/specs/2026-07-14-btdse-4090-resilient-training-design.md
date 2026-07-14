# BTD-SE RTX 4090 Resilient Training Design

## Goal

Run scratch RT-DETR-L with BTD-SE V2.5-S for 100 epochs on one RTX 4090 while protecting source code, datasets, checkpoints, metrics, and credentials against process crashes and server loss.

## Storage Boundaries

- Git `main`: source, configuration, tests, launchers, and documentation only.
- Persistent server disk: VisDrone, virtual environment, all run outputs, logs, `last.pt`, and per-epoch snapshots.
- GitHub Release `btdse-v2.5-s-4090-live`: the newest three validated resumable checkpoints and their SHA256 metadata.
- Git branch `training-results`: lightweight `results.csv`, diagnostics, arguments, and a publication manifest. Runtime files never modify `main`.

## Training And Recovery

The Linux supervisor launches scratch training with AMP and a conservative RTX 4090 batch size. Ultralytics writes `last.pt` and an independent `epochN.pt` after every completed epoch. After an abnormal exit, the supervisor deserializes `last.pt`; if it is incomplete, it selects the newest readable epoch snapshot. It then uses true Ultralytics resume so optimizer, scheduler, scaler, EMA, and epoch state are retained.

## Remote Protection

A watcher detects a newly completed checkpoint epoch, waits until the file is stable, deserializes it, computes SHA256, and uploads it under an epoch-specific Release asset name. Older remote checkpoints are deleted only after the new upload is verified, leaving the newest three. This avoids the no-backup interval caused by replacing one fixed `last.pt` asset.

After a successful checkpoint upload, lightweight run metadata is copied to an isolated results checkout, committed, and pushed to `training-results`. Push or upload failures retry without interrupting training.

## Credentials

The GitHub token is stored in a mode-600 file on the persistent disk and exported only to uploader processes. It is excluded from Git, logs, command lines, archives, and manifests. A fine-grained token needs repository Contents read/write permission only. Any token previously pasted into chat must be revoked and replaced.

## Verification

- Unit tests cover checkpoint validation, fallback ordering, retention, and publication manifests.
- A server preflight checks CUDA, GPU identity, disk space, dataset layout, Git state, and token file permissions.
- The operator can monitor the supervisor, training output, sync status, current metrics, GPU state, and available disk with documented commands.

# ACR-EG Round 1 Progress Handoff

Date: 2026-07-28 (Asia/Shanghai)

## Interpretation frozen for the paper

Fixed-SADED is an unpublished internal development diagnostic, not an external
baseline. The formal comparison is the complete ACR-EG method against the
original global RT-DETR-L baseline. Fixed-relative deltas remain useful for
diagnosing evidence retention, but they are not the sole formal success gate.

## Valid evidence already available

### Old SR-PEG (completed, not ACR-EG)

Source output:

`/home/ubuntu/gcte-srpeg-seed0-output-3b1f43a4/seed0-evaluation.json`

SHA256:

`7a4bab626aedf0b8e3b5c5aa3f09fc4bbcf4a6a094078413c218df11cbbb462b`

Relative to Global RT-DETR-L:

- mAP50-95: `+0.01059477545675891`
- AP-tiny: `+0.0134116943679079`
- tiny recall: `+0.054338202744577235`
- AP-medium: `-0.0016400479689508973`
- AP-large: `-0.00009592819312331802`

The medium/large values satisfy the protection budgets, but they are not
described as guaranteed improvements.

The old Full-versus-Fixed-SADED mAP delta was `-0.016578506475579652`. This is
retained as an internal evidence-retention diagnostic only.

### ACR-EG training state

The ACR-EG seed0 module-only training completed 10/10 epochs:

- source commit: `c966512f537d5671a6fd06584cd2774406e96320`
- source directory: `/home/ubuntu/gcte-acr-eg-c966512f`
- output directory: `/home/ubuntu/gcte-acr-eg-output-c966512f/train-seed0`
- best module: `/home/ubuntu/gcte-acr-eg-output-c966512f/train-seed0/best-module.pt`
- module SHA256: `427a7062a95f6ea44bf9f4fe67c88d1fd7dd0e64e2d7bcd2397016e0782a8a86`
- final train loss: `2.13051138`
- final calibration loss: `2.29111369`

No ACR-EG validation metric is claimed yet.

## Current round

The latest calibration/evaluation source is commit
`ada41f512945727d4ae60ba85ddada2eb495c5cb`, deployed independently at:

`/home/ubuntu/gcte-acr-eg-cal-ada41f51`

The source archive uploaded to the server has SHA256:

`db74239a2519676e6357b3684368308f83fe86fa67f3e4830cfbc0d928f363d0`

Remote focused regression before calibration: `11 passed`.

The first calibration launch failed before any data processing because the
entry point requires the cache manifest file, not the cache directory. The
error was:

`FileNotFoundError: /home/ubuntu/gcte-srpeg-seed0-output-e1a8b039/train-cache`

The contract was reproduced: the directory is rejected, while
`.../train-cache/manifest.json` loads 647 records with schema
`gcte-gcqf-evidence/v2`. The corrected three-route calibration completed:

- PID: `8561` (exited normally)
- log: `/home/ubuntu/gcte-acr-eg-output-c966512f/calibration.log`
- output: `/home/ubuntu/gcte-acr-eg-output-c966512f/calibration.json`
- cache manifest: `/home/ubuntu/gcte-srpeg-seed0-output-e1a8b039/train-cache/manifest.json`
- calibration SHA256:
  `212a9c969808d7f3cdc76dccb3c3b2158b0c0d709bc8b2b5509d061d0fdefd35`
- selected thresholds: `tiny_utility=0.5`,
  `non_tiny_risk=0.5`, `global_retain=0.6`

The corrected run reuses the sealed 647-record cache and the completed module;
it does not change the model or formal thresholds.

On the 129-image calibration holdout, the selected route relative to Global
gave:

- mAP50-95: `+0.009687565603029163`
- AP-tiny: `+0.01233865643379875`
- tiny recall: `+0.0751782242384964`
- AP-medium: `-0.0005373218279140501`
- AP-large: `+0.00043539309350792976`

These holdout numbers are diagnostic only; the formal decision remains pending
on the 548-image validation evaluation.

The 548-image five-state evaluation is now running:

- PID: `9611`
- log: `/home/ubuntu/gcte-acr-eg-output-c966512f/evaluation.log`
- output: `/home/ubuntu/gcte-acr-eg-output-c966512f/seed0-evaluation.json`

The evaluation completed successfully under the Global-relative formal policy:

- evaluation SHA256:
  `7606275a6196cc3f5af602dae36dea6ecdbd9d4d4f9c1cc68a26da4ea0171bd7`
- Full-GCQF mAP50-95 vs Global: `+0.010194439965272362`
- Full-GCQF AP-tiny-SBR vs Global: `+0.01321126596778198`
- Full-GCQF tiny recall vs Global: `+0.058211598052235414`
- Full-GCQF AP-medium-SBR vs Global: `-0.0015244507940789798`
- Full-GCQF AP-large-SBR vs Global: `-0.00008967181386335121`
- all formal gates: `passed=true`

The internal Full-GCQF minus Fixed-SADED mAP delta is
`-0.0169788419670662`. It is reported for evidence retention only and does not
block the formal comparison, because Fixed-SADED is unpublished internal
development evidence rather than an external baseline.

The complete metric, gate, coverage, and checksum record is tracked at
`docs/evidence/gcte-acr-eg-round1-evaluation.json`.

## Formal 100-epoch launch

The formal detector stage is now running from the verified new source:

- source commit: `098da04c7ef6f460ecf8298ab563ed70392bf97c`
- source archive SHA256:
  `8ec90544d0aeb6cad6b285fcf285aa817c5f10fe619623a99cb203e33c515d73`
- source directory: `/home/ubuntu/gcte-acr-eg-formal-098da04c`
- output directory:
  `/home/ubuntu/gcte-acr-eg-formal-output-098da04c/full-rtdetr-100`
- runner PID: `14324`
- training PID: `14325`
- log: `/home/ubuntu/gcte-acr-eg-formal-output-098da04c/formal.log`

The process has entered epoch 1 and reached batch `412/809` while using about
13.6 GiB on the RTX 4090. The log confirms 6,471 train images and 548
validation images, MuSGD, batch 8, workers 8, 640-pixel inputs, and
`pretrained=False`. The protocol manifest was corrected to the exact full
source commit and has SHA256
`2c73f2be0acaf3015d45a19c3a9e8192bc0b6230679edb62c17decd45e585e31`.

The structured launch record is tracked at
`docs/evidence/gcte-acr-eg-formal-launch.json`.

Epoch 1 completed in `490.196s` with train losses
`giou=2.36849`, `cls=0.45560`, `l1=0.94033`; the detector has continued into
the next epoch. The epoch-1 checkpoint SHA256 is
`9687babfb189518e5b4eceeb1ad0594024fa896addc276576259f6cb04032f61` and is
available as the
[epoch-1 GitHub Release asset](https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-formal-098da04c-epoch-001).
The structured epoch record is tracked at
`docs/evidence/gcte-acr-eg-formal-epoch-001.json`.

Epoch 2 also completed in `840.743s` with train losses
`giou=2.30258`, `cls=0.15589`, `l1=0.80998`; the run remains active. Its
checkpoint SHA256 is
`938c6b4f7239e3806a0fc8cb9a6b89fe8faeb016b9a0a42da00990e6630d11d6` and is
available as the
[epoch-2 GitHub Release asset](https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-formal-098da04c-epoch-002).
The structured epoch-2 record is tracked at
`docs/evidence/gcte-acr-eg-formal-epoch-002.json`.

The GitHub branch now includes a resume-capable formal entry. After downloading
the latest Release checkpoint, resume with the same protocol and a new output
name, for example:

```bash
python scripts/train_gcte_formal.py \
  --resume /absolute/path/to/epoch-002-last.pt \
  --data /mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml \
  --project /home/ubuntu/gcte-acr-eg-formal-output-resumed \
  --name full-rtdetr-100-resumed \
  --module /absolute/path/to/best-module.pt \
  --module-sha256 427a7062a95f6ea44bf9f4fe67c88d1fd7dd0e64e2d7bcd2397016e0782a8a86
```

## Reproducibility anchors

- train10 cache manifest SHA256:
  `bb629f7ae85dcad1b432b8f529cd423ed6c6e685f7c5f11e8af0ff1da60f0286`
- train10 dataset signature:
  `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`
- val cache manifest SHA256:
  `9e7f3a6361b8734cfab6667972e824c6f4c6ae2cedc0741006f8d2d0037bef86`
- val dataset signature:
  `A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A`
- baseline checkpoint SHA256:
  `54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b`

## Disk policy

Obsolete GCMV outputs were removed after checking their real paths and active
processes. Current GCTE caches, baseline checkpoint, and evidence outputs are
retained. The server root filesystem had approximately 4.6 GB free at formal
launch. After confirming that no active process referenced them, four more
authorized obsolete experiment directories were removed:

- `/home/ubuntu/tascv-runs`
- `/home/ubuntu/tsgr-p2-e1-final-9d9d7994`
- `/home/ubuntu/tsgr-p2-e1-8cb70c50`
- `/home/ubuntu/ascv-runs`

Current free space is approximately 16.0 GB. No current GCTE evidence, cache,
module, baseline checkpoint, or formal output was deleted.

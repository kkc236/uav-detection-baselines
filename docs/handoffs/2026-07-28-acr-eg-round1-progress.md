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
`gcte-gcqf-evidence/v2`. The corrected three-route calibration is now running:

- PID: `8561`
- log: `/home/ubuntu/gcte-acr-eg-output-c966512f/calibration.log`
- output: `/home/ubuntu/gcte-acr-eg-output-c966512f/calibration.json`
- cache manifest: `/home/ubuntu/gcte-srpeg-seed0-output-e1a8b039/train-cache/manifest.json`

The corrected run reuses the sealed 647-record cache and the completed module;
it does not change the model or formal thresholds.

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
retained. The server root filesystem is currently approximately 4.7 GB free;
no current GCTE evidence was deleted.

# BTD-SE V2.5-S Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, verify, and locally train BTD-SE V2.5-S inside the Ultralytics 8.4.90 RT-DETR-L P3 fusion path.

**Architecture:** A repository-owned module receives the existing `[G, P]` concatenation, predicts background reliability, computes normalized 9x9-minus-5x5 context, confirms the residual with a 32-channel semantic embedding, and returns `[G, P_out]`. A custom trainer/model add two training-only losses and preserve VisDrone ignored boxes through Ultralytics augmentations without modifying site-packages.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, VisDrone, RTX 4070 Laptop 8GB.

---

### Task 1: Core ring operator and module

**Files:**
- Create: `src/btd_se.py`
- Create: `tests/test_btd_se.py`

- [ ] Write tests proving ring pooling equals an explicit 9x9-minus-5x5 sum, border outputs are finite, output shape is unchanged, `gamma=0` makes `P_out=P`, and backward produces finite gradients.
- [ ] Run `C:\uav_env\Scripts\python.exe -m pytest tests/test_btd_se.py -q` and verify failure because `src.btd_se` does not exist.
- [ ] Implement `ring_sum` and `BTDSE` with cached `W_b`, `S`, `R`, and `Z` tensors for auxiliary loss computation.
- [ ] Re-run the test and require all cases to pass.

### Task 2: Auxiliary target maps and losses

**Files:**
- Create: `src/btd_se_targets.py`
- Create: `src/btd_se_loss.py`
- Create: `tests/test_btd_se_supervision.py`

- [ ] Write tests for object/ignore background masks, anisotropic Gaussian maxima, minimum sigma, ignore exclusion, focal loss finiteness, and gradients.
- [ ] Run `C:\uav_env\Scripts\python.exe -m pytest tests/test_btd_se_supervision.py -q` and verify the missing-module failure.
- [ ] Implement target rasterization from normalized augmented boxes and mean-reduced focal BCE losses.
- [ ] Re-run the supervision tests and require all cases to pass.

### Task 3: Preserve VisDrone ignored boxes

**Files:**
- Modify: `src/visdrone.py`
- Create: `src/btd_se_dataset.py`
- Modify: `tests/test_visdrone_conversion.py`
- Create: `tests/test_btd_se_dataset.py`

- [ ] Add failing tests for ignored-row conversion and sidecar loading as class `-1`.
- [ ] Generate `labels_ignore/<split>/<image>.txt` during VisDrone conversion.
- [ ] Implement a custom RT-DETR dataset that appends ignored boxes only in training mode, allowing the standard `Instances` transforms to move them with object boxes.
- [ ] Verify object classes remain unchanged and ignored boxes survive a dataset sample transform.

### Task 4: RT-DETR graph and loss integration

**Files:**
- Create: `configs/rtdetr-l-btdse.yaml`
- Create: `src/rtdetr_btdse.py`
- Create: `tests/test_rtdetr_btdse_integration.py`

- [ ] Write a failing test that constructs the custom YAML, confirms BTD-SE is immediately before the P3 RepC3 block, and runs a synthetic training loss with object and ignore boxes.
- [ ] Register the repository-owned module in `ultralytics.nn.tasks` at runtime without editing `site-packages`.
- [ ] Implement a custom RT-DETR model that filters class `-1` from detection targets and adds `lambda_b L_b + lambda_sal L_sal`.
- [ ] Implement a custom trainer using the custom dataset and return five logged loss items.
- [ ] Run the integration test and require finite forward/backward results.

### Task 5: Training entry point and monitoring

**Files:**
- Create: `scripts/train_rtdetr_btdse.py`
- Create: `scripts/run_btdse_local.ps1`
- Modify: `README.md`
- Modify: `requirements.txt`

- [ ] Add CLI options for epochs, image size, batch, workers, device, loss weights, fraction, name, and smoke mode.
- [ ] Pin `ultralytics==8.4.90` and document `C:\uav_env\Scripts\python.exe` as the local runtime.
- [ ] Add a PowerShell launcher that logs stdout/stderr, records `nvidia-smi`, and resumes only from a compatible BTD-SE checkpoint.
- [ ] Verify `--help` and a model-only dry run.

### Task 6: Local smoke and full training

**Files:**
- Runtime outputs only under ignored `logs/` and `runs/`.

- [ ] Regenerate VisDrone ignore sidecars.
- [ ] Run all tests: `C:\uav_env\Scripts\python.exe -m pytest -q`.
- [ ] Run a one-batch CUDA forward/backward at image size 640 with AMP and batch 1.
- [ ] Run one epoch with `batch=1`, `workers=2`, `cache=False`, and record peak memory plus `W_b`, `S`, `gamma`, and `Z` diagnostics.
- [ ] If stable, launch 100 epochs from YAML with `pretrained=False`; use gradient accumulation rather than lowering image size when effective-batch adjustment is required.
- [ ] Compare mAP50-95, mAP50, precision, recall, AP-small, parameters, peak memory, and latency against the scratch RT-DETR-L baseline.

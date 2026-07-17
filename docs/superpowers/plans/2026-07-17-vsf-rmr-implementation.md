# VSF-RMR Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen VSF-RMR module as a standalone RT-DETR-L experiment, with FP32 auxiliary supervision, exact identity initialization, and explicit isolation from BTD-SE and IOQC-SA.

**Architecture:** A pure PyTorch `VSFRMR` module consumes the three Hybrid Encoder features, predicts an ordered scale field, and adds zero-initialized residual corrections. A custom RT-DETR model injects the module immediately before the decoder head, while a custom trainer constructs that model without editing Ultralytics or YAML files.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics RT-DETR, pytest.

---

### Task 1: Ordered routing module

**Files:**
- Create: `src/vsf_rmr.py`
- Create: `tests/test_vsf_rmr.py`

- [ ] Add failing tests for three 256-channel inputs, shape validation, scale-field range, ordered adjacent weights, and output shapes.
- [ ] Run `C:\uav_env\Scripts\python.exe -m pytest tests/test_vsf_rmr.py -q` and confirm import failure.
- [ ] Implement independent GroupNorm, shared 256-to-32 projection, global and local field heads, and ordered routing weights.
- [ ] Add failing tests proving zero-initialized `gamma` makes every output exactly equal to its input and that the initial global prior is approximately 0.95.
- [ ] Implement residual restoration and per-level channel scales.
- [ ] Add failing tests for train-only cache creation, cache clearing at the next forward, and eval-mode cache suppression.
- [ ] Implement `pop_auxiliary_state()` and finite checks without retaining full-resolution tensors after loss consumption.
- [ ] Run the focused test file and commit the green state.

### Task 2: FP32 scale targets and auxiliary loss

**Files:**
- Create: `src/vsf_rmr_loss.py`
- Create: `tests/test_vsf_rmr_loss.py`

- [ ] Add failing tests for pixel-scale targets `clip(log2(sqrt(w*h)/8), 0.05, 1.95)` using actual batch height and width.
- [ ] Add failing tests for `align_corners=False` center sampling, SmoothL1 beta 1, and per-image balancing.
- [ ] Add failing tests proving FP16 inputs produce FP32 local/global losses and an empty-GT batch returns a graph-connected FP32 zero.
- [ ] Implement target extraction from RT-DETR batch dictionaries, FP32 sampling, local loss, global loss, finite validation, and diagnostic scalars.
- [ ] Run `C:\uav_env\Scripts\python.exe -m pytest tests/test_vsf_rmr_loss.py -q`.

### Task 3: RT-DETR model and trainer integration

**Files:**
- Create: `src/rtdetr_vsf_rmr.py`
- Create: `tests/test_rtdetr_vsf_rmr_integration.py`

- [ ] Add failing tests that the custom model preserves stock `predict` semantics, routes exactly the three `head.f` features, and returns normal decoder output.
- [ ] Add failing tests that training consumes the cached scale field once, adds `0.1 * (L_local + L_global)`, and validation skips auxiliary loss and cache retention.
- [ ] Add failing tests that `VSFRMRRTDETRTrainer.get_model()` constructs the custom model and loads optional weights.
- [ ] Implement a stock-compatible `predict` loop with a single routing hook immediately before the final decoder head.
- [ ] Implement custom loss and trainer construction without changing Ultralytics package files or official YAML.
- [ ] Run the focused integration tests.

### Task 4: Matched baseline and VSF-RMR training entry point

**Files:**
- Create: `scripts/train_rtdetr_vsf_rmr.py`
- Create: `tests/test_vsf_rmr_training_cli.py`

- [ ] Add failing parser/config tests for `--variant baseline|vsf-rmr`, scratch training, 100 epochs, 640 pixels, seed 0, `mosaic=0`, `mixup=0`, `scale=0.5`, and `perspective=0`.
- [ ] Add failing tests for separate run names and adaptive-state callback output.
- [ ] Implement the CLI so baseline uses stock RT-DETR and VSF-RMR uses the custom trainer with otherwise identical settings.
- [ ] Add epoch diagnostics for VSF losses, field statistics, routing weights, gamma norms, AMP state, batch size, and peak CUDA memory.
- [ ] Run focused CLI tests.

### Task 5: Innovation isolation and regression coverage

**Files:**
- Create: `tests/test_innovation_isolation.py`
- Modify only if required: `src/rtdetr_btdse.py`
- Modify only if required: `src/rtdetr_ioqc_sa.py`

- [ ] Add failing/green assertions that VSF-RMR imports neither BTD-SE nor IOQC-SA and that its model contains only the VSF-RMR innovation.
- [ ] Assert BTD-SE and IOQC-SA public entry points remain importable and do not construct VSF-RMR.
- [ ] Run the three innovation-focused suites together.

### Task 6: Stress and latency utilities

**Files:**
- Create: `src/vsf_rmr_stress.py`
- Create: `scripts/benchmark_vsf_rmr.py`
- Create: `tests/test_vsf_rmr_stress.py`
- Create: `tests/test_vsf_rmr_benchmark.py`

- [ ] Add failing tests for deterministic scale 0.75/1.25 and vertical homography +/-5e-4 box transforms, clipping, retention rules, and manifest hashes.
- [ ] Implement deterministic stress-set transformation helpers.
- [ ] Add failing tests for 50 warmups, 200 synchronized measurements, and mean/P50/P95 reporting at batch 1 and training batch.
- [ ] Implement the benchmark CLI and complexity comparison against the matched baseline.
- [ ] Verify the utilities with focused tests.

### Task 7: End-to-end verification

**Files:**
- Update: `README.md`

- [ ] Run all VSF-RMR unit and integration tests.
- [ ] Run existing BTD-SE, IOQC-SA, checkpoint recovery, and adaptive batch tests.
- [ ] Run one real 640-pixel RT-DETR-L forward/backward smoke test with AMP when CUDA is available.
- [ ] Confirm finite total loss, FP32 VSF losses, nonzero gamma gradients, and no retained cache.
- [ ] Document local baseline and VSF-RMR commands and the acceptance thresholds.


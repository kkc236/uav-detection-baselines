# ACR-EG Live Checkpoint Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a checksummed 548-image live comparison between the final integrated ACR-EG checkpoint and the sealed mature RT-DETR-L baseline.

**Architecture:** Normalize the two Ultralytics RT-DETR decoder output contracts inside the existing model adapter, without changing the training branch. A focused evaluator then drives the exact global-plus-four-local model path and feeds original-image predictions into the existing frozen SBR evaluator.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, TorchVision 0.20.1+cu121, Ultralytics 8.4.90, pytest, NumPy, PyYAML.

---

### Task 1: Support ACR-EG evaluation-mode decoder output

**Files:**
- Modify: `src/rtdetr_acr_eg.py`
- Test: `tests/test_rtdetr_acr_eg_integration.py`

- [ ] **Step 1: Write the failing output-contract test**

Add a test that passes `(postprocessed, raw_five_tuple)` to a wished-for `_require_raw_rtdetr_output(value, training=False)` helper and asserts that the raw tuple is returned.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_rtdetr_acr_eg_integration.py -q`

Expected: failure because `_require_raw_rtdetr_output` does not exist.

- [ ] **Step 3: Implement the minimal contract normalizer**

Add `_require_raw_rtdetr_output` that accepts the five-tensor tuple directly or the inference `(prediction_tensor, raw_five_tuple)` wrapper and rejects every other shape.

- [ ] **Step 4: Write and run the failing evaluation-decoding test**

The test constructs synthetic decoder boxes and scores, calls a wished-for `_decode_acr_eg_inference(raw, fused_scores, head)`, and asserts that the head receives the fused scores while the boxes remain unchanged.

- [ ] **Step 5: Implement evaluation decoding without changing training output**

In `ACREGDetectionModel.predict`, keep the current five-tensor return when `self.training` is true. Otherwise call `self.model[-1].postprocess(raw[0].squeeze(0), fused_scores.squeeze(0).sigmoid())` and return `(decoded, fused_raw)` exactly like stock RT-DETR inference.

- [ ] **Step 6: Verify focused and integration tests GREEN**

Run: `pytest tests/test_rtdetr_acr_eg_integration.py tests/test_acr_eg_resume_smoke.py -q`

Expected: all tests pass.

### Task 2: Build the live evaluator core

**Files:**
- Create: `src/acr_eg_live_evaluation.py`
- Test: `tests/test_acr_eg_live_evaluation.py`

- [ ] **Step 1: Write failing tests for prediction conversion and result schema**

Tests must assert normalized `xywh` becomes clipped source-pixel `xyxy`, that every record contains `pred_source` and `pred_query`, and that method-minus-baseline deltas are calculated for every numeric metric shared by both arms.

- [ ] **Step 2: Run the new test file and verify RED**

Run: `pytest tests/test_acr_eg_live_evaluation.py -q`

Expected: import failure because the module is absent.

- [ ] **Step 3: Implement pure helpers**

Implement SHA256 calculation, checkpoint identity checks, normalized prediction conversion, SBR image-row construction, numeric delta calculation and canonical JSON assembly. Keep CUDA/model loading outside these pure helpers.

- [ ] **Step 4: Verify helper tests GREEN**

Run: `pytest tests/test_acr_eg_live_evaluation.py -q`

Expected: all pure-helper tests pass.

### Task 3: Add the fail-closed CUDA evaluation CLI

**Files:**
- Create: `scripts/evaluate_acr_eg_live.py`
- Modify: `src/acr_eg_live_evaluation.py`
- Test: `tests/test_evaluate_acr_eg_live.py`

- [ ] **Step 1: Write failing CLI protocol tests**

Assert defaults freeze `device=0`, `batch=1`, `workers=0`, `imgsz=640`, `conf=0.001`, `max_det=300`, expected record count 548, expected dataset signature and expected baseline SHA256.

- [ ] **Step 2: Run the CLI tests and verify RED**

Run: `pytest tests/test_evaluate_acr_eg_live.py -q`

Expected: import failure because the CLI is absent.

- [ ] **Step 3: Implement model and dataset execution**

Load the integrated EMA and stock baseline on CUDA, construct the exact no-augmentation `GCMVRTDETRDataset`, execute baseline global inference and ACR-EG paired inference under autocast, assert `last_acr_eg_output` exists for every method image, measure latency/VRAM, evaluate both rows with `evaluate_dataset`, and atomically write JSONL, JSON and checksums.

- [ ] **Step 4: Verify CLI tests GREEN**

Run: `pytest tests/test_evaluate_acr_eg_live.py tests/test_acr_eg_live_evaluation.py -q`

Expected: all tests pass.

### Task 4: Execute real evidence run

**Files:**
- Create: `artifacts/acr-eg-live-final/*`
- Modify: `CODEX-START-HERE-GCTE-ACR-EG.md`

- [ ] **Step 1: Verify final identities**

Verify the final Release asset SHA256, baseline SHA256, 548 images, dataset signature, Python/package versions and CUDA device.

- [ ] **Step 2: Run a real one-image smoke**

Run the evaluator with `--limit 1 --smoke` and require both a baseline prediction row and a method row with non-null ACR-EG gate statistics.

- [ ] **Step 3: Run all 548 images**

Run the CLI without `--limit`, retaining the console log and structured outputs.

- [ ] **Step 4: Independently verify artifacts**

Recompute every SHA256, reload the two prediction JSONL files, assert 548 unique ordered image identities, recompute SBR metrics from the sealed rows and require exact agreement with `evaluation.json`.

- [ ] **Step 5: Update the handoff and push evidence**

Add a dedicated final-live-results section distinguishing it from cache diagnostics, then run `git diff --check`, the focused suite and broad regression before committing and pushing the code and lightweight evidence.

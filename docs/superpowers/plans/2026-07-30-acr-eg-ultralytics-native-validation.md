# ACR-EG Ultralytics Native Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce paired Ultralytics 8.4.90 Precision, Recall, AP50, AP75 and mAP50-95 for the mature baseline and final epoch-100 ACR-EG checkpoint without retraining.

**Architecture:** Reuse the sealed checkpoint and five-view dataset loaders, but replace SBR aggregation with Ultralytics `RTDETRValidator` postprocessing and `DetMetrics`. Keep the adapter isolated from training code and fail closed if the ACR-EG arm does not execute its learned five-view path.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, canonical JSON and SHA256.

---

### Task 1: Native metric result helpers

**Files:**
- Create: `src/acr_eg_ultralytics_native.py`
- Test: `tests/test_acr_eg_ultralytics_native.py`

- [ ] **Step 1: Write failing helper tests**

Add tests that construct a small metric-compatible object with `mp`, `mr`,
`map50`, `map75`, and `map`, then require exact extraction and exact paired
deltas. Add a protocol test that rejects a non-548 record count or changed
checkpoint digest.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv-eval\Scripts\python.exe -m pytest tests\test_acr_eg_ultralytics_native.py -q
```

Expected: collection failure because `src.acr_eg_ultralytics_native` does not
exist.

- [ ] **Step 3: Implement the minimal helpers**

Implement:

```python
def extract_requested_metrics(det_metrics) -> dict[str, float]:
    box = det_metrics.box
    return {
        "Precision": float(box.mp),
        "Recall": float(box.mr),
        "AP50": float(box.map50),
        "AP75": float(box.map75),
        "mAP50-95": float(box.map),
    }
```

Add protocol validation, numeric delta construction and canonical result
construction.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same focused pytest command and require zero failures.

### Task 2: Official RT-DETR validator loop

**Files:**
- Modify: `src/acr_eg_ultralytics_native.py`
- Modify: `tests/test_acr_eg_ultralytics_native.py`
- Create: `scripts/evaluate_acr_eg_ultralytics_native.py`

- [ ] **Step 1: Write a failing silent-fallback test**

Use a fake method model that returns valid RT-DETR predictions but leaves
`last_acr_eg_output=None`. Require the method-arm guard to raise
`ACR_EG_NATIVE_SILENT_STOCK_FALLBACK`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\.venv-eval\Scripts\python.exe -m pytest tests\test_acr_eg_ultralytics_native.py -q
```

Expected: failure because the method-arm guard is not implemented.

- [ ] **Step 3: Implement the minimal native loop**

Instantiate Ultralytics `RTDETRValidator` with the frozen arguments, initialize
its metrics with the checkpoint model, preprocess each batch through the
validator, perform one-view or five-view inference, call the validator's
`postprocess()` and `update_metrics()`, then call `get_stats()`. The CLI must
hard-code the expected checkpoint and dataset digests while accepting paths
and output directory as arguments.

- [ ] **Step 4: Run tests and verify GREEN**

Require the focused suite to pass with zero failures.

### Task 3: CUDA smoke and full paired evaluation

**Files:**
- Generate: `artifacts/acr-eg-ultralytics-native-smoke/evaluation.json`
- Generate: `artifacts/acr-eg-ultralytics-native-final/evaluation.json`
- Generate: `artifacts/acr-eg-ultralytics-native-final/checksums.sha256`

- [ ] **Step 1: Run one-image CUDA smoke**

Use the exact baseline and method checkpoints, dataset YAML and device `0`
with `--smoke --limit 1`. Require the baseline model identity,
`ACREGDetectionModel` identity, five-view execution and finite metrics.

- [ ] **Step 2: Run the complete 548-image evaluation**

Run the CLI without `--smoke` or `--limit`. Require exactly 548 images in both
arms and one canonical result.

- [ ] **Step 3: Verify output identities**

Recompute checkpoint SHA256, dataset signature and output SHA256. Parse the
JSON and require finite values for all five requested metrics in both arms and
all deltas.

### Task 4: Documentation and repository verification

**Files:**
- Modify: `CODEX-START-HERE-GCTE-ACR-EG.md`
- Create: `docs/evidence/gcte-acr-eg-ultralytics-native-final.json`

- [ ] **Step 1: Record only observed results**

Add the paired native metric table, exact deltas, checkpoint hashes, dataset
signature and protocol. Keep SBR and native metrics in separately labelled
sections.

- [ ] **Step 2: Run focused and broad regression tests**

Run:

```powershell
.\.venv-eval\Scripts\python.exe -m pytest tests\test_acr_eg_ultralytics_native.py tests\test_evaluate_acr_eg_live.py tests\test_acr_eg_live_evaluation.py -q
.\.venv-eval\Scripts\python.exe -m pytest -q
git diff --check
```

Require zero test failures and no whitespace errors.

- [ ] **Step 3: Commit and push**

Commit source, tests, evidence and documentation to
`codex/gcte-rtdetr-g0`, then push only after all fresh verification commands
pass.


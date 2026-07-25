# SADED Fresh-100 Stock Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start and supervise a fresh, fail-closed seed-0 RT-DETR-L 100-epoch stock baseline, then feed its sealed endpoint into the frozen SADED-SM evaluation path.

**Architecture:** A small stock-only protocol layer validates immutable provenance and calls the already tested `TASCVControlTrainer` at `FORMAL_100`. A background driver records PID, log, exit status, and final runtime canaries; evaluation begins only after exit 0.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, JSON/YAML manifests, NVIDIA RTX 4090.

---

### Task 1: Freeze the stock-only CLI contract

**Files:**
- Create: `tests/test_saded_stock_cli.py`
- Create: `src/saded_stock_cli.py`

- [ ] **Step 1: Write failing tests**

Test that the CLI exposes no hyperparameter switches, emits the exact frozen
100-epoch settings, accepts only seed 0/device 0, rejects test-dev and existing
targets, and verifies source/data/initial-state hashes.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_saded_stock_cli.py -q`

Expected: collection fails because `src.saded_stock_cli` does not exist.

- [ ] **Step 3: Implement the minimal validator**

Implement `build_parser`, `build_settings`, `source_closure`, and
`validate_protocol_inputs` with no scientific tuning options.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_saded_stock_cli.py -q`

Expected: all tests pass.

### Task 2: Add the runtime entry point

**Files:**
- Create: `tests/test_saded_stock_training.py`
- Create: `scripts/train_rtdetr_saded_stock.py`

- [ ] **Step 1: Write failing runtime-summary tests**

Test exact endpoint binding and the final checks for checkpoint existence,
80,900 successful batches, 10,556 optimizer attempts, AMP scale 128, MuSGD,
batch 8, workers 8, no test loader, and no internal evaluation.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_saded_stock_training.py -q`

Expected: collection fails because the training entry point does not exist.

- [ ] **Step 3: Implement the training entry**

Instantiate `TASCVControlTrainer(stage=FORMAL_100)`, record each successful
batch, train, validate all canaries, and atomically write
`saded_stock_training_summary.json`.

- [ ] **Step 4: Verify GREEN and regression**

Run: `python -m pytest tests/test_saded_stock_cli.py tests/test_saded_stock_training.py tests/test_rtdetr_tascv_integration.py -q`

Expected: all selected tests pass.

### Task 3: Seal, deploy, and launch

**Files:**
- Create at runtime: `/home/ubuntu/saded-fresh100-protocols/<run-id>/protocol_manifest.json`
- Create at runtime: `/home/ubuntu/saded-fresh100-runs/<run-id>/seed0`
- Create at runtime: `/home/ubuntu/saded-fresh100-logs/<run-id>/`

- [ ] **Step 1: Run the full local test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Commit and push the exact source**

Commit the specification, tests, validator, and training entry, then push the
`final-saded-tascv-preflight` branch.

- [ ] **Step 3: Deploy an isolated clean source tree**

Clone or extract a git bundle into a new `/home/ubuntu/repo-*` directory,
verify the exact commit, and run the full server test suite.

- [ ] **Step 4: Generate and validate the manifest**

Bind the live environment, source hashes, sealed seed-0 initial state, full
train-only YAML, and a fresh `/home/ubuntu` output endpoint.

- [ ] **Step 5: Launch and validate the first batches**

Start the single-GPU job in the background, record driver/trainer PIDs, and
confirm active GPU memory, RSS, log progress, and adequate disk space.

### Task 4: Complete training and sealed evaluation

**Files:**
- Create at runtime: final checkpoint, runtime summary, raw cache, routed
  predictions, sealed metrics, adjudication, and checksum anchors.

- [ ] **Step 1: Monitor without partial scientific metrics**

Report epoch/batch progress, GPU/RSS/disk, and exceptions. Do not read
test-dev or partial validation results.

- [ ] **Step 2: Verify the completed endpoint**

Require exit 0, a valid last checkpoint, the exact batch/optimizer counts,
fixed AMP scale, train-only proof, and checksums.

- [ ] **Step 3: Execute the frozen post-training path**

Run one raw cache, frozen GT-free SADED-SM routing, sealed dev-val evaluation,
five-gate adjudication, and checksum closure.

- [ ] **Step 4: Report the paper-facing result**

Report the single seed-0 metrics, five deltas and pass/fail decisions, artifact
paths, checksums, and limitations. Keep test-dev unopened.

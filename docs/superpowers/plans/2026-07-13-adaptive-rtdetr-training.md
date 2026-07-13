# Adaptive RT-DETR Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a self-healing RT-DETR supervisor that starts at batch 16, moves between batch levels from 10 through 20 in steps of two, preserves true resume state, uploads verified artifacts, and shuts down only after success.

**Architecture:** Pure policy logic in `src/adaptive_batch.py` decides the next batch from OOM and epoch telemetry. `scripts/train_rtdetr_adaptive.py` runs Ultralytics as a child, records epoch-level CUDA peaks through callbacks, exits at safe checkpoint boundaries for batch changes, and supervises retries. Packaging and GitHub publication remain separate completion steps so training cannot be marked complete by an upload failure.

**Tech Stack:** Python 3.12, PyTorch 2.11, Ultralytics 8.4.90, pytest, subprocess, JSON state files, GitHub API.

---

### Task 1: Adaptive batch policy

**Files:**
- Create: `src/adaptive_batch.py`
- Create: `tests/test_adaptive_batch.py`

- [ ] **Step 1: Write failing policy tests**

Test that the default starts at 16, every OOM reduces batch by two down to 10, stable epochs promote by two up to 18, cooldown grows 5/10/20, and a 29 GiB batch-18 epoch requests proactive demotion.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_adaptive_batch.py -v`

Expected: collection fails because `src.adaptive_batch` does not exist.

- [ ] **Step 3: Implement the minimal pure state machine**

Use a dataclass with JSON-serializable fields and pure methods `record_oom()`, `record_epoch()`, and `next_batch()`.

- [ ] **Step 4: Verify the policy tests pass**

Run: `pytest tests/test_adaptive_batch.py -v`

Expected: all adaptive policy tests pass.

### Task 2: True-resume supervisor

**Files:**
- Create: `scripts/train_rtdetr_adaptive.py`
- Create: `tests/test_adaptive_supervisor.py`

- [ ] **Step 1: Write failing command and state tests**

Test that every child command uses the current `last.pt`, `resume=True`, the selected batch, and never initializes RT-DETR from a YAML file after the starting checkpoint.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_adaptive_supervisor.py -v`

Expected: import fails because the supervisor does not exist.

- [ ] **Step 3: Implement supervisor and callback wiring**

Add a lock, atomic JSON state writes, a child process loop, OOM classification, planned epoch-boundary restart codes, heartbeat output, and unexpected-failure retry limits.

- [ ] **Step 4: Verify supervisor tests pass**

Run: `pytest tests/test_adaptive_supervisor.py -v`

Expected: all supervisor tests pass without requiring a GPU.

### Task 3: Artifact publication guard

**Files:**
- Create: `src/result_publisher.py`
- Create: `tests/test_result_publisher.py`

- [ ] **Step 1: Write failing completion-gate tests**

Test that publication is rejected below epoch 100, SHA256 values are generated, and shutdown permission remains false until commit push and release assets are both verified.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_result_publisher.py -v`

Expected: import fails because the publisher does not exist.

- [ ] **Step 3: Implement packaging and verification**

Package metrics/config/plots/logs, calculate checksums, push text artifacts, upload large weights as Release assets, and query GitHub to verify names and sizes.

- [ ] **Step 4: Verify publication tests pass**

Run: `pytest tests/test_result_publisher.py -v`

Expected: all completion-gate tests pass with local fake responses.

### Task 4: End-to-end verification and deployment

**Files:**
- Modify: `README.md`
- Create: `scripts/start_adaptive_rtdetr.sh`

- [ ] **Step 1: Run the full local test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run a fake-child dry run**

Run: `python scripts/train_rtdetr_adaptive.py --dry-run --simulate-oom-at 4 --target-epoch 8`

Expected: state transitions from 16 through lower and higher adjacent levels and completes without a stopped supervisor.

- [ ] **Step 3: Deploy files to the RTX 5090 server**

Copy the tested files into `/root/autodl-tmp/uav-detection-baselines`, verify Python package versions, dataset paths, checkpoint readability, disk space, GitHub credential permissions, and the absence of another trainer.

- [ ] **Step 4: Launch detached and verify activity**

Run: `nohup bash scripts/start_adaptive_rtdetr.sh > logs/adaptive_launcher.log 2>&1 &`

Expected: one supervisor and one trainer process, a fresh heartbeat, GPU utilization above zero, and `resume=True` logged from the original epoch-3 scratch checkpoint.

- [ ] **Step 5: Commit deployment code**

Run: `git add docs src scripts tests README.md && git commit -m "Add adaptive RT-DETR training supervisor"`

Expected: one commit containing policy, supervisor, publisher, tests, docs, and launch instructions.

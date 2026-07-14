# BTD-SE RTX 4090 Resilient Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a one-command RTX 4090 training workflow that resumes after crashes and protects each completed checkpoint and metric snapshot on persistent disk and GitHub.

**Architecture:** A Linux supervisor owns the training process and true-resume loop. A separate Python watcher validates completed Ultralytics checkpoints, uploads epoch-specific Release assets with rolling retention, and commits lightweight results through an isolated Git checkout. Server setup places the repository, dataset, runs, credentials, and result checkout on persistent storage.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, requests, Bash, Git, GitHub REST API, pytest.

---

### Task 1: Checkpoint Publication Domain

**Files:**
- Create: `src/github_checkpoint_sync.py`
- Create: `tests/test_github_checkpoint_sync.py`

- [ ] **Step 1: Write failing tests for checkpoint epoch extraction, asset ordering, retention, and manifest creation**

Use temporary `torch.save` checkpoints with optimizer and EMA state. Assert that checkpoint epoch zero is presented as completed epoch one, matching assets are sorted by completed epoch, and retention deletes only assets older than the newest three.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_github_checkpoint_sync.py -q`

Expected: collection fails because `src.github_checkpoint_sync` does not exist.

- [ ] **Step 3: Implement the GitHub Release client and pure helper functions**

Implement `checkpoint_metadata`, `checkpoint_asset_name`, `matching_checkpoint_assets`, `assets_to_delete`, `sha256_file`, `get_or_create_release`, `upload_asset`, and `publish_checkpoint`. Upload with raw `application/octet-stream`, verify response size, and delete old assets only after the new upload succeeds.

- [ ] **Step 4: Run the focused tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_github_checkpoint_sync.py -q`

Expected: all tests pass.

### Task 2: Continuous GitHub Watcher

**Files:**
- Create: `scripts/sync_btdse_checkpoint.py`
- Create: `tests/test_btdse_sync_cli.py`

- [ ] **Step 1: Write failing parser and artifact collection tests**

Assert secure defaults for token file, release tag, retention count, polling interval, status path, and the exact lightweight files copied from a run directory.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_btdse_sync_cli.py -q`

Expected: import fails because the watcher is not implemented.

- [ ] **Step 3: Implement one-shot and continuous modes**

Select the newest valid `last.pt` or `epochN.pt`, skip an already published epoch, upload the checkpoint, write an atomic JSON status file, copy only `results.csv`, `btdse_diagnostics.jsonl`, `args.yaml`, and the publication manifest, then invoke the isolated result commit helper. Catch errors in continuous mode, record them, and retry without terminating training.

- [ ] **Step 4: Run the focused tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_btdse_sync_cli.py -q`

Expected: all tests pass.

### Task 3: Persistent Server Supervisor

**Files:**
- Create: `scripts/setup_btdse_4090.sh`
- Create: `scripts/run_btdse_4090.sh`
- Modify: `scripts/train_rtdetr_btdse.py`
- Modify: `tests/test_btdse_training.py`

- [ ] **Step 1: Write failing tests for configurable project directory and server settings**

Assert that `--project`, `--workers`, and `--resume` are forwarded without placing auxiliary loss weights into Ultralytics overrides.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_btdse_training.py -q`

Expected: parser rejects `--project`.

- [ ] **Step 3: Add configurable output location and server launchers**

The setup script creates a Python 3.10 virtual environment, installs pinned CUDA wheels and requirements, configures the persistent Ultralytics dataset directory, prepares token permissions, and performs CUDA/disk checks. The supervisor starts the watcher, launches batch 8 training, validates a fallback checkpoint after any nonzero exit, resumes after a delay, and never deletes local checkpoints.

- [ ] **Step 4: Validate Bash syntax and focused tests**

Run: `bash -n scripts/setup_btdse_4090.sh scripts/run_btdse_4090.sh`

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_btdse_training.py -q`

Expected: syntax checks and tests pass.

### Task 4: Operator Guide

**Files:**
- Create: `docs/RTX4090_SERVER_GUIDE.md`
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Document clean-server setup and one-command launch**

Include persistent disk selection, repository clone, token creation and mode-600 storage, environment setup, VisDrone preparation, detached launch, live monitoring, recovery, GitHub verification, result download, shutdown, and migration to another server.

- [ ] **Step 2: Document data protection guarantees and limits**

State which data lives in Git, persistent disk, Release assets, and the results branch. Warn that abrupt power loss can lose the current incomplete epoch and that a previously exposed token must be revoked.

- [ ] **Step 3: Verify every command uses repository-relative paths or explicit environment variables**

Run a text review for `/root/autodl-tmp`, `$STORAGE_ROOT`, token file paths, run names, and Release tags. Remove provider-specific assumptions from mandatory steps.

### Task 5: Verification And Publication

**Files:**
- Modify: all files above as required by test findings

- [ ] **Step 1: Run the complete test suite**

Run: `C:\uav_env\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run Python compilation and Bash syntax checks**

Run: `C:\uav_env\Scripts\python.exe -m compileall src scripts`

Run: `bash -n scripts/setup_btdse_4090.sh scripts/run_btdse_4090.sh`

Expected: no syntax errors.

- [ ] **Step 3: Commit source and documentation**

Stage only source, configuration, tests, specs, plans, and documentation. Exclude datasets, runs, logs, downloaded papers, weights, token files, and local caches.

- [ ] **Step 4: Push the current branch and update `main` on GitHub**

Push the tested commit to `origin/main` only after confirming the remote branch has not advanced unexpectedly. Verify the resulting GitHub commit SHA.

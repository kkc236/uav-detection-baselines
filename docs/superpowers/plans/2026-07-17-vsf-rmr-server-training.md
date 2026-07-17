# VSF-RMR Resilient Server Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make matched-baseline and VSF-RMR training run unattended on 4090 or multi-GPU Linux servers with adaptive batch sizing, verified recovery, persistent storage, GitHub checkpoint exchange, and safe completion shutdown.

**Architecture:** A VSF-specific supervisor launches the matched training entry point, reads machine-readable epoch state, restarts planned batch changes, demotes on OOM/nonfinite failures, and resumes only from validated checkpoints. A separate sync process uploads rolling checkpoint assets and final metrics under innovation-specific GitHub tags so BTD-SE, IOQC-SA, baseline, and VSF-RMR artifacts never collide.

**Tech Stack:** Bash, Python, PyTorch distributed training, GitHub CLI/API, existing `gpu_adaptive_batch`, `checkpoint_recovery`, and checkpoint sync utilities.

---

### Task 1: Generalize and test adaptive batch policy

**Files:**
- Modify: `src/gpu_adaptive_batch.py`
- Create: `tests/test_vsf_rmr_adaptive_batch.py`

- [ ] Add tests for 24 GB 4090 and 8x4090 profiles, promotion after three stable epochs, immediate demotion at high memory, OOM demotion, numeric demotion with AMP disable, cooldown, and upper/lower bounds.
- [ ] Replace IOQC-specific user-facing wording with experiment-neutral wording without changing existing policy behavior.
- [ ] Run adaptive policy and IOQC regression tests.

### Task 2: VSF-RMR training supervisor

**Files:**
- Create: `scripts/supervise_vsf_rmr.py`
- Create: `tests/test_vsf_rmr_supervisor.py`

- [ ] Add failing tests for exact child commands for `baseline` and `vsf-rmr`, single/multi-GPU selection, resume arguments, and persistent storage paths.
- [ ] Add failing tests for exit 75 planned restart, CUDA OOM, nonfinite loss, ordinary failure, corrupt last checkpoint fallback, and completed 100-epoch detection.
- [ ] Implement the supervisor using the shared GPU policy and checkpoint validator.
- [ ] Ensure OOM lowers batch, nonfinite lowers batch and disables AMP, stable epochs can raise batch again, and every restart resumes from a validated checkpoint.
- [ ] Write atomic JSON status and append-only supervisor logs.

### Task 3: Setup and unattended run scripts

**Files:**
- Create: `scripts/setup_vsf_rmr_server.sh`
- Create: `scripts/run_vsf_rmr_server.sh`
- Create: `tests/test_vsf_rmr_server_scripts.py`

- [ ] Add static tests for persistent `/root/blockdata` or user-selected storage, branch selection, dependency installation, dataset preparation, token permission 600, and no embedded credentials.
- [ ] Implement CUDA/PyTorch selection including RTX 5090 support while retaining 4090 and A10 compatibility.
- [ ] Implement `VARIANT=baseline|vsf-rmr`, distinct project/run/tag/prefix values, sync watcher, supervisor, final publication verification, and optional shutdown only after verified upload.
- [ ] Run shell syntax checks and focused tests.

### Task 4: Checkpoint publication and cross-server restore

**Files:**
- Create: `scripts/restore_vsf_rmr_checkpoint.py`
- Create: `scripts/publish_vsf_rmr_results.py`
- Create: `tests/test_vsf_rmr_checkpoint_exchange.py`

- [ ] Add tests that baseline and VSF-RMR use separate GitHub tags and asset prefixes and that BTD-SE/IOQC-SA assets are never selected.
- [ ] Add tests for rolling three valid checkpoints, SHA-256 manifest, temporary download plus atomic rename, and corrupt newest fallback.
- [ ] Implement restore and final publication around existing generic checkpoint primitives.
- [ ] Verify a synthetic upload/download round trip without exposing the token in process output.

### Task 5: Server guide and recovery drill

**Files:**
- Create: `docs/VSF_RMR_SERVER_GUIDE.md`

- [ ] Document clone, setup, token creation, baseline launch, VSF-RMR launch, monitoring, graceful stop, emergency recovery, migration to another server, and final artifact locations.
- [ ] Include commands for `tail`, status JSON, `watch nvidia-smi`, checkpoint validation, and GitHub asset inspection.
- [ ] Run a local dry-run supervisor with a fake child process covering planned promotion, OOM demotion, restart, completion, and publication verification.
- [ ] Run all server, sync, checkpoint, adaptive batch, and innovation isolation tests.


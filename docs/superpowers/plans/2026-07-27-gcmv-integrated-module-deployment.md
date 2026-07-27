# GCMV Integrated Module Deployment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PLEC, GGLF, and PEG as one integrated GCMV-EI network
module and run one matched seed0 10-epoch end-to-end screen.

**Architecture:** Reuse the verified PLEC core and the exact augmentation
geometry/stop-gradient local path already implemented in the active worktree.
Replace the temporary reference adapter with a four-head fixed-window GGLF
interaction and a zero-guarded PEG residual injector. Keep RT-DETR query
selection, decoder, P4, P5, matcher, and prediction head unchanged. Add only
the frozen GCMV-internal tiny/gate/protect auxiliary supervision.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90, pytest,
OpenCV, RTX 4090.

---

### Task 1: Implement GGLF by TDD

**Files:**

- Create: `tests/test_gcmv_fusion.py`
- Create: `src/gcmv_fusion.py`

- [ ] **Step 1: Write failing GGLF tests**

Tests require exact output shapes, exact zero output at invalid locations,
normalized 3-by-3 attention, a bounded impulse response, finite diagnostics,
and gradients for query/key/value/correction/confidence families.

- [ ] **Step 2: Verify RED**

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_gcmv_fusion.py -q
```

Expected: import failure because `src.gcmv_fusion` does not exist.

- [ ] **Step 3: Implement fixed-window GGLF**

Use channel normalization, four-head 64-channel 1-by-1 query/key/value
projections, learned relative position bias,
`torch.nn.functional.unfold(kernel_size=3, padding=1)`, masked nine-position
softmax, an explicit `[A,G,A-G,abs(A-G)]` difference descriptor, a tiny-demand
map, and correspondence confidence. Reject any configured window other than
odd positive values; freeze the formal configuration to 3.

- [ ] **Step 4: Verify GREEN**

Run the focused test command and require zero failures.

### Task 2: Implement PEG and the integrated injector by TDD

**Files:**

- Modify: `tests/test_gcmv_fusion.py`
- Modify: `src/gcmv_fusion.py`

- [ ] **Step 1: Write failing PEG tests**

Tests require exact stock identity at zero residual scalar, spatial gates in
`[0,1]`, exact zero gates for invalid locations, raw initial gate 0.5, nonzero
scalar gradient at frozen initialization, and full-family gradients in an
audit with the scalar opened.

- [ ] **Step 2: Verify RED**

Run the focused test and observe missing PEG/injector classes.

- [ ] **Step 3: Implement PEG**

Use reduced global/evidence features and their absolute difference together
with tiny demand, correspondence, PLEC confidence, and edge reliability.
Reliability is normalized coverage times the cube root of the remaining three
priors. Use a zero-initialized scalar `rho`:

```python
enhanced = global_p3 + tanh(rho) * gate * evidence_projection
```

Return evidence, confidence, attention, tiny demand, raw/final gates, scalar
gamma, and enhanced P3 for diagnostics.

- [ ] **Step 4: Verify GREEN**

Run `tests/test_gcmv_fusion.py` and require zero failures.

### Task 3: Replace the reference adapter in RT-DETR integration

**Files:**

- Modify: `configs/rtdetr-l-gcmv-plec.yaml`
- Modify: `tests/test_rtdetr_gcmv_plec_integration.py`
- Modify: `src/rtdetr_gcmv_plec.py`
- Modify: `scripts/preflight_gcmv_plec.py`

- [ ] **Step 1: Write failing integration tests**

Require the YAML to expose PLEC/GGLF/PEG settings, no formal reference adapter,
exact bypass identity, exact zero-guard identity, detached local P3, and
nonzero audit gradients for every internal family.

- [ ] **Step 2: Verify RED**

Run the integration/preflight tests and observe the old reference-adapter
expectations fail.

- [ ] **Step 3: Integrate GCMV-EI**

Instantiate `GCMVEvidenceInjectionModule`, pass the complete `PLECOutput` to it,
replace `inject_local_p3` with `inject_gcmv_evidence`, retain diagnostics, and
delete reference-adapter use from prediction and preflight.

- [ ] **Step 4: Verify GREEN**

Run integration, fusion, and preflight unit tests with zero failures.

### Task 4: Add frozen internal supervision and complete the protocol runner

**Files:**

- Create: `src/gcmv_plec_protocol.py`
- Create: `src/gcmv_loss.py`
- Create: `tests/test_gcmv_loss.py`
- Modify: `tests/test_gcmv_plec_training_cli.py`
- Modify: `scripts/train_rtdetr_gcmv_plec.py`
- Modify: `src/rtdetr_gcmv_plec.py`

- [ ] **Step 1: Keep the existing formal-settings tests RED**

The active tests already require the frozen loss weights, authoritative
dataset/subset/initial-state hashes, 10 epochs, fraction 1, batch/workers 8,
fixed augmentations, and 145 optimizer attempts.

- [ ] **Step 2: Implement protocol validation**

Build the tiny Gaussian and non-tiny protection targets and add loss weights
`0.25/0.02/0.01`. Validate the seed0 scratch artifact, YAML/list hashes,
647-image semantic signature, environment, device, fixed AMP scale, loader
batch/workers, optimizer observation, and source commit. Allow missing
scratch-state keys only under `plec.` and `gcmv_injector.`.

- [ ] **Step 3: Verify protocol tests**

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_gcmv_plec_training_cli.py -q
```

Expected: zero failures.

### Task 5: Verify, deploy, and run

**Files:**

- Modify only when a failing test exposes a concrete defect.

- [ ] **Step 1: Run focused tests**

Run all GCMV data, geometry, PLEC, fusion, integration, preflight, CLI, and
evaluation tests.

- [ ] **Step 2: Run the full local suite**

```powershell
C:\uav_env\Scripts\python.exe -m pytest -q
```

Expected: zero failures and only the existing expected skip.

- [ ] **Step 3: Commit and deploy the exact source state**

Archive the committed tree, upload it to a new server directory, verify SHA256,
and bind every run artifact to the commit.

- [ ] **Step 4: Run batch-8 CUDA preflight**

Require stock identity, detached local path, preserved BatchNorm, all audit
gradients, fixed AMP, finite loss, and peak reserved memory below 23 GiB.

- [ ] **Step 5: Run the paired screen**

Run seed0 control and complete GCMV-EI from the same initial state, then evaluate
both on the same 548 validation images and apply the frozen advance/stop gate.

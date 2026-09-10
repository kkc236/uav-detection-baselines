# BPDD Capacity v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy paired Formal100 local-expert-only and quality-gated-BPDD arms while preserving the six-layer LRS-GFDR path and excluding DN queries from the expert.

**Architecture:** Extend the declarative FDR head to accept `[P2,P3,P4,P5]`, use `[P3,P4,P5]` unchanged for the stock decoder, and feed sampled `[P2,P3,P4]` tokens to a separately cached expert output. Train the expert from the original final assignment; use its detached output as a support-aware, decoded-IoU-gated localization and classification teacher for all identity-consistent base layers.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics 8.4.90, pytest, YAML, Formal100 MuSGD/fixed-AMP protocol, Windows local worktree and remote RTX 4090.

---

## File map

- Modify `src/bpdd_capacity.py`: sampled-token expert, RNG-isolated initialization, checkpointed attention.
- Modify `src/fdr_head.py`: four-input head, preserved six-layer output, normal-only expert cache and inference output.
- Create `src/bpdd_capacity_loss.py`: support mask, hybrid-IoU localization KD, classification KD and schedule.
- Modify `src/fdr_loss.py`: expose fixed-assignment expert direct loss without an extra matcher call.
- Modify `src/rtdetr_fdr.py`: cache/detach evidence and allow v2 private state.
- Create `src/rtdetr_bpdd_capacity.py`: v2 model/criterion/trainer integration and three-way gradient partition.
- Create `configs/rtdetr-l-lrs-gfdr-capacity-v2-expert.yaml` and `configs/rtdetr-l-lrs-gfdr-capacity-v2-bpdd.yaml`.
- Create `scripts/train_lrs_gfdr_capacity_v2.py`: Formal100 authority, arm selection, runtime recorder and fresh-only launch.
- Create `tests/test_bpdd_capacity_v2.py`, `tests/test_bpdd_capacity_loss.py`, `tests/test_rtdetr_bpdd_capacity.py`, and `tests/test_train_bpdd_capacity_v2.py`.
- Modify `src/lrs_runtime_evidence.py`: v2 teacher/KD/gradient/memory evidence fields without changing old-arm output.

### Task 1: Expert contract, RNG isolation and memory-oriented execution

**Files:**
- Modify: `tests/test_bpdd_capacity_v2.py`
- Modify: `src/bpdd_capacity.py`

- [ ] **Step 1: Write failing expert tests**

Add tests that construct `LocalBoundaryExpert(channels=(128,256,256), ...)` and assert sampled raw features are projected after `grid_sample`, zero-output identity, nonzero second-step internal gradients, and no CPU/CUDA RNG mutation. Add a checkpoint spy and assert both attention blocks execute through `checkpoint(..., use_reentrant=False)` in training. The normal-query API is:

```python
box_logits, class_logits = expert(
    query=normal_query,
    corners=normal_corners,
    classes=normal_classes,
    boxes=normal_boxes.detach(),
    features=[p2, p3, p4],
)
```

- [ ] **Step 2: Verify RED**

Run: `C:/uav_env/Scripts/python.exe -m pytest tests/test_bpdd_capacity_v2.py -q`

Expected: failures because current projections are full-map Conv2d, global RNG is mutated on CUDA, and checkpointing is absent.

- [ ] **Step 3: Implement sampled-token projections and deterministic private initialization**

Replace Conv2d projections with per-token Linear projections and initialize every private tensor from a CPU `torch.Generator` without calling global `torch.manual_seed`. Keep the two output layers exactly zero. Sample first, then project:

```python
sampled = sample_local_features(feature, local_grid)
tokens.append(projection(sampled) + self.position(position.to(query.dtype)))
```

Wrap each query chunk's two attention blocks with non-reentrant activation checkpointing in training; use the direct path during evaluation.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test command and require all tests to pass.

- [ ] **Step 5: Commit**

Commit `src/bpdd_capacity.py` and `tests/test_bpdd_capacity_v2.py` with message `feat: make local expert memory safe and rng isolated`.

### Task 2: Preserve base outputs and expose graph-native P2

**Files:**
- Modify: `tests/test_rtdetr_bpdd_capacity.py`
- Modify: `src/fdr_head.py`

- [ ] **Step 1: Write failing graph and output tests**

Construct the v2 head with actual channels `(128,256,256,256)`. Assert decoder input maps are the last three tensors, expert inputs are the first three tensors, training returns exactly six original layers, and cached expert tensors contain only 300 normal queries when DN is present. Assert initial cached expert output equals the original sixth layer and inference returns expert predictions.

- [ ] **Step 2: Verify RED**

Run: `C:/uav_env/Scripts/python.exe -m pytest tests/test_rtdetr_bpdd_capacity.py -q`

Expected: failure because the current head accepts only three features, overwrites layer six, and sends DN queries to the expert.

- [ ] **Step 3: Implement the four-input head and independent expert cache**

Override the head forward using the pinned Ultralytics 8.4.90 flow. Split `x` into `local_features=x[:3]` and `decoder_features=x[1:]`. Extend decoder forward with keyword-only `local_features` and `normal_query_count`; slice the final 300 queries, cache an `ExpertPrediction` containing box/corner/class tensors, keep all six base outputs unchanged during training, and substitute expert output only for inference.

- [ ] **Step 4: Clear cache defensively**

At each forward start set expert cache to `None`; validate four feature strides/channels and normal-query count. Never use a forward hook or retain a previous batch tensor.

- [ ] **Step 5: Verify GREEN and commit**

Run Task 2 tests plus `tests/test_fdr_head.py tests/test_rtdetr_fdr.py`; commit with message `feat: preserve FDR outputs with normal-only P2 expert`.

### Task 3: Quality-gated localization and classification distillation

**Files:**
- Create: `tests/test_bpdd_capacity_loss.py`
- Create: `src/bpdd_capacity_loss.py`

- [ ] **Step 1: Write failing pure-loss tests**

Test these pure functions and dataclasses:

```python
raw_targets, representable = raw_fdr_targets(reference, gt)
weight = bounded_improvement(student_error, teacher_error, margin=.02, tau=.1)
result = quality_gated_capacity_distillation(
    student_corners, student_classes, expert_corners.detach(),
    expert_classes.detach(), reference, gt_boxes, gt_classes,
    layer_matches, final_matches, options, epoch=completed_epoch,
)
```

Cover: exact-zero with no matches; out-of-support edge rejection; source/final identity rejection; teacher detach; per-layer normalization; schedule values at completed epochs 10/15/20; classification teacher-better gate; and the saved NLL-better/IoU-worse counterexample, which must yield zero localization KD.

- [ ] **Step 2: Verify RED**

Run: `C:/uav_env/Scripts/python.exe -m pytest tests/test_bpdd_capacity_loss.py -q`

Expected: import failure because `src.bpdd_capacity_loss` does not exist.

- [ ] **Step 3: Implement the pure loss module**

Add frozen `CapacityBPDDOptions` with `loc_weight=.15`, `cls_weight=.10`, `loc_margin=.02`, `cls_margin=.02`, `loc_tau=.1`, `cls_tau=.1`, `warmup_start=10`, `warmup_end=20`. Compute unclipped raw targets, a hybrid teacher/student box for the IoU gate, teacher-to-student edge KL, and matched-query Bernoulli KL. Return a connected scalar zero and detached unconditional/conditional statistics when nothing is eligible.

- [ ] **Step 4: Verify GREEN and commit**

Run Task 3 tests; commit with message `feat: add quality gated BPDD losses`.

### Task 4: Fixed-assignment direct supervision and model integration

**Files:**
- Modify: `tests/test_rtdetr_bpdd_capacity.py`
- Modify: `src/fdr_loss.py`
- Modify: `src/rtdetr_fdr.py`
- Create: `src/rtdetr_bpdd_capacity.py`

- [ ] **Step 1: Write failing integration tests**

Assert the criterion computes original stock/FGL losses, then expert classification/L1/GIoU/FGL from the already recorded original final assignment. Spy on matcher and assert no expert matcher call. Assert B has direct expert loss but exact zero KD; C adds scheduled localization/classification KD; teacher tensors have no KD gradient; base final output remains independently supervised.

- [ ] **Step 2: Verify RED**

Run: `C:/uav_env/Scripts/python.exe -m pytest tests/test_rtdetr_bpdd_capacity.py -q`

Expected: failure because the v2 criterion/model module is missing.

- [ ] **Step 3: Implement fixed-assignment criterion and detector**

Extend `FDRDetectionLoss` with a public method that accepts explicit match indices and a distinct postfix. Implement `CapacityBPDDDetectionLoss` and `CapacityBPDDDetectionModel`; pass cached expert evidence after DN splitting, reuse `normal_assignments[-1]`, and store only detached diagnostics. Make current completed epoch an explicit trainer-to-model value; do not infer it from validation.

- [ ] **Step 4: Detach retained loss evidence**

Change model-side runtime loss caches to detached tensors before the next forward, while returning the original connected total to the trainer. Add a two-iteration test that proves the previous autograd graph is not retained.

- [ ] **Step 5: Implement three-way gradient partition**

Return disjoint `gradient_norm`, `fdr_gradient_norm`, and `expert_gradient_norm` groups. Assert the 1.8M-class expert parameter set is entirely in the third group and optimizer coverage is exact.

- [ ] **Step 6: Verify GREEN and commit**

Run integration tests and the existing FDR/BPDD/LRS suites; commit with message `feat: integrate independent expert and capacity BPDD`.

### Task 5: Declarative configs, launcher and evidence

**Files:**
- Create: `tests/test_train_bpdd_capacity_v2.py`
- Create: `configs/rtdetr-l-lrs-gfdr-capacity-v2-expert.yaml`
- Create: `configs/rtdetr-l-lrs-gfdr-capacity-v2-bpdd.yaml`
- Create: `scripts/train_lrs_gfdr_capacity_v2.py`
- Modify: `src/lrs_runtime_evidence.py`

- [ ] **Step 1: Write failing config/launcher tests**

Assert configs differ only in v2 distillation enablement, both use `[1,21,24,27]`, LRS `.25`, geometry true, expert seed 30000, Formal100 settings, seed 0, create-only output, and the same initial-state loader. Assert the launch record labels `capacity-v2-expert` or `capacity-v2-bpdd`, and recorder fields use a v2 revision.

- [ ] **Step 2: Verify RED**

Run: `C:/uav_env/Scripts/python.exe -m pytest tests/test_train_bpdd_capacity_v2.py -q`

Expected: failure because v2 configs and launcher do not exist.

- [ ] **Step 3: Add configs and launcher**

Copy the frozen LRS-GFDR graph, extend only the head input/options and v2 loss mapping, and create a fresh-only launcher with arms `expert` and `bpdd`. Record config/source/initial-state/dataset hashes before training. Pass completed epoch into the model on each epoch start.

- [ ] **Step 4: Extend runtime evidence**

Record support, teacher advantage, active ratios, loc/cls KD, three gradient groups and CUDA memory when v2 fields exist. Preserve the old schema for old models.

- [ ] **Step 5: Verify GREEN and commit**

Run Task 5 tests, launcher dry-runs with fixture paths, and regression tests; commit with message `feat: add Formal100 capacity v2 arms`.

### Task 6: Full local verification and direct Formal100 deployment

**Files:**
- Modify: `docs/superpowers/plans/2026-09-10-bpdd-capacity-v2.md` checkboxes only
- Create outside source commit: deployment evidence under workspace `diagnostics/`

- [ ] **Step 1: Run full relevant verification**

Run all new tests plus existing FDR, FGL, BPDD, LRS, protocol, runtime and launcher suites. Run `git diff --check`, confirm clean tracked worktree, instantiate B/C at nc=10, compare shared/expert tensors, count parameters, and run CPU training/eval forward/backward and checkpoint roundtrip.

- [ ] **Step 2: Freeze source**

Commit all verified source, configs, tests and plan checkboxes. Create an archive and SHA-256 manifest under workspace `tmp/`; do not include credentials or datasets.

- [ ] **Step 3: Deploy create-only**

Upload to a new remote directory `/root/sj-tmp/lrs-v2/bpdd-capacity-v2-code`, verify normalized source hashes, environment, initial-state hash and dataset signature. Do not modify `/root/sj-tmp/lrs-v2/code`, v1 code, or earlier result directories.

- [ ] **Step 4: Start Formal100 directly as authorized**

Without an independent CUDA memory preflight, start B at batch 8/640/100 in a new result root. Queue C only after B exits successfully. Capture PID, exact commands, log paths and authority records. On OOM or nonzero exit, preserve evidence and do not auto-retry or lower batch.

- [ ] **Step 5: Confirm actual training**

Verify the Python process owns GPU memory, the log reaches real training batches, and authority/optimizer/runtime files begin to appear. This confirms launch only, not accuracy or eventual completion.

- [ ] **Step 6: Write deployment record**

Save local paths, remote commits/hashes, run identity, start time, PID, first observed batch and unresolved accuracy/OOM risk. Do not claim v2 success before Formal100 results exist.

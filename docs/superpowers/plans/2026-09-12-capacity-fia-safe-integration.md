# Capacity-FIA Safe Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separately named LRS-GFDR + Capacity-BPDD + FIA candidate that preserves the proven geometry and expert path, removes conflicting class distillation, fades localization distillation late, and emits trustworthy evaluation/runtime evidence.

**Architecture:** Keep every historical YAML and launcher unchanged. Extend `CapacityBPDDOptions` with an opt-in late linear decay whose defaults preserve v2 behavior, fix the expert-evaluation corner cache to use the same source as reported boxes/classes, and make runtime aggregation distinguish missing, non-finite, and real zero values. A new safe model/config/launcher enables `cls_weight=0` and BPDD decay from completed epoch 80 to 100; FIA and direct expert supervision remain active throughout.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics RT-DETR, pytest, YAML.

---

### Task 1: Add an opt-in BPDD late-decay schedule

**Files:**
- Modify: `tests/test_bpdd_capacity_loss.py`
- Modify: `src/bpdd_capacity_loss.py`
- Modify: `src/rtdetr_bpdd_capacity.py`

- [x] **Step 1: Write the failing schedule tests**

Add tests which construct `CapacityBPDDOptions(decay_start=80, decay_end=100)`, assert scale `1.0/0.5/0.0` at epochs `80/90/100`, reject half-specified or non-increasing decay bounds, and verify the result statistic reports the effective scale.

```python
def test_capacity_schedule_can_fade_bpdd_after_epoch_eighty():
    options = CapacityBPDDOptions(decay_start=80, decay_end=100)
    assert capacity_bpdd_schedule(80, options) == 1.0
    assert capacity_bpdd_schedule(90, options) == 0.5
    assert capacity_bpdd_schedule(100, options) == 0.0

@pytest.mark.parametrize("values", [(80, None), (None, 100), (100, 80)])
def test_capacity_options_reject_invalid_decay(values):
    with pytest.raises(ValueError, match="decay"):
        CapacityBPDDOptions(decay_start=values[0], decay_end=values[1])
```

- [x] **Step 2: Run the tests and verify RED**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py -k "schedule or invalid_decay" -q -p no:cacheprovider`

Expected: collection/import failure because `capacity_bpdd_schedule` and decay fields do not exist.

- [x] **Step 3: Implement the minimal schedule**

Add optional `decay_start`/`decay_end` fields, validate them as a pair, and expose:

```python
def capacity_bpdd_schedule(completed_epoch: int, options: CapacityBPDDOptions) -> float:
    scale = capacity_bpdd_warmup(completed_epoch, options.warmup_start, options.warmup_end)
    if options.decay_start is None:
        return scale
    if completed_epoch <= options.decay_start:
        return scale
    if completed_epoch >= options.decay_end:
        return 0.0
    return scale * (options.decay_end - completed_epoch) / (options.decay_end - options.decay_start)
```

Use this scale for only the two BPDD losses and report it as `schedule_scale`; direct expert loss is not changed. Add both keys to `_OPTION_KEYS`.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py -q -p no:cacheprovider`

Expected: all tests pass.

### Task 2: Make expert evaluation caches single-source

**Files:**
- Modify: `tests/test_rtdetr_bpdd_capacity.py`
- Modify: `src/fdr_head.py`

- [x] **Step 1: Change the regression assertion to the required contract**

In `test_eval_returns_cached_expert_prediction`, require:

```python
torch.testing.assert_close(head.decoder.last_corner_logits[-1], expert.corners)
```

- [x] **Step 2: Run the test and verify RED**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_rtdetr_bpdd_capacity.py::test_eval_returns_cached_expert_prediction -q -p no:cacheprovider`

Expected: FAIL because boxes/classes come from the expert while `last_corner_logits` still caches base cumulative corners.

- [x] **Step 3: Cache the reported corner source**

Change the decoder append site from `corner_logits.append(cumulative_corners)` to `corner_logits.append(reported_corners)`. Training behavior remains unchanged because training keeps `reported_corners` equal to base cumulative corners.

- [x] **Step 4: Run the test and verify GREEN**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_rtdetr_bpdd_capacity.py -q -p no:cacheprovider`

Expected: all tests pass.

### Task 3: Preserve missing and non-finite runtime evidence

**Files:**
- Modify: `tests/test_lrs_system_local_execution.py`
- Modify: `src/lrs_runtime_evidence.py`

- [x] **Step 1: Write failing evidence tests**

Add one zero-valued complete capacity observation and one observation containing `NaN`. Assert the zero observation contributes to the mean, the `NaN` observation does not, and the record exposes `capacity_nonfinite_values == 1`, `capacity_invalid_observations == 1`, plus `fia_gradient_norm` when supplied.

```python
assert record["capacity_observations"] == 1
assert record["capacity_invalid_observations"] == 1
assert record["capacity_nonfinite_values"] == 1
assert record["capacity_active_edge_ratio_mean"] == 0.0
assert record["fia_gradient_norm"] == pytest.approx(4.0)
```

- [x] **Step 2: Run the tests and verify RED**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_lrs_system_local_execution.py -k "runtime_recorder" -q -p no:cacheprovider`

Expected: FAIL because the recorder currently converts non-finite values to zero and omits FIA's norm.

- [x] **Step 3: Implement finite-only aggregation**

Validate the complete capacity field set before incrementing `capacity_observations`. Track missing and non-finite values independently, increment `capacity_invalid_observations` for rejected batches, retain genuine numeric zeros, and add `fia_gradient_norm`, `gradient_missing_values`, and `gradient_nonfinite_values` to the output record.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_lrs_system_local_execution.py -k "runtime_recorder" -q -p no:cacheprovider`

Expected: all selected tests pass.

### Task 4: Add the isolated safe joint candidate and launcher

**Files:**
- Create: `configs/rtdetr-l-lrs-gfdr-capacity-v3-safe-bpdd-fia.yaml`
- Modify: `src/rtdetr_bpdd_capacity_fia.py`
- Create: `scripts/train_lrs_gfdr_capacity_v3_safe_fia.py`
- Modify: `tests/test_capacity_fia.py`

- [x] **Step 1: Write failing config and launcher tests**

Assert the new YAML retains feasible geometry, P3-only FIA, P4 bypass, and the preserved local expert; assert `loc_weight=0.15`, `cls_weight=0.0`, `decay_start=80`, and `decay_end=100`. Assert the new launcher's settings keep the frozen Formal100 controls and point only to the new config.

- [x] **Step 2: Run the tests and verify RED**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_capacity_fia.py -q -p no:cacheprovider`

Expected: FAIL because the safe config/model/launcher do not exist.

- [x] **Step 3: Implement the candidate without mutating history**

Copy the v2 joint graph to the new YAML and change only the BPDD controls. Add `CapacityBPDDFIASafeDetectionModel` and `CapacityBPDDFIASafeTrainer` with revision `v3-safe-loc-kd-decay80-100-no-cls-kd`. Add a fresh-only launcher whose authority record names `lrs_gfdr_capacity_v3_safe_bpdd_fia` and hashes the new config, initial state, dataset, and source.

- [x] **Step 4: Run the tests and verify GREEN**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_capacity_fia.py tests/test_rtdetr_bpdd_capacity.py tests/test_bpdd_capacity_loss.py -q -p no:cacheprovider`

Expected: all selected tests pass.

### Task 5: Document and verify the complete candidate

**Files:**
- Create: `docs/CAPACITY_FIA_SAFE_CANDIDATE_2026-09-12_ZH.md`
- Modify: `diagnostics/2026-09-12-capacity-fia-joint-cause-analysis-zh.md` if present in this worktree

- [x] **Step 1: Document evidence, changes, and non-claims**

Record that real probes found strong base/expert gradient alignment, tiny inconsistent expert box gains, weak class-KD gradients with an actual target-direction counterexample, and late joint degradation. State explicitly that the candidate is untrained and cannot be claimed to improve mAP before a strict paired Formal100 run.

- [x] **Step 2: Run full relevant verification**

Run: `C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py tests/test_bpdd_capacity_v2.py tests/test_bpdd_capacity.py tests/test_rtdetr_bpdd_capacity.py tests/test_capacity_fia.py tests/test_lrs_system_local_execution.py tests/test_lrs_system_models.py -q -p no:cacheprovider`

Expected: zero failures.

- [x] **Step 3: Review the diff and commit**

Run: `git diff --check` and `git status --short`, inspect every changed file, then commit only the candidate, tests, and documentation with `git commit -m "feat: add conflict-safe capacity BPDD FIA candidate"`.

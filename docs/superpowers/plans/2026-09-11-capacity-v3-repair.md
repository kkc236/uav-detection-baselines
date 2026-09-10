# Capacity-v3 Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 capacity-v2 已复现的多匹配崩溃、分类蒸馏方向冲突、累计残差接线、非有限值吞错和训练/评估输出口径问题，形成可独立筛查的 `capacity-v3-repair` 备选实现。

**Architecture:** 保留现有 LRS-GFDR、normal-only 局部 expert、FDR 几何解码和初始化协议。把新版 BPDD 质量门控限定为 detached 的教师筛选；定位分布蒸馏增加 residual-only 可选路径，分类蒸馏改为按类别比较 IoU 质量感知目标，避免全类平均掩盖正类冲突。训练缓存拆成 base 与 expert 两套证据，recorder 对缺失、非有限和真实零值分别计数。

**Tech Stack:** Python 3.11, PyTorch, Ultralytics RT-DETR, pytest, YAML。

---

### Task 1: 修复多匹配形状契约

**Files:**
- Modify: `src/bpdd_capacity_loss.py:158-166,302-340`
- Modify: `src/bpdd_loss.py:110-139`
- Test: `tests/test_bpdd_capacity_loss.py`

- [ ] **Step 1: Write failing tests**

Add a parametrized real-function test that calls `quality_gated_capacity_distillation` with 0, 1, 2, 4, and 5 identity-consistent matches, plus disabled, warmup-zero, and zero-class-weight cases for two matches. Assert no shape exception, finite loss, and connected zero when disabled.

```python
@pytest.mark.parametrize("matches", [0, 1, 2, 4, 5])
def test_capacity_loss_accepts_multiple_identity_consistent_matches(matches):
    values = _inputs(queries=max(matches, 1))
    triple = (
        torch.zeros(matches, dtype=torch.long),
        torch.arange(matches, dtype=torch.long),
        torch.arange(matches, dtype=torch.long),
    )
    result = quality_gated_capacity_distillation(
        *values, [triple], triple, CapacityBPDDOptions(), epoch=20
    )
    assert torch.isfinite(result.loss)

@pytest.mark.parametrize("options,epoch", [
    (CapacityBPDDOptions(enabled=False), 20),
    (CapacityBPDDOptions(), 0),
    (CapacityBPDDOptions(cls_weight=0), 20),
])
def test_capacity_loss_short_circuits_after_validating_match_shapes(options, epoch):
    values = _inputs(queries=2)
    triple = (torch.zeros(2, dtype=torch.long), torch.arange(2), torch.arange(2))
    result = quality_gated_capacity_distillation(
        *values, [triple], triple, options, epoch=epoch
    )
    assert torch.isfinite(result.loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py -k "multiple_identity or short_circuits" -q -p no:cacheprovider`

Expected: FAIL for matches 2 and 5 with a dimension mismatch in `decoded_teacher_iou_gate`.

- [ ] **Step 3: Implement the minimal shape fix**

Pass a boolean edge mask with shape `[matches, 4]` to the classification IoU gate, and add explicit shape validation in `decoded_teacher_iou_gate`:

```python
if active_edges.shape != source_log.shape[:-1]:
    raise ValueError("active_edges must match [matches, edges]")
```

Use `torch.ones_like(loc_active, dtype=torch.bool)` for the classification gate. Do not change the gate's return semantics.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py -k "multiple_identity or short_circuits" -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/bpdd_capacity_loss.py src/bpdd_loss.py tests/test_bpdd_capacity_loss.py
git commit -m "fix: enforce capacity BPDD edge mask shape"
```

### Task 2: Make classification distillation VFL-quality aware per class

**Files:**
- Modify: `src/bpdd_capacity_loss.py:158-166,302-340`
- Test: `tests/test_bpdd_capacity_loss.py`

- [ ] **Step 1: Write failing tests**

Add tests showing that a teacher with a better decoded box but an overconfident positive class is rejected for that positive class, while a teacher that moves the positive probability toward the detached student IoU target is retained. Assert negative-class suppression remains independently eligible.

```python
def test_classification_gate_rejects_positive_teacher_farther_from_student_iou():
    values = list(_inputs())
    with torch.no_grad():
        values[1][..., 1] = -3.0
        values[3][..., 1] = 3.0
    result = quality_gated_capacity_distillation(
        *values, [_matches(), _matches()], _matches(),
        CapacityBPDDOptions(loc_weight=0.0), epoch=20,
    )
    assert result.statistics["active_classes"] == 0

def test_classification_gate_counts_positive_and_negative_classes_separately():
    values = list(_inputs())
    with torch.no_grad():
        values[1][..., 1] = -1.0
        values[3][..., 1] = -0.3
        values[1][..., 0] = 2.0
        values[3][..., 0] = 0.2
    result = quality_gated_capacity_distillation(
        *values, [_matches(), _matches()], _matches(),
        CapacityBPDDOptions(loc_weight=0.0), epoch=20,
    )
    assert result.statistics["active_classes"] >= 1
    assert result.statistics["classification_positive_active"] == 0
    assert result.statistics["classification_negative_active"] >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py -k "classification_gate_rejects_positive or counts_positive" -q -p no:cacheprovider`

Expected: FAIL because v2 uses one-hot full-class BCE and a match-level scalar reliability.

- [ ] **Step 3: Implement the minimal quality-aware objective**

Add a detached helper that decodes source and expert boxes and returns their IoU to the matched GT. Set the per-class VFL target to `one_hot * student_iou.detach()`. Compute BCE errors per class, derive bounded improvement per class, and apply the existing expert-box IoU gate as a match-level prerequisite. Compute Bernoulli KL per class and average only active classes. Record signed and absolute class advantages plus positive/negative active counts. Keep teacher logits detached and keep the existing warmup and weights.

```python
class_target = one_hot * student_iou.detach().unsqueeze(-1)
source_class_error = F.binary_cross_entropy_with_logits(
    source_class.float(), class_target, reduction="none"
)
teacher_class_error = F.binary_cross_entropy_with_logits(
    teacher_class, class_target, reduction="none"
)
class_reliability = bounded_improvement(
    source_class_error, teacher_class_error,
    margin=options.cls_margin, tau=options.cls_tau,
)
class_reliability = class_reliability * cls_quality_keep[:, None]
class_kl = _bernoulli_kl_per_class(teacher_class, source_class)
```

The target and per-class gate must be explicit in the docstring and statistics; do not call a full-class mean “VFL-aligned” without this per-class check.

- [ ] **Step 4: Run tests and the synthetic conflict probe**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py tests/test_bpdd_joint_mechanism.py -q -p no:cacheprovider`

Run: `& C:/uav_env/Scripts/python.exe -B diagnostics/audit_capacity_v3_design_20260911.py --root .`

Expected: new classification conflict test passes; the old production conflict probe is retained as a regression fixture and no longer represents the v3 classification gate.

- [ ] **Step 5: Commit**

```powershell
git add src/bpdd_capacity_loss.py tests/test_bpdd_capacity_loss.py diagnostics/audit_capacity_v3_design_20260911.py
git commit -m "fix: align capacity classification KD with VFL quality"
```

### Task 3: Connect residual-only to capacity localization KD

**Files:**
- Modify: `src/bpdd_capacity_loss.py:20-52,261-300`
- Modify: `src/rtdetr_bpdd_capacity.py:24-35,88-100`
- Modify: `configs/rtdetr-l-lrs-gfdr-capacity-v3-repair.yaml`
- Test: `tests/test_bpdd_capacity_loss.py`
- Test: `tests/test_rtdetr_bpdd_capacity.py`

- [ ] **Step 1: Write failing tests**

Add a two-layer cumulative-logit test that checks capacity `residual_gradient_only=True` keeps the same detached loss value, gives zero direct gradient to the previous cumulative layer, and retains current-layer gradient. Add parser/config tests that require the option to reach `CapacityBPDDOptions`.

```python
def test_capacity_residual_only_preserves_value_and_blocks_history_gradient():
    old_delta, old_inputs = _capacity_cumulative_case()
    new_delta, new_inputs = _capacity_cumulative_case()
    old = quality_gated_capacity_distillation(*old_inputs, CapacityBPDDOptions(), epoch=20)
    new = quality_gated_capacity_distillation(
        *new_inputs, CapacityBPDDOptions(residual_gradient_only=True), epoch=20
    )
    torch.testing.assert_close(old.loss, new.loss, rtol=0, atol=0)
    old.loss.backward()
    new.loss.backward()
    assert old_delta.grad[0].norm() > 0
    assert new_delta.grad[0].norm() == 0
    assert new_delta.grad[1].norm() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py tests/test_rtdetr_bpdd_capacity.py -k "residual_only or capacity_options" -q -p no:cacheprovider`

Expected: FAIL because capacity options do not expose or use `residual_gradient_only`.

- [ ] **Step 3: Implement the capacity residual path and config**

Add `residual_gradient_only: bool = False` to `CapacityBPDDOptions`, validate it, include it in `_OPTION_KEYS`, and pass it into the capacity localization branch only. Construct the same-forward-value detached residual expression before `source_log`; leave classification logits unchanged. Set the v3 YAML option to `true`, with `decoded_iou_gate: true` and `quality_aware_class_gate: true` as explicit keys.

- [ ] **Step 4: Run tests and integration checks**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity_loss.py tests/test_rtdetr_bpdd_capacity.py tests/test_bpdd_joint_mechanism.py -q -p no:cacheprovider`

Expected: PASS with identical forward values and expected direct gradient isolation.

- [ ] **Step 5: Commit**

```powershell
git add src/bpdd_capacity_loss.py src/rtdetr_bpdd_capacity.py configs/rtdetr-l-lrs-gfdr-capacity-v3-repair.yaml tests/test_bpdd_capacity_loss.py tests/test_rtdetr_bpdd_capacity.py
git commit -m "feat: connect residual-only capacity localization KD"
```

### Task 4: Make runtime evidence fail-visible and align train/eval output contracts

**Files:**
- Modify: `src/lrs_runtime_evidence.py:7-24,94-181,183-271`
- Modify: `src/fdr_head.py:535-640`
- Test: `tests/test_lrs_system_local_execution.py`
- Test: `tests/test_rtdetr_bpdd_capacity.py`

- [ ] **Step 1: Write failing tests**

Add recorder tests that pass NaN and missing values and assert they increment explicit counters instead of being accumulated as zero. Add eval tests that assert expert boxes, classes, and corners are aligned in the reported output, while training retains separate base and expert caches.

```python
def test_runtime_recorder_separates_nonfinite_from_zero():
    recorder = RuntimeEvidenceRecorder()
    model = SimpleNamespace(last_capacity_statistics={
        "localization_loss": torch.tensor(float("nan")),
        "active_edge_ratio": torch.tensor(0.0),
    })
    recorder.capture(SimpleNamespace(model=model))
    assert recorder.capacity_nonfinite_observations == 1
    assert recorder.capacity_active_sum == 0.0

def test_eval_expert_outputs_share_one_reported_contract():
    head = _head().eval()
    with torch.no_grad():
        _, raw = head(_features())
    expert = head.decoder.last_expert_prediction
    assert expert is not None
    reported_boxes, reported_scores, reported_corners = raw[:3]
    assert reported_boxes.shape == expert.boxes.shape
    assert reported_scores.shape == expert.classes.shape
    assert reported_corners.shape == expert.corners.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_lrs_system_local_execution.py tests/test_rtdetr_bpdd_capacity.py -k "nonfinite or reported_contract" -q -p no:cacheprovider`

Expected: FAIL because `_number(... ) or 0.0` turns NaN into zero and eval raw output retains base corners.

- [ ] **Step 3: Implement fail-visible recorder and explicit caches**

Add `capacity_nonfinite_observations`, `capacity_missing_observations`, and analogous expert geometry counters. When a field is absent or non-finite, increment the matching counter and do not add to a numeric sum. Preserve true zero. In `fdr_head.py`, store `last_base_prediction` and `last_expert_prediction` separately; in eval, append the expert triplet together, and in training append the base triplet while expert remains an independent direct-supervision input.

- [ ] **Step 4: Run targeted and full relevant tests**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_lrs_system_local_execution.py tests/test_rtdetr_bpdd_capacity.py tests/test_bpdd_capacity_loss.py tests/test_bpdd_capacity_v2.py -q -p no:cacheprovider`

Expected: PASS; existing tests that assert the old mixed eval cache must be updated to assert the explicit train/eval contract.

- [ ] **Step 5: Commit**

```powershell
git add src/lrs_runtime_evidence.py src/fdr_head.py tests/test_lrs_system_local_execution.py tests/test_rtdetr_bpdd_capacity.py
git commit -m "fix: make capacity evidence fail-visible and output-aligned"
```

### Task 5: Add the v3 launcher/config and same-capacity no-KD arm

**Files:**
- Create: `configs/rtdetr-l-lrs-gfdr-capacity-v3-repair.yaml`
- Modify: `src/rtdetr_bpdd_capacity.py:24-35,113-121`
- Create: `scripts/train_lrs_gfdr_capacity_v3.py`
- Test: `tests/test_rtdetr_bpdd_capacity.py`

- [ ] **Step 1: Write failing tests**

Add tests that the v3 config resolves to method revision `v3-fp32-extent-capacity-bpdd-repair`, keeps `feasible_geometry: true`, enables residual-only and quality-aware class gating, and can disable all KD while keeping the same local expert for the expert-only control.

- [ ] **Step 2: Run tests to verify they fail**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_rtdetr_bpdd_capacity.py -k "v3 or expert_only" -q -p no:cacheprovider`

Expected: FAIL because the v3 config and launcher do not exist.

- [ ] **Step 3: Implement config and launcher**

Copy the frozen v2 architecture and data settings into the v3 YAML, add explicit `capacity_bpdd_loss` keys for the repaired gates, and set a separate method revision. The launcher must accept `--arm repair|expert-only`, resolve the corresponding config, write an authority record containing config and initial-state SHA256, and refuse a dirty tracked worktree. It must not upload or change the current remote job.

- [ ] **Step 4: Run dry-run and integration tests**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_rtdetr_bpdd_capacity.py -q -p no:cacheprovider`

Run: `& C:/uav_env/Scripts/python.exe -B scripts/train_lrs_gfdr_capacity_v3.py --help`

Expected: tests PASS and help exits 0.

- [ ] **Step 5: Commit**

```powershell
git add configs/rtdetr-l-lrs-gfdr-capacity-v3-repair.yaml src/rtdetr_bpdd_capacity.py scripts/train_lrs_gfdr_capacity_v3.py tests/test_rtdetr_bpdd_capacity.py
git commit -m "feat: package capacity v3 repair backup arm"
```

### Task 6: Run the complete verification gate and publish the backup package locally

**Files:**
- Modify: `diagnostics/2026-09-11-capacity-v3-repair-design-audit.md`
- Create: `diagnostics/2026-09-11-capacity-v3-repair-implementation-report.md`

- [ ] **Step 1: Run all relevant tests**

Run: `& C:/uav_env/Scripts/python.exe -B -m pytest tests/test_bpdd_capacity.py tests/test_bpdd_capacity_v2.py tests/test_bpdd_capacity_loss.py tests/test_bpdd_capacity_loss_v3.py tests/test_bpdd_joint_mechanism.py tests/test_bpdd_iou_gate.py tests/test_rtdetr_bpdd_capacity.py tests/test_lrs_system_local_execution.py -q -p no:cacheprovider`

Expected: 0 failures; CUDA-only tests may be skipped only when CUDA is unavailable.

- [ ] **Step 2: Run source and config checks**

Run: `git diff --check`

Run: `& C:/uav_env/Scripts/python.exe -B -m compileall -q src scripts tests`

Expected: exit code 0.

- [ ] **Step 3: Run fixed CPU diagnostics**

Run: `& C:/uav_env/Scripts/python.exe -B diagnostics/audit_capacity_v3_design_20260911.py --root .`

Expected: multi-match, classification-gate, residual-path, recorder and output-contract probes complete with explicit PASS/REJECT labels; no production training is launched.

- [ ] **Step 4: Write an implementation report**

Record commit ids, test counts, skipped CUDA checks, config SHA256, method revision, and known limitations. State explicitly that no mAP or GPU peak-memory gain is established until a fresh paired training run.

- [ ] **Step 5: Commit the report**

```powershell
git add diagnostics/2026-09-11-capacity-v3-repair-design-audit.md diagnostics/2026-09-11-capacity-v3-repair-implementation-report.md
git commit -m "docs: record capacity v3 repair verification"
```

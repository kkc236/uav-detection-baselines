# IBER-BE v1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, deploy, and run the trajectory-free IBER-BE boundary refiner through B0–B3 Gate-1 and, only if Gate-1 passes, a fixed-subset seed0 30-epoch same-checkpoint stock/refined screen.

**Architecture:** Keep the matched RT-DETR-L checkpoint permanently frozen. Combine stride-8 F3 semantic boundary samples with sparse input-resolution RGB one-sided outside→edge and edge→inside directions in a private dual-route per-edge gate/residual head. Use separate `iber_*` protocols, checkpoints, reports, GitHub assets, and run roots so no I-TBER v1.1 evidence can be overwritten or reinterpreted.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.90, CUDA 12.1, pytest, Git/GitHub Releases, Bash deployment scripts, NVIDIA RTX 4090.

---

## File map

Create the following focused units:

- `src/iber_protocol.py`: immutable design/environment/data/checkpoint/training authority.
- `src/iber_sampling.py`: F3 and RGB boundary grids and sparse evidence extraction.
- `src/iber_head.py`: equal-capacity B0–B3 dual-route refiner and output diagnostics.
- `src/rtdetr_iber.py`: frozen detector adapter and last-layer-only evidence recorder.
- `src/iber_cache.py`: trajectory-free sharded Probe cache with RGB `uint8 CHW` records.
- `src/iber_probe.py`: 12-epoch B0–B3 private training, metrics, and Gate-1 decision.
- `src/iber_evaluation.py`: exact same-checkpoint stock/refined diagnostics and Gate-2 decision.
- `src/iber_publication.py`: transactional per-epoch GitHub publication and append-only ledger.
- `scripts/cache_iber_evidence.py`: deterministic train647/val548 Probe cache builder.
- `scripts/run_iber_canary.py`: Gate-0 stock identity, gradients, and detector-freeze canary.
- `scripts/evaluate_iber_stock.py`: three-repeat current-environment stock authority.
- `scripts/run_iber_probe.py`: B0–B3 fresh Probe CLI.
- `scripts/train_iber.py`: fixed-subset 30-epoch private training and resume supervisor.
- `scripts/evaluate_iber.py`: independent stock/refined evaluator.
- `scripts/benchmark_iber.py`: params, GFLOPs, and alternating latency measurement.
- `scripts/publish_iber_epoch.py`: credential-free publication CLI.
- `scripts/restore_iber_checkpoint.py`: verified Release restore CLI.
- `scripts/run_iber_pipeline.py`: authority → Gate-0 → stock → cache → Probe → Gate-1 → screen30 state machine.
- `scripts/audit_iber_deployment.py`: source/data/environment/config/run audit.
- `deploy/iber/*`: bare-server bootstrap, verification, bundle, and publication templates.
- `docs/IBER_BE_SERVER_GUIDE.md`: exact deployment, monitoring, recovery, and stop rules.

The implementation may import the already-tested pure functions from `src/itber_geometry.py`,
`src/itber_metrics.py`, and `src/itber_loss.py`. It must not modify those files or import
`ITBERRefiner`, `ITBERRecordingDecoder`, `trajectory_state`, or any I-TBER protocol/publication class.

### Task 1: Freeze the independent IBER-BE protocol

**Files:**
- Create: `src/iber_protocol.py`
- Create: `tests/test_iber_protocol.py`

- [ ] **Step 1: Write the failing protocol tests**

Add tests that require exact constants and reject I-TBER identities:

```python
from src.iber_protocol import (
    DESIGN_VERSION,
    PROBE_EPOCHS,
    SCREEN_EPOCHS,
    SCREEN_TRAIN_COUNT,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    execution_environment,
    validate_screen_contract,
)


def test_iber_protocol_is_independent_and_frozen():
    assert DESIGN_VERSION == "iber-be-v1.0"
    assert PROBE_EPOCHS == 12
    assert SCREEN_EPOCHS == 30
    assert SCREEN_TRAIN_COUNT == 647
    assert EXPECTED_BASELINE_SHA256.startswith("54CE6028")
    assert EXPECTED_DATASET_SHA256.startswith("FD92E9FF")
    assert EXPECTED_SUBSET_SHA256.startswith("52660F55")
    assert execution_environment()["driver"] == "550.142"
    assert execution_environment()["reported_memory_mib"] == 24564


def test_screen_contract_rejects_scientific_overrides():
    valid = validate_screen_contract({
        "seed": 0, "epochs": 30, "imgsz": 640, "batch": 8,
        "workers": 8, "amp_scale": 128.0, "mosaic": 1.0,
        "close_mosaic": 10, "max_det": 300, "nms": False,
    })
    assert valid["status"] == "passed_with_runtime_amendment"
    invalid = dict(valid["contract"], epochs=29)
    assert validate_screen_contract(invalid)["status"] == "engineering_invalid"
```

- [ ] **Step 2: Run the protocol tests and verify RED**

Run: `python -m pytest tests/test_iber_protocol.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.iber_protocol'`.

- [ ] **Step 3: Implement the protocol authority**

Define the exact public surface:

```python
DESIGN_VERSION = "iber-be-v1.0"
PROBES = frozenset(("b0", "b1", "b2", "b3"))
PROBE_EPOCHS = 12
SCREEN_EPOCHS = 30
SCREEN_TRAIN_COUNT = 647
SCREEN_VAL_COUNT = 548
PRIVATE_SEED = 10_000
PRIVATE_OPTIMIZER = {
    "name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
    "betas": (0.9, 0.999), "clip": 10.0,
}
EXPECTED_BASELINE_SHA256 = "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
EXPECTED_DATASET_SHA256 = "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
EXPECTED_SUBSET_SHA256 = "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
```

Reuse current execution-environment and runtime-amendment values by copying them into a new
canonical IBER payload; compute an IBER protocol SHA256 over canonical JSON. Implement
`execution_environment()`, `validate_screen_contract()`, `module_state_sha256()`,
`file_sha256()`, and `write_immutable_report()` without accepting CLI threshold overrides.

- [ ] **Step 4: Run protocol tests and verify GREEN**

Run: `python -m pytest tests/test_iber_protocol.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the protocol**

```bash
git add src/iber_protocol.py tests/test_iber_protocol.py
git commit -m "feat: freeze IBER-BE protocol"
```

### Task 2: Implement exact dual-resolution boundary sampling

**Files:**
- Create: `src/iber_sampling.py`
- Create: `tests/test_iber_sampling.py`

- [ ] **Step 1: Write failing grid and evidence tests**

Cover left/top/right/bottom normal orientation, near/far radii, exact output shapes,
`align_corners=False`, border padding, gradients, and no full-image learned operator:

```python
def test_rgb_boundary_evidence_has_exact_shape_and_one_sided_directions():
    image = torch.zeros(1, 3, 640, 640)
    image[:, :, :, 320:] = 1.0
    # The left edge is exactly on the synthetic x=0.5 intensity boundary.
    boxes = torch.tensor([[[0.625, 0.5, 0.25, 0.25]]])
    evidence = sample_rgb_boundary_evidence(image, boxes, image_size=640)
    assert evidence.shape == (1, 1, 4, 15)
    assert torch.isfinite(evidence).all()
    assert evidence[0, 0, 0, 3:6].abs().sum() > 0


def test_rgb_radii_follow_frozen_formula():
    boxes = torch.tensor([[[0.5, 0.5, 10 / 640, 20 / 640]]])
    near, far = rgb_normal_radii(boxes, image_size=640)
    torch.testing.assert_close(near, torch.tensor([[1 / 640]]))
    torch.testing.assert_close(far, torch.tensor([[2 / 640]]))
```

Also inspect the `IBERRefiner.rgb_encoder` subtree and the sampling source so the RGB path contains
no `Conv2d`, deformable attention, learned offsets, image pyramid, or full-image learned operator.

- [ ] **Step 2: Run sampling tests and verify RED**

Run: `python -m pytest tests/test_iber_sampling.py -q`

Expected: import failure for `src.iber_sampling`.

- [ ] **Step 3: Implement sampling functions**

Create these functions with detached boxes and FP32 grid construction:

```python
def rgb_normal_radii(boxes: torch.Tensor, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = boxes[..., 2:].unbind(-1)
    minimum = torch.minimum(width, height)
    near = (0.08 * minimum).clamp(1 / image_size, 4 / image_size)
    far = (0.20 * minimum).clamp(2 / image_size, 8 / image_size)
    return near, far


def sample_rgb_boundary_evidence(
    images: torch.Tensor, boxes: torch.Tensor, *, image_size: int
) -> torch.Tensor:
    """Return [edge, near outside→edge/edge→inside, far outside→edge/edge→inside] as [B,Q,4,15]."""


def sample_f3_boundary_evidence(
    features: torch.Tensor, boxes: torch.Tensor, *, image_size: int
) -> torch.Tensor:
    """Return [edge, outside→edge, edge→inside] F3 evidence as [B,Q,4,96]."""
```

Use three along-edge positions `(0.25, 0.50, 0.75)`, `grid_sample(...,
mode="bilinear", padding_mode="border", align_corners=False)`, and explicit outside/inside
orientation for all four edges. Do not clamp boxes before zero-correction identity handling.

- [ ] **Step 4: Run sampling tests and existing geometry regressions**

Run: `python -m pytest tests/test_iber_sampling.py tests/test_itber_geometry.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit sampling**

```bash
git add src/iber_sampling.py tests/test_iber_sampling.py
git commit -m "feat: add dual-resolution boundary sampling"
```

### Task 3: Build the equal-capacity B0–B3 private head

**Files:**
- Create: `src/iber_head.py`
- Create: `tests/test_iber_head.py`

- [ ] **Step 1: Write failing head tests**

Tests must instantiate every Probe with the same seed and assert:

```python
def test_b0_b3_have_equal_capacity_and_initial_state():
    heads = [IBERRefiner(256, 512, private_seed=10_000, probe=p) for p in PROBES]
    assert len({sum(x.numel() for x in h.parameters()) for h in heads}) == 1
    assert len({module_state_sha256(h) for h in heads}) == 1


def test_zero_outputs_preserve_out_of_image_stock_boxes_exactly():
    output = make_head("b3")(**out_of_image_evidence())
    torch.testing.assert_close(output.refined_boxes, output.stock_boxes, rtol=0, atol=0)


def test_iber_head_contains_no_trajectory_state():
    head = make_head("b3")
    names = {name for name, _ in head.named_modules()}
    assert all("trajectory" not in name.lower() for name in names)
    assert "box_l1" not in inspect.signature(head.forward).parameters
    assert "box_l2" not in inspect.signature(head.forward).parameters
```

Add B0/B1/B2/B3 masking tests, nonzero RGB/F3 gradient tests, and finite tiny/border-box tests.

- [ ] **Step 2: Run head tests and verify RED**

Run: `python -m pytest tests/test_iber_head.py -q`

Expected: import failure for `src.iber_head`.

- [ ] **Step 3: Implement `IBEROutput` and `IBERRefiner`**

Expose this exact forward signature:

```python
def forward(
    self,
    hidden: torch.Tensor,
    stock_boxes: torch.Tensor,
    stock_scores: torch.Tensor,
    f3: torch.Tensor,
    image_rgb: torch.Tensor,
) -> IBEROutput:
    ...
```

`IBEROutput` must expose `stock_boxes`, `refined_boxes`, `stock_edges`, `refined_edges`,
`gate_logits`, `gates`, `residual_raw`, `residuals`, `effective_correction`, `quality`, `entropy`,
`f3_boundary_features`, `rgb_boundary_features`, `boundary_features`, and the four separate
`base_gate_raw`, `boundary_gate_raw`, `base_residual_raw`, `boundary_residual_raw` tensors needed for
ablation and activity reports. Detach all five detector-owned inputs at the start of `forward()`.

Implement `PROBES = frozenset(("b0", "b1", "b2", "b3"))` and these exact routes:

```text
query = LayerNorm(256) -> Linear(256,64) -> SiLU
geometry = Linear(8,16) -> SiLU
edge = Embedding(4,8)
base = Linear(64+16+8,64) -> SiLU -> Linear(64,64) -> SiLU

F3 = Conv2d(C3,32,1) -> sparse sample [96] -> Linear(96,32) -> SiLU
RGB = sparse sample [15] -> Linear(15,16) -> LayerNorm(16) -> SiLU
boundary = Linear(32+16,32) -> SiLU
boundary_condition = concat(boundary, query[..., :32], edge)
boundary_trunk = Linear(32+32+8,64) -> SiLU -> Linear(64,64) -> SiLU
```

The first 32 dimensions of the common 64-d query projection are the parameter-free 32-d boundary
condition; do not introduce another query layer. Zero-initialize all four final base/boundary
gate/residual `Linear(64,1)` layers. Use `rho=0.05` and the fixed `apply_edge_update()`
zero-correction identity behavior. Expose `stock`, `refined`, and `boundary_off` output modes;
`boundary_off` retains the base route and disables only the boundary outputs.

Mask rules must be exact:

```python
use_f3 = self.probe in {"b1", "b3"}
use_rgb = self.probe in {"b2", "b3"}
f3_raw = sample_f3_boundary_evidence(...) if use_f3 else torch.zeros(..., 96)
rgb_raw = sample_rgb_boundary_evidence(...) if use_rgb else torch.zeros(..., 15)
```

- [ ] **Step 4: Run head/loss/geometry tests**

Run: `python -m pytest tests/test_iber_head.py tests/test_itber_loss.py tests/test_itber_geometry.py -q`

Expected: all tests pass; the reused private loss accepts `IBEROutput` structurally.

- [ ] **Step 5: Commit the head**

```bash
git add src/iber_head.py tests/test_iber_head.py
git commit -m "feat: add IBER-BE private head"
```

### Task 4: Record only final stock evidence from frozen RT-DETR

**Files:**
- Create: `src/rtdetr_iber.py`
- Create: `tests/test_rtdetr_iber.py`

- [ ] **Step 1: Write failing adapter tests**

Require the recording decoder to expose only `last_hidden`, `last_stock_scores`, and
`last_stock_boxes`; assert detector gradients remain `None` after private backward and that
the output mode switches stock/refined without changing scores.

```python
def test_recording_decoder_has_no_trajectory_buffers():
    decoder = IBERRecordingDecoder.from_stock(stock_decoder())
    assert not hasattr(decoder, "last_three_boxes")
    assert not hasattr(decoder, "trajectory")


def test_private_backward_never_updates_detector():
    adapter = build_adapter()
    losses = adapter.training_step(synthetic_batch())
    losses.total.backward()
    assert all(parameter.grad is None for parameter in adapter.detector.parameters())
    assert any(parameter.grad is not None for parameter in adapter.refiner.parameters())
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest tests/test_rtdetr_iber.py -q`

Expected: import failure for `src.rtdetr_iber`.

- [ ] **Step 3: Implement the recording decoder and adapter**

Use `src/rtdetr_itber.py:75` as the local parity reference for the exact Ultralytics 8.4.90 decoder
equations, but delete all trajectory accumulation and record only the final normal 300 queries.
`FrozenIBERAdapter.forward_evidence(image)` must call the frozen detector once, capture head input
F3 via a forward pre-hook, and pass the same detached `image` tensor to `IBERRefiner`. Use the
stock matcher once in `training_step()` and call the existing isolated private loss.

- [ ] **Step 4: Run adapter and stock decoder parity tests**

Run: `python -m pytest tests/test_rtdetr_iber.py tests/test_rtdetr_itber.py -q`

Expected: all tests pass and old I-TBER behavior remains unchanged.

- [ ] **Step 5: Commit the adapter**

```bash
git add src/rtdetr_iber.py tests/test_rtdetr_iber.py
git commit -m "feat: isolate RT-DETR evidence for IBER-BE"
```

### Task 5: Add the trajectory-free immutable Probe cache

**Files:**
- Create: `src/iber_cache.py`
- Create: `scripts/cache_iber_evidence.py`
- Create: `tests/test_iber_cache.py`

- [ ] **Step 1: Write failing cache tests**

Lock the record schema to:

```python
REQUIRED_RECORD_TENSORS = (
    "hidden", "stock_boxes", "stock_scores", "f3", "image_rgb",
    "target_edges", "match_source", "match_target",
)
```

Assert `image_rgb.dtype == torch.uint8`, shape `[3,640,640]`, no `box_l1/box_l2`, no train/val
overlap, contiguous indices, complete manifest published last, per-shard bytes/SHA256, and exact
authority including source commit and runtime amendment.

- [ ] **Step 2: Run cache tests and verify RED**

Run: `python -m pytest tests/test_iber_cache.py -q`

Expected: import failure for `src.iber_cache`.

- [ ] **Step 3: Implement cache read/write and builder CLI**

Use new `CACHE_FORMAT_VERSION = 1`, `DESIGN_VERSION = "iber-be-v1.0"`, shard size 16, and safe
relative paths. The builder must recreate deterministic 640 letterbox inputs, use the fixed hashed
647 train subset and all 548 val images, run the frozen detector once per batch, and store RGB as:

```python
image_rgb = images[local_index].mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
```

On load, verify every shard before exposing records; Probe conversion is exactly
`record["image_rgb"].float().div(255)`.

- [ ] **Step 4: Run cache tests**

Run: `python -m pytest tests/test_iber_cache.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the cache**

```bash
git add src/iber_cache.py scripts/cache_iber_evidence.py tests/test_iber_cache.py
git commit -m "feat: cache immutable IBER-BE evidence"
```

### Task 6: Implement B0–B3 Probe and frozen Gate-1

**Files:**
- Create: `src/iber_probe.py`
- Create: `scripts/run_iber_probe.py`
- Create: `tests/test_iber_probe.py`

- [ ] **Step 1: Write failing Gate-1 tests**

Construct synthetic reports and require all frozen conditions:

```python
def test_gate1_requires_full_b3_increment():
    reports = passing_reports()
    decision = evaluate_gate1(reports)
    assert decision["status"] == "passed"
    assert all(decision["conditions"].values())


@pytest.mark.parametrize("condition", [
    "edge_over_b0", "edge_over_b1", "matched_iou", "tiny_direction",
    "small_direction", "b3_best_primary", "finite_activity",
])
def test_each_gate1_condition_is_mandatory(condition):
    reports = reports_failing_only(condition)
    assert evaluate_gate1(reports)["status"] == "scientific_failed"
```

Also reject wrong epochs, unequal capacity/initialization, absent arms, nonfinite values, and best
epoch substitution.

- [ ] **Step 2: Run Probe tests and verify RED**

Run: `python -m pytest tests/test_iber_probe.py -q`

Expected: import failure for `src.iber_probe`.

- [ ] **Step 3: Implement Probe training and decision**

Fresh-train each arm for exactly 12 epochs with private seed 10000, batch 8, AdamW `1e-3`,
weight decay `1e-4`, betas `(0.9,0.999)`, clip 10, fixed AMP scale 128. Evaluate only epoch12.
Implement conditions without rounding:

```python
conditions = {
    "edge_over_b0": b3["edge_mae"] <= b0["edge_mae"] * 0.95,
    "edge_over_b1": b3["edge_mae"] <= b1["edge_mae"] * 0.985,
    "matched_iou": b3["matched_iou_delta"] >= 0.005,
    "tiny_direction": b3["tiny_direction_accuracy"] - b0["tiny_direction_accuracy"] >= 0.03,
    "small_direction": b3["small_direction_accuracy"] - b0["small_direction_accuracy"] >= 0.03,
    "b3_best_primary": b3["edge_mae"] == min(v["edge_mae"] for v in metrics.values())
        and b3["matched_iou_delta"] == max(v["matched_iou_delta"] for v in metrics.values()),
    "finite_activity": all_finite(b3)
        and b3["gradient_rms"] > 0.0
        and b3["gate_mean"] > 1e-4
        and 1e-3 < b3["gate_p95"] < 0.999
        and b3["residual_rms"] > 1e-4,
}
```

Write immutable per-arm reports/checkpoints and one Gate decision. Exit code 0 for pass, 2 for
scientific failure, 1 for engineering invalid.

- [ ] **Step 4: Run Probe/cache/head tests**

Run: `python -m pytest tests/test_iber_probe.py tests/test_iber_cache.py tests/test_iber_head.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Probe**

```bash
git add src/iber_probe.py scripts/run_iber_probe.py tests/test_iber_probe.py
git commit -m "feat: add IBER-BE Gate-1 Probe"
```

### Task 7: Add Gate-0 and stock authority

**Files:**
- Create: `scripts/run_iber_canary.py`
- Create: `scripts/evaluate_iber_stock.py`
- Create: `tests/test_iber_canary.py`
- Create: `tests/test_iber_stock_evaluation.py`

- [ ] **Step 1: Write failing canary/stock tests**

Require zero-init stock/refined exact equality, finite nonzero private gradients for F3 and RGB,
detector gradients `None`, no denoising queries in private loss, same stock matcher indices,
checkpoint stock/refined switching, three exact stock repeats, and runtime-amended authority.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_iber_canary.py tests/test_iber_stock_evaluation.py -q`

Expected: missing script/module imports fail.

- [ ] **Step 3: Implement Gate-0 and stock evaluator**

Adapt the proven I-TBER orchestration but instantiate `FrozenIBERAdapter`, report
`design_version="iber-be-v1.0"`, and refuse I-TBER paths/identities. Stock evaluation constants
remain `imgsz=640,batch=8,workers=8,conf=0.001,max_det=300,nms=False,half=False,repeats=3`.

- [ ] **Step 4: Run canary/stock tests**

Run: `python -m pytest tests/test_iber_canary.py tests/test_iber_stock_evaluation.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Gate-0**

```bash
git add scripts/run_iber_canary.py scripts/evaluate_iber_stock.py tests/test_iber_canary.py tests/test_iber_stock_evaluation.py
git commit -m "feat: add IBER-BE engineering authorities"
```

### Task 8: Implement fixed 30-epoch private training and resume

**Files:**
- Create: `scripts/train_iber.py`
- Create: `tests/test_iber_training.py`
- Create: `tests/test_iber_training_cli.py`

- [ ] **Step 1: Write failing training-contract tests**

Assert exact train647/val548, seed0, epochs30, batch8/workers8/imgsz640, fixed AMP128, AdamW private
optimizer, detector `eval/no_grad`, on-the-fly RGB/F3 evidence, common VisDrone augmentation including
`mosaic=1.0` and `close_mosaic=10`, save period 1, and no CLI scientific overrides.

Checkpoint tests must require:

```python
REQUIRED_CHECKPOINT_KEYS = {
    "format_version", "design_version", "stage", "probe", "seed", "epoch",
    "baseline_sha256", "dataset_sha256", "subset_sha256", "source_commit",
    "runtime_amendment_sha256", "protocol_sha256", "refiner", "optimizer",
    "scaler", "rng",
}
```

- [ ] **Step 2: Run training tests and verify RED**

Run: `python -m pytest tests/test_iber_training.py tests/test_iber_training_cli.py -q`

Expected: missing `scripts.train_iber` import fails.

- [ ] **Step 3: Implement exact training and resume**

Fresh screen initializes B3 at private seed10000. Each completed epoch must atomically write the
checkpoint, results row, diagnostic row, detector fingerprint, and optimizer evidence before invoking
publication. Resume accepts only the highest contiguous remotely verified epoch and restores optimizer,
scaler, RNG, completed epoch, and private state. Epoch30 must stop after publication verification.

- [ ] **Step 4: Run training tests**

Run: `python -m pytest tests/test_iber_training.py tests/test_iber_training_cli.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit training**

```bash
git add scripts/train_iber.py tests/test_iber_training.py tests/test_iber_training_cli.py
git commit -m "feat: train resumable IBER-BE screen"
```

### Task 9: Implement exact evaluation and Gate-2 decision

**Files:**
- Create: `src/iber_evaluation.py`
- Create: `scripts/evaluate_iber.py`
- Create: `tests/test_iber_evaluation.py`

- [ ] **Step 1: Write failing evaluator tests**

Require AP/AP50/AP75/AP-tiny/AP-small, edge MAE, matched IoU deltas, improvement/degradation counts,
matched/unmatched correction RMS, F3/RGB embedding RMS, gate/residual activity, detector SHA, and
three-repeat exactness. Gate-2 tests must independently fail each of: mAP `+0.0020`, AP75 `+0.0030`,
AP50 `-0.0005`, tiny-or-small positive, improved count greater than degraded count, unmatched RMS at
most 25% of matched RMS, finite/noncollapsed activity, exact three-repeat equality, and last-five
refined mean greater than the stock mean. They must also reject any checkpoint other than epoch30.

- [ ] **Step 2: Run evaluator tests and verify RED**

Run: `python -m pytest tests/test_iber_evaluation.py -q`

Expected: import failure for `src.iber_evaluation`.

- [ ] **Step 3: Implement evaluator and decision**

Use the same Ultralytics validator preprocessing/class mapping for stock/refined, identical stock
scores, `conf=0.001`, max_det300, NMS disabled, and no metric rounding before comparisons. Freeze the
epoch-30 decision exactly as:

```python
conditions = {
    "map": refined["map"] - stock["map"] >= 0.0020,
    "ap75": refined["ap75"] - stock["ap75"] >= 0.0030,
    "ap50": refined["ap50"] - stock["ap50"] >= -0.0005,
    "tiny_or_small": (refined["ap_tiny"] - stock["ap_tiny"] > 0.0)
        or (refined["ap_small"] - stock["ap_small"] > 0.0),
    "matched_counts": diagnostics["matched_improved"] > diagnostics["matched_degraded"],
    "unmatched_rms": diagnostics["unmatched_correction_rms"]
        <= 0.25 * diagnostics["matched_correction_rms"],
    "activity": finite_noncollapsed_activity(diagnostics),
    "repeatability": exact_three_repeat_equality(repeats),
    "last5": mean(last5_refined_map) > mean(last5_stock_map),
}
```

Report `passed`, `scientific_failed`, or `engineering_invalid`; never select a best epoch in place of
epoch30.

- [ ] **Step 4: Run evaluator tests**

Run: `python -m pytest tests/test_iber_evaluation.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit evaluation**

```bash
git add src/iber_evaluation.py scripts/evaluate_iber.py tests/test_iber_evaluation.py
git commit -m "feat: evaluate IBER-BE same-checkpoint gains"
```

### Task 10: Add transactional GitHub publication and restore

**Files:**
- Create: `src/iber_publication.py`
- Create: `scripts/publish_iber_epoch.py`
- Create: `scripts/restore_iber_checkpoint.py`
- Create: `tests/test_iber_publication.py`
- Create: `tests/test_iber_restore.py`

- [ ] **Step 1: Write failing publication/restore tests**

Lock `design_version="iber-be-v1.0"`, stage `screen`, probe `b3`, seed0, contiguous epochs1–30,
checkpoint/manifest pair upload, remote byte/SHA verification, append-only ledger, result-branch commit
verification, rolling retention3, credential redaction, and atomic restore.

- [ ] **Step 2: Run publication tests and verify RED**

Run: `python -m pytest tests/test_iber_publication.py tests/test_iber_restore.py -q`

Expected: missing IBER publication modules fail.

- [ ] **Step 3: Implement publication and restore**

Use tag `iber-be-v1-rtdetr-l-live`, asset prefix `iber-be-v1.0-screen-seed0-b3`, a dedicated
`iber-be-v1-results` branch, and a dedicated server results checkout. Read the token only from the
mode-600 token file; never serialize it into configs, logs, exceptions, or Git remotes.

- [ ] **Step 4: Run publication/restore tests**

Run: `python -m pytest tests/test_iber_publication.py tests/test_iber_restore.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit publication**

```bash
git add src/iber_publication.py scripts/publish_iber_epoch.py scripts/restore_iber_checkpoint.py tests/test_iber_publication.py tests/test_iber_restore.py
git commit -m "feat: publish and restore IBER-BE checkpoints"
```

### Task 11: Measure inference overhead honestly

**Files:**
- Create: `scripts/benchmark_iber.py`
- Create: `tests/test_iber_benchmark.py`

- [ ] **Step 1: Write failing benchmark tests**

Require positive stock denominator, exact private parameter accounting, FP16 `[1,3,640,640]`,
50 warmups, 200 measured iterations, CUDA synchronization, alternating stock/refined order, and
reported driver/GPU/memory/runtime amendment. Targets remain params/GFLOPs `<1%`, latency `<3%`.

- [ ] **Step 2: Run benchmark tests and verify RED**

Run: `python -m pytest tests/test_iber_benchmark.py -q`

Expected: benchmark module import fails.

- [ ] **Step 3: Implement benchmark CLI**

Count only new inference parameters and operations; include RGB and F3 `grid_sample`, private encoders,
and gate/residual heads. Measure stock/refined in one process with alternating order and report raw
samples, medians, percentiles, and percentage deltas.

- [ ] **Step 4: Run benchmark tests**

Run: `python -m pytest tests/test_iber_benchmark.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit benchmark**

```bash
git add scripts/benchmark_iber.py tests/test_iber_benchmark.py
git commit -m "feat: benchmark IBER-BE overhead"
```

### Task 12: Supervise the scientific state machine

**Files:**
- Create: `scripts/run_iber_pipeline.py`
- Create: `tests/test_iber_pipeline.py`

- [ ] **Step 1: Write failing state-machine tests**

Cover exact order:

```text
authority -> gate0 -> stock_authority -> cache -> probe -> screen30 -> screen_decision
```

Require engineering failures to stop for repair, scientific Gate-1 failure to stop before screen30,
scientific Gate-2 failure to stop after evaluation, verified resume of incomplete screen epochs, atomic
state/history JSON, PID/process-group tracking, and no manual phase skipping.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `python -m pytest tests/test_iber_pipeline.py -q`

Expected: missing pipeline module fails.

- [ ] **Step 3: Implement the pipeline**

Use terminal phases `engineering_invalid`, `scientific_failed`, and `screen_complete`. Subprocess logs
must contain exact commands but no credentials. The supervisor may launch screen30 only when authority,
Gate-0, stock authority, cache, and Gate-1 reports all validate against the same source commit.

- [ ] **Step 4: Run pipeline tests**

Run: `python -m pytest tests/test_iber_pipeline.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit pipeline**

```bash
git add scripts/run_iber_pipeline.py tests/test_iber_pipeline.py
git commit -m "feat: supervise IBER-BE scientific pipeline"
```

### Task 13: Add bare-server deployment and operator guide

**Files:**
- Create: `scripts/audit_iber_deployment.py`
- Create: `deploy/iber/__init__.py`
- Create: `deploy/iber/bootstrap_ubuntu.sh`
- Create: `deploy/iber/verify_host.sh`
- Create: `deploy/iber/build_wheelhouse.sh`
- Create: `deploy/iber/verify_bundle.py`
- Create: `deploy/iber/artifact-manifest.template.json`
- Create: `deploy/iber/publication-screen.template.json`
- Create: `docs/IBER_BE_SERVER_GUIDE.md`
- Create: `tests/test_iber_deploy_scripts.py`
- Create: `tests/test_iber_deployment_audit.py`

- [ ] **Step 1: Write failing deployment tests**

Require pinned Python/torch/torchvision/Ultralytics/CUDA, runtime-amended driver, GPU identity, data and
baseline hashes, source SHA, config mode 600, no embedded credentials, required scripts, new IBER run
roots, and rejection of I-TBER result/release names.

- [ ] **Step 2: Run deployment tests and verify RED**

Run: `python -m pytest tests/test_iber_deploy_scripts.py tests/test_iber_deployment_audit.py -q`

Expected: missing deployment files fail.

- [ ] **Step 3: Implement deployment assets and guide**

Document mirror-first bootstrap, exact host verification, immutable source checkout, cache/run/results
paths, launch command, phase/status inspection, GPU inspection, GitHub publication verification, resume,
engineering repair, scientific stop, and explicit prohibition on bypassing Gate-1.

- [ ] **Step 4: Run deployment tests**

Run: `python -m pytest tests/test_iber_deploy_scripts.py tests/test_iber_deployment_audit.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit deployment assets**

```bash
git add scripts/audit_iber_deployment.py deploy/iber docs/IBER_BE_SERVER_GUIDE.md tests/test_iber_deploy_scripts.py tests/test_iber_deployment_audit.py
git commit -m "docs: operationalize IBER-BE deployment"
```

### Task 14: Full verification, push, deployment, and launch

**Files:**
- Modify only if a failing regression identifies an IBER defect; do not alter scientific thresholds.

- [ ] **Step 1: Run focused IBER tests**

Run: `python -m pytest tests -q -k iber`

Expected: all IBER tests pass.

- [ ] **Step 2: Run the whole repository suite**

Run: `python -m pytest -q`

Expected: zero failures, including all existing I-TBER/LPR/BTD-SE/IOQC/VSF regressions.

- [ ] **Step 3: Verify source cleanliness and commit identities**

Run:

```bash
git status --short
git log --oneline --decorate -15
git diff main...HEAD --check
```

Expected: clean worktree, no whitespace errors, only intentional IBER/spec/plan changes.

- [ ] **Step 4: Push and verify the exact source SHA**

Push `codex/iber-be` without force, then compare local `git rev-parse HEAD` with GitHub branch SHA.
Do not deploy a source tree until both 40-character SHAs are identical.

- [ ] **Step 5: Deploy an immutable server checkout**

Use `/data/uav/source/uav-detection-baselines-<shortsha>`. Reuse the verified environment, dataset,
baseline checkpoint, runtime amendment, and mode-600 token file. Run server focused tests and
`verify_host.sh`; both must pass before creating a run root.

- [ ] **Step 6: Launch a new supervised run**

Create unique paths:

```text
/data/uav/cache/iber-be-v1-<shortsha>
/data/uav/runs/iber-be-v1/<shortsha>-seed0-amended
/data/uav/results/iber-be-v1-<shortsha>
/data/uav/logs/iber-be-v1-<shortsha>-pipeline.log
```

Start `scripts/run_iber_pipeline.py` under `nohup`, write `pipeline.pid`, and verify the process is alive
before reporting launch success. Never reuse or delete I-TBER run/cache paths.

- [ ] **Step 7: Monitor conditionally through Gate-1**

Poll `pipeline-state.json`, phase logs, GPU state, authority, stock report, cache manifest, B0–B3 reports,
and Gate-1 decision. Do not launch a second supervisor while the PID is alive. If Gate-1 is
`scientific_failed`, archive and push evidence and stop. If passed, confirm the supervisor starts fresh
screen30 and that epoch1 is remotely published before allowing continued unattended training.

- [ ] **Step 8: Verify the 30-epoch terminal report**

At completion, verify 30 contiguous result/diagnostic/ledger entries, epoch30 checkpoint identity,
three exact evaluations, Gate-2 decision, source/data/baseline hashes, and no active orphan process.
Archive both passed and failed results on `iber-be-v1-results` without changing the source branch.

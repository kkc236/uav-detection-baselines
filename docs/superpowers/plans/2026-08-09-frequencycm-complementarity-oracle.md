# FrequencyCM Complementarity Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic, non-trainable oracle that measures candidate complementarity between the frozen FDR epoch-100 detector and the frozen FrequencyCM epoch-100 detector.

**Architecture:** Reuse the existing RT-DETR raw-query extraction, immutable cache conventions, and independent detector evaluator. Add one pure mathematical module for same-class one-to-one assignment, coverage, oracle arms, selector utility, and frozen decision logic; add one source-bound CLI that caches both checkpoints, reconstructs stock metrics, runs the oracle once, and writes hashed JSON/CSV/Markdown evidence.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Torchvision 0.20.1, Ultralytics 8.4.90, SciPy Hungarian assignment, pytest, RTX 4090.

---

## File structure

- Create `src/rtdetr_complementarity_oracle.py`: pure tensor/record validation, one-to-one assignment, coverage aggregation, oracle arm construction, image selector, decision bands, immutable cache/report helpers.
- Create `scripts/run_rtdetr_complementarity_oracle.py`: authority verification, exact FDR/FrequencyCM model loading, raw-query caching, stock reconstruction, evaluator calls, report and SHA manifest generation.
- Create `tests/test_rtdetr_complementarity_oracle.py`: mathematical, cache, determinism, duplicate-control, scale/class, empty-input, and decision tests.
- Create `tests/test_run_rtdetr_complementarity_oracle.py`: CLI/source-lock and synthetic end-to-end report tests.
- Create at runtime `reports/frequencycm-complementarity-oracle-v1/`: immutable scientific evidence; do not commit generated binary caches.

### Task 1: Pure one-to-one assignment and coverage mathematics

**Files:**
- Create: `tests/test_rtdetr_complementarity_oracle.py`
- Create: `src/rtdetr_complementarity_oracle.py`

- [ ] **Step 1: Write failing tests for validated same-class IoU and deterministic assignment**

```python
from __future__ import annotations

import torch

from src.rtdetr_complementarity_oracle import (
    candidate_iou_matrix,
    one_to_one_same_class_assignment,
)


def test_assignment_is_same_class_one_to_one_and_maximizes_iou() -> None:
    predictions = torch.tensor(
        [[0.50, 0.50, 0.40, 0.40], [0.52, 0.50, 0.40, 0.40], [0.20, 0.20, 0.10, 0.10]],
        dtype=torch.float32,
    )
    classes = torch.tensor([0, 0, 1])
    targets = torch.tensor(
        [[0.50, 0.50, 0.40, 0.40], [0.20, 0.20, 0.10, 0.10]],
        dtype=torch.float32,
    )
    target_classes = torch.tensor([0, 1])

    matrix = candidate_iou_matrix(predictions, targets)
    assignment = one_to_one_same_class_assignment(
        matrix, classes, target_classes
    )

    assert assignment.prediction_indices.tolist() == [0, 2]
    assert assignment.target_indices.tolist() == [0, 1]
    assert torch.allclose(assignment.ious, torch.ones(2))


def test_assignment_returns_empty_tensors_for_empty_targets() -> None:
    result = one_to_one_same_class_assignment(
        torch.empty((3, 0)), torch.tensor([0, 1, 2]), torch.empty(0, dtype=torch.long)
    )
    assert result.ious.numel() == 0
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `pytest -q tests/test_rtdetr_complementarity_oracle.py`

Expected: collection fails because `src.rtdetr_complementarity_oracle` does not exist.

- [ ] **Step 3: Implement strict tensor validation, normalized cxcywh IoU, and class-wise Hungarian assignment**

```python
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Assignment:
    prediction_indices: torch.Tensor
    target_indices: torch.Tensor
    ious: torch.Tensor


def candidate_iou_matrix(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    for name, value in (("boxes", boxes), ("targets", targets)):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 4:
            raise ValueError(f"{name} must have shape [N,4]")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point")
        if not ((value >= 0) & (value <= 1)).all():
            raise ValueError(f"{name} must be normalized to [0,1]")
    if boxes.device != targets.device:
        raise ValueError("boxes and targets must share a device")
    dtype = torch.float64 if torch.float64 in (boxes.dtype, targets.dtype) else torch.float32
    boxes = boxes.detach().to(dtype)
    targets = targets.detach().to(dtype)
    box_lo, box_hi = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
    target_lo = targets[:, :2] - targets[:, 2:] / 2
    target_hi = targets[:, :2] + targets[:, 2:] / 2
    inter = (
        torch.minimum(box_hi[:, None], target_hi[None])
        - torch.maximum(box_lo[:, None], target_lo[None])
    ).clamp_min(0).prod(-1)
    union = boxes[:, 2:].prod(-1)[:, None] + targets[:, 2:].prod(-1)[None] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union)).clamp(0, 1)


def one_to_one_same_class_assignment(
    iou: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_classes: torch.Tensor,
) -> Assignment:
    if iou.ndim != 2 or iou.shape != (prediction_classes.numel(), target_classes.numel()):
        raise ValueError("IoU shape must equal prediction-by-target counts")
    if not torch.isfinite(iou).all() or not ((iou >= 0) & (iou <= 1)).all():
        raise ValueError("IoU must be finite and in [0,1]")
    if prediction_classes.dtype != torch.long or target_classes.dtype != torch.long:
        raise TypeError("classes must use torch.long")
    if iou.device != prediction_classes.device or iou.device != target_classes.device:
        raise ValueError("assignment tensors must share a device")
    selected: list[tuple[int, int, float]] = []
    classes = sorted(set(prediction_classes.tolist()) & set(target_classes.tolist()))
    for class_id in classes:
        pred = torch.where(prediction_classes == class_id)[0]
        target = torch.where(target_classes == class_id)[0]
        block = iou[pred][:, target].detach().cpu().double().numpy()
        row_bias = np.arange(block.shape[0], dtype=np.float64)[:, None] * 1e-12
        col_bias = np.arange(block.shape[1], dtype=np.float64)[None, :] * 1e-15
        rows, cols = linear_sum_assignment(-block + row_bias + col_bias)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            value = float(block[row, col])
            if value > 0:
                selected.append((int(pred[row]), int(target[col]), value))
    selected.sort(key=lambda item: (item[1], item[0]))
    device = iou.device
    return Assignment(
        prediction_indices=torch.tensor([item[0] for item in selected], device=device, dtype=torch.long),
        target_indices=torch.tensor([item[1] for item in selected], device=device, dtype=torch.long),
        ious=torch.tensor([item[2] for item in selected], device=device, dtype=iou.dtype),
    )
```

Use CPU float64 costs for SciPy assignment and return tensors on the original device. Reject booleans, non-integer classes, out-of-range boxes, NaN/Inf, and shape/device mismatches.

- [ ] **Step 4: Add tests for ties, cross-class rejection, malformed tensors, empty predictions, and repeatability**

```python
def test_assignment_breaks_equal_iou_ties_by_prediction_index() -> None:
    iou = torch.tensor([[0.8], [0.8]])
    result = one_to_one_same_class_assignment(iou, torch.tensor([0, 0]), torch.tensor([0]))
    assert result.prediction_indices.tolist() == [0]


def test_assignment_rejects_nonfinite_iou() -> None:
    with pytest.raises(ValueError, match="finite"):
        one_to_one_same_class_assignment(
            torch.tensor([[float("nan")]]), torch.tensor([0]), torch.tensor([0])
        )
```

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest -q tests/test_rtdetr_complementarity_oracle.py`

Expected: all assignment tests pass.

Commit: `git add src/rtdetr_complementarity_oracle.py tests/test_rtdetr_complementarity_oracle.py && git commit -m "feat: add complementarity assignment core"`

### Task 2: Coverage, scale buckets, and oracle candidate arms

**Files:**
- Modify: `src/rtdetr_complementarity_oracle.py`
- Modify: `tests/test_rtdetr_complementarity_oracle.py`

- [ ] **Step 1: Write failing tests for coverage and scale buckets**

```python
from src.rtdetr_complementarity_oracle import coverage_summary, visdrone_size_bucket


def test_visdrone_size_bucket_uses_frozen_pixel_area_boundaries() -> None:
    assert visdrone_size_bucket(15.0, 15.0) == "tiny"
    assert visdrone_size_bucket(16.0, 16.0) == "small"
    assert visdrone_size_bucket(32.0, 32.0) == "medium"
    assert visdrone_size_bucket(96.0, 96.0) == "large"


def test_union_coverage_counts_only_new_same_class_targets() -> None:
    summary = coverage_summary(
        fdr_best_iou=torch.tensor([0.8, 0.2]),
        frequencycm_best_iou=torch.tensor([0.7, 0.6]),
        thresholds=(0.5, 0.75),
    )
    assert summary["iou50"]["fdr_only"] == 1
    assert summary["iou50"]["frequencycm_only"] == 1
    assert summary["iou50"]["both"] == 0
```

- [ ] **Step 2: Verify the new tests fail**

Run: `pytest -q tests/test_rtdetr_complementarity_oracle.py -k "coverage or size_bucket"`

Expected: missing functions.

- [ ] **Step 3: Implement frozen scale buckets and coverage aggregation**

Use original-image pixel area: tiny `<256`, small `[256,1024)`, medium `[1024,9216)`, large `>=9216`. Produce raw best-IoU coverage and deterministic one-to-one matched recall for IoU 0.50 and 0.75, globally, by scale, and by class.

- [ ] **Step 4: Write failing duplicate-control and oracle-arm tests**

```python
from src.rtdetr_complementarity_oracle import build_matched_quality_arm


def test_duplicated_detector_has_exactly_neutral_oracle_candidates() -> None:
    arm = build_matched_quality_arm(
        boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        probabilities=torch.tensor([[0.9, 0.1]]),
        source_ranks=torch.tensor([0]),
        target_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        target_classes=torch.tensor([0]),
        max_det=2,
    )
    duplicated = build_matched_quality_arm(
        boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]),
        probabilities=torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        source_ranks=torch.tensor([0, 1]),
        target_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        target_classes=torch.tensor([0]),
        max_det=2,
    )
    assert torch.equal(arm[:1], duplicated[:1])
    assert duplicated[1, 4].item() == 0.0
```

- [ ] **Step 5: Implement one-to-one utility assignment and deterministic flattened top-300**

Expand `[Q,C]` into query-class pairs without losing the shared query box. Assign only same-class candidates one-to-one to targets, use matched IoU as score, set all unassigned scores to zero, sort by `(-utility, source_rank, query_index, class_index)`, and return Ultralytics-style `[x,y,w,h,score,class]` predictions.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/test_rtdetr_complementarity_oracle.py`

Expected: all tests pass.

Commit: `git add src/rtdetr_complementarity_oracle.py tests/test_rtdetr_complementarity_oracle.py && git commit -m "feat: add complementarity oracle arms"`

### Task 3: Immutable paired cache and frozen decision logic

**Files:**
- Modify: `src/rtdetr_complementarity_oracle.py`
- Modify: `tests/test_rtdetr_complementarity_oracle.py`

- [ ] **Step 1: Write failing cache round-trip and corruption tests**

```python
from src.rtdetr_complementarity_oracle import load_paired_cache, write_paired_cache


def test_paired_cache_is_create_only_authority_bound_and_hashed(tmp_path: Path) -> None:
    authority = {"fdr_sha256": "a" * 64, "frequencycm_sha256": "b" * 64, "dataset_sha256": "c" * 64}
    records = [synthetic_record("000001")]
    manifest = write_paired_cache(tmp_path / "cache", records, authority)
    assert manifest["record_count"] == 1
    assert load_paired_cache(tmp_path / "cache", authority)[0]["image_id"] == "000001"
    with pytest.raises(FileExistsError):
        write_paired_cache(tmp_path / "cache", [synthetic_record("changed")], authority)


def test_paired_cache_rejects_corrupted_payload(tmp_path: Path) -> None:
    authority = valid_authority()
    write_paired_cache(tmp_path / "cache", [synthetic_record("000001")], authority)
    (tmp_path / "cache" / "records.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_paired_cache(tmp_path / "cache", authority)
```

- [ ] **Step 2: Implement canonical authority JSON and create-only cache**

The record schema contains `image_id`, `original_shape`, `fdr_boxes`, `fdr_logits`, `frequencycm_boxes`, `frequencycm_logits`, `target_boxes`, and `target_classes`. Require exactly 300 queries and 10 classes per detector. Store CPU float32 tensors, reject symlinks, verify hashes before deserialization, and compare every authority field exactly.

- [ ] **Step 3: Write and implement frozen Red/Yellow/Green decision tests**

```python
from src.rtdetr_complementarity_oracle import decide_complementarity


@pytest.mark.parametrize(
    ("map_delta", "recall_delta", "expected"),
    [(0.0029, 0.0099, "red"), (0.0030, 0.0, "yellow"), (0.0, 0.0100, "yellow"), (0.0100, 0.0, "green"), (0.0, 0.0200, "green")],
)
def test_decision_boundaries_are_exact(map_delta: float, recall_delta: float, expected: str) -> None:
    assert decide_complementarity(map_delta, recall_delta)["decision"] == expected
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_rtdetr_complementarity_oracle.py`

Expected: all tests pass.

Commit: `git add src/rtdetr_complementarity_oracle.py tests/test_rtdetr_complementarity_oracle.py && git commit -m "feat: add immutable complementarity cache"`

### Task 4: Source-bound CLI and exact raw-query extraction

**Files:**
- Create: `tests/test_run_rtdetr_complementarity_oracle.py`
- Create: `scripts/run_rtdetr_complementarity_oracle.py`

- [ ] **Step 1: Write failing CLI contract tests**

```python
def test_cli_exposes_only_frozen_artifact_and_output_paths() -> None:
    args = parse_args([
        "--fdr-checkpoint", "fdr.pt",
        "--frequencycm-checkpoint", "cm.pt",
        "--dataset-root", "VisDrone",
        "--cache-root", "cache",
        "--report-root", "report",
        "--device", "0",
    ])
    assert not hasattr(args, "threshold")
    assert not hasattr(args, "alpha")
    assert not hasattr(args, "max_det")


def test_cli_source_contains_frozen_checkpoint_hashes() -> None:
    source = Path("scripts/run_rtdetr_complementarity_oracle.py").read_text(encoding="utf-8")
    assert "c2f638744508adfe7b6c4a1ef3e08c503273f628062e4650ad59ffff4c6588c2" in source
    assert "2bbcd6057fefed5792f786a18e603f8feca3ec426a6f68938f5f8ada1603a141" in source
```

- [ ] **Step 2: Implement CLI constants and authority verification**

Expose only the six arguments in the test. Verify checkpoint hashes, 548 validation images, dataset SHA, ten-class mapping, CUDA/runtime versions, exact source commit, and the frozen evaluator constants before loading models.

- [ ] **Step 3: Implement exact model loading and raw-query extraction**

Reuse the final decoder tuple path used by `scripts/run_rtdetr_quality_oracle.py`. Load FDR with `FDRRTDETRDetectionModel` and FrequencyCM with `FDRFrequencyCMDetectionModel`; use each checkpoint's EMA weights, run `eval()` under inference mode, and cache final-layer `[B,300,4]` normalized boxes plus `[B,300,10]` logits before Ultralytics flattened Top-K.

Add runtime assertions that shared preprocessing tensors, image IDs, target boxes, and target classes match exactly between the two detector passes.

- [ ] **Step 4: Implement stock reconstruction gate**

Use `src.rtdetr_quality_oracle.flattened_topk` and `src.iber_evaluation.compute_detection_metrics` to reconstruct ordinary FDR and FrequencyCM metrics. Abort unless FDR reproduces its frozen endpoint/evaluator authority and FrequencyCM reproduces its checkpoint endpoint within the explicitly stored tolerance. Record both training-endpoint and independent-evaluator identities separately rather than treating them as interchangeable.

- [ ] **Step 5: Run CLI tests and commit**

Run: `pytest -q tests/test_run_rtdetr_complementarity_oracle.py`

Expected: all tests pass without requiring CUDA or real checkpoints.

Commit: `git add scripts/run_rtdetr_complementarity_oracle.py tests/test_run_rtdetr_complementarity_oracle.py && git commit -m "feat: add complementarity oracle runner"`

### Task 5: Full report generation and synthetic end-to-end proof

**Files:**
- Modify: `src/rtdetr_complementarity_oracle.py`
- Modify: `scripts/run_rtdetr_complementarity_oracle.py`
- Modify: `tests/test_rtdetr_complementarity_oracle.py`
- Modify: `tests/test_run_rtdetr_complementarity_oracle.py`

- [ ] **Step 1: Write failing report-contract test**

```python
def test_synthetic_run_writes_every_frozen_output(tmp_path: Path) -> None:
    report = run_from_records(synthetic_paired_records(), tmp_path)
    expected = {
        "oracle-summary.json",
        "coverage-by-scale.csv",
        "coverage-by-class.csv",
        "missed-target-categories.csv",
        "oracle-arms.csv",
        "frequencycm-complementarity-report.md",
        "SHA256SUMS.txt",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert report["interpretation"] == "non_deployable_design_selection_evidence"
```

- [ ] **Step 2: Implement image selector and all report tables**

For each image, compute deterministic one-to-one matched-IoU sum for FDR and FrequencyCM; choose FDR on exact ties; preserve the selected arm's original scores. Generate stock, selector, FDR-oracle, FrequencyCM-oracle, duplicate-FDR, and union-oracle metric rows. Write all files create-only using canonical JSON and stable CSV column order.

- [ ] **Step 3: Implement SHA manifest and human-readable report**

The Markdown report must state the frozen checkpoints, ordinary detector results, coverage deltas, candidate-oracle deltas, scale/class concentration, decision, and explicit warning that the oracle uses ground truth and the official validation set for design selection.

- [ ] **Step 4: Run all oracle tests and the existing quality-oracle regression suite**

Run: `pytest -q tests/test_rtdetr_complementarity_oracle.py tests/test_run_rtdetr_complementarity_oracle.py tests/test_rtdetr_quality_oracle.py tests/test_run_rtdetr_quality_oracle.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit: `git add src/rtdetr_complementarity_oracle.py scripts/run_rtdetr_complementarity_oracle.py tests/test_rtdetr_complementarity_oracle.py tests/test_run_rtdetr_complementarity_oracle.py && git commit -m "feat: complete FrequencyCM complementarity oracle"`

### Task 6: Preflight, immutable deployment, execution, and publication

**Files:**
- Runtime only: `/data/uav/source/uav-detection-baselines-<commit>/`
- Runtime only: `/data/uav/cache/frequencycm-complementarity-oracle-v1/`
- Runtime only: `/data/uav/reports/frequencycm-complementarity-oracle-v1/`

- [ ] **Step 1: Run the complete local test suite and record the source commit**

Run: `pytest -q tests/test_rtdetr_complementarity_oracle.py tests/test_run_rtdetr_complementarity_oracle.py tests/test_rtdetr_quality_oracle.py tests/test_run_rtdetr_quality_oracle.py tests/test_rtdetr_fdr_frequencycm.py`

Expected: all selected tests pass and `git status --short` is empty.

- [ ] **Step 2: Verify the server host key and runtime readiness**

Require the previously frozen ED25519 fingerprint, one RTX 4090, sufficient disk, VisDrone dataset, both exact checkpoint hashes, Python 3.10.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, CUDA 12.1, and Ultralytics 8.4.90. Stop on any authority mismatch.

- [ ] **Step 3: Deploy a git bundle into a new immutable source directory**

Create a bundle from the committed source, transfer it after host verification, clone it into `/data/uav/source/uav-detection-baselines-<commit>`, and record bundle/source SHA-256. Do not modify the completed FDR or FrequencyCM run directories.

- [ ] **Step 4: Run one real-batch CUDA preflight**

Execute cache extraction for one validation batch, reconstruct both stock top-300 outputs, verify finite tensors and exact shapes, then delete only the explicitly named disposable preflight cache directory after verifying its resolved path is below `/data/uav/cache/`.

- [ ] **Step 5: Run the full deterministic oracle once**

```bash
python scripts/run_rtdetr_complementarity_oracle.py \
  --fdr-checkpoint /data/uav/checkpoints/fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt \
  --frequencycm-checkpoint /data/uav/checkpoints/fdr-frequencycm-formal-d3655b14-epoch-0100.pt \
  --dataset-root /data/uav/datasets/VisDrone \
  --cache-root /data/uav/cache/frequencycm-complementarity-oracle-v1 \
  --report-root /data/uav/reports/frequencycm-complementarity-oracle-v1 \
  --device 0
```

Expected: stock reconstruction passes, duplicate-FDR oracle is neutral, all reports and hashes are written, and the decision is exactly Red, Yellow, or Green.

- [ ] **Step 6: Independently verify reports and publish evidence**

Recompute every SHA-256, rerun report loading without model inference, inspect decision inputs against raw CSV values, commit lightweight evidence to the results branch, and upload the full report plus cache manifest to a GitHub Release. Large caches remain on the server unless explicitly required; their manifests and hashes are published.

- [ ] **Step 7: Commit the final evidence index**

Commit: `git add reports/frequencycm-complementarity-oracle-v1 && git commit -m "evidence: publish FrequencyCM complementarity oracle"`

The final handoff reports the exact decision, candidate-oracle mAP delta, tiny/small recall deltas, artifact URLs, and whether CM-v2 proceeds. It must not claim a deployable gain.

# EBC-QP v1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen EBC-QP v1.0 query-protection method, its diagnostics, and the D0-D3 screening protocol on the matched Ultralytics RT-DETR-L baseline without changing the decoder, Hungarian matcher, or 300-query budget.

**Architecture:** Keep the stock P3/P4/P5 path intact and add one isolated P2 side branch inside a repository-owned `RTDETRDecoder` subclass. Pure tensor modules own deterministic center matching, sparse P2/EBC losses, fixed-budget query competition, and unique-GT statistics; a thin model/trainer/validator layer owns Ultralytics lifecycle, optimizer-update monitoring, custom AP-tiny, checkpoint state, and persistent logs.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.90 RT-DETR, SciPy linear assignment, pytest, VisDrone YOLO dataset.

---

## File Map

- Create `src/ebc_qp_config.py`: frozen constants, source-lock verification, initialization-state fingerprints.
- Create `src/ebc_qp_matching.py`: deterministic maximum-cardinality/minimum-distance center matching and local P2 assignment.
- Create `src/ebc_qp_loss.py`: sparse P2 VFL/L1/GIoU and zero-margin EBC loss.
- Create `src/ebc_qp_queries.py`: stable Top-K selection, complete-tuple global competition, and unique-GT replacement statistics.
- Create `src/ebc_qp_decoder.py`: isolated P2 adapter/head, stock-compatible encoder path, warm-up switch, and forward diagnostics.
- Create `src/ebc_qp_metrics.py`: validation-size tiny masks and project-specific AP-tiny accumulation.
- Create `src/rtdetr_ebc_qp.py`: Ultralytics model, trainer, validator, checkpoint/resume, update monitor, and logging integration.
- Create `configs/rtdetr-l-ebc-qp.yaml`: stock RT-DETR-L graph with C2 added only to the decoder input list.
- Create `scripts/diagnose_ebc_qp.py`: D0 zero-training coverage/rank/failure diagnosis.
- Create `scripts/train_rtdetr_ebc_qp.py`: D1/D2/D3/formal run launcher and protocol guards.
- Create focused tests under `tests/test_ebc_qp_*.py`; modify no existing innovation implementation.

### Task 1: Freeze Runtime Contract and Source Provenance

**Files:**
- Create: `src/ebc_qp_config.py`
- Create: `tests/test_ebc_qp_config.py`

- [ ] **Step 1: Write failing tests for immutable defaults and source hashes**

```python
from pathlib import Path

import pytest

from src.ebc_qp_config import EBCQPConfig, assert_ultralytics_source_lock


def test_v1_defaults_are_the_frozen_values():
    cfg = EBCQPConfig()
    assert cfg.query_budget == 300
    assert cfg.p2_candidates == 50
    assert cfg.warmup_epochs == 3
    assert cfg.tiny_radius == 16.0
    assert cfg.p2_anchor_size == 0.025
    assert cfg.lambda_p2 == 0.25
    assert cfg.lambda_ebc == 0.05
    assert cfg.local_radius == 1


def test_source_lock_rejects_a_changed_file(tmp_path: Path):
    changed = tmp_path / "head.py"
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source lock mismatch"):
        assert_ultralytics_source_lock({"head.py": changed})
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.ebc_qp_config'`.

- [ ] **Step 3: Implement the frozen dataclass and SHA256 verifier**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping


ULTRALYTICS_VERSION = "8.4.90"
SOURCE_SHA256 = {
    "head.py": "5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F",
    "tasks.py": "B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92",
    "rtdetr-l.yaml": "85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83",
}


@dataclass(frozen=True)
class EBCQPConfig:
    query_budget: int = 300
    p2_candidates: int = 50
    warmup_epochs: int = 3
    tiny_radius: float = 16.0
    p2_anchor_size: float = 0.025
    lambda_p2: float = 0.25
    lambda_ebc: float = 0.05
    local_radius: int = 1
    update_ratio_limit: float = 10.0
    update_ratio_patience: int = 20
    update_monitor_steps: int = 200
    epsilon: float = 1e-12

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def assert_ultralytics_source_lock(paths: Mapping[str, Path]) -> None:
    mismatches = [name for name, path in paths.items() if file_sha256(path) != SOURCE_SHA256[name]]
    if mismatches:
        raise RuntimeError(f"Ultralytics source lock mismatch: {', '.join(sorted(mismatches))}")
```

- [ ] **Step 4: Run the config tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_config.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the runtime contract**

```powershell
git add src/ebc_qp_config.py tests/test_ebc_qp_config.py
git commit -m "Add frozen EBC-QP runtime contract"
```

### Task 2: Deterministic Stock Coverage and P2 Local Assignment

**Files:**
- Create: `src/ebc_qp_matching.py`
- Create: `tests/test_ebc_qp_matching.py`

- [ ] **Step 1: Write failing tests for cardinality, distance, tie order, uniqueness, and unassigned GTs**

```python
import torch

from src.ebc_qp_matching import assign_local_p2, match_centers_inside_boxes


def test_matching_maximizes_gt_coverage_before_distance():
    centers = torch.tensor([[0.20, 0.20], [0.28, 0.20]])
    boxes = torch.tensor([[0.24, 0.20, 0.10, 0.10], [0.28, 0.20, 0.04, 0.04]])
    result = match_centers_inside_boxes(centers, boxes)
    assert result.pairs.tolist() == [[0, 0], [1, 1]]


def test_equal_cost_tie_prefers_low_gt_then_low_token():
    centers = torch.tensor([[0.45, 0.50], [0.55, 0.50]])
    boxes = torch.tensor([[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20]])
    result = match_centers_inside_boxes(centers, boxes)
    assert result.pairs.tolist() == [[0, 0], [1, 1]]


def test_local_assignment_uses_unique_cells_and_reports_unassigned():
    result = assign_local_p2(
        height=4,
        width=4,
        boxes=torch.tensor([[0.375, 0.375, 0.05, 0.05], [0.375, 0.375, 0.05, 0.05]]),
        valid_mask=torch.ones(16, dtype=torch.bool),
        radius=1,
    )
    assert result.pairs.shape == (2, 2)
    assert result.pairs[:, 1].unique().numel() == 2
    assert result.unassigned_gt.numel() == 0


def test_empty_and_unreachable_inputs_return_typed_empty_tensors():
    result = match_centers_inside_boxes(torch.empty(0, 2), torch.tensor([[0.5, 0.5, 0.1, 0.1]]))
    assert result.pairs.shape == (0, 2)
    assert result.unassigned_gt.tolist() == [0]
```

- [ ] **Step 2: Run matching tests and verify the missing module failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_matching.py -q`

Expected: collection fails because `src.ebc_qp_matching` does not exist.

- [ ] **Step 3: Implement one matching API for both stock and P2 paths**

```python
@dataclass(frozen=True)
class CenterMatch:
    pairs: torch.Tensor  # columns: gt index, candidate index
    unassigned_gt: torch.Tensor

    @property
    def covered_gt(self) -> torch.Tensor:
        return self.pairs[:, 0]


def normalized_center_cost(centers: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    scale = boxes[:, 2:].prod(-1).sqrt().clamp_min(1e-6)
    return torch.cdist(boxes[:, :2].float(), centers.float()) / scale[:, None]


def match_centers_inside_boxes(centers: torch.Tensor, boxes: torch.Tensor) -> CenterMatch:
    half = boxes[:, None, 2:] / 2
    delta = (centers[None] - boxes[:, None, :2]).abs()
    legal = (delta <= half).all(-1)
    costs = normalized_center_cost(centers, boxes)
    pairs = _lexicographic_assignment(costs, legal)
    covered = set(pairs[:, 0].tolist())
    unassigned = torch.tensor(
        [index for index in range(len(boxes)) if index not in covered],
        dtype=torch.long,
        device=boxes.device,
    )
    return CenterMatch(pairs=pairs, unassigned_gt=unassigned)


def assign_local_p2(
    height: int,
    width: int,
    boxes: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int = 1,
) -> CenterMatch:
    centers = p2_grid_centers(height, width, boxes.device)
    cell_x = (boxes[:, 0] * width).floor().long().clamp(0, width - 1)
    cell_y = (boxes[:, 1] * height).floor().long().clamp(0, height - 1)
    legal = torch.zeros((len(boxes), height * width), dtype=torch.bool, device=boxes.device)
    for gt_index, (x, y) in enumerate(zip(cell_x.tolist(), cell_y.tolist())):
        for row in range(max(0, y - radius), min(height, y + radius + 1)):
            for column in range(max(0, x - radius), min(width, x + radius + 1)):
                flat = row * width + column
                legal[gt_index, flat] = bool(valid_mask[flat])
    costs = normalized_center_cost(centers, boxes)
    pairs = _lexicographic_assignment(costs, legal)
    covered = set(pairs[:, 0].tolist())
    unassigned = torch.tensor(
        [index for index in range(len(boxes)) if index not in covered],
        dtype=torch.long,
        device=boxes.device,
    )
    return CenterMatch(pairs=pairs, unassigned_gt=unassigned)
```

Define `_lexicographic_assignment(costs, legal)` with `scipy.optimize.linear_sum_assignment`. Add one dummy column per remaining GT; set every dummy cost to `penalty=(max_legal_cost+1)*(remaining_gt+1)` and forbidden real edges to `2*penalty`. This makes unmatched count dominate distance. Then perform deterministic prefix fixing: for each GT in ascending input index, try legal tokens in ascending flattened index followed by unmatched; re-solve the remaining matrix and keep the first option that preserves optimum cardinality and total distance within `1e-10`. Return pairs sorted by GT index on the input device. Define `p2_grid_centers` with `(column+0.5)/width, (row+0.5)/height` and flattened row-major order.

- [ ] **Step 4: Run matching tests, including five repeated tie runs**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_matching.py -q`

Expected: all tests pass and the tie test returns the same pairs on every repetition.

- [ ] **Step 5: Commit deterministic matching**

```powershell
git add src/ebc_qp_matching.py tests/test_ebc_qp_matching.py
git commit -m "Add deterministic EBC-QP center matching"
```

### Task 3: Sparse P2 and Boundary-Calibrated Losses

**Files:**
- Create: `src/ebc_qp_loss.py`
- Create: `tests/test_ebc_qp_loss.py`

- [ ] **Step 1: Write failing tests for sparse classification positions and loss edge cases**

```python
import torch

from src.ebc_qp_loss import P2Targets, compute_ebc_loss, compute_sparse_p2_loss


def test_sparse_p2_keeps_top50_union_positive_and_excludes_inside_gt_negatives():
    logits = torch.zeros(1, 6, 2, requires_grad=True)
    boxes = torch.tensor([[[0.1, 0.1, 0.1, 0.1]] * 6], requires_grad=True)
    targets = P2Targets(
        gt_boxes=[torch.tensor([[0.5, 0.5, 0.2, 0.2]])],
        gt_classes=[torch.tensor([1])],
        assigned_pairs=[torch.tensor([[0, 4]])],
        topk_indices=torch.tensor([[0, 1, 2]]),
        anchor_centers=torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7]]),
    )
    result = compute_sparse_p2_loss(logits, boxes, targets)
    assert result.classification_indices[0].tolist() == [0, 2, 4]


def test_vfl_iou_target_is_detached_but_box_losses_train_boxes():
    result = make_single_positive_loss()
    result.total.backward()
    assert result.vfl_target.requires_grad is False
    assert result.pred_boxes_grad_is_expected


def test_no_positive_loss_is_finite_and_differentiable():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    boxes = torch.zeros(1, 3, 4, requires_grad=True)
    result = compute_sparse_p2_loss(logits, boxes, empty_targets(topk=[0, 1]))
    result.total.backward()
    assert torch.isfinite(result.total)
    assert boxes.grad is not None


def test_ebc_uses_correct_class_logit_only_for_uncovered_assigned_gt():
    logits = torch.tensor([[[-2.0, 0.4], [3.0, -1.0]]], requires_grad=True)
    loss = compute_ebc_loss(
        p2_logits=logits,
        assigned_pairs=[torch.tensor([[0, 0], [1, 1]])],
        gt_classes=[torch.tensor([1, 0])],
        uncovered=[torch.tensor([True, False])],
        stock_boundary=torch.tensor([1.0]),
    )
    torch.testing.assert_close(loss, torch.tensor(0.6))
```

- [ ] **Step 2: Run loss tests and verify the missing module failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_loss.py -q`

Expected: collection fails because `src.ebc_qp_loss` does not exist.

- [ ] **Step 3: Implement typed loss results and frozen formulas**

```python
@dataclass(frozen=True)
class P2LossResult:
    total: torch.Tensor
    vfl: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    positive_count: torch.Tensor
    classification_indices: list[torch.Tensor]
    vfl_target: torch.Tensor


def differentiable_zero(*tensors: torch.Tensor) -> torch.Tensor:
    return sum((tensor.sum() * 0.0 for tensor in tensors), start=tensors[0].new_zeros(()))


def compute_sparse_p2_loss(
    logits: torch.Tensor,
    boxes_xywh: torch.Tensor,
    targets: P2Targets,
) -> P2LossResult:
    image_results = [
        _compute_image_p2_loss(logits[b], boxes_xywh[b], targets.image(b))
        for b in range(logits.shape[0])
    ]
    return P2LossResult.from_images(image_results, logits, boxes_xywh)


def compute_ebc_loss(
    p2_logits: torch.Tensor,
    assigned_pairs: list[torch.Tensor],
    gt_classes: list[torch.Tensor],
    uncovered: list[torch.Tensor],
    stock_boundary: torch.Tensor,
) -> torch.Tensor:
    eligible_logits = collect_correct_class_logits(p2_logits, assigned_pairs, gt_classes, uncovered)
    if not eligible_logits:
        return differentiable_zero(p2_logits)
    values = [torch.relu(stock_boundary[b].detach() - score) for b, score in eligible_logits]
    return torch.stack(values).mean()
```

Use Ultralytics `VarifocalLoss` and `bbox_iou(pred_boxes, target_boxes, GIoU=True, xywh=True)`. `P2Targets.image(index)` returns the image's GT boxes/classes, assigned pairs, Top-50 indices, and shared anchor centers; `P2LossResult.from_images` stacks the scalar terms, concatenates detached VFL targets, and returns the batch mean. Force matching and IoU-target construction to FP32 under AMP; cast the final scalar to FP32 before adding it to the stock loss.

- [ ] **Step 4: Run the loss tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_loss.py -q`

Expected: all finite-zero, detach, masking, weighting, and EBC eligibility tests pass.

- [ ] **Step 5: Commit the loss layer**

```powershell
git add src/ebc_qp_loss.py tests/test_ebc_qp_loss.py
git commit -m "Add sparse P2 and boundary losses"
```

### Task 4: Stable Fixed-Budget Query Competition and Replacement Statistics

**Files:**
- Create: `src/ebc_qp_queries.py`
- Create: `tests/test_ebc_qp_queries.py`

- [ ] **Step 1: Write failing tests for stock tie priority, tuple integrity, and unique GT gain/loss**

```python
import torch

from src.ebc_qp_queries import QuerySet, compete_queries, replacement_statistics


def test_competition_keeps_budget_and_prefers_stock_on_equal_scores():
    stock = make_query_set(scores=[0.9, 0.5, 0.4], source="stock")
    p2 = make_query_set(scores=[0.8, 0.5], source="p2")
    mixed = compete_queries(stock, p2, budget=3)
    assert mixed.source.tolist() == [0, 1, 0]
    assert mixed.source_index.tolist() == [0, 0, 1]
    assert mixed.features[:, 0].tolist() == [stock.features[0, 0], p2.features[0, 0], stock.features[0, 1]]


def test_duplicate_p2_candidates_count_one_new_gt():
    stats = replacement_statistics(
        stock_centers=torch.tensor([[0.1, 0.1]]),
        final_centers=torch.tensor([[0.5, 0.5], [0.51, 0.5]]),
        gt_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        tiny_mask=torch.tensor([True]),
    )
    assert stats.n_gain == 1
    assert stats.n_loss == 0
    assert stats.value == 1


def test_loss_counts_non_tiny_gt_removed_by_competition():
    stats = replacement_statistics_for_lost_large_gt()
    assert stats.n_gain == 1
    assert stats.n_loss == 1
    assert stats.value == 0
```

- [ ] **Step 2: Run query tests and verify the missing module failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_queries.py -q`

Expected: collection fails because `src.ebc_qp_queries` does not exist.

- [ ] **Step 3: Implement stable Top-K over complete query records**

```python
@dataclass(frozen=True)
class QuerySet:
    features: torch.Tensor
    reference_logits: torch.Tensor
    boxes: torch.Tensor
    logits: torch.Tensor
    ranking_score: torch.Tensor
    centers: torch.Tensor
    source: torch.Tensor  # 0 stock, 1 P2
    source_level: torch.Tensor
    source_index: torch.Tensor


def stable_rank_indices(scores: torch.Tensor, source: torch.Tensor, source_index: torch.Tensor, k: int) -> torch.Tensor:
    selected = []
    for batch_index in range(scores.shape[0]):
        order = sorted(
            range(scores.shape[1]),
            key=lambda index: (
                -float(scores[batch_index, index].detach().cpu()),
                int(source[batch_index, index].detach().cpu()),
                int(source_index[batch_index, index].detach().cpu()),
            ),
        )[:k]
        selected.append(order)
    return torch.tensor(selected, dtype=torch.long, device=scores.device)


def compete_queries(stock: QuerySet, p2: QuerySet, budget: int = 300) -> QuerySet:
    merged = concatenate_query_sets(stock, p2)
    return gather_query_set(merged, stable_rank_indices(merged.ranking_score, merged.source, merged.source_index, budget))


@dataclass(frozen=True)
class ReplacementStats:
    n_gain: int
    n_loss: int
    value: int


def replacement_statistics(stock_centers, final_centers, gt_boxes, tiny_mask) -> ReplacementStats:
    stock_gt = set(match_centers_inside_boxes(stock_centers, gt_boxes).covered_gt.tolist())
    final_gt = set(match_centers_inside_boxes(final_centers, gt_boxes).covered_gt.tolist())
    tiny_gt = set(torch.where(tiny_mask)[0].tolist())
    gain = len((final_gt - stock_gt) & tiny_gt)
    loss = len(stock_gt - final_gt)
    return ReplacementStats(gain, loss, gain - loss)


@dataclass(frozen=True)
class P2DiversityStats:
    foreground_at_50: int
    unique_gt_at_50: int
    duplicate_rate_at_50: float
    background_rate_at_50: float


def p2_diversity_statistics(p2_centers: torch.Tensor, tiny_boxes: torch.Tensor) -> P2DiversityStats:
    # Associate every center to the nearest containing tiny GT without enforcing
    # one-to-one uniqueness. Count background separately from repeated GT hits.
    association = nearest_containing_gt(p2_centers.detach(), tiny_boxes.detach())
    foreground = association >= 0
    foreground_count = int(foreground.sum())
    unique_count = int(association[foreground].unique().numel())
    duplicate = 0.0 if foreground_count == 0 else 1.0 - unique_count / foreground_count
    return P2DiversityStats(
        foreground_at_50=foreground_count,
        unique_gt_at_50=unique_count,
        duplicate_rate_at_50=duplicate,
        background_rate_at_50=1.0 - foreground_count / max(len(p2_centers), 1),
    )
```

- [ ] **Step 4: Run query tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_queries.py -q`

Expected: budget, stable ties, complete-field gathering, unique-GT replacement, and foreground-versus-duplicate P2 statistics pass.

- [ ] **Step 5: Commit query competition**

```powershell
git add src/ebc_qp_queries.py tests/test_ebc_qp_queries.py
git commit -m "Add fixed-budget EBC-QP query competition"
```

### Task 5: Isolated P2 Decoder Side Branch

**Files:**
- Create: `src/ebc_qp_decoder.py`
- Create: `src/rtdetr_ebc_qp.py` (registration and model-construction shell)
- Create: `configs/rtdetr-l-ebc-qp.yaml`
- Create: `tests/test_ebc_qp_decoder.py`

- [ ] **Step 1: Write failing decoder tests before registering the custom head**

```python
from pathlib import Path

import torch

from src.ebc_qp_decoder import EBCQPDecoder, register_ebc_qp_decoder
from src.rtdetr_ebc_qp import EBCQPDetectionModel


CONFIG = Path(__file__).parents[1] / "configs" / "rtdetr-l-ebc-qp.yaml"


def test_yaml_passes_c2_p3_p4_p5_to_one_custom_decoder():
    model = EBCQPDetectionModel(CONFIG, ch=3, nc=10, verbose=False)
    head = model.model[-1]
    assert isinstance(head, EBCQPDecoder)
    assert head.f == [1, 21, 24, 27]
    assert len(head.input_proj) == 3
    assert head.p2_adapter[0].in_channels == 128


def test_disabled_path_matches_stock_indices_and_outputs_fp32():
    stock, ebc = build_elementwise_identical_models(enabled=False)
    image = torch.rand(2, 3, 160, 160)
    stock.eval(); ebc.eval()
    with torch.no_grad():
        stock_output = stock.predict(image)
        ebc_output = ebc.predict(image)
    torch.testing.assert_close(ebc_output[0], stock_output[0], rtol=1e-5, atol=1e-6)
    assert torch.equal(ebc.model[-1].last_state.stock_topk_indices, captured_stock_topk(stock))


def test_p2_only_backward_isolates_stock_parameters_and_side_inputs():
    head, c2, p3, p4, p5, batch = make_training_head_inputs()
    state = head.forward_with_state([c2, p3, p4, p5], batch)
    state.p2_loss.backward()
    assert grad_nonzero(head.p2_adapter)
    assert grad_nonzero(head.p2_bbox_head)
    assert grad_zero(head.enc_output)
    assert grad_zero(head.enc_score_head)
    assert c2.grad is None and p3.grad is None


def test_warmup_and_active_query_integrity():
    head, inputs, batch = make_training_case()
    head.set_progress(epoch=2)
    warm = head.forward_with_state(inputs, batch)
    assert warm.p2_entry_count == 0 and warm.ebc_active is False
    head.set_progress(epoch=3)
    active = head.forward_with_state(inputs, batch)
    assert active.ordinary_query_count == 300
    assert 0 <= active.p2_entry_count <= 50
    assert active.encoder_aux_source_is_stock
```

- [ ] **Step 2: Run decoder tests and verify the missing-module failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_decoder.py -q`

Expected: collection fails because the custom decoder/model files do not exist.

- [ ] **Step 3: Add the stock graph with only the final input list changed**

Copy the locked stock `rtdetr-l.yaml` into `configs/rtdetr-l-ebc-qp.yaml`. Preserve layers 0-27 byte-for-byte apart from comments, and set only the final layer to:

```yaml
  - [[1, 21, 24, 27], 1, RTDETRDecoder, [nc]]
```

- [ ] **Step 4: Implement `EBCQPDecoder` with explicit per-forward state**

```python
class EBCQPDecoder(RTDETRDecoder):
    def __init__(self, nc=80, ch=(), *args, ebc_config: EBCQPConfig | None = None, **kwargs):
        c2_channels, *stock_channels = ch
        super().__init__(nc=nc, ch=tuple(stock_channels), *args, **kwargs)
        self.ebc_config = ebc_config or EBCQPConfig()
        self.p2_adapter = nn.Sequential(
            nn.Conv2d(c2_channels, self.hidden_dim, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
        )
        self.p2_bbox_head = deepcopy(self.enc_bbox_head)
        self.ebc_epoch = 0
        self.ebc_enabled = True
        self.last_state: EBCQPForwardState | None = None

    def set_progress(self, epoch: int) -> None:
        self.ebc_epoch = int(epoch)

    def _p2_features(self, c2: torch.Tensor, projected_p3: torch.Tensor) -> torch.Tensor:
        lateral = self.p2_adapter(c2.detach())
        context = F.interpolate(projected_p3.detach(), size=lateral.shape[-2:], mode="nearest")
        return F.silu(lateral + context)

    def _detached_stock_transform(self, p2_tokens: torch.Tensor) -> torch.Tensor:
        linear, norm = self.enc_output
        value = F.linear(p2_tokens, linear.weight.detach(), linear.bias.detach())
        return F.layer_norm(value, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)

    def _detached_stock_scores(self, p2_embed: torch.Tensor) -> torch.Tensor:
        return F.linear(p2_embed, self.enc_score_head.weight.detach(), self.enc_score_head.bias.detach())
```

Override `forward` as a close, source-locked copy of Ultralytics 8.4.90 lines 1543-1593. It must:

1. project only P3/P4/P5 through `input_proj` and reproduce stock anchors/features/scores/top-300;
2. save stock anchor centers before inverse sigmoid and derive `u_g` from tiny GT matching;
3. build P2 anchors with `grid_size=0.025`, mask invalid anchors to `-inf`, and take stable P2 Top-50;
4. compute sparse P2 state in every training epoch;
5. before epoch index 3, send stock Top-300 to the decoder and set EBC to differentiable zero;
6. from epoch index 3, merge complete stock/P2 query tuples and send exactly 300 ordinary queries;
7. prepend denoising inputs exactly as stock does;
8. return original stock `enc_bboxes/enc_scores` to the criterion;
9. detach ordinary query content/reference boxes before the decoder exactly as stock does;
10. replace `last_state` atomically on each forward so stale tensors cannot survive an exception.

Register without editing site-packages:

```python
def register_ebc_qp_decoder() -> None:
    ultralytics_tasks.RTDETRDecoder = EBCQPDecoder
```

Because `parse_model` compares `m is RTDETRDecoder`, replace the module global before `RTDETRDetectionModel.__init__`; this preserves Ultralytics' decoder-channel argument injection for the four-input YAML.

Add the construction shell used by the decoder tests:

```python
class EBCQPDetectionModel(RTDETRDetectionModel):
    def __init__(self, cfg="configs/rtdetr-l-ebc-qp.yaml", ch=3, nc=None, verbose=True, ebc_config=None):
        register_ebc_qp_decoder()
        self.ebc_config = ebc_config or EBCQPConfig()
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @property
    def ebc_head(self) -> EBCQPDecoder:
        return self.model[-1]
```

- [ ] **Step 5: Run decoder tests in FP32, then add the CUDA AMP equivalence case**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_decoder.py -q`

Expected: FP32 tolerance passes; CUDA-only AMP test passes at `rtol=1e-4, atol=1e-5` or is skipped when CUDA is unavailable.

- [ ] **Step 6: Commit the decoder and YAML**

```powershell
git add src/ebc_qp_decoder.py src/rtdetr_ebc_qp.py configs/rtdetr-l-ebc-qp.yaml tests/test_ebc_qp_decoder.py
git commit -m "Add isolated EBC-QP P2 decoder branch"
```

### Task 6: Detection Loss, Warm-up, and Checkpoint Lifecycle

**Files:**
- Modify: `src/rtdetr_ebc_qp.py`
- Create: `tests/test_rtdetr_ebc_qp_integration.py`

- [ ] **Step 1: Write failing model/trainer lifecycle tests**

```python
import torch

from src.rtdetr_ebc_qp import EBCQPDetectionModel, EBCQPTrainer


def test_model_adds_weighted_losses_but_keeps_stock_encoder_auxiliary_output():
    model, batch = build_small_training_model_and_batch()
    total, items = model.loss(batch)
    state = model.ebc_head.last_state
    expected = state.stock_loss + 0.25 * state.p2_loss + 0.05 * state.ebc_loss
    torch.testing.assert_close(total, expected)
    assert items.shape == (5,)
    assert state.encoder_aux_source_is_stock


def test_epoch_callback_restores_activation_boundary_after_resume():
    trainer = make_trainer(epoch=3)
    trainer._set_ebc_progress()
    assert unwrap(trainer.model).ebc_head.ebc_epoch == 3
    assert unwrap(trainer.model).ebc_head.competition_active


def test_checkpoint_round_trip_preserves_p2_parameters_and_progress(tmp_path):
    restored = round_trip_checkpoint(make_model(epoch=7), tmp_path)
    assert restored.ebc_head.ebc_epoch == 7
    assert_state_dict_equal(restored.ebc_head.p2_adapter, original.ebc_head.p2_adapter)
    assert_state_dict_equal(restored.ebc_head.p2_bbox_head, original.ebc_head.p2_bbox_head)
```

- [ ] **Step 2: Run integration tests and verify the missing model failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_rtdetr_ebc_qp_integration.py -q`

Expected: collection fails because `src.rtdetr_ebc_qp` does not exist.

- [ ] **Step 3: Implement model loss and trainer progress hooks**

```python
LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "p2_loss", "ebc_loss")


class EBCQPDetectionModel(RTDETRDetectionModel):
    def __init__(self, cfg="configs/rtdetr-l-ebc-qp.yaml", ch=3, nc=None, verbose=True, ebc_config=None):
        register_ebc_qp_decoder()
        self.ebc_config = ebc_config or EBCQPConfig()
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.loss_names = LOSS_NAMES

    @property
    def ebc_head(self) -> EBCQPDecoder:
        return self.model[-1]

    def set_ebc_progress(self, epoch: int) -> None:
        self.ebc_head.set_progress(epoch)

    def loss(self, batch: dict, preds=None):
        stock_loss, stock_items = super().loss(batch, preds=preds)
        state = self.ebc_head.last_state
        if state is None:
            raise RuntimeError("EBC-QP forward state was not populated")
        ebc = state.ebc_loss if state.competition_active else state.ebc_loss * 0.0
        total = stock_loss.float() + self.ebc_config.lambda_p2 * state.p2_loss.float() + self.ebc_config.lambda_ebc * ebc.float()
        state.stock_loss = stock_loss.detach()
        return total, torch.cat((stock_items, state.p2_loss.detach()[None], ebc.detach()[None]))


class EBCQPTrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = EBCQPDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1, ebc_config=self.ebc_config)
        if weights:
            model.load(weights)
        return model

    def _set_ebc_progress(self) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.set_ebc_progress(int(self.epoch))

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return EBCQPValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
```

Call `_set_ebc_progress()` from `on_train_epoch_start`. Save `ebc_epoch`, the full config dictionary, source hashes, and an implementation-version string in the checkpoint metadata; on resume, reject a changed config/source lock before training continues.

- [ ] **Step 4: Run model/trainer lifecycle tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_rtdetr_ebc_qp_integration.py -q`

Expected: loss composition, warm-up boundary, P2 state restoration, EMA construction, and resume rejection tests pass.

- [ ] **Step 5: Commit model lifecycle integration**

```powershell
git add src/rtdetr_ebc_qp.py tests/test_rtdetr_ebc_qp_integration.py
git commit -m "Integrate EBC-QP model and trainer lifecycle"
```

### Task 7: Actual Optimizer-Update Ratio Monitor

**Files:**
- Modify: `src/rtdetr_ebc_qp.py`
- Create: `tests/test_ebc_qp_update_monitor.py`

- [ ] **Step 1: Write failing tests around actual parameter deltas**

```python
import pytest
import torch

from src.rtdetr_ebc_qp import NormalizedUpdateMonitor


def test_monitor_uses_post_optimizer_delta_not_raw_gradient():
    stock = torch.nn.Parameter(torch.tensor([10.0]))
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    monitor = NormalizedUpdateMonitor([p2], [stock], limit=10.0, patience=2, max_steps=200)
    monitor.snapshot()
    with torch.no_grad():
        p2.add_(0.2); stock.add_(0.1)
    record = monitor.observe()
    assert record.u_p2 == pytest.approx(0.2)
    assert record.u_stock == pytest.approx(0.01)
    assert record.ratio == pytest.approx(20.0)


def test_one_spike_does_not_abort_but_twenty_consecutive_steps_do():
    monitor = make_monitor(limit=10.0, patience=20)
    for _ in range(19):
        assert observe_ratio(monitor, 11.0).abort is False
    assert observe_ratio(monitor, 9.0).abort is False
    for _ in range(19):
        assert observe_ratio(monitor, 11.0).abort is False
    assert observe_ratio(monitor, 11.0).abort is True


def test_monitor_stops_collecting_after_step_200():
    monitor = make_monitor(max_steps=200)
    for _ in range(201):
        record = observe_ratio(monitor, 1.0)
    assert record.monitored is False
```

- [ ] **Step 2: Run update-monitor tests and verify the missing class failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_update_monitor.py -q`

Expected: import fails because `NormalizedUpdateMonitor` is not implemented.

- [ ] **Step 3: Implement exact pre/post-step snapshots and Trainer override**

```python
class NormalizedUpdateMonitor:
    def snapshot(self) -> None:
        self._before_p2 = [p.detach().to(device="cpu", dtype=torch.float32).clone() for p in self.p2 if p.requires_grad]
        self._before_stock = [p.detach().to(device="cpu", dtype=torch.float32).clone() for p in self.stock if p.requires_grad]

    @staticmethod
    def _relative(before: list[torch.Tensor], after: list[nn.Parameter], eps: float) -> float:
        delta2 = sum(float((p.detach().float().cpu() - old).square().sum()) for old, p in zip(before, after))
        theta2 = sum(float(old.square().sum()) for old in before)
        return delta2**0.5 / (theta2**0.5 + eps)

    def observe(self) -> UpdateRecord:
        u_p2 = self._relative(self._before_p2, self.p2, self.eps)
        u_stock = self._relative(self._before_stock, self.stock, self.eps)
        ratio = u_p2 / (u_stock + self.eps)
        self.consecutive = self.consecutive + 1 if ratio > self.limit else 0
        self.step += 1
        return UpdateRecord(u_p2, u_stock, ratio, self.step <= self.max_steps, self.consecutive >= self.patience)
```

In `EBCQPTrainer.optimizer_step()`, snapshot only while `monitor.step < 200`, call the stock sequence exactly (`unscale_`, clip, `scaler.step`, `scaler.update`), observe before zeroing gradients, then zero and update EMA. Divide parameters by stable names: names containing `.p2_adapter.` or `.p2_bbox_head.` are P2; every other trainable model parameter is stock. Raise `RuntimeError` with the 20-step trace only when `abort` is true. CPU FP32 snapshots avoid adding full-model copies to GPU memory and exist only for the first 200 optimizer steps.

- [ ] **Step 4: Run monitor tests and a scaler-skipped-step case**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_update_monitor.py -q`

Expected: exact delta math, reset-on-normal-step, 20-step patience, 200-step cutoff, and zero-delta scaler-skip behavior pass.

- [ ] **Step 5: Commit the update monitor**

```powershell
git add src/rtdetr_ebc_qp.py tests/test_ebc_qp_update_monitor.py
git commit -m "Monitor normalized EBC-QP optimizer updates"
```

### Task 8: Custom AP-Tiny and Validation Diagnostics

**Files:**
- Create: `src/ebc_qp_metrics.py`
- Modify: `src/rtdetr_ebc_qp.py`
- Create: `tests/test_ebc_qp_metrics.py`

- [ ] **Step 1: Write failing tests for resize-only radius and out-of-range ignores**

```python
import torch

from src.ebc_qp_metrics import TinyDetectionMetrics, resized_radius


def test_radius_uses_actual_xy_gain_and_padding_changes_no_size():
    boxes_xyxy = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    radius = resized_radius(boxes_xyxy, gain=(0.5, 0.25))
    torch.testing.assert_close(radius, torch.tensor([(10.0 * 10.0) ** 0.5]))


def test_gt_larger_than_16_is_ignored_not_counted_as_false_positive():
    metric = TinyDetectionMetrics(iouv=torch.linspace(0.5, 0.95, 10))
    metric.update(
        pred_boxes=torch.tensor([[0.0, 0.0, 40.0, 40.0]]),
        pred_conf=torch.tensor([0.9]),
        pred_cls=torch.tensor([0]),
        gt_boxes=torch.tensor([[0.0, 0.0, 40.0, 40.0]]),
        gt_cls=torch.tensor([0]),
        radius=torch.tensor([20.0]),
    )
    result = metric.compute()
    assert result.target_count == 0
    assert result.false_positive_count == 0


def test_custom_groups_are_r_lt_8_and_8_through_16():
    metric = metric_for_radii([7.99, 8.0, 16.0, 16.01])
    assert metric.extreme_tiny_count == 1
    assert metric.tiny_8_16_count == 2
```

- [ ] **Step 2: Run metric tests and verify the missing module failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_metrics.py -q`

Expected: collection fails because `src.ebc_qp_metrics` does not exist.

- [ ] **Step 3: Implement custom tiny accumulation and validator hook**

```python
def resized_radius(boxes_xyxy: torch.Tensor, gain: tuple[float, float]) -> torch.Tensor:
    width = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * gain[0]
    height = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * gain[1]
    return (width.clamp_min(0) * height.clamp_min(0)).sqrt()


class EBCQPValidator(RTDETRValidator):
    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self.tiny_metrics = TinyDetectionMetrics(self.iouv.cpu())

    def update_metrics(self, preds, batch) -> None:
        super().update_metrics(preds, batch)
        for image_index, pred in enumerate(preds):
            prepared = self._prepare_batch(image_index, batch)
            ratio_pad = prepared["ratio_pad"]
            gain = validation_xy_gain(prepared["ori_shape"], prepared["imgsz"], ratio_pad)
            self.tiny_metrics.update_from_prepared(self._prepare_pred(pred), prepared, gain)

    def get_stats(self) -> dict:
        results = super().get_stats()
        tiny = self.tiny_metrics.compute()
        results.update({
            "metrics/AP-tiny": tiny.map,
            "metrics/Recall-tiny": tiny.recall,
            "metrics/AP-r<8": tiny.extreme_map,
            "metrics/AP-8<=r<=16": tiny.tiny_8_16_map,
        })
        return results
```

`TinyDetectionMetrics.update` must first remove out-of-range GTs and predictions matched to those ignored GTs, then run the same class-aware IoU matching thresholds as Ultralytics. It must never label the metric COCO small. Add distributed gather support mirroring `DetectionValidator.gather_stats`.

- [ ] **Step 4: Run metrics tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_metrics.py -q`

Expected: anisotropic resize, padding invariance, boundary groups, ignored larger GT handling, empty sets, and AP/Recall accumulation pass.

- [ ] **Step 5: Commit custom validation metrics**

```powershell
git add src/ebc_qp_metrics.py src/rtdetr_ebc_qp.py tests/test_ebc_qp_metrics.py
git commit -m "Add custom EBC-QP AP-tiny validation"
```

### Task 9: D0 Diagnosis, D1-D3 Launchers, Initialization Lock, and Logs

**Files:**
- Create: `scripts/diagnose_ebc_qp.py`
- Create: `scripts/train_rtdetr_ebc_qp.py`
- Create: `tests/test_ebc_qp_cli.py`

- [ ] **Step 1: Write failing CLI/protocol tests**

```python
from scripts.train_rtdetr_ebc_qp import build_parser, build_settings, validate_protocol


def test_d2_defaults_are_frozen_ten_epoch_ten_percent_scratch_settings():
    args = build_parser().parse_args(["--stage", "d2", "--arm", "a2", "--initial-state", "init.pt"])
    settings = build_settings(args)
    assert settings["model"].endswith("configs/rtdetr-l-ebc-qp.yaml")
    assert settings["epochs"] == 10
    assert settings["fraction"] == 0.10
    assert settings["pretrained"] is False
    assert settings["seed"] == 0
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["max_det"] == 300


def test_d2_control_uses_stock_yaml_and_the_same_initial_state():
    args = build_parser().parse_args(["--stage", "d2", "--arm", "control", "--initial-state", "init.pt"])
    settings = build_settings(args)
    assert settings["model"] == "rtdetr-l.yaml"
    assert settings["epochs"] == 10
    assert settings["fraction"] == 0.10
    assert settings["seed"] == 0


def test_d3_forces_zero_ebc_and_requires_passing_d2_manifest(tmp_path):
    args = build_parser().parse_args(["--stage", "d3", "--d2-manifest", str(tmp_path / "d2.json")])
    with pytest.raises(SystemExit, match="passing D2 manifest"):
        validate_protocol(args)


def test_formal_a1_requires_a2_seed0_manifest_and_exact_initial_state(tmp_path):
    assert_formal_guard_rejects_missing_or_changed_a2_manifest(tmp_path)


def test_seed_one_rejects_unfrozen_commit_or_dataset_signature(tmp_path):
    assert_multiseed_guard_rejects_changed_artifacts(tmp_path)
```

- [ ] **Step 2: Run CLI tests and verify the missing scripts failure**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_cli.py -q`

Expected: collection fails because the EBC-QP scripts do not exist.

- [ ] **Step 3: Implement stage-specific settings and manifest guards**

```python
STAGES = {
    "d1": Stage(epochs=3, fraction=0.10, scratch=False, stock_frozen=True, lambda_ebc=0.0, inject_p2=False),
    "d2": Stage(epochs=10, fraction=0.10, scratch=True, stock_frozen=False, lambda_ebc=0.05, inject_p2=True),
    "d3": Stage(epochs=10, fraction=0.10, scratch=True, stock_frozen=False, lambda_ebc=0.0, inject_p2=True),
    "a1": Stage(epochs=100, fraction=1.00, scratch=True, stock_frozen=False, lambda_ebc=0.0, inject_p2=True),
    "a2": Stage(epochs=100, fraction=1.00, scratch=True, stock_frozen=False, lambda_ebc=0.05, inject_p2=True),
}


def build_settings(args: argparse.Namespace) -> dict:
    stage = STAGES[args.stage]
    model = "rtdetr-l.yaml" if args.arm == "control" else str(ROOT / "configs" / "rtdetr-l-ebc-qp.yaml")
    return {
        "model": model,
        "data": "VisDrone.yaml",
        "epochs": stage.epochs,
        "fraction": stage.fraction,
        "imgsz": 640,
        "batch": 8,
        "workers": args.workers,
        "device": args.device,
        "pretrained": False,
        "deterministic": True,
        "seed": args.seed,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "amp": True,
        "save": True,
        "save_period": 1,
        "val": True,
    }
```

Generate one initial-state artifact before D2 containing common stock tensors, P2 tensors generated under a separate fixed seed, config JSON, source hashes, git commit, dataset file-list/label hash, and subset-index hash. The stock control and A2 both load common tensors element-by-element and assert no missing/unexpected common keys. D1 loads archived baseline best, freezes every stock parameter, calls `eval()` on stock BN modules after every `train()` transition, and never emits an initializer accepted by D2.

Use repository-local immutable artifact names: `artifacts/ebc-qp/baseline-seed0-best.pt` for the SHA256-verified release checkpoint and `artifacts/ebc-qp/d2-seed0-initial-state.pt` for generated common initialization. The launcher command `--create-initial-state artifacts/ebc-qp/d2-seed0-initial-state.pt` creates the latter once and refuses to overwrite it.

The parser exposes `--arm control|a2` for D2 and `--arm a1|a2` for formal runs. The D2 manifest is complete only after both `control` and `a2` records reference the same common-initialization fingerprint, subset hash, seed, and augmentation protocol. A control run uses the locked stock YAML and never constructs P2 parameters.

`validate_protocol` must enforce:

- D3 accepts only a D2 manifest whose frozen criteria are marked passed;
- A1 accepts only the completed A2 seed-0 initial-state fingerprint and full protocol;
- seeds 1/2 accept only the frozen commit/config/data/evaluator signature recorded after seed 0;
- a changed signature starts a new version instead of resuming or combining trajectories.

- [ ] **Step 4: Implement D0 from archived baseline weights**

`scripts/diagnose_ebc_qp.py` loads the archived matched baseline seed-0 checkpoint and full validation set, captures all stock token centers/raw logits plus Top-300 indices, and writes one JSON summary with:

```python
record = {
    "all_token_tiny_coverage": all_token_covered / max(tiny_gt, 1),
    "stock_top300_tiny_coverage": top300_covered / max(tiny_gt, 1),
    "stock_survival_r_lt_8": survival["r_lt_8"],
    "stock_survival_r_8_16": survival["r_8_16"],
    "stock_survival_r_16_32": survival["r_16_32"],
    "stock_survival_r_gt_32": survival["r_gt_32"],
    "tiny_relevant_rank_histogram": rank_histogram,
    "tiny_final_decoder_recall": final_detected / max(tiny_gt, 1),
    "checkpoint_sha256": file_sha256(args.weights),
    "dataset_signature": dataset_signature,
    "source_sha256": SOURCE_SHA256,
}
```

Use anchor centers for all-token and Top-300 coverage. Final decoder failure is reported separately and never substituted for anchor-center coverage.

- [ ] **Step 5: Add one compact JSONL diagnostics writer**

At each train epoch end, aggregate and append only:

```python
record = {
    "epoch": trainer.epoch + 1,
    "ap_tiny": metrics["metrics/AP-tiny"],
    "tiny_recall": metrics["metrics/Recall-tiny"],
    "stock_top300_coverage": aggregate.stock_coverage,
    "local_assign_rate": aggregate.local_assign_rate,
    "p2_entry_count": aggregate.p2_entry_count,
    "n_gain": aggregate.n_gain,
    "n_loss": aggregate.n_loss,
    "v_replace": aggregate.n_gain - aggregate.n_loss,
    "effective_p2_entry_rate": aggregate.n_gain / max(aggregate.p2_entry_count, 1),
    "boundary_gap_mean": aggregate.boundary_gap_mean,
    "boundary_gap_positive_ratio": aggregate.boundary_gap_positive_ratio,
    "p2_foreground_at_50": aggregate.p2_foreground_at_50,
    "p2_unique_gt_at_50": aggregate.p2_unique_gt_at_50,
    "p2_duplicate_rate_at_50": aggregate.p2_duplicate_rate_at_50,
    "p2_background_rate_at_50": aggregate.p2_background_rate_at_50,
    "score_iou_spearman": aggregate.score_iou_spearman,
    "score_nwd_spearman": aggregate.score_nwd_spearman,
    "score_quality_sample_count": aggregate.score_quality_sample_count,
    "assigned_entry_mean_iou": aggregate.assigned_entry_mean_iou,
    "assigned_entry_mean_nwd": aggregate.assigned_entry_mean_nwd,
    "unassigned_entry_rate": aggregate.unassigned_entry_rate,
    "low_quality_entry_rate": aggregate.low_quality_entry_rate,
    "c2_p3_rms_ratio": aggregate.c2_p3_rms_ratio,
    "p2_loss": aggregate.p2_loss,
    "ebc_loss": aggregate.ebc_loss,
    "precision": metrics["metrics/precision(B)"],
    "recall": metrics["metrics/recall(B)"],
    "map50": metrics["metrics/mAP50(B)"],
    "map50_95": metrics["metrics/mAP50-95(B)"],
}
```

All diagnostic tensors are detached. NWD is metric-only with `C=12.8/640`; it does not enter v1.0 loss or ranking. Spearman values are `null` when fewer than three samples exist or either input is constant, and the sample count is always logged. `LowQualityEntryRate` uses the frozen diagnostic threshold `IoU<0.1` and never gates training. Do not serialize P2 maps or per-candidate tensors. Persist the first-200-step update-monitor trace as a test/abort artifact, not an epoch metric.

- [ ] **Step 6: Run CLI and protocol tests**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_cli.py -q`

Expected: all stage defaults, D3/A1/multiseed guards, subset hashes, identical common initialization, D1 freezing, and compact log-schema tests pass.

- [ ] **Step 7: Commit launchers and protocol guards**

```powershell
git add scripts/diagnose_ebc_qp.py scripts/train_rtdetr_ebc_qp.py tests/test_ebc_qp_cli.py
git commit -m "Add frozen EBC-QP experiment protocol"
```

### Task 10: End-to-End Integrity, Resume, and Screening Gate

**Files:**
- Modify: `tests/test_rtdetr_ebc_qp_integration.py`
- Create: `tests/test_ebc_qp_end_to_end.py`
- Modify: `docs/superpowers/specs/2026-07-22-ebc-qp-v1-design.md`

- [ ] **Step 1: Add failing end-to-end tests for the whole logic chain**

```python
def test_training_step_preserves_stock_aux_and_updates_only_allowed_p2_parameters():
    trainer = build_cpu_smoke_trainer(epoch=3)
    before = named_parameter_snapshot(trainer.model)
    trainer.run_one_optimizer_step()
    after = named_parameter_snapshot(trainer.model)
    assert changed(after, before, include="p2_adapter")
    assert changed(after, before, include="p2_bbox_head")
    assert trainer.model.ebc_head.last_state.encoder_aux_source_is_stock
    assert trainer.model.ebc_head.last_state.ordinary_query_count == 300


def test_resume_reproduces_next_batch_query_selection(tmp_path):
    uninterrupted, resumed = run_split_resume_pair(tmp_path, split_after_steps=2)
    assert torch.equal(uninterrupted.stock_indices, resumed.stock_indices)
    assert torch.equal(uninterrupted.p2_indices, resumed.p2_indices)
    assert torch.equal(uninterrupted.final_sources, resumed.final_sources)


def test_d2_gate_requires_net_unique_gt_gain_without_map_regression():
    decision = evaluate_d2_gate(make_epoch_records(
        coverage_improved=True,
        tiny_recall_improved=True,
        n_gain=8,
        n_loss=3,
        effective_entry_rate=0.12,
        max_p2_entries=17,
        final3_map_delta=0.0,
        late_decline=False,
    ))
    assert decision.passed
```

- [ ] **Step 2: Run the end-to-end tests and confirm they fail at the first uncovered integration gap**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_end_to_end.py tests/test_rtdetr_ebc_qp_integration.py -q`

Expected: tests fail on the first missing training-step/resume/gate behavior, not on fixture construction.

- [ ] **Step 3: Complete the minimal integration needed by those failures**

Implement `evaluate_d2_gate(records)` with every frozen condition as an explicit boolean and return all failure reasons. Add checkpoint RNG, optimizer, scaler, EMA, current epoch, config fingerprint, and subset sampler state restoration required for the split/resume comparison. Add a final spec appendix containing the implementation commit, test command, and exact D0/D1/D2 commands; do not change any frozen algorithm value.

- [ ] **Step 4: Run focused and full regression suites**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_ebc_qp_config.py tests/test_ebc_qp_matching.py tests/test_ebc_qp_loss.py tests/test_ebc_qp_queries.py tests/test_ebc_qp_decoder.py tests/test_rtdetr_ebc_qp_integration.py tests/test_ebc_qp_update_monitor.py tests/test_ebc_qp_metrics.py tests/test_ebc_qp_cli.py tests/test_ebc_qp_end_to_end.py -q
C:\uav_env\Scripts\python.exe -m pytest -q
```

Expected: all EBC-QP tests pass; the repository regression suite has no new failures.

- [ ] **Step 5: Run one real-data smoke command before D1**

Run:

```powershell
C:\uav_env\Scripts\python.exe scripts\train_rtdetr_ebc_qp.py --stage d1 --weights artifacts\ebc-qp\baseline-seed0-best.pt --smoke --device 0
```

Expected: one bounded smoke epoch finishes; losses and diagnostics are finite; P2 adapter and box head update; stock parameters/BN statistics remain fixed; P2 decoder entry and EBC remain zero; `last.pt` can be resumed for one additional smoke batch with identical next-batch selections.

- [ ] **Step 6: Run D0 and permit D1/D2 only after artifact review**

Run:

```powershell
C:\uav_env\Scripts\python.exe scripts\diagnose_ebc_qp.py --weights artifacts\ebc-qp\baseline-seed0-best.pt --data VisDrone.yaml --device 0
C:\uav_env\Scripts\python.exe scripts\train_rtdetr_ebc_qp.py --stage d1 --weights artifacts\ebc-qp\baseline-seed0-best.pt --device 0
C:\uav_env\Scripts\python.exe scripts\train_rtdetr_ebc_qp.py --stage d2 --arm a2 --create-initial-state artifacts\ebc-qp\d2-seed0-initial-state.pt --device 0
C:\uav_env\Scripts\python.exe scripts\train_rtdetr_ebc_qp.py --stage d2 --arm control --initial-state artifacts\ebc-qp\d2-seed0-initial-state.pt --device 0
C:\uav_env\Scripts\python.exe scripts\train_rtdetr_ebc_qp.py --stage d2 --arm a2 --initial-state artifacts\ebc-qp\d2-seed0-initial-state.pt --device 0
```

Expected: D0 writes the locked diagnosis; D1 passes gradient/isolation health; D2 runs only with matching control/A2 initialization manifests. Do not launch D3 unless the D2 gate passes, and do not launch a 100-epoch run from an unreviewed screen.

- [ ] **Step 7: Commit the verified implementation state**

```powershell
git add tests/test_ebc_qp_end_to_end.py tests/test_rtdetr_ebc_qp_integration.py docs/superpowers/specs/2026-07-22-ebc-qp-v1-design.md
git commit -m "Verify EBC-QP v1 screening chain"
```

## Plan Self-Review

- Spec coverage: Tasks 1-10 cover source locking, C2/P3 detaches, fixed coefficient-one fusion, parameter-level detached stock heads, independent P2 box head, deterministic stock/P2 matching, sparse VFL masking, zero-margin EBC, stable fixed-budget competition, stock-only encoder auxiliary loss, warm-up, custom AP-tiny, scale survival, P2 diversity, score-quality diagnostics, unique-GT gain/loss, normalized optimizer updates, resume, D0-D3, formal A1/A2, and multiseed freeze.
- Scope control: no learnable fusion scalar, P2 NMS, local-peak selection, hard-negative ranking, reserved quota, dynamic assignment, decoder/Hungarian modification, ignore-region branch, NWD training loss, or BTD-SE code changes are included.
- Type consistency: `EBCQPConfig`, `CenterMatch`, `P2LossResult`, `QuerySet`, `ReplacementStats`, `EBCQPForwardState`, `EBCQPDecoder`, `EBCQPDetectionModel`, `EBCQPTrainer`, and `EBCQPValidator` are introduced once and reused under the same names.
- Execution boundary: automated tests and one bounded smoke run precede D1/D2; 10-epoch results remain screening evidence and never enter the formal 100-epoch ablation table.

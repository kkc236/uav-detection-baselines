# CSHC-RTDETR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trainable, pre-Top-300 high-resolution sparse candidate module to RT-DETR-L and run only a gated smoke/short paired screen before any long training.

**Architecture:** The module receives the existing stride-4 C2 feature and the decoded stride-8 F3 feature. A DySample upsampler and depthwise fusion produce a small C2 candidate map. Its top-K locations create token embeddings and reference boxes that are concatenated with stock encoder proposals *before* RT-DETR's unchanged 300-query selection. A class-agnostic tiny-center loss trains the candidate-location map; final boxes and classes remain under the stock RT-DETR criterion.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics 8.4.90 RT-DETR, pytest, CUDA 12.x server.

---

## File structure

- Create `src/cshc.py`: DySample, C2 fusion, sparse-token generator, and the RT-DETR decoder subclass.
- Create `src/cshc_targets.py`: normalized tiny-box center-map target construction.
- Create `src/cshc_loss.py`: numerically stable focal loss on candidate logits.
- Create `src/rtdetr_cshc.py`: parser registration, model/trainer classes, and the auxiliary-loss integration.
- Create `configs/rtdetr-l-cshc.yaml`: RT-DETR-L graph with C2 fusion and the registered decoder.
- Create `scripts/train_rtdetr_cshc.py`: smoke, paired short-screen, and formal-run configuration construction.
- Create `scripts/audit_cshc_coverage.py`: immutable JSON coverage audit for newly generated candidates only.
- Create `tests/test_cshc.py`, `tests/test_cshc_targets.py`, `tests/test_rtdetr_cshc_integration.py`, `tests/test_cshc_training.py`, and `tests/test_cshc_coverage.py`.

### Task 1: Specify the candidate primitives with failing tests

**Files:**
- Create: `tests/test_cshc.py`
- Create: `src/cshc.py`

- [x] **Step 1: Write failing tests for the two pure network primitives.**

```python
import torch

from src.cshc import C2CandidateFusion, DySample, SparseC2CandidateGenerator


def test_dysample_preserves_shape_and_has_finite_gradients():
    module = DySample(channels=8, scale=2, groups=4)
    feature = torch.randn(2, 8, 5, 7, requires_grad=True)
    output = module(feature)
    assert output.shape == (2, 8, 10, 14)
    output.square().mean().backward()
    assert torch.isfinite(feature.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_sparse_generator_emits_topk_tokens_valid_logit_anchors_and_map():
    module = SparseC2CandidateGenerator(channels=8, hidden_dim=16, candidates=6, anchor_size=0.025)
    feature = torch.zeros(1, 8, 3, 4)
    with torch.no_grad():
        module.objectness.weight.zero_()
        module.objectness.bias.zero_()
        module.objectness.bias[0] = -2.0
        module.objectness.weight[0, 0, 1, 1] = 5.0
    feature[0, 0, 1, 2] = 1.0
    output = module(feature)
    assert output.tokens.shape == (1, 6, 16)
    assert output.class_logits.shape == (1, 6, 10)
    assert output.anchor_logits.shape == (1, 6, 4)
    assert output.objectness_logits.shape == (1, 1, 3, 4)
    assert torch.isfinite(output.anchor_logits).all()
    assert torch.all((output.anchor_logits.sigmoid() > 0) & (output.anchor_logits.sigmoid() < 1))


def test_c2_fusion_outputs_requested_channels_at_c2_resolution():
    fusion = C2CandidateFusion(in_channels=12, out_channels=8)
    output = fusion(torch.randn(2, 12, 10, 14))
    assert output.shape == (2, 8, 10, 14)
```

- [x] **Step 2: Run the new test file and verify RED.**

Run: `python -m pytest tests/test_cshc.py -q`

Expected: import failure for `src.cshc`.

- [x] **Step 3: Implement only the tested primitives in `src/cshc.py`.**

```python
@dataclass(frozen=True)
class SparseCandidates:
    tokens: torch.Tensor
    anchor_logits: torch.Tensor
    class_logits: torch.Tensor
    objectness_logits: torch.Tensor


class C2CandidateFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.out_norm(self.pointwise(F.silu(self.norm(self.depthwise(x))))))


class DySample(nn.Module):
    def __init__(self, channels: int, scale: int = 2, groups: int = 4) -> None: ...
    def forward(self, feature: torch.Tensor) -> torch.Tensor: ...  # offset Conv -> pixel shuffle -> grid_sample


class SparseC2CandidateGenerator(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, candidates: int, anchor_size: float, nc: int = 10) -> None: ...
    def forward(self, feature: torch.Tensor) -> SparseCandidates: ...  # top-K C2 cells, linear token/class/box heads
```

Initialize the class head to the stock RT-DETR class prior, the box head to zero, and the objectness bias to `log(0.01 / 0.99)`; reject tensors with the wrong rank or too few C2 cells.

- [x] **Step 4: Run the primitive tests and verify GREEN.**

Run: `python -m pytest tests/test_cshc.py -q`

Expected: `3 passed`.

### Task 2: Define tiny-center supervision with failing tests

**Files:**
- Create: `tests/test_cshc_targets.py`
- Create: `src/cshc_targets.py`
- Create: `src/cshc_loss.py`

- [x] **Step 1: Write failing target and loss tests.**

```python
import torch

from src.cshc_loss import focal_binary_logits
from src.cshc_targets import build_tiny_center_targets


def test_tiny_centers_land_in_the_expected_cells_and_large_boxes_are_excluded():
    target = build_tiny_center_targets(
        bboxes=torch.tensor([[0.25, 0.75, 0.02, 0.02], [0.75, 0.25, 0.20, 0.20]]),
        batch_idx=torch.tensor([0, 0]), batch_size=1, height=4, width=4, tiny_area_threshold=0.0025,
    )
    assert target.shape == (1, 1, 4, 4)
    assert target[0, 0, 3, 1] == 1
    assert target.sum() == 1


def test_focal_binary_logits_is_finite_and_penalizes_wrong_high_confidence():
    target = torch.tensor([[[[1.0, 0.0]]]])
    good = focal_binary_logits(torch.tensor([[[[6.0, -6.0]]]]), target)
    bad = focal_binary_logits(torch.tensor([[[[-6.0, 6.0]]]]), target)
    assert torch.isfinite(good) and torch.isfinite(bad)
    assert good < bad
```

- [x] **Step 2: Run RED.**

Run: `python -m pytest tests/test_cshc_targets.py -q`

Expected: import failure for the two new modules.

- [x] **Step 3: Implement target construction and focal loss.**

`build_tiny_center_targets` must use normalized `xywh`, require finite values in `[0,1]`, use `floor(center * size)` clamped to the grid, and write one positive only for `w*h <= 0.0025`. `focal_binary_logits` must be `BCEWithLogitsLoss(reduction="none")` multiplied by `(1 - p_t)**2` and use `alpha=0.25` for positives.

- [x] **Step 4: Run GREEN.**

Run: `python -m pytest tests/test_cshc_targets.py -q`

Expected: `2 passed`.

### Task 3: Inject new candidates before the stock 300-query selection

**Files:**
- Modify: `src/cshc.py`
- Create: `tests/test_rtdetr_cshc_integration.py`

- [x] **Step 1: Write the failing decoder-contract tests.**

```python
import torch

from src.cshc import CSHCRTDDETRDecoder


def test_decoder_keeps_three_stock_memory_levels_but_selects_from_extra_c2_candidates():
    decoder = CSHCRTDDETRDecoder(nc=10, ch=[64, 256, 256, 256], nq=300, candidates=512)
    decoder.train()
    raw = decoder([
        torch.randn(2, 64, 160, 160),
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 256, 20, 20),
    ], batch={"cls": torch.empty(0, 1, dtype=torch.long), "bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0, dtype=torch.long)})
    assert raw[0].shape[-2:] == (300, 4)
    assert raw[1].shape[-2:] == (300, 10)
    assert decoder.last_candidates is not None
    assert decoder.last_candidates.objectness_logits.shape[-2:] == (160, 160)
```

- [x] **Step 2: Run RED.**

Run: `python -m pytest tests/test_rtdetr_cshc_integration.py -q`

Expected: import failure for `CSHCRTDDETRDecoder`.

- [x] **Step 3: Implement `CSHCRTDDETRDecoder`.**

Subclass `RTDETRDecoder`; call `super().__init__(nc=nc, ch=tuple(ch[1:]), ...)`, store `SparseC2CandidateGenerator(channels=ch[0], hidden_dim=hd, candidates=candidates, anchor_size=0.025, nc=nc)`, and override `forward` plus decoder-input preparation. The override must:

```python
c2_feature, *stock_features = x
feats, shapes = self._get_encoder_input(stock_features)
stock_features = self.enc_output(self.valid_mask * feats)
stock_scores = self.enc_score_head(stock_features)
candidates = self.c2_candidates(c2_feature)
all_features = torch.cat((stock_features, candidates.tokens), dim=1)
all_scores = torch.cat((stock_scores, candidates.class_logits), dim=1)
all_anchors = torch.cat((self.anchors.expand(bs, -1, -1), candidates.anchor_logits), dim=1)
topk = torch.topk(all_scores.max(-1).values, self.num_queries, dim=1).indices
```

Gather `topk` from all three tensors, create references as `enc_bbox_head(top_features) + top_anchors`, return the unchanged five-item RT-DETR loss contract, and use only `feats/shapes` from the stock three levels in deformable cross-attention. Store `last_candidates` for loss/audit; do not alter final score postprocessing.

- [x] **Step 4: Run GREEN.**

Run: `python -m pytest tests/test_rtdetr_cshc_integration.py -q`

Expected: `1 passed`.

### Task 4: Register the decoder, add the model graph, and attach its loss

**Files:**
- Create: `src/rtdetr_cshc.py`
- Create: `configs/rtdetr-l-cshc.yaml`
- Modify: `tests/test_rtdetr_cshc_integration.py`

- [x] **Step 1: Add failing model tests.**

```python
from pathlib import Path
import torch

from src.rtdetr_cshc import CSHCDetectionModel


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-cshc.yaml"


def test_yaml_builds_registered_cshc_decoder_and_preserves_300_queries():
    model = CSHCDetectionModel(CONFIG, ch=3, nc=10, verbose=False).eval()
    assert model.model[-1].num_queries == 300
    with torch.no_grad():
        prediction = model.predict(torch.rand(1, 3, 160, 160))
    assert prediction[0].shape == (1, 300, 6)


def test_model_loss_adds_finite_candidate_map_term():
    model = CSHCDetectionModel(CONFIG, ch=3, nc=10, verbose=False).train()
    batch = {"img": torch.rand(1, 3, 160, 160), "cls": torch.tensor([[0]]), "bboxes": torch.tensor([[0.25, 0.75, 0.02, 0.02]]), "batch_idx": torch.tensor([0])}
    loss, items = model.loss(batch)
    assert torch.isfinite(loss)
    assert items.shape[-1] == 4
```

- [x] **Step 2: Run RED.**

Run: `python -m pytest tests/test_rtdetr_cshc_integration.py -q`

Expected: import failure for `src.rtdetr_cshc`.

- [x] **Step 3: Implement registration/model/YAML.**

`register_cshc_decoder()` must temporarily assign `ultralytics.nn.tasks.RTDETRDecoder = CSHCRTDDETRDecoder` before `RTDETRDetectionModel.__init__`, so Ultralytics passes the YAML input-channel list to the subclass. The model loss must call `super().loss`, build a tiny-center target matching `last_candidates.objectness_logits`, add `0.25 * focal_binary_logits(...)`, and report `("giou_loss", "cls_loss", "l1_loss", "c2_candidate_loss")`.

The YAML must preserve the stock F3/F4/F5 neck, create `DySample(F3, 2)`, project the stride-4 backbone output to 64 channels, concatenate the two tensors, fuse to 64 channels, and feed `[c2_candidate, F3, F4, F5]` to `RTDETRDecoder`.

- [x] **Step 4: Run GREEN.**

Run: `python -m pytest tests/test_cshc.py tests/test_cshc_targets.py tests/test_rtdetr_cshc_integration.py -q`

Expected: all listed tests pass.

### Task 5: Create deterministic coverage auditing and launch configuration

**Files:**
- Create: `scripts/audit_cshc_coverage.py`
- Create: `scripts/train_rtdetr_cshc.py`
- Create: `tests/test_cshc_coverage.py`
- Create: `tests/test_cshc_training.py`

- [x] **Step 1: Write failing tests for immutable coverage accounting and training modes.**

```python
def test_coverage_counts_only_new_c2_candidates_against_supplied_stock_misses():
    from scripts.audit_cshc_coverage import summarize_new_candidate_coverage
    result = summarize_new_candidate_coverage(
        missed=[{"image_id": "a", "class_id": 1, "box": [0.4, 0.4, 0.6, 0.6]}],
        candidates=[{"image_id": "a", "class_id": 1, "box": [0.41, 0.41, 0.59, 0.59]}], iou_threshold=0.5,
    )
    assert result == {"missed_tiny": 1, "covered_by_new_candidates": 1, "coverage": 1.0}


def test_screen_mode_uses_five_epochs_and_never_claims_formal_results():
    from scripts.train_rtdetr_cshc import build_settings, build_parser
    settings = build_settings(build_parser().parse_args(["--screen"]))
    assert settings["epochs"] == 5
    assert settings["val"] is True
    assert settings["nms"] is False
    assert settings["max_det"] == 300
```

- [x] **Step 2: Run RED.**

Run: `python -m pytest tests/test_cshc_coverage.py tests/test_cshc_training.py -q`

Expected: import failure for the two scripts.

- [x] **Step 3: Implement the audit and training settings.**

The audit must use class-aware one-to-one IoU at `0.50`, receive the frozen stock-missed JSONL as input, count only the C2 branch candidates before the combined Top-300, and write identity hashes plus the ratio. The trainer must expose `--smoke` (`1 epoch`, `fraction=0.01`), `--screen` (`5 epochs`, `fraction=0.10`), and `--formal` (`100 epochs`, `fraction=1.0`); all modes use `imgsz=640`, `max_det=300`, `nms=False`, `seed=0`, and an explicit project path.

- [x] **Step 4: Run GREEN.**

Run: `python -m pytest tests/test_cshc_coverage.py tests/test_cshc_training.py -q`

Expected: `2 passed`.

### Task 6: Verify locally, then use the GPU only for gated execution

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-cshc-querydet.md` (check the boxes with actual command output)

- [x] **Step 1: Run the focused suite and parser smoke locally.**

Run: `python -m pytest tests/test_cshc.py tests/test_cshc_targets.py tests/test_rtdetr_cshc_integration.py tests/test_cshc_coverage.py tests/test_cshc_training.py -q`

Expected: all tests pass before any remote upload.

- [ ] **Step 2: Push the clean branch, checkout it on the GPU host, and run only the one-epoch 1% smoke.**

Run remotely: `python scripts/train_rtdetr_cshc.py --smoke --project /root/data/uav/runs/cshc-smoke --device 0 --workers 8 --batch 8`

Expected: checkpoint, results file, finite C2 candidate loss, and native validator output; no formal claim.

- [ ] **Step 3: Run the coverage audit and make the go/no-go decision.**

Run remotely: `python scripts/audit_cshc_coverage.py --checkpoint <smoke-checkpoint> --stock-misses <frozen-bqp-misses.jsonl> --output /root/data/uav/runs/cshc-smoke/coverage.json`

Expected: a new-candidate-only coverage record. Do not reuse BQP's 7.43% replacement-pair metric and do not start `--screen` if coverage is non-positive, model outputs are nonfinite, or the baseline path does not reproduce.

- [ ] **Step 4: Only after a positive smoke and a matched five-epoch control, run the five-epoch 10% screen.**

Run both arms with identical seed/data/batch: stock RT-DETR continuation and `scripts/train_rtdetr_cshc.py --screen`. Compare only Ultralytics native mAP/AP50/AP75/precision/recall and candidate coverage; SBR is diagnostic only.

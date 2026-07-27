# PVC Core Implementation Plan

> **Superseded:** Continue from the approved PLEC freeze using
> `2026-07-27-plec-implementation.md`. This file is retained as implementation
> history for the two already completed geometry/reference commits.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and locally verify the first GCMV-RTDETR network module, PVC (Phase-Preserving View Canonicalizer), as an isolated trainable PyTorch component before any GRCA, QCVR, RT-DETR integration, YAML work, or server training.

**Architecture:** A frozen geometry builder maps the four source-resolution 60%-overlap local views onto the full-view P3 lattice with exact crop and letterbox transforms. PVC samples a fixed row-major `3 x 3` phase pattern for every valid view, adds learned phase/view/magnification-edge embeddings, compresses the nine samples with grouped/depthwise/pointwise convolutions, and performs masked learned overlap fusion. A separate bilinear/uniform reference path supplies the required non-learned ablation. The module returns one canonical feature `C3`, a geometric `valid_count`, and a fused `edge_prior`; every location with no valid local evidence is exactly zero.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90 geometry conventions, NumPy, pytest.

---

## Frozen scope and contracts

This plan implements PVC only.

PVC may:

- consume four local semantic P3 tensors produced by a shared backbone/Hybrid Encoder;
- consume exact, non-trainable crop and letterbox metadata;
- resample local features onto the global P3 lattice;
- learn feature, phase, view, magnification, edge, and overlap transformations;
- backpropagate into both PVC parameters and the four local P3 tensors.

PVC must not:

- read or alter the global P3 feature;
- read P4/P5;
- select image crops;
- predict objectness, boxes, classes, or query scores;
- add or remove RT-DETR queries;
- import GRCA, QCVR, or a custom RT-DETR model;
- change any Ultralytics package file or model YAML.

The first implementation uses these public contracts:

```python
@dataclass(frozen=True)
class PVCGeometry:
    sample_grid: torch.Tensor       # [B, 4, 9, Hg, Wg, 2], align_corners=False
    sample_valid: torch.Tensor      # bool [B, 4, 9, Hg, Wg]
    center_valid: torch.Tensor      # bool [B, 4, 1, Hg, Wg]
    subcell_offset: torch.Tensor    # [B, 4, 9, 2, Hg, Wg], values in [-0.5, 0.5)
    magnification: torch.Tensor     # [B, 4, 2, Hg, Wg], positive and non-quantized
    edge_distance: torch.Tensor     # [B, 4, 1, Hg, Wg], values in [0, 1]
    local_feature_shape: tuple[int, int]
    global_feature_shape: tuple[int, int]

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PVCGeometry": ...


@dataclass(frozen=True)
class PVCOutput:
    canonical: torch.Tensor         # C3: [B, C, Hg, Wg]
    valid_count: torch.Tensor       # [B, 1, Hg, Wg]
    edge_prior: torch.Tensor        # [B, 1, Hg, Wg]
    overlap_weights: torch.Tensor   # [B, 4, 1, Hg, Wg], diagnostic/ablation output
```

```python
def build_pvc_geometry(
    *,
    source_shapes: Sequence[tuple[int, int]],  # (height, width), one per image
    tiles: Sequence[Sequence[Tile]],           # exactly TL/TR/BL/BR per image
    global_transforms: Sequence[LetterboxTransform],
    local_transforms: Sequence[Sequence[LetterboxTransform]],
    global_feature_shape: tuple[int, int],
    local_feature_shape: tuple[int, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> PVCGeometry:
    ...
```

```python
class PhasePreservingViewCanonicalizer(nn.Module):
    def forward(
        self,
        local_features: Sequence[torch.Tensor],  # four [B, C, Hl, Wl]
        geometry: PVCGeometry,
    ) -> PVCOutput:
        ...
```

The phase order is frozen as row-major:

```text
(-1/3,-1/3), (0,-1/3), (+1/3,-1/3),
(-1/3,   0), (0,   0), (+1/3,   0),
(-1/3,+1/3), (0,+1/3), (+1/3,+1/3)
```

Offsets are fractions of a global P3 cell. They are mapped through the actual global letterbox inverse, crop translation, local letterbox, and local feature shape. No integer scale ratio is allowed.

---

### Task 1: Exact tensor geometry for the global-to-local P3 correspondence

**Files:**

- Create: `src/gcmv_geometry.py`
- Create: `tests/test_gcmv_geometry.py`
- Reuse unchanged: `src/sbr_geometry.py`

- [ ] Write an import-failing test for the public geometry contract:

```python
from src.gcmv_geometry import PVCGeometry, build_pvc_geometry
from src.sbr_geometry import LetterboxTransform, Tile
```

- [ ] Add `test_identity_transform_places_center_phase_on_same_lattice()`. Use one batch item, four identical full-image tiles, explicit no-padding transforms, and equal global/local feature shapes. Assert:

```python
center = geometry.sample_grid[0, 0, 4]
expected_x = 2.0 * (torch.arange(width) + 0.5) / width - 1.0
expected_y = 2.0 * (torch.arange(height) + 0.5) / height - 1.0
torch.testing.assert_close(center[..., 0], expected_x.expand(height, width))
torch.testing.assert_close(center[..., 1], expected_y[:, None].expand(height, width))
torch.testing.assert_close(
    geometry.subcell_offset[0, 0, 4],
    torch.zeros_like(geometry.subcell_offset[0, 0, 4]),
)
```

- [ ] Run the focused test and confirm it fails because `src.gcmv_geometry` does not exist:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_geometry.py -q
```

- [ ] Create `PVCGeometry` and the strict argument validators. Reject:

  - an empty batch;
  - anything other than four tiles/transforms per image;
  - tile order whose `Tile.index` is not `[0, 1, 2, 3]`;
  - source, network, or feature dimensions that are non-positive;
  - non-positive letterbox gains;
  - a dtype that is not floating point;
  - a tile extending beyond its source image;
  - inconsistent transform source dimensions.

- [ ] Implement `PVCGeometry.to()` so cached geometry can follow feature
  device/dtype without reconstruction. Floating tensors change dtype, boolean
  masks remain boolean, shape tuples remain unchanged, and returned geometry
  tensors still have `requires_grad=False`.

- [ ] Implement the coordinate chain using tensor arithmetic:

```text
global P3 phase point
  -> global network pixel coordinate
  -> inverse global letterbox
  -> source-image coordinate
  -> subtract local tile origin
  -> local network pixel coordinate
  -> local feature coordinate
  -> align_corners=False normalized grid
```

Use pixel-center formulas, never `round()`, `floor()`, or an assumed stride:

```python
x_global_net = (x_global_feature + 0.5) * global_network_width / Wg
x_source = (x_global_net - global_pad_x) / global_gain_x
x_local_net = (x_source - tile.left) * local_gain_x + local_pad_x
x_local_feature = x_local_net * Wl / local_network_width - 0.5
x_grid = 2.0 * (x_local_feature + 0.5) / Wl - 1.0
```

- [ ] Add `test_phase_order_is_row_major_and_spans_one_global_cell()`. Under identity geometry, assert all nine displacements equal the frozen phase table divided by `(Wg, Hg)` in normalized grid coordinates.

- [ ] Add `test_non_integer_magnification_is_preserved()`. Construct transforms whose analytic local/global feature-scale ratio is non-integer, then assert the two `magnification` channels match the analytic derivatives within `1e-6` and are not truncated.

- [ ] Implement magnification as the positive derivative from one global feature cell to local feature cells:

```python
mag_x = (
    global_network_width / Wg / global_gain_x
    * local_gain_x * Wl / local_network_width
)
mag_y = (
    global_network_height / Hg / global_gain_y
    * local_gain_y * Hl / local_network_height
)
```

- [ ] Add `test_60_percent_tiles_produce_expected_coverage_and_edge_distance()`. Use `overlapping_tiles()` rather than reimplementing tile placement. Assert:

  - exclusive corner regions have `valid_count == 1`;
  - horizontal/vertical overlaps have at least two valid views;
  - the central four-view overlap has `valid_count == 4`;
  - center `edge_distance` is larger than boundary `edge_distance`;
  - all edge distances lie in `[0, 1]`.

- [ ] Implement masks from the exact source tile bounds and the normalized local grid. A phase sample is valid only when it lies inside both. Define `center_valid` from phase index `4`.

- [ ] Implement normalized edge distance at the center phase:

```python
distance = min(
    x_source - tile.left,
    tile.right - x_source,
    y_source - tile.top,
    tile.bottom - y_source,
)
edge_distance = clamp(
    distance / (0.5 * min(tile.width, tile.height)),
    min=0.0,
    max=1.0,
)
```

Mask invalid centers to exact zero.

- [ ] Add tests that geometry tensors have `requires_grad=False`, stay on the requested device/dtype (boolean masks remain boolean), and contain no NaN/Inf.

- [ ] Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_geometry.py tests/test_sbr_geometry.py -q
```

- [ ] Commit the green geometry slice:

```powershell
git add src/gcmv_geometry.py tests/test_gcmv_geometry.py
git commit -m "feat: add exact PVC feature geometry"
```

---

### Task 2: Shared phase sampler and bilinear/uniform reference

**Files:**

- Create: `src/gcmv_pvc.py`
- Create: `tests/test_gcmv_pvc.py`

- [ ] Write failing imports for:

```python
from src.gcmv_pvc import (
    PVCOutput,
    sample_local_phases,
    uniform_bilinear_canonicalize,
)
```

- [ ] Add `test_phase_sampler_matches_analytic_ramp_values()`. Fill each local feature with an `x + 10*y + 100*view_id` ramp, use simple identity geometry, and assert all nine sampled phases equal analytic bilinear values.

- [ ] Run and confirm the import failure:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_pvc.py -q
```

- [ ] Implement strict local-feature validation:

  - exactly four tensors;
  - every tensor is `[B, C, Hl, Wl]`;
  - shared batch, channel, spatial, device, and dtype;
  - spatial size equals `geometry.local_feature_shape`;
  - batch/global output sizes match the geometry;
  - floating feature dtype only.

Geometry and features must share a device. Geometry is normally moved through
`PVCGeometry.to(device=features.device, dtype=features.dtype)` by the caller.
When CUDA autocast supplies FP16 features with an FP32 grid, the sampler must
use PyTorch's autocast-safe `grid_sample` path; outside autocast, explicitly
convert the grid to the feature dtype before sampling.

- [ ] Implement one vectorized `grid_sample` operation by stacking the views into `[B*4, C, Hl, Wl]` and flattening phase into the output height. Call:

```python
F.grid_sample(
    stacked_features,
    flattened_grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
)
```

Return `[B, 4, 9, C, Hg, Wg]` in the frozen phase order and multiply by `sample_valid` after sampling.

- [ ] Add `test_invalid_phase_samples_are_exact_zero()` and an `unittest.mock.patch` assertion that `align_corners=False` is explicitly passed.

- [ ] Add `test_uniform_reference_uses_center_phase_and_uniform_valid_views()`. Give each view a distinct constant. Assert the reference canonical feature is:

  - that view's value where one view is valid;
  - the arithmetic mean where views overlap;
  - exact zero where no view is valid.

- [ ] Implement `uniform_bilinear_canonicalize()` from center phase index `4` only. Compute safe uniform weights without a softmax over all-invalid values:

```python
valid = geometry.center_valid.to(feature_dtype)
denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
weights = valid / denominator
```

Return `PVCOutput`; derive `valid_count` from geometry and `edge_prior` from the same uniform weights.

- [ ] Add a backward test showing the canonical sum has finite, nonzero gradients on every contributing local feature and no gradient path through geometry tensors.

- [ ] Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_pvc.py -q
```

- [ ] Commit:

```powershell
git add src/gcmv_pvc.py tests/test_gcmv_pvc.py
git commit -m "feat: add PVC phase sampler and reference"
```

---

### Task 3: Trainable per-view phase encoder

**Files:**

- Modify: `src/gcmv_pvc.py`
- Modify: `tests/test_gcmv_pvc.py`

- [ ] Add a failing construction test for:

```python
module = PhasePreservingViewCanonicalizer(
    channels=256,
    embedding_hidden=64,
    overlap_hidden=64,
    use_phase_embedding=True,
    use_view_embedding=True,
    use_metadata_embedding=True,
    learned_overlap=True,
)
```

Assert the module contains:

- one `nn.Embedding(4, 256)`;
- a `2 -> 64 -> 256` phase-offset MLP;
- a `3 -> 64 -> 256` magnification/edge MLP;
- a grouped `1 x 1` phase reducer with `in_channels=9*256`, `out_channels=256`, `groups=256`;
- a depthwise `3 x 3` convolution with `groups=256`;
- a pointwise `1 x 1` convolution;
- a trainable overlap head;
- an output normalization layer.

Use channel-only normalization at each spatial location (BHWC
`nn.LayerNorm(channels)` wrapped back to BCHW), not `GroupNorm`, so the number
of invalid spatial cells cannot alter valid-cell statistics.

- [ ] Add `test_embeddings_are_added_per_sample_before_phase_reduction()`. Set tiny deterministic channel counts and hand-set weights so the expected contribution of phase, view, and metadata embeddings can be computed exactly.

- [ ] Implement sample enrichment:

```text
enriched =
    sampled_feature
  + view_embedding[view_id]
  + phase_mlp(subcell_offset)
  + metadata_mlp([log2(mag_x), log2(mag_y), edge_distance])
```

Invalid phase samples must be remasked after enrichment so embedding biases cannot leak into invalid locations.

- [ ] Reorder from `[B, 4, 9, C, Hg, Wg]` to channel-major phase groups `[B*4, 9*C, Hg, Wg]` so each grouped `1 x 1` filter sees the nine phases of exactly one channel. Do not use phase-major flattening.

- [ ] Apply:

```text
grouped 1x1 phase reduction
  -> SiLU
  -> depthwise 3x3 spatial mixing
  -> SiLU
  -> pointwise 1x1 projection
```

Remask the encoded view with `center_valid` after every bias-producing boundary needed to preserve exact invalid zeros.

- [ ] Add `test_channel_major_grouping_does_not_mix_feature_channels()` with hand-set grouped weights.

- [ ] Add `test_ablation_flags_disable_only_the_requested_embedding()`. The four required experiment variants must be constructible without editing source:

```text
bilinear/uniform reference                -> uniform_bilinear_canonicalize
phase samples without phase/view embeds   -> phase=False, view=False
PVC without learned overlap               -> learned_overlap=False
full PVC                                  -> all True
```

- [ ] Run the focused suite and commit:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_pvc.py -q
git add src/gcmv_pvc.py tests/test_gcmv_pvc.py
git commit -m "feat: add trainable PVC phase encoder"
```

---

### Task 4: Masked learned overlap fusion and exact output contract

**Files:**

- Modify: `src/gcmv_pvc.py`
- Modify: `tests/test_gcmv_pvc.py`

- [ ] Add `test_learned_overlap_weights_sum_to_one_only_on_valid_views()`. Assert:

```python
invalid_weights = weights.masked_select(~center_valid)
assert torch.equal(invalid_weights, torch.zeros_like(invalid_weights))
torch.testing.assert_close(
    weights.sum(dim=1),
    (valid_count > 0).to(weights.dtype),
)
```

- [ ] Add `test_empty_locations_remain_exact_zero_despite_biases()`. Deliberately set all MLP, convolution, overlap-head, and normalization biases nonzero, then assert all four outputs are exactly zero at mask-empty locations.

- [ ] Implement the overlap logit head from each encoded view concatenated with its `edge_distance`. Use a safe masked softmax:

```python
masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
weights = torch.softmax(masked_logits, dim=1) * valid.to(logits.dtype)
weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
    torch.finfo(logits.dtype).eps
)
```

For `learned_overlap=False`, use uniform valid-view weights from Task 2.

- [ ] Fuse:

```python
canonical = (weights * encoded_views).sum(dim=1)
canonical = output_norm(canonical)
canonical = canonical * any_valid
valid_count = center_valid.sum(dim=1).to(canonical.dtype)
edge_prior = (weights * edge_distance).sum(dim=1) * any_valid
```

Return all values through `PVCOutput`. `overlap_weights` is exposed only for diagnostics and ablations; it is not an extra prediction branch.

- [ ] Add output-shape tests for `B=1` and `B=2`, non-square global P3, and non-square local P3.

- [ ] Add error-contract tests for wrong view count, channels, spatial shape, batch size, device, geometry dtype, and malformed masks.

- [ ] Run and commit:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_pvc.py -q
git add src/gcmv_pvc.py tests/test_gcmv_pvc.py
git commit -m "feat: add masked PVC overlap fusion"
```

---

### Task 5: Prove that PVC is a trainable network module, not post-processing

**Files:**

- Modify: `tests/test_gcmv_pvc.py`
- Create: `tests/test_pvc_innovation_isolation.py`

- [ ] Add `test_full_pvc_backpropagates_into_features_and_every_parameter_family()`. Use a geometry region with at least two valid views and a spatially varying loss so softmax does not cancel. Assert finite nonzero gradients for:

  - all four contributing local feature tensors;
  - `view_embedding`;
  - both layers of `phase_mlp`;
  - both layers of `metadata_mlp`;
  - grouped phase reducer;
  - depthwise spatial mixer;
  - pointwise projection;
  - overlap head;
  - output normalization.

- [ ] Add `test_pvc_parameter_budget_is_lightweight()`. For `channels=256`, freeze the acceptance limit:

```python
trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
assert 0 < trainable <= 200_000
```

If the implementation exceeds the limit, simplify the hidden projections; do not raise the limit without revising the design specification.

- [ ] Add `test_pvc_is_isolated_from_future_modules_and_detector_logic()`. Parse/import the module and assert it does not import:

  - `src.gcmv_grca`;
  - `src.gcmv_qcvr`;
  - `ultralytics.models.rtdetr`;
  - any SADED post-processing or fusion module.

- [ ] Add a mixed-precision smoke test when CUDA is available. Under `torch.autocast("cuda", dtype=torch.float16)`, assert finite output and gradients; skip cleanly on CPU-only machines.

- [ ] Add `test_state_dict_roundtrip_preserves_output()` in eval mode.

- [ ] Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_geometry.py `
  tests/test_gcmv_pvc.py `
  tests/test_pvc_innovation_isolation.py -q
```

- [ ] Commit:

```powershell
git add tests/test_gcmv_pvc.py tests/test_pvc_innovation_isolation.py
git commit -m "test: prove PVC trainability and isolation"
```

---

### Task 6: Local verification and handoff gate before Module 2

**Files:**

- Update: `docs/superpowers/specs/2026-07-27-gcmv-rtdetr-design.md` only if implementation reveals a necessary contract correction
- Create: `docs/evidence/pvc-local-verification.md`

- [ ] Run formatting/static sanity checks without rewriting unrelated files:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m compileall -q src/gcmv_geometry.py src/gcmv_pvc.py
git diff --check
```

- [ ] Run focused PVC and reused geometry tests:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_geometry.py `
  tests/test_gcmv_pvc.py `
  tests/test_pvc_innovation_isolation.py `
  tests/test_sbr_geometry.py -q
```

- [ ] Run the full local regression suite:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest -q
```

- [ ] Record in `docs/evidence/pvc-local-verification.md`:

  - git commit and environment versions;
  - exact focused/full commands and pass/skip counts;
  - trainable parameter count;
  - input/output tensor contracts;
  - proof that empty locations are exact zero;
  - proof that all parameter families receive finite nonzero gradients;
  - whether CUDA autocast ran or skipped;
  - explicit statement that no RT-DETR integration or accuracy claim has yet been made.

- [ ] Inspect the final diff. It must contain only PVC geometry, PVC module, PVC tests, the verification record, and an unavoidable design-contract correction if one was discovered.

- [ ] Commit the verified PVC module:

```powershell
git add docs/evidence/pvc-local-verification.md
git commit -m "docs: record PVC local verification"
```

- [ ] Stop at the Module 1 gate. Do not begin GRCA until all of the following are true:

  - focused PVC tests pass;
  - the full local regression suite has no new failure;
  - PVC has trainable parameters below the frozen budget;
  - all trainable parameter families and contributing inputs receive finite nonzero gradients;
  - non-integer geometry, overlap masking, and exact empty-output behavior are proven;
  - the user has received the PVC verification result.

Server training is intentionally outside this plan. A green local PVC module proves structural correctness and trainability, not an AP improvement.

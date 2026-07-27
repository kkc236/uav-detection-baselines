# PLEC Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish and locally verify PLEC, the first frozen GCMV-RTDETR network module, without implementing GGLF, PEG, RT-DETR integration, or server training.

**Architecture:** Exact crop/letterbox geometry maps four local semantic P3 tensors to nine phase samples per full-view P3 cell. A trainable PLEC enriches those samples with phase, view, magnification, and boundary metadata; performs grouped/depthwise/pointwise feature encoding; and fuses valid overlapping views with a masked learned softmax. A parameter-free bilinear/uniform path remains the scientific reference.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90 geometry conventions, pytest.

---

## File structure and frozen boundary

Files owned by this plan:

```text
src/gcmv_geometry.py                 exact PLEC correspondence tensors
src/gcmv_plec.py                     sampler, reference, and trainable PLEC only
tests/test_gcmv_geometry.py          geometry contract
tests/test_gcmv_plec.py              PLEC behavior and gradients
tests/test_plec_innovation_isolation.py
docs/evidence/plec-local-verification.md
```

PLEC must not import RT-DETR, GGLF, PEG, SADED post-processing, query logic, or
training code. The public names are:

```python
PLECGeometry
build_plec_geometry
PLECOutput
PhasePreservingLocalEvidenceCanonicalizer
```

No PVC compatibility aliases are retained in this isolated branch; stale names
should fail tests so paper and code terminology cannot drift.

## Completed prerequisites

- [x] Exact non-integer crop/letterbox feature geometry implemented and tested
  in commit `be29f983`.
- [x] One-call nine-phase `grid_sample` and parameter-free bilinear/uniform
  reference implemented and tested in commit `ba832a8a`.
- [x] Frozen PLEC/GGLF/PEG design approved in
  `docs/superpowers/specs/2026-07-27-gcmv-rtdetr-frozen-design.md`.

---

### Task 1: Migrate internal PVC names to the frozen PLEC vocabulary

**Files:**

- Modify: `src/gcmv_geometry.py`
- Move: `src/gcmv_pvc.py` -> `src/gcmv_plec.py`
- Modify: `tests/test_gcmv_geometry.py`
- Move: `tests/test_gcmv_pvc.py` -> `tests/test_gcmv_plec.py`

- [ ] **Step 1: Preserve the current red construction test**

The existing uncommitted test must import:

```python
from src.gcmv_plec import (
    ChannelLayerNorm,
    PLECOutput,
    PhasePreservingLocalEvidenceCanonicalizer,
    sample_local_phases,
    uniform_bilinear_canonicalize,
)
```

It must construct `PhasePreservingLocalEvidenceCanonicalizer(...)` and currently
fail because the class is not implemented.

- [ ] **Step 2: Perform the mechanical public rename**

Use these exact replacements:

```text
PVCGeometry                              -> PLECGeometry
build_pvc_geometry                       -> build_plec_geometry
PVCOutput                                -> PLECOutput
src/gcmv_pvc.py                          -> src/gcmv_plec.py
tests/test_gcmv_pvc.py                   -> tests/test_gcmv_plec.py
```

Do not change tensor math during this step.

- [ ] **Step 3: Run the completed geometry/reference slice**

Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_geometry.py `
  tests/test_gcmv_plec.py `
  tests/test_sbr_geometry.py -q
```

Expected: geometry and reference tests pass; the construction test remains red
only because `ChannelLayerNorm` and the trainable PLEC class are absent.

- [ ] **Step 4: Commit terminology migration only after the next task turns the
  construction test green**

The red test and mechanical rename stay uncommitted until Task 2.

---

### Task 2: Construct the frozen trainable PLEC layer families

**Files:**

- Modify: `src/gcmv_plec.py`
- Modify: `tests/test_gcmv_plec.py`

- [ ] **Step 1: Verify the existing construction test is red**

Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_plec.py::test_full_plec_constructs_the_frozen_trainable_layers -q
```

Expected: import failure for `ChannelLayerNorm` or
`PhasePreservingLocalEvidenceCanonicalizer`.

- [ ] **Step 2: Implement channel-only normalization**

Add:

```python
class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.norm = nn.LayerNorm(channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.norm.normalized_shape[0]:
            raise ValueError("ChannelLayerNorm expects BxCxHxW with configured channels")
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
```

- [ ] **Step 3: Implement the constructor**

Add `nn` imports and this public signature:

```python
class PhasePreservingLocalEvidenceCanonicalizer(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        embedding_hidden: int = 64,
        overlap_hidden: int = 64,
        *,
        use_phase_embedding: bool = True,
        use_view_embedding: bool = True,
        use_metadata_embedding: bool = True,
        learned_overlap: bool = True,
    ) -> None:
```

Reject non-positive channel/hidden values. Store the four flags and construct:

```python
self.view_embedding = nn.Embedding(4, channels) if use_view_embedding else None
self.phase_mlp = (
    nn.Sequential(
        nn.Linear(2, embedding_hidden),
        nn.SiLU(),
        nn.Linear(embedding_hidden, channels),
    )
    if use_phase_embedding
    else None
)
self.metadata_mlp = (
    nn.Sequential(
        nn.Linear(3, embedding_hidden),
        nn.SiLU(),
        nn.Linear(embedding_hidden, channels),
    )
    if use_metadata_embedding
    else None
)
self.phase_reducer = nn.Conv2d(
    9 * channels, channels, kernel_size=1, groups=channels, bias=False
)
self.spatial_mixer = nn.Conv2d(
    channels, channels, kernel_size=3, padding=1, groups=channels
)
self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
self.overlap_head = (
    nn.Sequential(
        nn.Conv2d(channels + 1, overlap_hidden, kernel_size=1),
        nn.SiLU(),
        nn.Conv2d(overlap_hidden, 1, kernel_size=1),
    )
    if learned_overlap
    else None
)
self.output_norm = ChannelLayerNorm(channels)
```

- [ ] **Step 4: Run the construction test**

Run the Step 1 command.

Expected: PASS.

- [ ] **Step 5: Commit the frozen terminology and layer structure**

```powershell
git add src/gcmv_geometry.py src/gcmv_plec.py `
  tests/test_gcmv_geometry.py tests/test_gcmv_plec.py
git commit -m "feat: freeze PLEC naming and layer structure"
```

---

### Task 3: Encode phase, view, magnification, and boundary evidence

**Files:**

- Modify: `src/gcmv_plec.py`
- Modify: `tests/test_gcmv_plec.py`

- [ ] **Step 1: Write a failing channel-major encoding test**

Use `channels=2`, replace SiLU with identity for the test, zero every embedding,
and configure grouped reducer weights so output channel 0 reads only its nine
channel-0 phases and output channel 1 reads only its nine channel-1 phases.
Capture the reducer input with a forward pre-hook and assert its channel order:

```python
expected = sampled.permute(0, 1, 3, 2, 4, 5).reshape(
    batch * 4, channels * 9, height, width
)
torch.testing.assert_close(captured[0], expected)
```

- [ ] **Step 2: Run the new test and verify red**

Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_plec.py::test_channel_major_phase_groups_do_not_mix_channels -q
```

Expected: FAIL because `forward()` is absent.

- [ ] **Step 3: Implement sample enrichment**

Inside `forward()`, validate configured channels, call
`sample_local_phases()`, and construct:

```python
sample_mask = geometry.sample_valid.unsqueeze(3).to(sampled.dtype)
enriched = sampled

if self.view_embedding is not None:
    view_ids = torch.arange(4, device=sampled.device)
    view = self.view_embedding(view_ids).view(1, 4, 1, self.channels, 1, 1)
    enriched = enriched + view

if self.phase_mlp is not None:
    phase = geometry.subcell_offset.permute(0, 1, 2, 4, 5, 3)
    phase = self.phase_mlp(phase).permute(0, 1, 2, 5, 3, 4)
    enriched = enriched + phase

if self.metadata_mlp is not None:
    eps = torch.finfo(geometry.magnification.dtype).eps
    metadata = torch.cat(
        (
            torch.log2(geometry.magnification.clamp_min(eps)),
            geometry.edge_distance,
        ),
        dim=2,
    ).permute(0, 1, 3, 4, 2)
    metadata = self.metadata_mlp(metadata).permute(0, 1, 4, 2, 3)
    enriched = enriched + metadata.unsqueeze(2)

enriched = enriched * sample_mask
```

Geometry floating tensors must be moved/cast to the sampled tensor's
device/dtype before MLP use.

- [ ] **Step 4: Implement channel-major phase compression**

```python
phase_input = enriched.permute(0, 1, 3, 2, 4, 5).reshape(
    batch_size * 4,
    self.channels * 9,
    global_height,
    global_width,
)
center_mask = geometry.center_valid.reshape(
    batch_size * 4, 1, global_height, global_width
).to(phase_input.dtype)

encoded = F.silu(self.phase_reducer(phase_input)) * center_mask
encoded = F.silu(self.spatial_mixer(encoded)) * center_mask
encoded = self.pointwise(encoded) * center_mask
encoded = encoded.reshape(
    batch_size, 4, self.channels, global_height, global_width
)
```

- [ ] **Step 5: Add and pass ablation-flag tests**

Assert:

```python
assert no_phase.phase_mlp is None
assert no_view.view_embedding is None
assert no_metadata.metadata_mlp is None
assert uniform_overlap.overlap_head is None
```

The remaining enabled families must still exist.

- [ ] **Step 6: Run the focused suite and commit**

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_plec.py -q
git add src/gcmv_plec.py tests/test_gcmv_plec.py
git commit -m "feat: encode phase-preserving local evidence"
```

---

### Task 4: Add masked learned overlap fusion and exact-zero outputs

**Files:**

- Modify: `src/gcmv_plec.py`
- Modify: `tests/test_gcmv_plec.py`

- [ ] **Step 1: Write failing learned-overlap tests**

Create a geometry with:

- one cell covered by one view;
- one cell covered by two views;
- one cell covered by no view.

Assert output shapes and:

```python
invalid = output.overlap_weights.masked_select(~geometry.center_valid)
assert torch.equal(invalid, torch.zeros_like(invalid))
torch.testing.assert_close(
    output.overlap_weights.sum(dim=1),
    (output.valid_count > 0).to(output.overlap_weights.dtype),
)
```

- [ ] **Step 2: Write the bias-leak regression**

Set every existing bias, including LayerNorm bias, to `2.0`. At an all-invalid
cell assert exact zero for `canonical`, `valid_count`, `edge_prior`, and all
four overlap weights.

- [ ] **Step 3: Run the two tests and verify red**

Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_plec.py -k "learned_overlap or bias_leak" -q
```

Expected: FAIL because `forward()` does not yet return `PLECOutput`.

- [ ] **Step 4: Implement safe overlap weights**

```python
valid = geometry.center_valid.to(device=encoded.device)
edge = geometry.edge_distance.to(device=encoded.device, dtype=encoded.dtype)

if self.overlap_head is None:
    numeric_valid = valid.to(encoded.dtype)
    weights = numeric_valid / numeric_valid.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
else:
    logits_in = torch.cat((encoded, edge), dim=2).reshape(
        batch_size * 4,
        self.channels + 1,
        global_height,
        global_width,
    )
    logits = self.overlap_head(logits_in).reshape(
        batch_size, 4, 1, global_height, global_width
    )
    masked_logits = logits.masked_fill(
        ~valid, torch.finfo(logits.dtype).min
    )
    weights = torch.softmax(masked_logits, dim=1) * valid.to(logits.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).eps
    )
```

- [ ] **Step 5: Implement the exact output contract**

```python
valid_count = valid.sum(dim=1).to(encoded.dtype)
any_valid = (valid_count > 0).to(encoded.dtype)
canonical = (weights * encoded).sum(dim=1)
canonical = self.output_norm(canonical) * any_valid
edge_prior = (weights * edge).sum(dim=1) * any_valid

return PLECOutput(
    canonical=canonical,
    valid_count=valid_count,
    edge_prior=edge_prior,
    overlap_weights=weights,
)
```

- [ ] **Step 6: Add strict geometry-shape validation**

Before sampling, require exact shapes:

```text
sample_grid       [B,4,9,Hg,Wg,2]
sample_valid      [B,4,9,Hg,Wg]
center_valid      [B,4,1,Hg,Wg]
subcell_offset    [B,4,9,2,Hg,Wg]
magnification     [B,4,2,Hg,Wg]
edge_distance     [B,4,1,Hg,Wg]
```

Reject malformed tensors with `ValueError` naming the field.

- [ ] **Step 7: Add batch/non-square and input-error tests**

Test `B=2`, non-square local/global shapes, wrong view count, channel count,
batch size, spatial shape, integer features, and device mismatch.

- [ ] **Step 8: Run and commit**

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest tests/test_gcmv_plec.py -q
git add src/gcmv_plec.py tests/test_gcmv_plec.py
git commit -m "feat: fuse valid PLEC overlap evidence"
```

---

### Task 5: Prove PLEC trainability, budget, serialization, and isolation

**Files:**

- Modify: `tests/test_gcmv_plec.py`
- Create: `tests/test_plec_innovation_isolation.py`

- [ ] **Step 1: Write the full-gradient test**

Use random non-symmetric features, an overlap region with four valid views, and
a spatial/channel weighting tensor:

```python
loss = (
    output.canonical
    * torch.linspace(
        0.3, 1.7, output.canonical.numel(), device=output.canonical.device
    ).reshape_as(output.canonical)
).square().mean()
loss.backward()
```

Assert finite nonzero gradients for all four input features and parameters whose
names start with:

```text
view_embedding
phase_mlp
metadata_mlp
phase_reducer
spatial_mixer
pointwise
overlap_head
output_norm
```

- [ ] **Step 2: Run the gradient test and verify red if any family is detached**

Run:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_plec.py::test_full_plec_backpropagates_to_every_parameter_family -q
```

Expected before fixes: FAIL naming the detached family. Fix production tensor
permutations/masks, never weaken the assertion.

- [ ] **Step 3: Add the frozen parameter-budget test**

```python
module = PhasePreservingLocalEvidenceCanonicalizer(channels=256)
trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
assert 0 < trainable <= 200_000
```

- [ ] **Step 4: Add state-dict roundtrip**

Save `state_dict()` in memory, load it into a fresh same-config module in eval
mode, and assert all `PLECOutput` tensors match for identical inputs.

- [ ] **Step 5: Add CUDA autocast smoke**

When CUDA is available, move features, geometry, and module to CUDA. Under:

```python
with torch.autocast("cuda", dtype=torch.float16):
    output = module(features, geometry)
    loss = output.canonical.square().mean()
loss.backward()
```

Assert finite output and gradients. Skip only when CUDA is unavailable.

- [ ] **Step 6: Add source-level isolation test**

Parse `src/gcmv_plec.py` imports with `ast`. Fail if it imports any path
containing:

```text
ultralytics.models.rtdetr
gcmv_gglf
gcmv_peg
saded
sbr_fusion
query
```

Also assert the source contains no class or function named PVC, GRCA, or QCVR.

- [ ] **Step 7: Run and commit**

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_geometry.py `
  tests/test_gcmv_plec.py `
  tests/test_plec_innovation_isolation.py -q
git add tests/test_gcmv_plec.py tests/test_plec_innovation_isolation.py `
  src/gcmv_plec.py
git commit -m "test: prove PLEC trainability and isolation"
```

---

### Task 6: Record fresh local verification and stop before server work

**Files:**

- Create: `docs/evidence/plec-local-verification.md`

- [ ] **Step 1: Run static verification**

```powershell
& 'C:\uav_env\Scripts\python.exe' -m compileall -q `
  src/gcmv_geometry.py src/gcmv_plec.py
git diff --check
```

Expected: exit code `0`.

- [ ] **Step 2: Run the focused gate**

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_geometry.py `
  tests/test_gcmv_plec.py `
  tests/test_plec_innovation_isolation.py `
  tests/test_sbr_geometry.py -q
```

Expected: zero failures.

- [ ] **Step 3: Run the full local regression suite**

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest -q
```

Expected: zero new failures. Existing environmental skips are recorded rather
than hidden.

- [ ] **Step 4: Write the evidence record**

Record:

- branch and commit;
- Python/PyTorch/Ultralytics versions;
- focused/full commands with pass/fail/skip counts;
- trainable parameter count;
- exact input/output contracts;
- exact-zero empty-region result;
- nonzero gradient families;
- CUDA autocast status;
- explicit statements that RT-DETR integration, GGLF, PEG, dataset training,
  AP improvement, and publication claims have not been performed.

- [ ] **Step 5: Commit evidence**

```powershell
git add docs/evidence/plec-local-verification.md
git commit -m "docs: record PLEC local verification"
```

- [ ] **Step 6: Stop**

Do not start a server or begin GGLF. Report the local PLEC gate result to the
user and wait for the next instruction.

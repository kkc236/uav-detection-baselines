# GCTE-RTDETR G0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `GCTENetworkModule` as one trainable network module containing DSTE, GCQL, and APFG, then run the shortest valid G0 anchor-reproduction and learned-gate screen from the mature RT-DETR-L baseline.

**Architecture:** Development starts from commit `81e6f495db58e4bf5eeb0984d8fb886e44badc0b`, which already contains the verified GCMV/SADED geometry, evaluator, and mature-baseline protocol. The global detector remains frozen. GCTE consumes global/local decoder evidence, maps local queries into global coordinates, and learns a bounded residual around the fixed successful SADED admission policy.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90, pytest, CUDA 12.1, RTX 4090.

---

## File map

Create:

- `src/gcte_types.py`: immutable tensor contracts used by all GCTE stages.
- `src/gcte_dste.py`: detection-supervised local query adapter.
- `src/gcte_gcql.py`: deterministic box lifting plus learnable geometry embedding.
- `src/gcte_apfg.py`: fixed-anchor protected gate and bounded learned residual.
- `src/gcte_network.py`: the single exported `GCTENetworkModule`.
- `src/gcte_cache.py`: sealed cached-evidence schema and SHA-256 validation.
- `src/gcte_loss.py`: direct local-query and gate supervision.
- `src/rtdetr_gcte.py`: mature RT-DETR evidence extraction and frozen integration.
- `scripts/cache_gcte_evidence.py`: cache global/local decoder evidence.
- `scripts/train_gcte_g0.py`: module-only G0-B trainer.
- `scripts/evaluate_gcte_g0.py`: Control / Anchor / Method-On / Method-Off evaluation.
- `scripts/run_gcte_g0.sh`: fail-closed server workflow.
- `configs/rtdetr-l-gcte.yaml`: one top-level GCTE configuration block.
- `tests/test_gcte_types.py`
- `tests/test_gcte_dste.py`
- `tests/test_gcte_gcql.py`
- `tests/test_gcte_apfg.py`
- `tests/test_gcte_network.py`
- `tests/test_gcte_cache.py`
- `tests/test_gcte_loss.py`
- `tests/test_rtdetr_gcte.py`
- `tests/test_gcte_g0_cli.py`

Reuse without semantic changes:

- `src/saded.py`
- `src/saded_stock_postprocess.py`
- `src/sbr_geometry.py`
- `src/sbr_metrics.py`
- `src/gcmv_warmstart.py`
- `scripts/evaluate_gcmv_warmstart.py`

## Task 1: Isolated implementation worktree

**Files:**

- Base commit: `81e6f495db58e4bf5eeb0984d8fb886e44badc0b`
- Worktree: `C:\Users\16946\Documents\OBJECTIVE CHECK PAPER-gcte`

- [ ] **Step 1: Create the isolated branch and worktree**

```powershell
git worktree add `
  -b codex/gcte-rtdetr-g0 `
  "C:\Users\16946\Documents\OBJECTIVE CHECK PAPER-gcte" `
  81e6f495db58e4bf5eeb0984d8fb886e44badc0b
```

Expected: a clean worktree on `codex/gcte-rtdetr-g0`.

- [ ] **Step 2: Bring the frozen design and plan into the worktree**

```powershell
git -C "C:\Users\16946\Documents\OBJECTIVE CHECK PAPER-gcte" `
  cherry-pick 700b709 bce9f13
```

After this plan is committed, cherry-pick its commit as a third commit.

- [ ] **Step 3: Verify the base suite before editing**

```powershell
python -m pytest -q
```

Expected: the full suite from commit `81e6f495` passes.

## Task 2: Tensor contracts

**Files:**

- Create: `src/gcte_types.py`
- Create: `tests/test_gcte_types.py`

- [ ] **Step 1: Write failing shape and finiteness tests**

```python
import pytest
import torch

from src.gcte_types import QueryEvidence


def test_query_evidence_accepts_matching_shapes():
    value = QueryEvidence(
        queries=torch.zeros(2, 8, 256),
        logits=torch.zeros(2, 8, 10),
        boxes=torch.full((2, 8, 4), 0.5),
        quality=torch.ones(2, 8, 1),
    )
    assert value.batch_size == 2
    assert value.query_count == 8


def test_query_evidence_rejects_nonfinite_boxes():
    boxes = torch.zeros(1, 2, 4)
    boxes[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        QueryEvidence(
            queries=torch.zeros(1, 2, 256),
            logits=torch.zeros(1, 2, 10),
            boxes=boxes,
            quality=torch.ones(1, 2, 1),
        )
```

- [ ] **Step 2: Run the tests and confirm import failure**

```powershell
python -m pytest tests/test_gcte_types.py -q
```

Expected: FAIL because `src.gcte_types` does not exist.

- [ ] **Step 3: Implement validated immutable contracts**

Implement:

```python
@dataclass(frozen=True)
class QueryEvidence:
    queries: torch.Tensor
    logits: torch.Tensor
    boxes: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        if self.queries.ndim != 3:
            raise ValueError("queries must be [B,Q,C]")
        batch, count, _ = self.queries.shape
        expected = (batch, count)
        if self.logits.shape[:2] != expected:
            raise ValueError("logits must share [B,Q]")
        if self.boxes.shape != (*expected, 4):
            raise ValueError("boxes must be normalized xywh [B,Q,4]")
        if self.quality.shape != (*expected, 1):
            raise ValueError("quality must be [B,Q,1]")
        for tensor in (self.queries, self.logits, self.boxes, self.quality):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("query evidence must be finite floating point")
```

Also define `CropGeometry`, `GCTEStageOutput`, and `GCTENetworkOutput` with explicit shape validation.

- [ ] **Step 4: Run the focused tests**

```powershell
python -m pytest tests/test_gcte_types.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gcte_types.py tests/test_gcte_types.py
git commit -m "Add GCTE tensor contracts"
```

## Task 3: DSTE local expert adapter

**Files:**

- Create: `src/gcte_dste.py`
- Create: `tests/test_gcte_dste.py`

- [ ] **Step 1: Write failing identity, gradient, and residual-cap tests**

```python
def test_dste_zero_initialization_is_identity():
    module = DetectionSupervisedTinyExpert(query_dim=256, num_classes=10)
    evidence = make_query_evidence(batch=2, queries=12)
    output = module(evidence)
    torch.testing.assert_close(output.queries, evidence.queries)


def test_dste_parameters_receive_gradient():
    module = DetectionSupervisedTinyExpert(query_dim=256, num_classes=10)
    evidence = make_query_evidence(batch=1, queries=4, requires_grad=True)
    output = module(evidence)
    output.logits.sum().backward()
    assert any(p.grad is not None for p in module.parameters())


def test_dste_query_residual_is_bounded():
    module = DetectionSupervisedTinyExpert(
        query_dim=256,
        num_classes=10,
        residual_cap=0.2,
    )
    evidence = make_query_evidence(batch=1, queries=4)
    delta = module(evidence).queries - evidence.queries
    assert delta.abs().max() <= 0.2 + 1e-6
```

- [ ] **Step 2: Confirm the tests fail**

```powershell
python -m pytest tests/test_gcte_dste.py -q
```

Expected: FAIL because DSTE is missing.

- [ ] **Step 3: Implement the minimal DSTE**

Use `LayerNorm(256) -> Linear(256,128) -> SiLU -> Linear(128,256)` with the final layer zero initialized. Add copied class, box, and quality heads. Return a new `QueryEvidence`.

The query residual must be:

```python
adapted = evidence.queries + self.residual_cap * torch.tanh(
    self.adapter(self.norm(evidence.queries))
)
```

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_gcte_dste.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gcte_dste.py tests/test_gcte_dste.py
git commit -m "Add detection-supervised tiny expert adapter"
```

## Task 4: GCQL geometry lifting

**Files:**

- Create: `src/gcte_gcql.py`
- Create: `tests/test_gcte_gcql.py`

- [ ] **Step 1: Write failing coordinate and identity tests**

Cover:

- a full-image identity crop;
- TL/TR/BL/BR crop translations;
- boxes touching crop boundaries;
- zero-initialized geometry residual;
- gradients reaching the geometry MLP but not crop metadata.

Core assertion:

```python
expected = torch.tensor([[[0.25, 0.25, 0.10, 0.10]]])
output = module(local, geometry_for_top_left_half())
torch.testing.assert_close(output.boxes, expected, atol=1e-6, rtol=0)
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest tests/test_gcte_gcql.py -q
```

Expected: FAIL because GCQL is missing.

- [ ] **Step 3: Implement deterministic lifting plus learnable embedding**

For normalized local `xywh`:

```python
global_x = crop_x0 + local_x * crop_width
global_y = crop_y0 + local_y * crop_height
global_w = local_w * crop_width
global_h = local_h * crop_height
```

Normalize by source width/height, then add a bounded zero-initialized geometry MLP residual to queries. Preserve boxes deterministically; the MLP never modifies geometry.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_gcte_gcql.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/gcte_gcql.py tests/test_gcte_gcql.py
git commit -m "Add geometry-canonical query lifting"
```

## Task 5: APFG anchored protected gate

**Files:**

- Create: `src/gcte_apfg.py`
- Create: `tests/test_gcte_apfg.py`

- [ ] **Step 1: Write failing protection and anchor tests**

Tests must prove:

- learned residual disabled reproduces the fixed anchor;
- global non-tiny entries remain byte-identical;
- local candidates larger than the emitted-size threshold are rejected;
- local boundary fragments overlapping protected global entries are rejected;
- zero-initialized residual does not change anchor scores;
- APFG parameters receive gradients when residual learning is enabled.

Representative assertion:

```python
output = gate(
    global_evidence,
    local_evidence,
    learned_residual_enabled=False,
)
torch.testing.assert_close(
    output.protected_global_boxes,
    expected_protected_boxes,
    rtol=0,
    atol=0,
)
assert output.anchor_fingerprint == expected_fingerprint
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest tests/test_gcte_apfg.py -q
```

Expected: FAIL because APFG is missing.

- [ ] **Step 3: Implement the fixed anchor**

Port the validated SADED-SM rules without changing thresholds:

- effective-size threshold `16`;
- same-class IoU match threshold `0.5`;
- protected overlap IoS threshold `0.5`;
- stable score/source/query/index order;
- final maximum detections `300`.

- [ ] **Step 4: Add the bounded residual gate**

Use:

```python
delta_score, admit_logit = self.head(features).split((1, 1), dim=-1)
delta_score = self.residual_cap * torch.tanh(delta_score)
admission = torch.sigmoid(admit_logit)
score = anchor_mask * base_score * torch.exp(delta_score) * admission
```

Initialize `delta_score` to zero and initialize `admit_logit` so the enabled network reproduces the fixed anchor within `1e-6`.

- [ ] **Step 5: Run focused tests**

```powershell
python -m pytest tests/test_gcte_apfg.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/gcte_apfg.py tests/test_gcte_apfg.py
git commit -m "Add anchored protected fusion gate"
```

## Task 6: Single top-level GCTENetworkModule

**Files:**

- Create: `src/gcte_network.py`
- Create: `tests/test_gcte_network.py`
- Create: `configs/rtdetr-l-gcte.yaml`

- [ ] **Step 1: Write failing composition tests**

Test:

- exactly one exported top-level module;
- `dste`, `gcql`, and `apfg` are registered submodules;
- `enabled=False` returns global evidence exactly;
- each internal stage can be disabled for ablation;
- all enabled stages receive gradients;
- config values match the frozen design.

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest tests/test_gcte_network.py -q
```

Expected: FAIL because the top-level module is missing.

- [ ] **Step 3: Implement composition**

```python
class GCTENetworkModule(nn.Module):
    def forward(...):
        if not self.enabled:
            return baseline_output(global_evidence)
        local = self.dste(local_evidence) if self.dste_enabled else local_evidence
        canonical = self.gcql(local, crop_geometry)
        return self.apfg(
            global_evidence,
            canonical,
            learned_residual_enabled=learned_residual_enabled,
        )
```

The public import must be:

```python
from src.gcte_network import GCTENetworkModule
```

- [ ] **Step 4: Add the one-block YAML configuration**

Add only the frozen `gcte:` block from the design, including internal ablation switches.

- [ ] **Step 5: Run focused tests**

```powershell
python -m pytest tests/test_gcte_network.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/gcte_network.py tests/test_gcte_network.py configs/rtdetr-l-gcte.yaml
git commit -m "Compose GCTE as one network module"
```

## Task 7: Direct supervision and cached evidence

**Files:**

- Create: `src/gcte_loss.py`
- Create: `src/gcte_cache.py`
- Create: `tests/test_gcte_loss.py`
- Create: `tests/test_gcte_cache.py`

- [ ] **Step 1: Write failing loss tests**

Prove:

- matched tiny local queries reduce classification and box losses;
- protected non-tiny overlaps produce protection loss;
- unmatched background local queries produce admission negatives;
- no loss term sends gradients into frozen global evidence.

- [ ] **Step 2: Implement the loss**

Return:

```python
@dataclass(frozen=True)
class GCTELoss:
    total: torch.Tensor
    local_cls: torch.Tensor
    local_l1: torch.Tensor
    local_giou: torch.Tensor
    admission: torch.Tensor
    quality: torch.Tensor
    protect: torch.Tensor
```

The G0-B total is:

\[
\mathcal L
=
\mathcal L_{admit}
+0.5\mathcal L_{quality}
+0.25\mathcal L_{protect}.
\]

G0-C adds the standard local RT-DETR classification, L1, and GIoU terms.

- [ ] **Step 3: Write failing cache integrity tests**

Test schema version, source checkpoint SHA-256, dataset signature, view count, tensor shapes, record count, and per-shard SHA-256.

- [ ] **Step 4: Implement the sealed cache**

Use one manifest JSON plus `.pt` shards. Reject any missing, extra, or mismatched shard before training.

- [ ] **Step 5: Run focused tests**

```powershell
python -m pytest tests/test_gcte_loss.py tests/test_gcte_cache.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/gcte_loss.py src/gcte_cache.py tests/test_gcte_loss.py tests/test_gcte_cache.py
git commit -m "Add GCTE supervision and sealed evidence cache"
```

## Task 8: RT-DETR evidence extraction

**Files:**

- Create: `src/rtdetr_gcte.py`
- Create: `scripts/cache_gcte_evidence.py`
- Create: `tests/test_rtdetr_gcte.py`

- [ ] **Step 1: Write failing integration tests**

Prove:

- the formal baseline loads without changing detector tensors;
- global and four local views use the same detector state;
- BatchNorm buffers remain unchanged;
- cached evidence is detached;
- each view retains at most 64 local queries after deterministic preselection;
- the detector has no gradient during G0-B.

- [ ] **Step 2: Implement read-only evidence extraction**

Reuse the existing GCMV view construction and BatchNorm preservation. Extract decoder query embeddings, logits, boxes, and quality under `torch.no_grad()`.

- [ ] **Step 3: Implement cache CLI**

Required arguments:

```text
--checkpoint
--data
--split train|val
--output
--batch 8
--device 0
--max-local-queries 64
```

Print the final manifest SHA-256 and cache record count.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_rtdetr_gcte.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/rtdetr_gcte.py scripts/cache_gcte_evidence.py tests/test_rtdetr_gcte.py
git commit -m "Extract frozen RT-DETR evidence for GCTE"
```

## Task 9: G0 trainer and four-state evaluator

**Files:**

- Create: `scripts/train_gcte_g0.py`
- Create: `scripts/evaluate_gcte_g0.py`
- Create: `tests/test_gcte_g0_cli.py`

- [ ] **Step 1: Write failing CLI and gate tests**

Test:

- module-only optimizer contains no detector parameter;
- fixed AMP scale is 128;
- seed is 0;
- output includes Control, Raw-Union, Anchor, Method-On, and Method-Off;
- advance gate uses exact unrounded values;
- failure exits nonzero and writes `G0_FAILED`;
- success writes `G0_COMPLETE`.

- [ ] **Step 2: Implement module-only training**

G0-B defaults:

```text
epochs=3
batch=64 cached records
optimizer=AdamW
lr=1e-3
weight_decay=1e-4
seed=0
deterministic=True
amp_scale=128
```

The trainer saves `best-module.pt`, `last-module.pt`, `results.csv`, and a manifest.

- [ ] **Step 3: Implement evaluation and deltas**

Write `evaluation/g0-five-state.json` with:

- absolute metrics;
- Method-On − Control;
- Method-On − Anchor;
- Method-On − Method-Off;
- Anchor − Control;
- latency and candidate counts;
- all frozen gate booleans.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_gcte_g0_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/train_gcte_g0.py scripts/evaluate_gcte_g0.py tests/test_gcte_g0_cli.py
git commit -m "Add GCTE G0 training and adjudication"
```

## Task 10: Fail-closed server workflow

**Files:**

- Create: `scripts/run_gcte_g0.sh`
- Test: `tests/test_gcte_g0_cli.py`

- [ ] **Step 1: Add workflow contract tests**

Assert the script:

- validates code commit and dirty state;
- validates baseline SHA-256;
- runs preflight tests;
- caches train and val evidence once;
- runs anchor reproduction before training;
- runs G0-B only if anchor reproduction passes;
- never launches fresh100;
- never shuts down or restarts the server.

- [ ] **Step 2: Implement the workflow**

Status files:

```text
PIPELINE_STATUS
G0_FAILED
G0_COMPLETE
checksums.sha256
```

Use `set -euo pipefail` and write a distinct failed stage before exiting.

- [ ] **Step 3: Run full local verification**

```powershell
python -m pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` produces no output.

- [ ] **Step 4: Commit**

```powershell
git add scripts/run_gcte_g0.sh tests/test_gcte_g0_cli.py
git commit -m "Add fail-closed GCTE G0 workflow"
```

## Task 11: Server deployment and G0 execution

**Files:**

- Local artifact directory: `artifacts/gcte-g0-<commit>`
- Remote source directory: `/home/ubuntu/gcte-g0-<commit>`
- Remote output directory: `/home/ubuntu/gcte-g0-output-<commit>`

- [ ] **Step 1: Package the exact committed source**

```powershell
git archive --format=tar.gz `
  --output "$env:TEMP\gcte-g0-source.tar.gz" `
  HEAD
```

Record the archive SHA-256.

- [ ] **Step 2: Deploy without altering existing GCMV results**

Upload to a new source and output directory. Do not reuse or delete `/home/ubuntu/gcmv-warmstart-output-7d44a725`.

- [ ] **Step 3: Run server preflight**

Run:

```bash
python -m pytest -q
nvidia-smi
df -h /
```

Require zero test failures, an RTX 4090, and enough disk for caches and checkpoints.

- [ ] **Step 4: Launch the bounded G0 workflow**

Use the existing formal baseline:

```text
/home/ubuntu/matched-baseline-best-epoch-0100.pt
```

Do not launch any 100-epoch job.

- [ ] **Step 5: Monitor to a terminal state**

Inspect:

- `PIPELINE_STATUS`;
- `G0_FAILED` / `G0_COMPLETE`;
- cache logs;
- anchor reproduction log;
- trainer log;
- evaluation log;
- GPU utilization.

- [ ] **Step 6: Verify and download results**

On success or failure:

```bash
sha256sum -c checksums.sha256
```

Download manifests, logs, module weights, and `evaluation/g0-five-state.json` to the local artifact directory. Keep the server running.

## Task 12: Final scientific decision

**Files:**

- Create: `artifacts/gcte-g0-<commit>/RESULTS.md`

- [ ] **Step 1: Apply frozen gates**

Required:

- `Δ AP-tiny-SBR >= +0.010`;
- `Δ mAP50-95 >= +0.005`;
- `Δ tiny recall >= +0.020`;
- `Δ AP-medium-SBR >= -0.002`;
- `Δ AP-large-SBR >= -0.005`;
- learned gate positive against Raw Local Union;
- Method-Off reproduces Anchor;
- protected global predictions have zero drift;
- latency ratio `<=3.0`.

- [ ] **Step 2: Write one of two decisions**

If every core gate passes:

```text
GCTE_G0_ADVANCE
```

Otherwise:

```text
GCTE_G0_STOP
```

Do not reinterpret failed thresholds after reading the result.

- [ ] **Step 3: Verify the local evidence package**

Recompute all local SHA-256 values and compare with the remote manifest before reporting the decision.


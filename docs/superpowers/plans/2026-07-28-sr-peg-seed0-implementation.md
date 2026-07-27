# SR-PEG Seed0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GCQF's third stage with the trainable SR-PEG medium-protection gate, generate supervised train10 evidence, and run one sealed seed0 G0 evaluation.

**Architecture:** Keep the frozen RT-DETR detector, geometry projector, and local-to-global query interaction. Add one SR-PEG stage with local tiny/risk/residual heads and a reverse-attention global-retain head; train only GCQF on cached evidence, calibrate three thresholds on a deterministic 518/129 train10 split, and evaluate once on the existing 548-image validation cache.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90, CUDA 12.1, pytest, VisDrone YOLO labels, existing SBR evaluator.

---

## File map

- Create `src/sr_peg.py`: SR-PEG tensor output and trainable third-stage module.
- Create `src/sr_peg_targets.py`: source-frame tiny, risk, and retain targets.
- Create `src/sr_peg_routing.py`: learned routing around the sealed Fixed-SADED fallback.
- Create `scripts/calibrate_sr_peg_g0.py`: deterministic 27-point train10 calibration.
- Create `scripts/run_sr_peg_seed0.py`: fail-closed seed0 stage orchestration.
- Modify `src/gcqf.py`: register SR-PEG as GCQF's third child and expose outputs.
- Modify `src/gcqf_cache.py`: support optional supervised v2 train records while reading the sealed v1 validation cache.
- Modify `src/gcqf_training.py`: collate SR-PEG targets and deterministic 518/129 split.
- Modify `src/gcqf_loss.py`: add three weighted BCE-with-logits terms.
- Modify `scripts/cache_gcqf_evidence.py`: attach SR-PEG targets to train records.
- Modify `scripts/train_gcqf_g0.py`: seed0-only supervised module training.
- Modify `scripts/evaluate_gcqf_g0.py`: route five states with calibrated SR-PEG outputs.
- Modify `configs/rtdetr-l-gcte.yaml`: freeze third-stage name and dimensions.

### Task 1: Trainable SR-PEG stage

**Files:**
- Create: `src/sr_peg.py`
- Modify: `src/gcqf.py`
- Modify: `configs/rtdetr-l-gcte.yaml`
- Test: `tests/test_sr_peg.py`
- Test: `tests/test_gcqf.py`

- [ ] **Step 1: Write the failing SR-PEG shape, bypass, bounds, and gradient tests**

```python
def test_sr_peg_emits_four_trainable_query_outputs():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    output = module(
        canonical_queries=torch.randn(2, 12, 32),
        global_context=torch.randn(2, 12, 32),
        geometry_embedding=torch.randn(2, 12, 32),
        local_scores=torch.full((2, 12, 1), 0.5),
        global_queries=torch.randn(2, 3, 32),
        global_boxes=torch.full((2, 3, 4), 0.2),
        global_scores=torch.full((2, 3, 1), 0.5),
        local_valid_mask=torch.ones(2, 12, dtype=torch.bool),
        residual_enabled=True,
    )
    assert output.tiny_utility_logits.shape == (2, 12, 1)
    assert output.non_tiny_risk_logits.shape == (2, 12, 1)
    assert output.global_retain_logits.shape == (2, 3, 1)
    assert output.score_residual.abs().max() <= 1
    output.adjusted_local_scores.sum().backward()
    assert module.local_trunk[0].weight.grad is not None
    assert module.global_attention.in_proj_weight.grad is not None
```

```python
def test_gcqf_registers_exactly_three_stages_with_sr_peg_third():
    module = GCQF(query_dim=32, num_classes=3, num_heads=4)
    assert tuple(dict(module.named_children())) == (
        "geometry_projector",
        "query_interaction",
        "sr_peg",
    )
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest -q tests/test_sr_peg.py tests/test_gcqf.py
```

Expected: import or child-name failures because `ScaleRiskProtectedEvidenceGate` does not exist.

- [ ] **Step 3: Implement the minimal registered module**

Define:

```python
@dataclass(frozen=True)
class SRPEGOutput:
    tiny_utility_logits: torch.Tensor
    non_tiny_risk_logits: torch.Tensor
    global_retain_logits: torch.Tensor
    score_residual: torch.Tensor
    adjusted_local_scores: torch.Tensor


class ScaleRiskProtectedEvidenceGate(nn.Module):
    def __init__(self, *, query_dim: int, num_heads: int, residual_eta: float = 0.2):
        self.local_trunk = nn.Sequential(
            nn.Linear(query_dim * 3 + 1, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.tiny_utility_head = nn.Linear(query_dim, 1)
        self.non_tiny_risk_head = nn.Linear(query_dim, 1)
        self.score_residual_head = nn.Linear(query_dim, 1)
        self.global_attention = nn.MultiheadAttention(
            query_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.global_box_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.global_retain_head = nn.Sequential(
            nn.Linear(query_dim * 2 + 64 + 1, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1),
        )
```

Zero-initialize the three final local heads and the retain head. When
`residual_enabled=False`, return the original local score tensor object and a
zero residual without disabling utility/risk/retain logits.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest -q tests/test_sr_peg.py tests/test_gcqf.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/sr_peg.py src/gcqf.py configs/rtdetr-l-gcte.yaml tests/test_sr_peg.py tests/test_gcqf.py
git commit -m "Add trainable SR-PEG third stage"
```

### Task 2: Source-frame SR-PEG supervision

**Files:**
- Create: `src/sr_peg_targets.py`
- Test: `tests/test_sr_peg_targets.py`

- [ ] **Step 1: Write failing target tests**

```python
def test_targets_distinguish_true_tiny_from_underestimated_medium():
    targets = build_sr_peg_targets(
        global_boxes=torch.tensor([[[0.50, 0.50, 0.02, 0.02]]]),
        global_logits=torch.tensor([[[9.0, -9.0]]]),
        local_boxes=torch.tensor([[[0.50, 0.50, 0.02, 0.02]]]),
        local_logits=torch.tensor([[[9.0, -9.0]]]),
        gt_boxes=torch.tensor([[0.50, 0.50, 0.08, 0.08]]),
        gt_classes=torch.tensor([0]),
        source_shape=(640, 640),
    )
    assert targets.local_non_tiny_risk.item() == 1.0
    assert targets.local_tiny_utility.item() == 0.0
    assert targets.global_retain.item() == 1.0
```

Also test an exact 12 px GT gives positive soft tiny utility and zero risk,
empty GT yields all zeros, wrong-class global evidence does not receive a
retain target, and invalid shapes fail closed.

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_sr_peg_targets.py
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement target tensors**

Define:

```python
@dataclass(frozen=True)
class SRPEGTargets:
    local_tiny_utility: torch.Tensor      # [1,1200,1], float
    local_non_tiny_risk: torch.Tensor     # [1,1200,1], float
    global_retain: torch.Tensor           # [1,300,1], float
```

Use normalized `xywh`, pairwise IoU, pairwise intersection-over-smaller, and
`sqrt(width * height) * 640` effective size. Utility requires same class,
tiny GT, and IoU at least 0.5; risk uses class-agnostic IoS at least 0.5 with
non-tiny GT; retain uses same-class IoS at least 0.5 with non-tiny GT.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_sr_peg_targets.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/sr_peg_targets.py tests/test_sr_peg_targets.py
git commit -m "Add SR-PEG scale-risk targets"
```

### Task 3: Backward-compatible supervised evidence cache

**Files:**
- Modify: `src/gcqf_cache.py`
- Modify: `scripts/cache_gcqf_evidence.py`
- Test: `tests/test_gcqf_cache.py`
- Test: `tests/test_cache_gcqf_evidence_cli.py`

- [ ] **Step 1: Write failing v1/v2 compatibility tests**

```python
def test_v2_train_cache_round_trips_sr_peg_targets(tmp_path):
    record = _record("train/a.jpg", sr_peg_targets=_sr_targets())
    manifest = write_evidence_cache(
        output=tmp_path / "cache",
        records=[record],
        baseline_sha256="A" * 64,
        dataset_signature="B" * 64,
        split="train10",
    )
    cache = VerifiedEvidenceCache(manifest)
    loaded = next(cache.iter_records())
    assert cache.manifest["schema_version"] == "gcte-gcqf-evidence/v2"
    assert loaded.sr_peg_targets.global_retain.shape == (1, 300, 1)
```

```python
def test_existing_v1_val_cache_remains_readable(tmp_path):
    manifest = _write_v1_record_without_sr_targets(tmp_path)
    loaded = next(VerifiedEvidenceCache(manifest).iter_records())
    assert loaded.sr_peg_targets is None
```

Add a fail-closed test that one cache cannot mix supervised and
unsupervised records.

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_gcqf_cache.py tests/test_cache_gcqf_evidence_cli.py
```

Expected: missing `sr_peg_targets` and v2 schema support.

- [ ] **Step 3: Add optional targets and schema selection**

Add:

```python
SRPEG_CACHE_SCHEMA_VERSION = "gcte-gcqf-evidence/v2"

@dataclass(frozen=True)
class GCQFEvidenceRecord:
    image_id: str
    global_evidence: QueryEvidence
    local_evidence: QueryEvidence
    geometry: ViewGeometry
    anchor_mask: torch.Tensor
    quality_targets: torch.Tensor
    equivariance_pairs: torch.Tensor
    fixed_anchor_payload: dict[str, Any]
    sr_peg_targets: SRPEGTargets | None = None
```

The writer selects v2 only when every record has targets; it rejects mixed
records. The reader accepts exactly v1 or v2 and validates target presence
against the manifest version. Preserve FP16 query storage and all v1 hashes.

In `cache_gcqf_evidence.py`, call `build_sr_peg_targets` for `split=train`
and leave validation records v1-compatible.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_gcqf_cache.py tests/test_cache_gcqf_evidence_cli.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/gcqf_cache.py scripts/cache_gcqf_evidence.py tests/test_gcqf_cache.py tests/test_cache_gcqf_evidence_cli.py
git commit -m "Add supervised SR-PEG train cache"
```

### Task 4: Batch collation and frozen multi-head loss

**Files:**
- Modify: `src/gcqf_training.py`
- Modify: `src/gcqf_loss.py`
- Test: `tests/test_gcqf_training.py`
- Test: `tests/test_gcqf_loss.py`

- [ ] **Step 1: Write failing collation and loss tests**

```python
def _base_loss_inputs():
    return {
        "adjusted_scores": torch.full(
            (1, 3, 1), 0.5, requires_grad=True
        ),
        "quality_targets": torch.tensor([[[1.0], [0.0], [0.5]]]),
        "canonical_queries": torch.randn(
            1, 3, 8, requires_grad=True
        ),
        "equivariance_pairs": torch.empty((0, 3), dtype=torch.long),
        "score_residual": torch.zeros(
            1, 3, 1, requires_grad=True
        ),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
        "anchor_mask": torch.ones(1, 3, 1, dtype=torch.bool),
    }


def test_sr_peg_loss_uses_frozen_weights_and_all_heads():
    tiny_logits = torch.zeros(1, 3, 1, requires_grad=True)
    risk_logits = torch.zeros(1, 3, 1, requires_grad=True)
    retain_logits = torch.zeros(1, 2, 1, requires_grad=True)
    result = compute_gcqf_loss(
        **_base_loss_inputs(),
        tiny_utility_logits=tiny_logits,
        tiny_utility_targets=torch.tensor([[[1.0], [0.0], [0.0]]]),
        non_tiny_risk_logits=risk_logits,
        non_tiny_risk_targets=torch.tensor([[[0.0], [1.0], [0.0]]]),
        global_retain_logits=retain_logits,
        global_retain_targets=torch.tensor([[[1.0], [0.0]]]),
        positive_weights={"tiny": 3.0, "risk": 2.0, "retain": 4.0},
    )
    torch.testing.assert_close(
        result.total,
        result.quality + 0.1 * result.equivariance
        + 0.01 * result.residual_regularization
        + result.tiny_utility + 2.0 * result.non_tiny_risk
        + 2.0 * result.global_retain,
    )
    result.total.backward()
    assert tiny_logits.grad is not None
    assert risk_logits.grad is not None
    assert retain_logits.grad is not None
```

Test that `collate_evidence_records` rejects unsupervised records when
`require_sr_peg_targets=True`.

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_gcqf_training.py tests/test_gcqf_loss.py
```

- [ ] **Step 3: Implement batching, positive weights, and losses**

Extend `GCQFBatch` with three optional target tensors. Add:

```python
def compute_positive_weights(records) -> dict[str, float]:
    # clip(Nneg / max(Npos, 1), 1, 20) independently for each head
```

Extend `GCQFLoss` with `tiny_utility`, `non_tiny_risk`, and
`global_retain`. Use `binary_cross_entropy_with_logits` and detached target
tensors; no target receives gradients.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_gcqf_training.py tests/test_gcqf_loss.py
```

- [ ] **Step 5: Commit**

```powershell
git add src/gcqf_training.py src/gcqf_loss.py tests/test_gcqf_training.py tests/test_gcqf_loss.py
git commit -m "Train SR-PEG with scale-risk supervision"
```

### Task 5: Learned routing with exact fallback

**Files:**
- Create: `src/sr_peg_routing.py`
- Modify: `src/gcqf_routing.py`
- Test: `tests/test_sr_peg_routing.py`
- Test: `tests/test_gcqf_routing.py`

- [ ] **Step 1: Write failing routing tests**

```python
def test_protected_global_rejects_class_conflicting_local_fragment():
    routed = route_sr_peg_record(
        record,
        score_residual=torch.zeros(1, 1200, 1),
        tiny_utility=torch.ones(1, 1200, 1),
        non_tiny_risk=torch.zeros(1, 1200, 1),
        global_retain=torch.ones(1, 300, 1),
        thresholds=SRPEGThresholds(0.5, 0.5, 0.5),
        residual_enabled=True,
    )
    assert routed.invariants["protected_identity_exact"]
    assert routed.invariants["no_class_conflicting_fragment"]
    assert len(routed.output) <= 300
```

Also test stable ties, same-class deduplication, learned protection of a
small global box, rejection by risk, utility rejection, and:

```python
assert route_sr_peg_record(record, learned_outputs=None).output == \
       route_gcqf_record(record, score_residual=None).output
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_sr_peg_routing.py tests/test_gcqf_routing.py
```

- [ ] **Step 3: Implement query-to-row mapping and stable Top-300**

Define:

```python
@dataclass(frozen=True)
class SRPEGThresholds:
    tiny_utility: float
    non_tiny_risk: float
    global_retain: float
```

Map decoder-query outputs to each postprocessed row through the cached
`selected_query_indices`. Preserve every global detection above 16 px and
every learned-retained global detection. Reject any eligible local detection
with class-agnostic IoS at least 0.5 against protected global. Deduplicate
same-class IoU above 0.5 and fill remaining slots by
`(-score, source_order, query_index, original_index)`.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_sr_peg_routing.py tests/test_gcqf_routing.py
```

- [ ] **Step 5: Commit**

```powershell
git add src/sr_peg_routing.py src/gcqf_routing.py tests/test_sr_peg_routing.py tests/test_gcqf_routing.py
git commit -m "Add protected SR-PEG evidence routing"
```

### Task 6: Deterministic seed0 split and training

**Files:**
- Modify: `src/gcqf_training.py`
- Modify: `scripts/train_gcqf_g0.py`
- Test: `tests/test_gcqf_training.py`
- Test: `tests/test_train_gcqf_g0_cli.py`

- [ ] **Step 1: Write failing split and protocol tests**

```python
def test_seed0_split_is_exact_stable_and_disjoint():
    ids = [f"train/{i:04d}.jpg" for i in range(647)]
    train, calibration = split_seed0_records(ids)
    assert len(train) == 518
    assert len(calibration) == 129
    assert set(train).isdisjoint(calibration)
    assert split_seed0_records(list(reversed(ids))) == (train, calibration)
```

Assert the CLI accepts only `seed=0`, `epochs=10`, `batch=8`, `MuSGD`,
AMP scale 128, and a supervised v2 train cache.

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_gcqf_training.py tests/test_train_gcqf_g0_cli.py
```

- [ ] **Step 3: Implement the split and supervised epoch**

Sort records by:

```python
sha256(f"seed0:{record.image_id}".encode()).hexdigest()
```

Use the first 518 for training and the remaining 129 only for calibration.
Compute positive weights from the 518 records. Train only GCQF parameters for
10 epochs and save `best-module.pt`, `last-module.pt`, `losses.csv`, and a
manifest binding the source commit, train-cache hash, split identities, and
positive weights.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_gcqf_training.py tests/test_train_gcqf_g0_cli.py
```

- [ ] **Step 5: Commit**

```powershell
git add src/gcqf_training.py scripts/train_gcqf_g0.py tests/test_gcqf_training.py tests/test_train_gcqf_g0_cli.py
git commit -m "Train SR-PEG on sealed seed0 split"
```

### Task 7: Calibration without validation tuning

**Files:**
- Create: `scripts/calibrate_sr_peg_g0.py`
- Test: `tests/test_calibrate_sr_peg_g0.py`

- [ ] **Step 1: Write failing calibration tests**

```python
def test_calibration_prefers_budget_safe_map_then_tiny_then_thresholds():
    rows = [
        _candidate((0.5, 0.5, 0.5), medium=-0.001, large=-0.004, map=0.010, tiny=0.02),
        _candidate((0.4, 0.4, 0.4), medium=-0.003, large=-0.001, map=0.020, tiny=0.03),
    ]
    selected = select_calibration(rows)
    assert selected["thresholds"] == {
        "tiny_utility": 0.5,
        "non_tiny_risk": 0.5,
        "global_retain": 0.5,
    }
```

Assert exactly 27 unique combinations are evaluated and all metrics use only
the 129 calibration identities from the training artifact.

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_calibrate_sr_peg_g0.py
```

- [ ] **Step 3: Implement deterministic selection**

Evaluate `{0.4,0.5,0.6}^3`, discard settings below `medium=-0.002` or
`large=-0.005` relative to calibration Global, then maximize:

```python
(mAP50_95, AP_tiny_SBR, tiny_recall, -tiny_threshold,
 risk_threshold, -retain_threshold)
```

Write `calibration.json` with all 27 rows, selected thresholds, image
identities, module checksum, and cache checksum. Exit nonzero if no setting
satisfies both budgets.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_calibrate_sr_peg_g0.py
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/calibrate_sr_peg_g0.py tests/test_calibrate_sr_peg_g0.py
git commit -m "Calibrate SR-PEG on train10 holdout"
```

### Task 8: Five-state seed0 evaluation and hard gate

**Files:**
- Modify: `scripts/evaluate_gcqf_g0.py`
- Test: `tests/test_evaluate_gcqf_g0.py`

- [ ] **Step 1: Write failing learned-gate evaluation tests**

Assert:

```python
assert STATES == [
    "Global", "Raw-Union", "Fixed-SADED", "Residual-Off", "Full-GCQF"
]
```

Test the final gate requires:

```python
full_minus_global["mAP50-95"] >= 0.005
full_minus_global["AP-tiny-SBR"] >= 0.010
full_minus_global["tiny_recall"] >= 0.020
full_minus_global["AP-medium-SBR"] >= -0.002
full_minus_global["AP-large-SBR"] >= -0.005
full_minus_anchor["AP-medium-SBR"] >= 0.008
full_minus_anchor["mAP50-95"] >= 0.0
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_evaluate_gcqf_g0.py
```

- [ ] **Step 3: Route and serialize all learned outputs**

Load thresholds from `calibration.json`. Store per-image utility, risk,
retain, and residual tensors. Route `Residual-Off` with the three learned
gates active and residual disabled; route `Full-GCQF` with all outputs.
Report gate probability histograms, acceptance counts, exact-global
invariants, five metrics, all pairwise deltas, and the hard gate.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_evaluate_gcqf_g0.py
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate_gcqf_g0.py tests/test_evaluate_gcqf_g0.py
git commit -m "Evaluate learned SR-PEG seed0"
```

### Task 9: Fail-closed one-seed runner

**Files:**
- Create: `scripts/run_sr_peg_seed0.py`
- Test: `tests/test_run_sr_peg_seed0.py`

- [ ] **Step 1: Write failing stage-transition tests**

Test that the runner:

- validates source commit, baseline SHA, train10 signature, val signature,
  Ultralytics version, GPU model, and free disk;
- never accepts seed1 or seed2;
- never regenerates a verified val cache;
- stops on cache, training, calibration, or evaluation failure;
- writes `PIPELINE_COMPLETE` only after a passing seed0 evaluation.

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest -q tests/test_run_sr_peg_seed0.py
```

- [ ] **Step 3: Implement idempotent stage dispatch**

Expose:

```python
STAGES = ("TRAIN_CACHE", "TRAIN_SEED0", "CALIBRATE", "EVALUATE")
```

Every stage receives exact input and output paths, writes its own log and
checksum file, and creates one completion marker. Existing verified markers
are resumed; incomplete output directories fail closed instead of being
deleted. The runner contains no shutdown, reboot, process-kill, seed1, seed2,
or fresh100 code path.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_run_sr_peg_seed0.py
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_sr_peg_seed0.py tests/test_run_sr_peg_seed0.py
git commit -m "Add fail-closed SR-PEG seed0 runner"
```

### Task 10: Verification, deployment, and run

**Files:**
- Verify all modified source and tests.
- Produce local artifact archive and remote outputs only.

- [ ] **Step 1: Run the full SR-PEG and GCQF test set**

```powershell
python -m pytest -q tests/test_sr_peg.py tests/test_sr_peg_targets.py tests/test_sr_peg_routing.py tests/test_gcqf.py tests/test_gcqf_cache.py tests/test_gcqf_training.py tests/test_gcqf_loss.py tests/test_cache_gcqf_evidence_cli.py tests/test_train_gcqf_g0_cli.py tests/test_calibrate_sr_peg_g0.py tests/test_evaluate_gcqf_g0.py tests/test_run_sr_peg_seed0.py
git diff --check
git status --short
```

Expected: all tests pass, no whitespace errors, clean worktree.

- [ ] **Step 2: Build and verify an exact Git archive**

```powershell
$commit = git rev-parse --short=8 HEAD
$archive = Join-Path $env:TEMP "gcte-srpeg-$commit.tar.gz"
git archive --format=tar.gz --output=$archive HEAD
Get-FileHash -Algorithm SHA256 $archive
```

- [ ] **Step 3: Deploy to a new server source directory**

Upload the archive as `/home/ubuntu/gcte-srpeg-${commit}.tar.gz`, extract it
to `/home/ubuntu/gcte-srpeg-${commit}`, and verify the archive SHA. Here
`commit` is the exact eight-character value printed in Step 2. Do not modify
`/home/ubuntu/gcmv-warmstart-output-7d44a725` or any earlier GCTE
source/output.

- [ ] **Step 4: Run the complete server focused tests**

```bash
commit="$(basename /home/ubuntu/gcte-srpeg-*.tar.gz .tar.gz | sed 's/^gcte-srpeg-//' | sort | tail -1)"
cd "/home/ubuntu/gcte-srpeg-${commit}"
/mnt/uav/venv/bin/python -m pytest -q \
  tests/test_sr_peg*.py tests/test_gcqf*.py tests/test_gcte*.py \
  tests/test_cache_gcqf_evidence_cli.py \
  tests/test_train_gcqf_g0_cli.py \
  tests/test_calibrate_sr_peg_g0.py \
  tests/test_evaluate_gcqf_g0.py \
  tests/test_run_sr_peg_seed0.py
```

- [ ] **Step 5: Start only the seed0 runner**

Use:

```bash
commit="$(basename /home/ubuntu/gcte-srpeg-*.tar.gz .tar.gz | sed 's/^gcte-srpeg-//' | sort | tail -1)"
/mnt/uav/venv/bin/python -m scripts.run_sr_peg_seed0 \
  --source "/home/ubuntu/gcte-srpeg-${commit}" \
  --output "/home/ubuntu/gcte-srpeg-seed0-output-${commit}" \
  --checkpoint /home/ubuntu/matched-baseline-best-epoch-0100.pt \
  --data /mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml \
  --train-images /mnt/uav/protocols/tsgr-p2-e1/d2-train-10pct.txt \
  --val-cache /home/ubuntu/gcte-g0-output-0e10f1f1/val-cache/manifest.json \
  --seed 0
```

Launch in the background, record its PID, monitor stage markers and GPU, and
never start a duplicate.

- [ ] **Step 6: Download and verify final evidence**

Download `calibration.json`, `seed0-evaluation.json`, training losses,
manifests, checksums, and logs to:

```text
C:\Users\16946\Documents\OBJECTIVE CHECK PAPER-gcte\artifacts\sr-peg-seed0-$commit
```

Report the five states, Full-minus-Global, Full-minus-Fixed-SADED, every hard
gate, and the final pass/fail decision.

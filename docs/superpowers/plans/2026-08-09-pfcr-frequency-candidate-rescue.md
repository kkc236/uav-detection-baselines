# PFCR Frequency Candidate Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a deterministic learned gate that rescues complementary FrequencyCM candidates while preserving the frozen FDR stock predictions.

**Architecture:** Extract paired train evidence from the two completed epoch-100 detectors into an immutable sharded cache. Train one small detached MLP on deterministic train/dev evidence to adjust only FrequencyCM logits, use a protected Top-300 merge, and open the existing official-val cache once only after the internal Gate passes.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.90, SciPy assignment, pytest, RTX 4090.

---

## File structure

- Create `src/pfcr.py`: split, feature, MLP, teacher, RankNet/BCE loss, protected merge, selection, and decisions.
- Create `src/pfcr_cache.py`: create-only sharded paired evidence with strict authority and SHA-256 validation.
- Create `scripts/run_pfcr_probe.py`: authority checks, train extraction, deterministic training, delayed val evaluation, atomic reports, and CLI.
- Create `tests/test_pfcr.py`: mathematical, model, merge, loss, split, and decision tests.
- Create `tests/test_pfcr_cache.py`: immutable cache, corruption, traversal, resume, and schema tests.
- Create `tests/test_run_pfcr_probe.py`: CLI, extraction isolation, state machine, report, and source-binding tests.

### Task 1: Split, candidate features, and gate identity

**Files:**
- Create: `tests/test_pfcr.py`
- Create: `src/pfcr.py`

- [ ] **Step 1: Write failing split and feature tests**

```python
def test_split_is_hash_deterministic_and_disjoint():
    ids = [f"img-{index:04d}" for index in range(100)]
    first = {name: pfcr_split(name) for name in ids}
    second = {name: pfcr_split(name) for name in reversed(ids)}
    assert first == second
    assert set(first.values()) == {"train", "dev"}
    assert all((int(sha256(name.encode()).hexdigest(), 16) % 5 == 0) ==
               (split == "dev") for name, split in first.items())


def test_pfcr_features_have_frozen_shape_and_are_detached():
    fdr_boxes, fdr_logits, cm_boxes, cm_logits = synthetic_pair()
    features = pfcr_features(fdr_boxes, fdr_logits, cm_boxes, cm_logits)
    assert features.shape == (300, 10, 35)
    assert not features.requires_grad
    assert torch.isfinite(features).all()
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_pfcr.py -k 'split or features'`  
Expected: collection fails because `src.pfcr` is absent.

- [ ] **Step 3: Implement deterministic split and 35-value features**

```python
PFCR_FEATURE_DIM = 35


def pfcr_split(image_id: str) -> str:
    normalized = Path(image_id).name
    value = int(hashlib.sha256(normalized.encode("utf-8")).hexdigest(), 16)
    return "dev" if value % 5 == 0 else "train"


def pfcr_features(fdr_boxes, fdr_logits, cm_boxes, cm_logits):
    fdr_boxes, fdr_logits = validate_detector_pair(fdr_boxes, fdr_logits)
    cm_boxes, cm_logits = validate_detector_pair(cm_boxes, cm_logits)
    fdr_prob, cm_prob = fdr_logits.sigmoid(), cm_logits.sigmoid()
    overlap = pairwise_valid_box_iou(cm_boxes, fdr_boxes)
    match_quality = overlap[:, :, None] * fdr_prob[None, :, :]
    match_index = match_quality.argmax(dim=1)
    class_index = torch.arange(10, device=fdr_boxes.device)[None, :].expand(300, -1)
    matched_logits = fdr_logits[match_index, class_index]
    matched_prob = fdr_prob[match_index, class_index]
    matched_boxes = fdr_boxes[match_index]
    cm_stats = query_class_statistics(cm_logits)
    fdr_stats = gather_query_class_statistics(fdr_logits, match_index, class_index)
    cm_geometry = expanded_box_geometry(cm_boxes, classes=10)
    cross = cross_model_geometry(cm_boxes, matched_boxes, cm_prob, matched_prob)
    one_hot = F.one_hot(class_index, num_classes=10).float()
    result = torch.cat((cm_stats, cm_geometry, fdr_stats, cross, one_hot), dim=-1)
    if result.shape != (300, 10, 35) or not torch.isfinite(result).all():
        raise RuntimeError("PFCR feature schema drift")
    return result.contiguous().detach()
```

Reject invalid width/height as feature evidence while forcing their overlap feature to
zero; do not clamp the raw detector box stored in cache.

- [ ] **Step 4: Write gate identity and bound tests**

```python
def test_gate_is_zero_initialized_and_bounded():
    model = PFCRGate()
    features = torch.randn(2, 300, 10, 35)
    residual = model(features)
    assert torch.equal(residual, torch.zeros_like(residual))
    with torch.no_grad():
        model.network[-1].bias.fill_(1000)
    assert torch.all(model(features) <= 2.0)
```

- [ ] **Step 5: Implement the gate and verify GREEN**

```python
class PFCRGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(35, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        raw = self.network(features.detach()).squeeze(-1)
        return 2.0 * torch.tanh(raw / 2.0)
```

Run: `python -m pytest -q tests/test_pfcr.py -k 'split or features or gate'`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pfcr.py tests/test_pfcr.py
git commit -m "feat: add PFCR candidate representation"
```

### Task 2: Protected merge, one-to-one teacher, and loss

**Files:**
- Modify: `src/pfcr.py`
- Modify: `tests/test_pfcr.py`

- [ ] **Step 1: Write failing protected-merge tests**

```python
@pytest.mark.parametrize("slots", [0, 15, 30, 60])
def test_protected_merge_preserves_the_fdr_prefix(slots):
    fdr = synthetic_fdr_pairs()
    cm = synthetic_cm_pairs()
    merged = protected_merge(fdr, cm, rescue_slots=slots)
    if slots == 0:
        assert torch.equal(merged, fdr[:300])
    else:
        assert torch.equal(merged[:300-slots], fdr[:300-slots])
    assert merged.shape == (300, 6)


def test_protected_merge_rejects_unregistered_budget():
    with pytest.raises(ValueError):
        protected_merge(synthetic_fdr_pairs(), synthetic_cm_pairs(), rescue_slots=29)
```

- [ ] **Step 2: Run RED merge tests**

Run: `python -m pytest -q tests/test_pfcr.py -k protected_merge`  
Expected: FAIL because `protected_merge` is absent.

- [ ] **Step 3: Implement exact protected merge**

```python
RESCUE_SLOT_GRID = (0, 15, 30, 60)


def protected_merge(fdr_stock, cm_pairs, *, rescue_slots):
    if rescue_slots not in RESCUE_SLOT_GRID:
        raise ValueError("unregistered rescue budget")
    if rescue_slots == 0:
        return fdr_stock[:300].clone()
    protected = fdr_stock[:300-rescue_slots]
    pool = torch.cat((fdr_stock[300-rescue_slots:300], cm_pairs), dim=0)
    tie_rank = torch.arange(pool.shape[0], device=pool.device)
    order = torch.argsort(pool[:, 4], descending=True, stable=True)
    selected = pool[order[:rescue_slots]]
    return torch.cat((protected, selected), dim=0)
```

- [ ] **Step 4: Write one-to-one teacher and loss tests**

```python
def test_duplicate_candidates_receive_only_one_positive_teacher():
    teacher = one_to_one_teacher(duplicate_union(), one_target())
    assert (teacher > 0).sum().item() == 1


def test_pfcr_loss_prefers_teacher_boundary_order():
    preferred = torch.tensor([1.0, -1.0])
    reversed_order = -preferred
    teacher = torch.tensor([0.8, 0.1])
    assert pfcr_loss(preferred, teacher) < pfcr_loss(reversed_order, teacher)
```

- [ ] **Step 5: Implement teacher and boundary loss**

Reuse `one_to_one_same_class_assignment` from `src.rtdetr_complementarity_oracle` and
the validated pair orientation pattern from `src.rtdetr_oar`. Return zero utility for
unassigned duplicates and invalid boxes. Build only CM-versus-FDR-tail and adjacent
teacher-boundary pairs. The loss is teacher-gap normalized RankNet plus `0.25` weighted
BCE on the same hard candidates.

- [ ] **Step 6: Run all pure tests and commit**

Run: `python -m pytest -q tests/test_pfcr.py`  
Expected: PASS.

```bash
git add src/pfcr.py tests/test_pfcr.py
git commit -m "feat: add protected PFCR ranking"
```

### Task 3: Immutable sharded train evidence cache

**Files:**
- Create: `src/pfcr_cache.py`
- Create: `tests/test_pfcr_cache.py`

- [ ] **Step 1: Write failing create-only and corruption tests**

```python
def test_cache_is_create_only_authority_bound_and_sharded(tmp_path):
    authority = valid_authority()
    writer = PFCRCacheWriter(tmp_path / "cache", authority, shard_size=64)
    writer.append_many(valid_records(65))
    manifest = writer.finalize()
    assert len(manifest["shards"]) == 2
    loaded = load_pfcr_cache(tmp_path / "cache", authority)
    assert len(loaded) == 65
    with pytest.raises(FileExistsError):
        PFCRCacheWriter(tmp_path / "cache", authority, shard_size=64)


def test_cache_rejects_corrupted_shard_before_torch_load(tmp_path):
    root = write_valid_cache(tmp_path)
    (root / "shards/train-00000.pt").write_bytes(b"corrupt")
    with pytest.raises(PFCRCacheViolation, match="sha256"):
        load_pfcr_cache(root, valid_authority())
```

- [ ] **Step 2: Run RED cache tests**

Run: `python -m pytest -q tests/test_pfcr_cache.py`  
Expected: collection fails because `src.pfcr_cache` is absent.

- [ ] **Step 3: Implement streaming create-only shards**

```python
class PFCRCacheWriter:
    def __init__(self, root, authority, *, shard_size=64):
        self.root = validate_new_cache_root(root)
        self.authority = canonical_authority(authority)
        self.shard_size = validate_shard_size(shard_size)
        self.pending = []
        self.shards = []

    def append_many(self, records):
        for record in records:
            self.pending.append(validate_record(record))
            if len(self.pending) == self.shard_size:
                self.shards.append(write_next_create_only_shard(
                    self.root, self.pending, self.authority, self.shards
                ))
                self.pending = []

    def finalize(self):
        if self.pending:
            self.shards.append(write_next_create_only_shard(
                self.root, self.pending, self.authority, self.shards
            ))
            self.pending = []
        return publish_manifest_no_replace(
            self.root, self.authority, self.shard_size, self.shards
        )


def load_pfcr_cache(root, authority, *, splits=("train", "dev")):
    manifest = load_and_validate_manifest(root, canonical_authority(authority))
    verified = verify_all_shards_before_deserialization(root, manifest)
    records_by_split = {name: [] for name in splits}
    for shard, path in verified:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        for record in artifact["records"]:
            checked = validate_record(record)
            records_by_split[pfcr_split(checked["image_id"])].append(checked)
    return records_by_split
```

Each shard is fsynced and create-only. `manifest.json` is published last with
no-replace atomic semantics. Incomplete roots have no manifest and are resumable only
when every existing shard matches the append journal.

- [ ] **Step 4: Add traversal, symlink, shape, dtype, and resume tests**

Test exact record keys: `image_id`, `original_shape`, `resized_shape`, `fdr_boxes`,
`fdr_logits`, `frequencycm_boxes`, `frequencycm_logits`, `target_boxes`, and
`target_classes`. Require CPU contiguous float32 evidence and integer classes.

- [ ] **Step 5: Run and commit**

Run: `python -m pytest -q tests/test_pfcr_cache.py`  
Expected: PASS.

```bash
git add src/pfcr_cache.py tests/test_pfcr_cache.py
git commit -m "feat: add immutable PFCR cache"
```

### Task 4: Source-bound extraction runner

**Files:**
- Create: `scripts/run_pfcr_probe.py`
- Create: `tests/test_run_pfcr_probe.py`

- [ ] **Step 1: Write the frozen CLI test**

```python
def test_cli_contains_only_artifact_and_runtime_paths():
    args = parse_args([
        "--fdr-checkpoint", "fdr.pt",
        "--frequencycm-checkpoint", "cm.pt",
        "--dataset-root", "VisDrone",
        "--train-cache-root", "cache",
        "--val-cache-root", "val-cache",
        "--run-root", "run",
        "--report-root", "report",
    ])
    assert args.device == "0"
    assert not hasattr(args, "threshold")
    assert not hasattr(args, "rescue_slots")
```

- [ ] **Step 2: Run RED runner test**

Run: `python -m pytest -q tests/test_run_pfcr_probe.py -k cli`  
Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement authority and generic train loader**

Reuse checkpoint loading and final-decoder extraction from
`scripts/run_rtdetr_complementarity_oracle.py`. Add a generic split loader that asserts
6,471 train images, 640 square preprocessing, batch 8, workers 8, shared batch tensors,
and exact image/target identity across both detector passes. Freeze runtime, checkpoint,
dataset, evaluator, category mapping, feature schema, and source hashes.

- [ ] **Step 4: Implement cache extraction state**

The runner state machine is:

```text
authority -> train cache extraction/resume -> cache verification
          -> internal controls -> gate epochs 1..20 -> internal decision
          -> optional contextual arm only when authorized
          -> one official val evaluation -> final decision -> report
```

The existing official val cache is loaded read-only under ancestor-source authority; it
is never rewritten.

- [ ] **Step 5: Test extraction isolation**

Assert both detector state hashes and every parameter gradient are unchanged before and
after extraction. Assert reconstructed stock output equals model postprocess exactly.

- [ ] **Step 6: Run and commit**

Run: `python -m pytest -q tests/test_run_pfcr_probe.py -k 'cli or authority or extract'`  
Expected: PASS.

```bash
git add scripts/run_pfcr_probe.py tests/test_run_pfcr_probe.py
git commit -m "feat: extract paired PFCR evidence"
```

### Task 5: Deterministic gate training and internal selection

**Files:**
- Modify: `scripts/run_pfcr_probe.py`
- Modify: `tests/test_run_pfcr_probe.py`

- [ ] **Step 1: Write failing optimizer, checkpoint, and selection tests**

```python
def test_internal_selection_uses_only_dev_and_smallest_near_best_budget():
    history = [
        {"epoch": 1, "split": "dev", "slots": 15, "map": .2129, "ap75": .20, "ap50": .35},
        {"epoch": 1, "split": "dev", "slots": 30, "map": .2130, "ap75": .21, "ap50": .36},
        {"epoch": 1, "split": "train", "slots": 60, "map": .99, "ap75": .99, "ap50": .99},
    ]
    assert select_internal_checkpoint(history) == {"epoch": 1, "slots": 15}


def test_training_saves_exactly_twenty_create_only_checkpoints(tmp_path):
    train_gate(synthetic_cache(), tmp_path, epochs=20)
    assert len(list((tmp_path / "checkpoints").glob("epoch-*.pt"))) == 20


def test_detector_tensors_never_enter_optimizer_groups():
    gate = PFCRGate()
    detector_parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    optimizer = build_probe_optimizer(gate)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(detector_parameter) not in optimized
    assert optimized == {id(p) for p in gate.parameters()}


def test_resume_reproduces_uninterrupted_history(tmp_path):
    direct = train_gate(synthetic_cache(), tmp_path / "direct", epochs=20)
    train_gate(synthetic_cache(), tmp_path / "resume", epochs=10)
    resumed = train_gate(synthetic_cache(), tmp_path / "resume", epochs=20, resume=True)
    assert direct == resumed
```

- [ ] **Step 2: Run RED training tests**

Run: `python -m pytest -q tests/test_run_pfcr_probe.py -k 'train or selection or resume'`  
Expected: FAIL because training is absent.

- [ ] **Step 3: Implement the frozen optimizer and epoch loop**

Use AdamW only for this detached learnability probe: learning rate `1e-3`, weight decay
`1e-4`, batch size eight images, 20 epochs, seed zero, deterministic algorithms, and
gradient norm cap `1.0`. This optimizer is not carried into Screen30; any integrated
detector uses the already frozen MuSGD baseline protocol.

At every epoch, evaluate C0, C1, and PFCR on internal dev for budgets 15/30/60. Save
checkpoint, optimizer, history, RNG state, source hash, and cache manifest hash with
create-only names. Select lexicographically by `(mAP, AP75, AP50, -epoch)` and then apply
the smallest-within-`0.0002` budget rule.

- [ ] **Step 4: Implement exact internal decision**

```python
def decide_internal(c0, c1, candidate, tiny_small):
    passed = (
        candidate["map"] - max(c0["map"], c1["map"]) >= 0.0020
        and candidate["ap75"] > max(c0["ap75"], c1["ap75"])
        and candidate["ap50"] > c0["ap50"]
        and tiny_small["candidate"] >= tiny_small["c0"]
    )
    return {
        "status": "passed" if passed else "scientific_failed",
        "observed": {"c0": dict(c0), "c1": dict(c1), "candidate": dict(candidate)},
        "tiny_small": dict(tiny_small),
        "thresholds": {"map": "0.0020", "ap75": "strict", "ap50": "strict"},
    }
```

- [ ] **Step 5: Run and commit**

Run: `python -m pytest -q tests/test_pfcr.py tests/test_pfcr_cache.py tests/test_run_pfcr_probe.py`  
Expected: PASS.

```bash
git add scripts/run_pfcr_probe.py tests/test_run_pfcr_probe.py
git commit -m "feat: train deterministic PFCR gate"
```

### Task 6: Delayed official evaluation and atomic evidence

**Files:**
- Modify: `scripts/run_pfcr_probe.py`
- Modify: `tests/test_run_pfcr_probe.py`

- [ ] **Step 1: Write state-machine and atomic-report tests**

```python
def test_internal_failure_never_opens_val_cache(monkeypatch):
    opened = []
    monkeypatch.setattr(module, "load_official_val_cache", lambda path: opened.append(path))
    module.advance_after_internal({"status": "scientific_failed"}, Path("val"))
    assert opened == []


def test_official_val_is_evaluated_once_after_pass(monkeypatch):
    opened = []
    monkeypatch.setattr(module, "load_official_val_cache", lambda path: opened.append(path) or [])
    module.advance_after_internal({"status": "passed"}, Path("val"))
    assert opened == [Path("val")]


def test_report_failure_leaves_no_partial_report_root(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "write_report_file", lambda *args: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        module.publish_reports(tmp_path / "report", valid_reports())
    assert not (tmp_path / "report").exists()


def test_official_decision_requires_positive_map_and_nonnegative_ap75():
    assert decide_official({"map": .20, "ap75": .18}, {"map": .200001, "ap75": .18})["eligible"]
    assert not decide_official({"map": .20, "ap75": .18}, {"map": .20, "ap75": .19})["eligible"]
    assert not decide_official({"map": .20, "ap75": .18}, {"map": .21, "ap75": .179999})["eligible"]
```

- [ ] **Step 2: Run RED official tests**

Run: `python -m pytest -q tests/test_run_pfcr_probe.py -k 'official or report'`  
Expected: FAIL because delayed evaluation/reporting is absent.

- [ ] **Step 3: Implement one-shot val and decision**

Load and verify the existing 548-record complementarity cache only after an immutable
internal pass. Reproduce both stock authorities exactly, evaluate the selected checkpoint
and budget once, and decide:

```python
eligible = candidate["map"] > fdr["map"] and candidate["ap75"] >= fdr["ap75"]
```

Write `internal-history.csv`, `internal-selection.json`, `internal-decision.json`,
`official-metrics.json`, `pfcr-decision.json`, `environment.json`, `authority.json`, and
`SHA256SUMS.txt` into a staging directory, fsync, and publish with no-replace semantics.

- [ ] **Step 4: Run the complete relevant suite and commit**

Run:

```bash
python -m pytest -q \
  tests/test_pfcr.py tests/test_pfcr_cache.py tests/test_run_pfcr_probe.py \
  tests/test_rtdetr_complementarity_oracle.py \
  tests/test_run_rtdetr_complementarity_oracle.py \
  tests/test_rtdetr_oar.py tests/test_iber_evaluation.py
```

Expected: all pass.

```bash
git add scripts/run_pfcr_probe.py tests/test_run_pfcr_probe.py
git commit -m "feat: gate PFCR official evaluation"
```

### Task 7: Immutable deployment, execution, and publication

**Files:**
- Runtime: `/data/uav/source/uav-detection-baselines-<commit>/`
- Runtime: `/data/uav/cache/pfcr-train-v1/`
- Runtime: `/data/uav/runs/pfcr-probe-<commit>/`
- Runtime: `/data/uav/reports/pfcr-probe-v1/`

- [ ] **Step 1: Build and verify a complete git bundle**

Create `pfcr-probe-<commit>.bundle`, run `git bundle verify`, record SHA-256, verify the
pinned server fingerprint, upload, and clone to a new immutable source directory.

- [ ] **Step 2: Run one real train-batch CUDA preflight**

Verify both checkpoint hashes, exact `[8,300,4]`/`[8,300,10]` evidence, detached features,
finite loss, gate-only gradients, optimizer step, protected identity, and checkpoint
round-trip. Publish preflight JSON before full extraction.

- [ ] **Step 3: Generate the train cache without leaving the GPU idle**

Run the canonical CLI under a supervisor. Inspect PID/GPU, shard journal, disk, and log.
Engineering failures resume from verified shards in a new immutable run; they never
rewrite completed shards or open val early.

- [ ] **Step 4: Complete the internal decision**

Train all 20 epochs and publish every checkpoint sidecar. If the internal decision is
scientific failure, publish it and stop without loading official val. If it passes, the
same canonical process evaluates official val exactly once.

- [ ] **Step 5: Independently verify and publish**

Recompute report and cache-manifest SHA-256 values, recalculate the decision directly
from raw metrics, verify source/checkpoint identities, and upload reports, selected gate,
cache manifest, and source bundle to GitHub Release `pfcr-probe-v1`. Git transport failure
queues branch publication but never blocks the Release API path.

- [ ] **Step 6: Decide the next stage**

An eligible official result authorizes a separate design for the YAML-pluggable shared
network and paired Screen30. A valid scientific failure freezes PFCR-v1; it does not
authorize validation retuning or an immediate formal100 run.

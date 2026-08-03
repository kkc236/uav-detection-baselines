# IBER P2 Boundary Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic held-out P2 boundary-direction oracle that decides whether a stride-4 boundary branch is scientifically justified before any further 30/100-epoch training.

**Architecture:** Hook the frozen RT-DETR-L layer-1 P2 feature, sample compact matched-edge normal profiles with stock matching, write an immutable authority-bound cache, and train final-epoch-only P2-only and P2-plus-context classifiers. The oracle is diagnostic and cannot bypass or alter Gate-1.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, CUDA 12.1, RTX 4090.

---

## File map

- Create `src/iber_p2_oracle.py`: profile sampler, immutable cache schema, oracle model, deterministic training, and decision report.
- Create `scripts/run_iber_p2_oracle.py`: frozen detector hook, fixed dataset extraction, oracle execution, and immutable artifacts.
- Create `tests/test_iber_p2_oracle.py`: geometry, labels, cache authority, determinism, threshold, and detector-isolation tests.

### Task 1: Lock profile geometry and labels

**Files:**
- Create: `tests/test_iber_p2_oracle.py`
- Create: `src/iber_p2_oracle.py`

- [ ] **Step 1: Write failing sampler tests**

Test a coordinate-ramp P2 tensor with one centered stock box. Assert output shape
`[1, 4, 7, C]`, fixed normal offsets, left/right/top/bottom ordering,
`align_corners=False`, finite border behavior, the existing Gate validity mask, and exact
`sign(target-stock)` labels.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_iber_p2_oracle.py -q`

Expected: collection fails because `src.iber_p2_oracle` does not exist.

- [ ] **Step 3: Implement the minimal sampler**

Implement these public contracts:

```python
P2_NORMAL_OFFSETS_PX = (-12, -8, -4, 0, 4, 8, 12)
P2_TANGENT_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)

def sample_p2_edge_profiles(
    p2: torch.Tensor,
    stock_boxes_cxcywh: torch.Tensor,
    *,
    image_size: int = 640,
) -> torch.Tensor: ...

def correction_direction_targets(
    stock_edges: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    image_size: int = 640,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

- [ ] **Step 4: Run sampler tests**

Run: `python -m pytest tests/test_iber_p2_oracle.py -q`

Expected: sampler and label tests pass.

### Task 2: Lock cache and oracle decision semantics

**Files:**
- Modify: `tests/test_iber_p2_oracle.py`
- Modify: `src/iber_p2_oracle.py`

- [ ] **Step 1: Add failing cache and decision tests**

Assert immutable create-only writes, uppercase SHA-256, exact authority rejection,
train/validation image disjointness, safe `weights_only=True` loading, final-epoch-only
selection, and exact decision thresholds 0.624866/0.634066.

- [ ] **Step 2: Verify the new tests fail**

Run: `python -m pytest tests/test_iber_p2_oracle.py -q`

Expected: cache and decision symbols are missing.

- [ ] **Step 3: Implement cache, model, trainer, and report**

Use a versioned schema with detached CPU tensors and a canonical JSON manifest. Expose:

```python
def write_p2_oracle_cache(root: Path, *, train: list[dict], val: list[dict], authority: Mapping[str, str]) -> Path: ...
def load_p2_oracle_cache(root: Path, *, authority: Mapping[str, str]) -> dict[str, tuple[dict, ...]]: ...
def train_p2_oracles(cache: Mapping[str, Sequence[dict]], *, device: torch.device, epochs: int = 20) -> dict: ...
def decide_p2_viability(report: Mapping[str, object]) -> dict: ...
```

Initialize both models from the same fixed seed, use deterministic batches, fixed AdamW,
and report only epoch 20 validation metrics.

- [ ] **Step 4: Run all oracle unit tests**

Run: `python -m pytest tests/test_iber_p2_oracle.py -q`

Expected: all tests pass.

### Task 3: Integrate the frozen detector extraction CLI

**Files:**
- Modify: `tests/test_iber_p2_oracle.py`
- Create: `scripts/run_iber_p2_oracle.py`

- [ ] **Step 1: Add failing CLI source-contract tests**

Assert layer index 1 is hooked, the baseline/dataset/subset hashes are checked, only the
fixed 647/548 splits are accepted, stock matcher indices are reused, detector parameters
remain frozen, output roots cannot be overwritten, and no CLI option can change epochs,
thresholds, layers, offsets, or seed.

- [ ] **Step 2: Verify the CLI tests fail**

Run: `python -m pytest tests/test_iber_p2_oracle.py -q`

Expected: the CLI file or required contracts are absent.

- [ ] **Step 3: Implement the extraction and execution CLI**

The only public arguments are baseline checkpoint, dataset root, cache root, report root,
and device. Hook `detector.model[1]`, run the existing frozen adapter and stock matcher,
sample only matched stock queries, remove the hook in `finally`, verify every detector
gradient is `None`, then train both oracles and write an immutable report and decision.

- [ ] **Step 4: Run focused and regression tests**

Run: `python -m pytest tests/test_iber_p2_oracle.py tests/test_iber_cache.py tests/test_rtdetr_iber.py -q`

Expected: all tests pass.

### Task 4: Publish and execute on the RTX 4090 server

**Files:**
- Modify: `docs/IBER_BE_SERVER_GUIDE.md`

- [ ] **Step 1: Commit and push the verified source**

Run: `git add docs/superpowers/specs/2026-08-03-iber-p2-boundary-oracle-design.md docs/superpowers/plans/2026-08-03-iber-p2-boundary-oracle.md src/iber_p2_oracle.py scripts/run_iber_p2_oracle.py tests/test_iber_p2_oracle.py && git commit -m "experiment: add P2 boundary evidence oracle" && git push origin codex/iber-be`

Expected: remote branch resolves to the local commit SHA.

- [ ] **Step 2: Deploy to a new immutable source and run root**

Create `/data/uav/source/uav-detection-baselines-<sha12>` and
`/data/uav/runs/iber-be-p2-oracle/<sha12>-seed10000`; never modify a prior run root.

- [ ] **Step 3: Run the oracle under the pinned runtime**

Run the CLI with device `0`, the matched baseline checkpoint, and VisDrone root. Record
GPU identity, runtime packages, process exit status, cache/report SHA-256, and stderr.

- [ ] **Step 4: Apply the frozen decision**

If both final held-out metrics meet 0.624866/0.634066, proceed to a separately versioned
minimal P2 branch and four-arm Gate-1 implementation. Otherwise publish
`scientific_failed` and stop boundary-only candidates without changing thresholds.

- [ ] **Step 5: Publish evidence transactionally**

Upload the design, source commit, cache manifest, training history, final report, decision,
and artifact hashes to the existing results branch. Verify the returned remote commit
before considering the diagnostic complete.

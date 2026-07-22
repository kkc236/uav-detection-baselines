# EBC-QP A1 D2 Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the existing A0/fusion-gamma A2 evidence, add an A1 D2 arm that only disables EBC, verify gradient/query integrity, run A1 for 10 epochs, and produce a reproducible A0/A1/A2 decision artifact.

**Architecture:** Keep the model implementation unchanged and express A1 through the existing EBC-QP configuration with `lambda_ebc=0.0` and `learnable_fusion_gamma=True`. Add small artifact-freezing and tri-arm comparison utilities so runtime evidence is hashed and the branch decision is reproducible. Training and validation remain on the existing RTX 4090 server with the frozen D2 subset and initial state.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, GitHub Release assets, JSON/SHA256 manifests.

---

## File Map

- Modify `scripts/train_rtdetr_ebc_qp.py`: permit D2 A1 and resolve A1/A2 EBC weights without changing shared settings.
- Modify `tests/test_ebc_qp_cli.py`: lock the A1/A2 configuration delta and protocol guards.
- Modify `tests/test_ebc_qp_decoder.py`: prove initial query identity for A1 and A2.
- Modify `tests/test_rtdetr_ebc_qp_integration.py`: prove one-batch A1 gradient flow and EBC exclusion.
- Create `scripts/freeze_ebc_qp_d2_results.py`: hash and freeze existing A0/A2 evidence without overwriting a changed manifest.
- Create `tests/test_freeze_ebc_qp_d2_results.py`: test freeze immutability and artifact hashing.
- Create `scripts/compare_ebc_qp_d2_arms.py`: verify the common protocol and summarize A0/A1/A2 metrics and mechanisms.
- Create `tests/test_compare_ebc_qp_d2_arms.py`: test isolation deltas and branch classification.

### Task 1: Freeze A0 and Fusion-Gamma A2 Evidence

**Files:**
- Create: `scripts/freeze_ebc_qp_d2_results.py`
- Create: `tests/test_freeze_ebc_qp_d2_results.py`

- [x] **Step 1: Write the failing immutability test**

Create temporary A0/A2 files, call `build_freeze_manifest()`, and assert every
entry records the resolved path, byte size, and uppercase SHA256. Write the
manifest once, then change one input and assert `write_freeze_manifest()` refuses
to overwrite the existing manifest.

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_freeze_ebc_qp_d2_results.py -q`

Expected: FAIL because `scripts.freeze_ebc_qp_d2_results` does not exist.

- [x] **Step 3: Implement the manifest utility**

Expose:

```python
def artifact_record(path: Path) -> dict[str, object]: ...
def build_freeze_manifest(*, variant: str, protocol_signature: str, artifacts: dict[str, Path]) -> dict: ...
def write_freeze_manifest(path: Path, payload: dict) -> None: ...
```

`write_freeze_manifest` writes atomically when absent, accepts an identical
second write, and raises `FileExistsError` if existing content differs.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest tests/test_freeze_ebc_qp_d2_results.py -q`

Expected: PASS.

### Task 2: Add the D2 A1 Configuration

**Files:**
- Modify: `tests/test_ebc_qp_cli.py`
- Modify: `scripts/train_rtdetr_ebc_qp.py`

- [x] **Step 1: Write failing A1 isolation tests**

Parse matched D2 A1/A2 commands with `--learnable-fusion-gamma`. Assert
`build_settings()` returns identical dictionaries and a new
`build_ebc_config()` returns dictionaries differing only at `lambda_ebc`:

```python
assert a1_config.lambda_ebc == 0.0
assert a2_config.lambda_ebc == 0.05
assert a1_config.learnable_fusion_gamma
assert a2_config.learnable_fusion_gamma
```

Also assert protocol validation accepts D2 A1 and rejects quality-weighted EBC
for A1.

- [x] **Step 2: Run the CLI tests and verify RED**

Run: `python -m pytest tests/test_ebc_qp_cli.py -q`

Expected: FAIL because D2 A1 is rejected and `build_ebc_config` is absent.

- [x] **Step 3: Implement minimal A1 resolution**

Allow `arm=a1` for D2, allow fusion gamma for A1/A2, keep quality-weighted EBC
A2-only, and construct `EBCQPConfig` through:

```python
def build_ebc_config(args: argparse.Namespace) -> EBCQPConfig:
    stage_key = args.arm if args.stage == "formal" or (args.stage == "d2" and args.arm == "a1") else args.stage
    return EBCQPConfig(
        lambda_ebc=STAGES[stage_key].lambda_ebc,
        quality_weighted_ebc=args.quality_weighted_ebc,
        learnable_fusion_gamma=args.learnable_fusion_gamma,
    )
```

Use this helper when constructing `EBCQPTrainer`.

- [x] **Step 4: Run the CLI tests and verify GREEN**

Run: `python -m pytest tests/test_ebc_qp_cli.py -q`

Expected: PASS.

### Task 3: Prove Single-Batch Gradient and Query Integrity

**Files:**
- Modify: `tests/test_ebc_qp_decoder.py`
- Modify: `tests/test_rtdetr_ebc_qp_integration.py`

- [x] **Step 1: Write the A1/A2 query identity test**

Build small gamma-enabled A1 and A2 decoders from the same state, activate epoch
4, run the same batch, and assert both return exactly `num_queries`, identical
`final_sources`, identical `final_source_indices`, and identical P2 entry counts.

- [x] **Step 2: Write the one-batch A1 gradient test**

Run `EBCQPDetectionModel.loss()` with `lambda_ebc=0.0`, backpropagate, and assert
finite nonzero gradients for `p2_adapter`, `p2_bbox_head`, and
`p2_fusion_gamma`. Assert the scalar objective equals stock loss plus
`lambda_p2 * p2_loss` and contains no EBC term.

- [x] **Step 3: Run both focused tests**

Run: `python -m pytest tests/test_ebc_qp_decoder.py tests/test_rtdetr_ebc_qp_integration.py -q`

Expected: PASS without production model changes.

### Task 4: Add Reproducible Tri-Arm Comparison

**Files:**
- Create: `scripts/compare_ebc_qp_d2_arms.py`
- Create: `tests/test_compare_ebc_qp_d2_arms.py`

- [x] **Step 1: Write failing comparison tests**

Use synthetic A0/A1/A2 exact-metric and mechanism JSON records. Assert the output
contains A1-A0, A2-A1, and A2-A0 deltas for precision, recall, mAP50,
mAP50-95, AP-tiny, and tiny recall. Assert `classify_a1()` returns
`P2_EFFECTIVE`, `QUERY_INJECTION_UNCLEAR`, or `P2_INEFFECTIVE` from the frozen
rules in the design.

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_compare_ebc_qp_d2_arms.py -q`

Expected: FAIL because the comparison module does not exist.

- [x] **Step 3: Implement comparison and protocol checks**

Require all three arm manifests to report the same initial-state SHA256, D2
subset SHA256, seed, augmentation signature, and validation protocol. Emit an
atomic JSON report containing source hashes, exact values, deltas, mechanism
fields, and branch decision. Never infer missing fields as zero.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest tests/test_compare_ebc_qp_d2_arms.py -q`

Expected: PASS.

### Task 5: Verify and Deploy A1 D2

**Files:**
- Modify only generated runtime artifacts under `/mnt/uav/protocols`, `/mnt/uav/runs`, and `/mnt/uav/logs` on the server.

- [x] **Step 1: Run the full local regression suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [x] **Step 2: Commit, push, and synchronize the server**

Push `codex/ebc-qp`, fetch it on `/mnt/uav/repo`, and verify local/server commit
identity plus a clean server worktree.

- [x] **Step 3: Freeze A0/A2**

Create an immutable freeze manifest from the existing gamma protocol directory,
A0 checkpoint/results/revalidation, A2 checkpoint/results/diagnostics, logs, and
D2 gate. Verify every SHA256 after creation and publish the resumable checkpoint
assets and lightweight manifest to GitHub.

- [x] **Step 4: Run the server preflight tests**

Run the CLI, decoder, integration, freeze, and comparison tests in
`/mnt/uav/venv`, including CUDA query integrity where available.

Expected: all focused tests pass on RTX 4090.

- [x] **Step 5: Run A1 for 10 epochs**

Launch D2 A1 with seed 0, the existing gamma initial state, D2 data YAML,
protocol manifest, and `--learnable-fusion-gamma`. Keep batch 8, workers 8, AMP,
MuSGD, and every frozen augmentation setting. Save every epoch.

- [x] **Step 6: Revalidate and diagnose A1**

Run exact checkpoint revalidation and read-only mechanism diagnostics against the
same D2 validation YAML and protocol used by A0/A2. Hash all outputs.

- [x] **Step 7: Generate the tri-arm decision**

Run `compare_ebc_qp_d2_arms.py` and freeze the resulting A0/A1/A2 report. Based
on its predeclared decision, either begin a separate QG-P2 design, schedule
A1-no-injection, or stop the current P2 formulation. Do not launch 100 epochs.

Result: `QUERY_INJECTION_UNCLEAR`; schedule A1-no-injection.

### Task 6: Run the Triggered A1-No-Injection Isolation

**Files:**
- Modify: `src/ebc_qp_config.py`
- Modify: `src/ebc_qp_decoder.py`
- Modify: `src/rtdetr_ebc_qp.py`
- Modify: `scripts/train_rtdetr_ebc_qp.py`
- Modify: `tests/test_ebc_qp_cli.py`
- Modify: `tests/test_ebc_qp_decoder.py`
- Modify: `tests/test_rtdetr_ebc_qp_integration.py`

- [x] **Step 1: Write and verify failing no-injection tests**

Assert that the no-injection arm differs from A1 only at
`query_injection_enabled`, keeps P2/gamma gradients, preserves exactly 300 final
stock queries, and rejects the switch for any arm other than D2 A1.

- [x] **Step 2: Implement the isolated query-injection switch**

Add `query_injection_enabled=True` to `EBCQPConfig`, make
`competition_active` require that flag, preserve legacy checkpoint compatibility,
and expose `--disable-query-injection` only for D2 A1.

- [x] **Step 3: Run the full local regression suite**

Run: `python -m pytest -q`

Result: `275 passed`.

- [ ] **Step 4: Commit, push, and synchronize the server**

Commit the tested no-injection isolation, push `codex/ebc-qp`, synchronize the
server by fast-forward, and verify commit identity.

- [ ] **Step 5: Run A1-no-injection for 10 epochs**

Use the same gamma initial state, D2 dataset, seed, protocol manifest, batch,
workers, AMP, optimizer, scheduler, and augmentation settings as A1. Add only
`--disable-query-injection`. Protect every completed epoch in a separate GitHub
Release with optimizer checkpoint, SHA256 manifest, and results snapshot.

- [ ] **Step 6: Revalidate and compare A0/A1-no-injection/A1/A2**

Run exact checkpoint validation and read-only diagnostics, freeze all outputs,
and report whether no-injection reproduces A0 and whether A1 query injection
causes a consistent metric or mechanism change.

### Task 7: Completion Verification

- [ ] **Step 1: Re-run the full server suite**

Run: `/mnt/uav/venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Verify evidence integrity**

Recompute every freeze and tri-arm artifact SHA256, confirm GitHub assets match
their manifests, confirm no training process remains, and record GPU/software
fingerprints plus the deterministic CUDA warning.

- [ ] **Step 3: Update this plan**

Mark completed checkboxes and record the exact training commit, artifact paths,
hashes, metrics, mechanism values, and branch decision.

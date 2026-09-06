# VisDrone Full Transfer Initial-State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a safe `nc=10` weights-only `initial-state.pt` that preserves all 969 tensors from `DFR-BDPP-FIA-best.pt` and can warm-start only the current Full arm.

**Architecture:** A focused transfer-state module owns schema validation, tensor fingerprinting, strict loading and file loading. The existing FIA trainer detects the explicit transfer role and otherwise retains its paired-scratch loading behavior. A one-off conversion extracts the trusted legacy checkpoint state with its historical source tree, builds the new safe artifact, and verifies it against the current Full graph.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics 8.4.90, pytest.

---

### Task 1: Define and integrate the transfer-state contract

**Files:**
- Create: `src/full_transfer_state.py`
- Modify: `src/rtdetr_lrs_system.py`
- Modify: `scripts/train_visdrone_lrs_system.py`
- Create: `tests/test_full_transfer_state.py`
- Modify: `tests/test_lrs_system_launcher.py`

- [ ] **Step 1: Write failing schema, loading and launcher-boundary tests**

Cover a valid 969-tensor-style mapping with a small test model, fingerprint
corruption, wrong role, wrong `nc`, missing/extra/shape-mismatched tensors,
weights-only file loading, exact strict model loading, and launcher rejection of
the transfer artifact for arms `f/g/h`.

```python
artifact = build_full_transfer_artifact(
    model.state_dict(),
    source_checkpoint_sha256="A" * 64,
    source_checkpoint_bytes=67_871_120,
    nc=10,
)
loaded = load_full_transfer_initial_state(model, artifact, expected_nc=10)
assert loaded["tensor_count"] == len(model.state_dict())
assert loaded["tensor_mismatch_count"] == 0
```

- [ ] **Step 2: Run the focused tests and require RED**

Run:

```bash
python -m pytest tests/test_full_transfer_state.py tests/test_lrs_system_launcher.py -q
```

Expected: collection fails because `src.full_transfer_state` does not exist.

- [ ] **Step 3: Implement the minimal transfer-state module**

Implement `FORMAT_VERSION = 1`,
`ARTIFACT_ROLE = "visdrone_full_transfer"`, and four public functions:
`build_full_transfer_artifact(state, source_checkpoint_sha256,
source_checkpoint_bytes, nc=10, channels=3)`,
`validate_full_transfer_artifact(artifact, target_state=None,
expected_nc=10)`, `load_full_transfer_file(path)`, and
`load_full_transfer_initial_state(model, artifact, expected_nc=10)`.

Use `public_state_sha256` for deterministic tensor fingerprints, clone tensors
to CPU, require uppercase 64-character source SHA-256, load files with
`weights_only=True`, and call `model.load_state_dict(state, strict=True)` only
after key/shape/dtype validation.

- [ ] **Step 4: Integrate without weakening paired-scratch loading**

In `_load_fia_artifact`, safely deserialize once. If
`artifact_role == "visdrone_full_transfer"`, require the Full FIA model and
load it through `load_full_transfer_initial_state`; otherwise call the existing
`load_fia_initial_state` unchanged. In the VisDrone launcher, pass the selected
arm into initial-state validation and reject transfer artifacts unless
`arm == "i"`.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
python -m pytest tests/test_full_transfer_state.py tests/test_lrs_system_models.py tests/test_lrs_system_launcher.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the implementation**

```bash
git add src/full_transfer_state.py src/rtdetr_lrs_system.py scripts/train_visdrone_lrs_system.py tests/test_full_transfer_state.py tests/test_lrs_system_launcher.py
git commit -m "feat: support VisDrone Full transfer initial state"
```

### Task 2: Generate and verify the VisDrone artifact

**Files:**
- Create outside Git tracking: `artifacts/visdrone-full-transfer/initial-state.pt`
- Create: `local-validation/visdrone-full-transfer-initial-state.json`

- [ ] **Step 1: Extract the trusted legacy state as tensors only**

Run the trusted checkpoint under the historical source tree, select EMA then
model, convert it to float, and save only its 969-entry CPU state dictionary.
Require the source file SHA-256 to equal
`FD972586E2833A28AA02D04AC9E460B0D7C8CFC4E6E8BCB48DA273F0E905593C`
before deserialization.

- [ ] **Step 2: Build the safe transfer artifact**

Instantiate `LRSFDRBPDDFIADetectionModel(nc=10, ch=3)` and require exact key,
shape and dtype equality with the extracted source state. Call the transfer
artifact builder and write the resulting mapping with `torch.save` to the
artifact path.

- [ ] **Step 3: Verify the finished artifact independently**

Load with `weights_only=True`, run schema validation, strict-load a fresh
current Full model, check all 969 tensors byte-for-byte, and run a finite CPU
forward on a deterministic `1x3x128x128` input. Record artifact path, size,
SHA-256, source authority, tensor fingerprint, tensor count and verification
results in the JSON report.

- [ ] **Step 4: Run the full relevant regression suite**

Run:

```bash
python -m pytest tests/test_full_transfer_state.py tests/test_lrs_system_models.py tests/test_lrs_system_launcher.py tests/test_no_server_correctness.py -q
```

Expected: all tests pass with no failure.

- [ ] **Step 5: Commit only reproducible code and evidence**

```bash
git add local-validation/visdrone-full-transfer-initial-state.json
git commit -m "test: verify VisDrone Full transfer initial state"
```

The binary remains outside normal Git tracking because `*.pt` is ignored and
the file exceeds GitHub's ordinary 100 MB file limit.

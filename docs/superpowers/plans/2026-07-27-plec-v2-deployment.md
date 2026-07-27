# PLEC-v2 Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair PLEC's gradient and augmentation boundaries, verify the frozen
batch-8 protocol, deploy it to the RTX 4090 server, and start the seed0 paired
10-epoch screen.

**Architecture:** Keep the verified PLEC core. Add exact augmentation
provenance to the method data path, generalize PLEC geometry from a fixed
letterbox inverse to a recorded affine inverse, and detach all local P3
features before PLEC. Reuse the frozen scratch initialization and matched
training controls from the existing formal experiment infrastructure.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90, pytest,
OpenCV, YAML, SSH, NVIDIA RTX 4090.

---

## File structure

- `src/gcmv_data.py`: paired global/local sample construction and augmentation
  provenance.
- `src/gcmv_geometry.py`: exact global-network-to-source affine mapping.
- `src/rtdetr_gcmv_plec.py`: stop-gradient local extraction and matched trainer.
- `src/gcmv_plec_protocol.py`: frozen seed0 screen constants and input checks.
- `scripts/train_rtdetr_gcmv_plec.py`: formal method/control entry point.
- `scripts/preflight_gcmv_plec.py`: structural, gradient, AMP, and memory gate.
- `tests/test_gcmv_data.py`: data/RNG/provenance behavior.
- `tests/test_gcmv_geometry.py`: affine geometry cases.
- `tests/test_rtdetr_gcmv_plec_integration.py`: gradient ownership.
- `tests/test_gcmv_plec_training_cli.py`: frozen CLI settings.
- `tests/test_gcmv_plec_preflight.py`: fail-closed checks.

### Task 1: Record the stock augmentation geometry without changing the global arm

**Files:**

- Modify: `tests/test_gcmv_data.py`
- Modify: `src/gcmv_data.py`

- [ ] **Step 1: Write failing augmentation-provenance tests**

Add tests which construct the traced RandomPerspective and RandomFlip
transforms, seed Python/NumPy RNG identically, and assert:

```python
assert np.array_equal(traced_result["img"], stock_result["img"])
assert np.array_equal(
    traced_result["instances"].bboxes,
    stock_result["instances"].bboxes,
)
assert traced_result["_gcmv_source_to_global"].shape == (3, 3)
```

Add an HSV replay test:

```python
assert np.array_equal(
    traced_result["_gcmv_source_image"],
    separately_replayed_source,
)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_gcmv_data.py -q
```

Expected: fail because traced transforms and
`_gcmv_source_to_global` do not exist.

- [ ] **Step 3: Implement the minimal traced transforms**

Add subclasses with the same parameter sampling and application order as
Ultralytics 8.4.90:

```python
class GCMVRandomPerspective(RandomPerspective):
    def __call__(self, labels):
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels["_gcmv_affine_matrix"] = params["M"].copy()
        labels["_gcmv_pre_affine_shape"] = tuple(params["orig_shape"])
        return labels


class GCMVRandomFlip(RandomFlip):
    def __call__(self, labels):
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels[f"_gcmv_flip_{self.direction}"] = bool(params["flip"])
        return labels
```

The HSV subclass replays the saved NumPy RNG state on a copy of
`_gcmv_source_image`, restores the post-global RNG state, and therefore neither
adds nor removes a global RNG draw.

Recursively replace only RandomPerspective, RandomHSV, and RandomFlip objects
created by `super().build_transforms(hyp)`. Reject missing affine provenance,
active mosaic, rotation, shear, or perspective when constructing the method
sample.

Compose the continuous edge-coordinate matrix:

```python
source_to_global = flip @ affine @ resize
```

and collate it as `[B, 3, 3]`.

- [ ] **Step 4: Run data tests and verify GREEN**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_gcmv_data.py -q
```

Expected: all data tests pass.

### Task 2: Generalize exact PLEC geometry to recorded affine provenance

**Files:**

- Modify: `tests/test_gcmv_geometry.py`
- Modify: `src/gcmv_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Add identity, scale/translation, horizontal-flip, singular-matrix, and
off-diagonal rejection cases. The identity case must show that the center phase
of each global cell maps to the same local feature cell:

```python
geometry = build_plec_geometry(
    ...,
    global_to_source=[torch.eye(3)],
)
assert torch.allclose(
    geometry.sample_grid[:, :, 4],
    expected_center_grid,
    atol=1e-6,
)
```

- [ ] **Step 2: Run geometry tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_gcmv_geometry.py -q
```

Expected: fail because `global_to_source` is not accepted.

- [ ] **Step 3: Implement affine inverse mapping**

Keep the old `LetterboxTransform` interface for existing callers and add an
optional per-image 3-by-3 `global_to_source` matrix. Validate:

```python
assert matrix.shape == (3, 3)
assert torch.isfinite(matrix).all()
assert abs(torch.det(matrix)) > 1e-12
assert abs(matrix[0, 1]) <= 1e-8
assert abs(matrix[1, 0]) <= 1e-8
assert torch.allclose(matrix[2], torch.tensor([0.0, 0.0, 1.0]))
```

Map homogeneous global network coordinates through the matrix, then reuse the
existing crop-to-local-letterbox and align-corners-false calculations.
Magnification uses the absolute affine x/y derivatives.

- [ ] **Step 4: Run geometry tests and verify GREEN**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_gcmv_geometry.py -q
```

Expected: all geometry tests pass.

### Task 3: Isolate local gradients while preserving PLEC gradients

**Files:**

- Modify: `tests/test_rtdetr_gcmv_plec_integration.py`
- Modify: `scripts/preflight_gcmv_plec.py`
- Modify: `src/rtdetr_gcmv_plec.py`

- [ ] **Step 1: Write the failing gradient-ownership test**

Run `_local_feature_passes()` in training mode and assert:

```python
assert all(not feature.requires_grad for feature in local_p3)
assert all(feature.grad_fn is None for feature in local_p3)
assert batchnorm_buffer_fingerprint(model) == before
```

Then inject detached synthetic local P3 tensors with `gamma_ref=1`, backpropagate
a scalar loss, and retain the existing assertions that all PLEC and adapter
families receive finite nonzero gradients.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_rtdetr_gcmv_plec_integration.py -q
```

Expected: fail because local P3 currently retains a checkpoint/autograd graph.

- [ ] **Step 3: Implement stop-gradient local extraction**

Replace checkpointed local passes with:

```python
with torch.no_grad(), preserve_batchnorm_buffers(self):
    feature = self._extract_local_p3(local_view).detach()
```

Pass the collated inverse affine matrices into `build_plec_geometry`. Remove the
old requirement that local pixels and local P3 receive gradients from the
preflight; replace it with a hard assertion that they do not.

- [ ] **Step 4: Run integration and preflight unit tests**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_rtdetr_gcmv_plec_integration.py `
  tests/test_gcmv_plec_preflight.py -q
```

Expected: all selected tests pass.

### Task 4: Freeze the seed0 screen runner

**Files:**

- Create: `src/gcmv_plec_protocol.py`
- Modify: `tests/test_gcmv_plec_training_cli.py`
- Modify: `scripts/train_rtdetr_gcmv_plec.py`
- Modify: `src/rtdetr_gcmv_plec.py`

- [ ] **Step 1: Write failing formal-settings tests**

Assert the runner emits exactly:

```python
{
    "epochs": 10,
    "fraction": 1.0,
    "batch": 8,
    "workers": 8,
    "imgsz": 640,
    "optimizer": "MuSGD",
    "warmup_epochs": 3.0,
    "nbs": 64,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "scale": 0.5,
    "translate": 0.1,
    "fliplr": 0.5,
}
```

Also assert batch drift, seed drift, fraction drift, automatic OOM batch
reduction, wrong initial-state hash, wrong subset count, and wrong environment
all raise before training.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_gcmv_plec_training_cli.py -q
```

Expected: fail on the old 3-epoch, 3%, batch-1 development defaults.

- [ ] **Step 3: Implement the frozen runner**

Load the scratch artifact's `common_state` into the stock portion of the GCMV
model and allow missing keys only under:

```python
("plec.", "reference_adapter.")
```

Use the same GCMV model and global data path for method and control, with
`gcmv_enabled=False` in control. Lock batch 8, fixed AMP scale 128, MuSGD,
optimizer-attempt accounting, no internal validation, and a train-only loader.
Write a run manifest with source commit, hashes, settings, parameter counts,
batch canaries, AMP range, optimizer observations, and completion state.

- [ ] **Step 4: Run formal-runner tests and verify GREEN**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_gcmv_plec_training_cli.py -q
```

Expected: all runner tests pass.

### Task 5: Verify, deploy, preflight, and launch

**Files:**

- Modify only if a failing test exposes a defect.

- [ ] **Step 1: Run focused tests**

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_gcmv_data.py `
  tests/test_gcmv_geometry.py `
  tests/test_gcmv_plec.py `
  tests/test_rtdetr_gcmv_plec_integration.py `
  tests/test_gcmv_plec_preflight.py `
  tests/test_gcmv_plec_training_cli.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run the complete local suite**

```powershell
C:\uav_env\Scripts\python.exe -m pytest -q
```

Expected: zero failures, with only the existing expected skip.

- [ ] **Step 3: Commit and deploy the exact commit**

Commit only PLEC-v2 files, create an archive from that commit, upload it to a
new server directory, and verify the archive/source SHA256 after extraction.
Do not overwrite an earlier experiment directory.

- [ ] **Step 4: Run server preflight**

Run one real batch-8 CUDA forward/backward with fixed AMP. Require:

```text
stock identity: true
local BatchNorm preserved: true
local path gradients: absent
PLEC/adapter gradients: finite and nonzero
peak reserved memory: < 23 GiB
batch size: exactly 8
```

- [ ] **Step 5: Launch seed0 control and method**

Start the matched control first, followed by the method after control
completion. Each arm uses the same seed0 initial state and fixed subset.
Persist PID, command, log, manifest, and exit status. Do not start GGLF or PEG.

- [ ] **Step 6: Evaluate the completed pair**

Evaluate both checkpoints on the same 548-image validation set and report
mAP50-95, AP50, AP75, AP-tiny, AP-medium, AP-large, tiny recall, absolute
deltas, runtime, and the frozen advance/stop decision.

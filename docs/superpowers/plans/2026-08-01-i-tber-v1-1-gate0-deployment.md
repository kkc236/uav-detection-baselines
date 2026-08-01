# I-TBER v1.1 Gate 0 and Bare-Server Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the inference-real I-TBER v1.1 core, prove detector isolation with Gate 0, and produce a reproducible deployment bundle for a blank Ubuntu RTX 4090 server.

**Architecture:** Pure tensor geometry, sampling, and private-head modules remain independent of Ultralytics integration. A recording decoder exposes detached stock evidence without changing stock outputs; a frozen adapter owns the baseline checkpoint and the private refiner. A separate deployment layer installs the exact Python/CUDA user-space stack, verifies immutable data and checkpoint authorities, and refuses to start when the host or artifacts drift.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.90, pytest, Bash, NVIDIA RTX 4090.

---

## File map

- Create `src/itber_geometry.py`: box conversion, trajectory encoding, v1.1 targets, and bounded edge update.
- Create `src/itber_sampling.py`: normalized four-edge grids and F3 inside/edge/outside sampling.
- Create `src/itber_head.py`: equal-capacity P0-P3 private head and typed output record.
- Create `src/itber_loss.py`: isolated matched/unmatched private losses.
- Create `src/rtdetr_itber.py`: exact stock recording decoder and frozen detector/refiner adapter.
- Create `src/itber_protocol.py`: immutable checkpoint, environment, dataset, and source authority.
- Create `scripts/run_itber_canary.py`: Gate 0 executable and immutable evidence report.
- Create `requirements-itber.lock`: exact deploy-time Python package pins.
- Create `deploy/itber/verify_host.sh`: read-only OS/GPU/disk validation.
- Create `deploy/itber/build_wheelhouse.sh`: mirror-first Linux wheel download with official PyTorch fallback.
- Create `deploy/itber/bootstrap_ubuntu.sh`: idempotent bare-server environment bootstrap.
- Create `deploy/itber/verify_bundle.py`: SHA256 verification for source, wheelhouse, data, and checkpoint manifests.
- Create `docs/ITBER_BARE_SERVER_GUIDE.md`: blank-server handoff and launch checklist.
- Create focused `tests/test_itber_*.py` files for every boundary above.

### Task 1: v1.1 box geometry and target factorization

**Files:**
- Create: `src/itber_geometry.py`
- Test: `tests/test_itber_geometry.py`

- [ ] **Step 1: Write failing tests for target factorization and stable updates**

```python
import torch

from src.itber_geometry import apply_edge_update, correction_targets, cxcywh_to_xyxy


def test_gate_magnitude_times_direction_reconstructs_clipped_correction() -> None:
    stock = torch.tensor([[[0.40, 0.40, 0.60, 0.60]]])
    target = torch.tensor([[[0.39, 0.42, 0.61, 0.58]]])
    magnitude, direction, normalized = correction_targets(stock, target, rho=0.05)
    torch.testing.assert_close(magnitude * direction, normalized)
    assert torch.all((magnitude >= 0) & (magnitude <= 1))
    assert set(torch.unique(direction).tolist()) <= {-1.0, 0.0, 1.0}


def test_zero_correction_has_zero_direction_and_identity_update() -> None:
    box = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    edges = cxcywh_to_xyxy(box)
    magnitude, direction, _ = correction_targets(edges, edges, rho=0.05)
    refined = apply_edge_update(edges, magnitude, direction, rho=0.05)
    torch.testing.assert_close(refined, edges, rtol=0, atol=0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_itber_geometry.py -q`

Expected: collection fails because `src.itber_geometry` does not exist.

- [ ] **Step 3: Implement the minimal public API**

```python
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes.split(2, dim=-1)
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_cxcywh(edges: torch.Tensor) -> torch.Tensor:
    lower, upper = edges.split(2, dim=-1)
    return torch.cat(((lower + upper) / 2, upper - lower), dim=-1)


def edge_scale(edges: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    width = (edges[..., 2] - edges[..., 0]).clamp_min(eps)
    height = (edges[..., 3] - edges[..., 1]).clamp_min(eps)
    return torch.stack((width, height, width, height), dim=-1)


def correction_targets(stock_edges, target_edges, *, rho, eps=1e-6):
    normalized = ((target_edges - stock_edges) / (rho * edge_scale(stock_edges, eps) + eps)).clamp(-1, 1)
    magnitude = normalized.abs()
    direction = torch.where(magnitude > eps, normalized / magnitude.clamp_min(eps), torch.zeros_like(normalized))
    return magnitude, direction, normalized


def apply_edge_update(stock_edges, gate, residual, *, rho, eps=1e-6):
    candidate = stock_edges + rho * edge_scale(stock_edges, eps) * gate * residual
    left = candidate[..., 0].clamp(eps, 1 - eps)
    top = candidate[..., 1].clamp(eps, 1 - eps)
    right = torch.maximum(candidate[..., 2].clamp(eps, 1), left + eps).clamp(max=1)
    bottom = torch.maximum(candidate[..., 3].clamp(eps, 1), top + eps).clamp(max=1)
    return torch.stack((left, top, right, bottom), dim=-1)


def trajectory_state(edge_l2, edge_l1, edge_l, *, eps=1e-6):
    scale = edge_scale(edge_l, eps)
    v1 = (edge_l1 - edge_l2) / (scale + eps)
    v2 = (edge_l - edge_l1) / (scale + eps)
    return torch.stack((v1, v2, v1.abs() + v2.abs(), v2 - v1, v1 * v2, v2.abs() / (v1.abs() + eps)), dim=-1)
```

`correction_targets` computes `u=clip((gt-stock)/(rho*scale+eps),-1,1)`, `magnitude=abs(u)`, and zero-safe `direction=sign(u)`. `apply_edge_update` clips coordinates, enforces positive width/height with `eps`, and preserves exact identity when the effective correction is zero.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_itber_geometry.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/itber_geometry.py tests/test_itber_geometry.py
git commit -m "feat: add I-TBER v1.1 edge geometry"
```

### Task 2: Four-edge sparse sampling

**Files:**
- Create: `src/itber_sampling.py`
- Test: `tests/test_itber_sampling.py`

- [ ] **Step 1: Write failing sampling tests**

```python
import torch

from src.itber_sampling import boundary_sample_grid, sample_boundary_evidence


def test_grid_has_four_edges_three_positions_and_three_normal_offsets() -> None:
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.1]]])
    grid = boundary_sample_grid(boxes, image_size=640)
    assert grid.shape == (1, 1, 4, 3, 3, 2)
    assert torch.isfinite(grid).all()
    assert grid.min() >= -1 and grid.max() <= 1


def test_constant_feature_produces_zero_inside_outside_difference() -> None:
    f3 = torch.ones(1, 32, 80, 80)
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    evidence = sample_boundary_evidence(f3, boxes, image_size=640)
    assert evidence.shape == (1, 1, 4, 96)
    torch.testing.assert_close(evidence[..., 32:], torch.zeros_like(evidence[..., 32:]))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_sampling.py -q`

Expected: import failure for `src.itber_sampling`.

- [ ] **Step 3: Implement fixed-coordinate sampling**

`boundary_sample_grid` converts letterboxed normalized `cxcywh` boxes to four edge grids, uses along-edge positions `(0.25,0.5,0.75)`, applies `d=clip(0.08*min(w,h),1/640,4/640)`, orders normal samples as `(outside, edge, inside)`, maps `[0,1]` to `[-1,1]`, and clamps the final grid. `sample_boundary_evidence` calls `torch.nn.functional.grid_sample(..., mode="bilinear", padding_mode="border", align_corners=False)`, averages each group of three along-edge positions, and returns `[edge, inside-outside, abs(inside-outside)]`.

- [ ] **Step 4: Verify GREEN and gradients**

Run: `python -m pytest tests/test_itber_sampling.py -q`

Expected: all tests pass; the test suite also proves gradients reach F3 while boxes remain detached.

- [ ] **Step 5: Commit**

```bash
git add src/itber_sampling.py tests/test_itber_sampling.py
git commit -m "feat: add I-TBER boundary evidence sampling"
```

### Task 3: Equal-capacity P0-P3 private head

**Files:**
- Create: `src/itber_head.py`
- Test: `tests/test_itber_head.py`

- [ ] **Step 1: Write failing head tests**

```python
import torch

from src.itber_head import ITBERRefiner


def test_zero_outputs_are_exact_stock_identity() -> None:
    model = ITBERRefiner(hidden_dim=256, f3_channels=256, private_seed=10000)
    output = model.synthetic_forward(batch=2, queries=5)
    torch.testing.assert_close(output.refined_boxes, output.stock_boxes, rtol=0, atol=0)


def test_all_probe_modes_have_identical_parameter_count() -> None:
    counts = {
        mode: sum(p.numel() for p in ITBERRefiner(256, 256, 10000, probe=mode).parameters())
        for mode in ("p0", "p1", "p2", "p3")
    }
    assert len(set(counts.values())) == 1
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_head.py -q`

Expected: import failure for `src.itber_head`.

- [ ] **Step 3: Implement the private head**

Create an `ITBEROutput` dataclass containing stock/refined boxes, gate logits/gates, residual raw/residuals, quality, trajectory, and effective corrections. `ITBERRefiner` contains the exact 64/32/16/6/8 feature slots from the specification, zero-fills disabled Probe modalities, shares the two-layer 64-wide fusion across edges, and zero-initializes gate/residual output layers inside `torch.random.fork_rng`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_itber_head.py tests/test_itber_sampling.py tests/test_itber_geometry.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/itber_head.py tests/test_itber_head.py
git commit -m "feat: add equal-capacity I-TBER refiner"
```

### Task 4: Isolated v1.1 private loss

**Files:**
- Create: `src/itber_loss.py`
- Test: `tests/test_itber_loss.py`

- [ ] **Step 1: Write failing loss tests**

```python
from types import SimpleNamespace

import torch

from src.itber_geometry import apply_edge_update
from src.itber_loss import itber_private_loss


def _output(query_count: int):
    stock = torch.full((1, query_count, 4), 0.4)
    gate_logits = torch.zeros(1, query_count, 4, requires_grad=True)
    residual_raw = torch.zeros(1, query_count, 4, requires_grad=True)
    gates = gate_logits.sigmoid()
    residuals = residual_raw.tanh()
    refined = apply_edge_update(stock, gates, residuals, rho=0.05)
    return SimpleNamespace(
        stock_edges=stock,
        refined_edges=refined,
        gate_logits=gate_logits,
        gates=gates,
        residual_raw=residual_raw,
        residuals=residuals,
        quality=torch.ones(1, query_count, 1),
    )


def test_private_loss_reuses_match_and_never_touches_detector() -> None:
    detector_tensor = torch.rand(1, 5, 4, requires_grad=True)
    output = _output(query_count=5)
    output.stock_edges = detector_tensor.detach()
    output.refined_edges = apply_edge_update(
        output.stock_edges, output.gates, output.residuals, rho=0.05
    )
    losses = itber_private_loss(
        output,
        target_edges=torch.tensor([[0.35, 0.35, 0.55, 0.55]]),
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        rho=0.05,
    )
    losses.total.backward()
    assert detector_tensor.grad is None
    assert output.gate_logits.grad is not None


def test_positive_and_negative_gate_means_are_separately_normalized() -> None:
    target = torch.tensor([[0.35, 0.35, 0.55, 0.55]])
    match = [(torch.tensor([0]), torch.tensor([0]))]
    small = itber_private_loss(_output(3), target_edges=target, match_indices=match, rho=0.05)
    large = itber_private_loss(_output(300), target_edges=target, match_indices=match, rho=0.05)
    torch.testing.assert_close(small.gate, large.gate)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_loss.py -q`

Expected: import failure for `src.itber_loss`.

- [ ] **Step 3: Implement `ITBERLosses` and `itber_private_loss`**

The implementation gathers only last-layer normal-query stock matches, calculates `L_box`, magnitude-weighted normalized `L_dir`, separately normalized positive/negative soft BCE gate terms, and score-weighted unmatched no-op. It returns named finite scalars and `total = box + direction + 0.25*gate + 0.05*noop`. Empty positive or negative sets produce a graph-connected zero, not NaN.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_itber_loss.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/itber_loss.py tests/test_itber_loss.py
git commit -m "feat: add supervised I-TBER private loss"
```

### Task 5: Exact RT-DETR evidence recording and frozen adapter

**Files:**
- Create: `src/rtdetr_itber.py`
- Test: `tests/test_rtdetr_itber.py`

- [ ] **Step 1: Write failing integration tests**

```python
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.rtdetr_itber import FrozenITBERAdapter, ITBERRecordingDecoder


def _stock_head():
    return RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).model[-1]


def test_recording_decoder_preserves_stock_outputs_exactly() -> None:
    torch.manual_seed(0)
    stock = _stock_head().eval()
    wrapped = _stock_head().eval()
    wrapped.load_state_dict(stock.state_dict())
    wrapped.decoder = ITBERRecordingDecoder.from_stock(wrapped.decoder)
    features = [torch.randn(1, 256, size, size) for size in (20, 10, 5)]
    with torch.no_grad():
        expected = stock(features)
        actual = wrapped(features)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_frozen_adapter_has_no_detector_gradients() -> None:
    detector = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    adapter = FrozenITBERAdapter.from_detector(detector, private_seed=10000)
    batch = {
        "img": torch.rand(1, 3, 640, 640),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0.0]),
    }
    losses = adapter.training_step(batch)
    losses.total.backward()
    assert all(parameter.grad is None for parameter in adapter.detector.parameters())
    assert any(parameter.grad is not None for parameter in adapter.refiner.parameters())
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rtdetr_itber.py -q`

Expected: import failure for `src.rtdetr_itber`.

- [ ] **Step 3: Implement wrapper and adapter**

`ITBERRecordingDecoder.from_stock` reuses the exact decoder layers and reproduces the Ultralytics 8.4.90 forward path while recording the final three normal-query boxes and final hidden state. `FrozenITBERAdapter` loads the mature checkpoint, locks `eval()` and `requires_grad_(False)`, records the highest-resolution RT-DETR head input F3, obtains stock matcher indices without a second match, invokes `ITBERRefiner`, and exposes `set_output_mode("stock"|"refined")`.

- [ ] **Step 4: Verify GREEN and existing compatibility**

Run: `python -m pytest tests/test_rtdetr_itber.py tests/test_lpr_g_decoder.py tests/test_rtdetr_lpr_g_integration.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rtdetr_itber.py tests/test_rtdetr_itber.py
git commit -m "feat: expose frozen RT-DETR evidence for I-TBER"
```

### Task 6: Immutable protocol and Gate 0 Canary

**Files:**
- Create: `src/itber_protocol.py`
- Create: `scripts/run_itber_canary.py`
- Test: `tests/test_itber_protocol.py`
- Test: `tests/test_itber_canary.py`

- [ ] **Step 1: Write failing authority tests**

Tests must reject a changed baseline SHA, dataset SHA, category mapping, Ultralytics source SHA, non-4090 GPU, wrong package version, trainable detector tensor, changed detector state after one private optimizer step, and a mutable report path. A success fixture requires baseline SHA `54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b`, dataset SHA `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`, subset SHA `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`, and environment versions from the specification.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_protocol.py tests/test_itber_canary.py -q`

Expected: imports fail for the new protocol and Canary modules.

- [ ] **Step 3: Implement protocol and CLI**

`scripts/run_itber_canary.py` accepts only operational paths, not scientific hyperparameters. It validates authorities, runs stock-wrapper equality, zero-init equality, one private backward/step, detector gradient/state invariance, finite edge cases, checkpoint round-trip, and writes immutable JSON containing every check, SHA, environment field, and `status: passed|engineering_invalid`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_itber_protocol.py tests/test_itber_canary.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/itber_protocol.py scripts/run_itber_canary.py tests/test_itber_protocol.py tests/test_itber_canary.py
git commit -m "feat: add immutable I-TBER Gate 0 Canary"
```

### Task 7: Exact dependency lock and artifact bundle verifier

**Files:**
- Create: `requirements-itber.lock`
- Create: `deploy/itber/verify_bundle.py`
- Test: `tests/test_itber_bundle.py`

- [ ] **Step 1: Write failing bundle tests**

Tests create a manifest with relative POSIX paths, bytes, and SHA256, verify a valid tree, then prove missing files, path traversal, changed bytes, duplicate paths, and a mismatched baseline authority are rejected.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_bundle.py -q`

Expected: import failure for `deploy.itber.verify_bundle`.

- [ ] **Step 3: Implement lock and verifier**

Pin Python packages needed by the repository, including `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `ultralytics==8.4.90`, pytest, requests, numpy, scipy, pandas, opencv-python-headless, pyyaml, psutil, and thop. The verifier accepts `--root` and `--manifest`, resolves every path under root, streams SHA256, and emits machine-readable JSON without reading any secret file.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_itber_bundle.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add requirements-itber.lock deploy/itber/verify_bundle.py tests/test_itber_bundle.py
git commit -m "build: lock and verify I-TBER deployment artifacts"
```

### Task 8: Blank Ubuntu host verification and bootstrap

**Files:**
- Create: `deploy/itber/verify_host.sh`
- Create: `deploy/itber/build_wheelhouse.sh`
- Create: `deploy/itber/bootstrap_ubuntu.sh`
- Test: `tests/test_itber_deploy_scripts.py`

- [ ] **Step 1: Write failing script-contract tests**

Tests read the Bash files and require `set -euo pipefail`, absolute task-specific roots, no `$HOME`/`~`, no embedded credentials, hidden background processes, Python 3.10 venv creation, mirror-first wheel download, official PyTorch CUDA 12.1 fallback, `nvidia-smi` RTX 4090/driver checks, minimum disk checks, token permission 600 checks, and idempotent marker files tied to manifest SHA.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_deploy_scripts.py -q`

Expected: required deployment files are absent.

- [ ] **Step 3: Implement scripts**

`verify_host.sh` is read-only and reports OS, architecture, GPU, driver, disk, RAM, network/DNS, Git, and Python availability as JSON. `build_wheelhouse.sh` downloads Linux wheels to an explicit staging root, first using configurable mainland mirrors and then `https://download.pytorch.org/whl/cu121` for PyTorch. `bootstrap_ubuntu.sh` installs apt prerequisites, installs Python 3.10 without replacing system Python, creates `/data/uav/venvs/itber-v1.1`, installs from a verified local wheelhouse when available, falls back to locked online indexes, and runs import/version/CUDA checks.

- [ ] **Step 4: Verify GREEN and Bash syntax**

Run: `python -m pytest tests/test_itber_deploy_scripts.py -q`

Run on a Bash-capable environment: `bash -n deploy/itber/verify_host.sh deploy/itber/build_wheelhouse.sh deploy/itber/bootstrap_ubuntu.sh`

Expected: tests pass and Bash syntax exits 0.

- [ ] **Step 5: Commit**

```bash
git add deploy/itber/verify_host.sh deploy/itber/build_wheelhouse.sh deploy/itber/bootstrap_ubuntu.sh tests/test_itber_deploy_scripts.py
git commit -m "build: bootstrap I-TBER on a blank 4090 server"
```

### Task 9: Bare-server handoff and dry-run audit

**Files:**
- Create: `docs/ITBER_BARE_SERVER_GUIDE.md`
- Create: `scripts/audit_itber_deployment.py`
- Test: `tests/test_itber_deployment_audit.py`

- [ ] **Step 1: Write failing audit tests**

The audit must report `ready_waiting_for_server` only when source commit, lockfile, deploy scripts, baseline manifest, dataset manifest, Gate 0 command template, GitHub publication configuration template, recovery command, disk budget, and secret-file policy are present. It must never claim the remote host or GPU is verified before connection.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_deployment_audit.py -q`

Expected: import or missing-artifact failure.

- [ ] **Step 3: Implement audit and guide**

The guide documents a new host-key pinning step, exact directory layout under `/data/uav`, source clone/bundle options, wheelhouse transfer, dataset/baseline rsync with post-transfer SHA verification, token creation with mode 600, bootstrap, Gate 0, Probe launch, monitoring, recovery, and the rule that no host-changing command runs before the user supplies and authorizes the new server endpoint. The audit emits only local readiness and unresolved remote fields.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_itber_deployment_audit.py tests/test_itber_deploy_scripts.py tests/test_itber_bundle.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add docs/ITBER_BARE_SERVER_GUIDE.md scripts/audit_itber_deployment.py tests/test_itber_deployment_audit.py
git commit -m "docs: add I-TBER bare-server deployment handoff"
```

### Task 10: Gate 0 package verification

**Files:**
- Modify only files created by Tasks 1-9 when verification identifies a concrete defect.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_itber_geometry.py tests/test_itber_sampling.py tests/test_itber_head.py tests/test_itber_loss.py tests/test_rtdetr_itber.py tests/test_itber_protocol.py tests/test_itber_canary.py tests/test_itber_bundle.py tests/test_itber_deploy_scripts.py tests/test_itber_deployment_audit.py -q`

Expected: zero failures.

- [ ] **Step 2: Run the complete repository suite**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 3: Run source and deployment checks**

Run: `git diff --check`

Run: `python -m compileall -q src scripts deploy/itber`

Run: `python scripts/audit_itber_deployment.py --output tmp/itber-deployment-readiness.json`

Expected: clean diff, successful compilation, and local status `ready_waiting_for_server` with remote checks explicitly unresolved.

- [ ] **Step 4: Commit only evidence-backed corrections**

```bash
git add --all
git commit -m "test: verify I-TBER Gate 0 deployment package"
```

Do not create an empty commit when no correction was needed.

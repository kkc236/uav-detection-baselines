# LPR-RTDETR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an identity-initialized localization-prior residual head to Ultralytics RT-DETR-L, screen it for 10 epochs on VisDrone, optimize only if the frozen gate fails, then resume the passing run to 100 epochs.

**Architecture:** Replace only the decoder container with a repository-owned wrapper that reuses every stock decoder layer and head. Each layer computes its stock output box first, sends only the supervised/output copy through a small geometry-conditioned refiner, and keeps the stock reference box for the next layer. Training, matching, denoising, classification, encoder predictions, and postprocessing remain stock.

**Tech Stack:** Python 3.12/3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, VisDrone, RTX 4090, tmux.

---

## File map

- Create `src/lpr_head.py`: geometry encoding, identity-gated residual refiner, and decoder wrapper.
- Create `src/rtdetr_lpr.py`: RT-DETR model/trainer integration and runtime diagnostics.
- Create `scripts/train_rtdetr_lpr.py`: frozen 10/100 epoch CLI and callbacks.
- Create `scripts/evaluate_lpr_gate.py`: deterministic screen/final gate report.
- Create `scripts/benchmark_lpr.py`: parameter, GFLOPs, and latency comparison.
- Create `tests/test_lpr_head.py`: refiner and decoder unit tests.
- Create `tests/test_rtdetr_lpr_integration.py`: model/trainer/checkpoint tests.
- Create `tests/test_lpr_training_cli.py`: frozen protocol and callback tests.
- Create `tests/test_lpr_gate.py`: decision-boundary tests.

### Task 1: Geometry prior and identity-gated refiner

**Files:**
- Create: `tests/test_lpr_head.py`
- Create: `src/lpr_head.py`

- [ ] **Step 1: Write failing tests for geometry, identity, gradients, bounds, and RNG preservation**

```python
import torch

from src.lpr_head import LocalizationPriorRefiner, box_geometry_prior


def test_geometry_prior_is_finite_for_tiny_boxes():
    boxes = torch.tensor([[[0.5, 0.25, 1e-12, 2e-12]]])
    prior = box_geometry_prior(boxes)
    assert prior.shape == (1, 1, 6)
    assert torch.isfinite(prior).all()
    torch.testing.assert_close(prior[..., :2], torch.tensor([[[0.0, -0.5]]]))


def test_zero_gate_is_bitwise_identity_and_alpha_gets_gradient():
    module = LocalizationPriorRefiner(hidden_dim=256, seed=3407)
    hidden = torch.randn(2, 5, 256, requires_grad=True)
    boxes = torch.rand(2, 5, 4).mul(0.8).add(0.1).requires_grad_()
    refined = module(hidden, boxes)
    assert torch.equal(refined, boxes)
    refined.sum().backward()
    assert module.alpha.grad is not None
    assert module.alpha.grad.abs().item() > 0


def test_refiner_construction_does_not_advance_global_rng():
    torch.manual_seed(17)
    expected = torch.rand(4)
    torch.manual_seed(17)
    LocalizationPriorRefiner(hidden_dim=256, seed=3407)
    actual = torch.rand(4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_lpr_head.py -q
```

Expected: collection fails because `src.lpr_head` does not exist.

- [ ] **Step 3: Implement the minimal refiner**

Implement these public interfaces in `src/lpr_head.py`:

```python
def box_geometry_prior(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    detached = boxes.detach()
    center = detached[..., :2].mul(2).sub(1)
    width, height = detached[..., 2:].clamp_min(eps).unbind(-1)
    scale = torch.stack(
        (width.log(), height.log(), (width * height).log(), (width / height).log()),
        dim=-1,
    ).clamp(-12, 12)
    return torch.cat((center, scale), dim=-1)


class LocalizationPriorRefiner(nn.Module):
    def __init__(self, hidden_dim: int, seed: int, max_logit_delta: float = 0.5):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.query_path = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64), nn.SiLU())
            self.geometry_path = nn.Sequential(nn.Linear(6, 16), nn.SiLU())
            self.residual_head = nn.Linear(80, 4)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.max_logit_delta = float(max_logit_delta)

    def forward(self, hidden: torch.Tensor, stock_boxes: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry_path(box_geometry_prior(stock_boxes).to(hidden.dtype))
        residual = torch.tanh(self.residual_head(torch.cat((self.query_path(hidden), geometry), dim=-1)))
        candidate = torch.sigmoid(torch.logit(stock_boxes.clamp(1e-6, 1 - 1e-6)) + self.max_logit_delta * residual)
        gate = 0.5 * torch.tanh(self.alpha)
        return (stock_boxes + gate * (candidate - stock_boxes)).clamp(1e-6, 1 - 1e-6)
```

If the bitwise identity test reveals that `clamp` changes an extreme stock value, move the clamp into a branch-free correction that leaves every stock sigmoid output unchanged at `alpha=0`; do not weaken the identity assertion.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_lpr_head.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/lpr_head.py tests/test_lpr_head.py
git commit -m "feat: add identity-gated localization refiner"
```

### Task 2: Output-isolated decoder wrapper

**Files:**
- Modify: `src/lpr_head.py`
- Modify: `tests/test_lpr_head.py`

- [ ] **Step 1: Add a failing decoder equivalence test**

Create deterministic fake decoder layers and heads in the test file. Feed the same tensors to the stock Ultralytics decoder and `LPRDeformableTransformerDecoder.from_stock(stock)`. Assert:

```python
assert torch.equal(lpr_boxes, stock_boxes)
assert torch.equal(lpr_scores, stock_scores)
assert recorded_stock_references == recorded_lpr_references
assert len(lpr.lpr_refiners) == stock.num_layers
```

Add a second test that sets only the final refiner `alpha` to `0.2`, then asserts the final output box changes while every recorded reference passed into decoder layers remains equal to stock.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest tests/test_lpr_head.py -q
```

Expected: failure because `LPRDeformableTransformerDecoder` is missing.

- [ ] **Step 3: Implement the wrapper without copying stock parameters**

`from_stock` must re-register the existing layer list and preserve these attributes:

```python
wrapper.layers = stock.layers
wrapper.num_layers = stock.num_layers
wrapper.hidden_dim = stock.hidden_dim
wrapper.eval_idx = stock.eval_idx
wrapper.lpr_refiners = nn.ModuleList(
    LocalizationPriorRefiner(stock.hidden_dim, seed=3407 + index, max_logit_delta=max_logit_delta)
    for index in range(stock.num_layers)
)
```

Implement the forward loop line-for-line from Ultralytics 8.4.90, with one controlled difference:

```python
stock_output_bbox = refined_bbox if index == 0 else torch.sigmoid(
    bbox_delta + inverse_sigmoid(last_refined_bbox)
)
dec_bboxes.append(self.lpr_refiners[index](output, stock_output_bbox))
```

For evaluation, apply the matching refiner only at `eval_idx`. Always update `last_refined_bbox` and `refer_bbox` from stock `refined_bbox`, never from the LPR output.

- [ ] **Step 4: Verify train/eval equivalence and isolation**

Run:

```bash
python -m pytest tests/test_lpr_head.py -q
```

Expected: all refiner and decoder tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/lpr_head.py tests/test_lpr_head.py
git commit -m "feat: isolate LPR to decoder output boxes"
```

### Task 3: RT-DETR model and trainer integration

**Files:**
- Create: `src/rtdetr_lpr.py`
- Create: `tests/test_rtdetr_lpr_integration.py`

- [ ] **Step 1: Write failing integration tests**

Cover the real `rtdetr-l.yaml` model:

```python
def test_lpr_model_replaces_only_decoder_container():
    model = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    head = model.model[-1]
    assert isinstance(head.decoder, LPRDeformableTransformerDecoder)
    assert len(head.decoder.lpr_refiners) == 6
    names = [type(module).__name__ for module in model.modules()]
    forbidden = ("BTDSE", "VSFRMR", "IOQC", "NWD")
    assert all(token not in name for name in names for token in forbidden)


def test_zero_gate_model_matches_stock_eval_output():
    torch.manual_seed(0)
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    lpr = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    lpr.load_state_dict(stock.state_dict(), strict=False)
    image = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        stock_output = stock.predict(image)
        lpr_output = lpr.predict(image)
    torch.testing.assert_close(lpr_output[0], stock_output[0], rtol=0, atol=0)
```

Also test that `LPRTrainer.get_model()` returns the custom model, stock checkpoints load with only LPR keys missing, and a saved LPR state dict reloads with no missing LPR keys.

- [ ] **Step 2: Run the integration tests and verify RED**

```bash
python -m pytest tests/test_rtdetr_lpr_integration.py -q
```

Expected: import failure for `src.rtdetr_lpr`.

- [ ] **Step 3: Implement model and trainer**

Create:

```python
class LPRRTDETRDetectionModel(RTDETRDetectionModel):
    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True, max_logit_delta=0.5):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        head.decoder = LPRDeformableTransformerDecoder.from_stock(
            head.decoder, max_logit_delta=max_logit_delta
        )
        self.nc = self.yaml["nc"]


class LPRTrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = LPRRTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            max_logit_delta=self.max_logit_delta,
        )
        if weights:
            model.load(weights)
        return model
```

Reuse `apply_resume_runtime_overrides` from `src.rtdetr_vsf_rmr` rather than duplicating its list.

- [ ] **Step 4: Run integration and existing RT-DETR tests**

```bash
python -m pytest tests/test_rtdetr_lpr_integration.py tests/test_rtdetr_vsf_rmr_integration.py tests/test_rtdetr_nwd_integration.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/rtdetr_lpr.py tests/test_rtdetr_lpr_integration.py
git commit -m "feat: integrate LPR with RT-DETR trainer"
```

### Task 4: Frozen training CLI and diagnostics

**Files:**
- Create: `scripts/train_rtdetr_lpr.py`
- Create: `tests/test_lpr_training_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Assert the exact protocol from the design, including:

```python
assert settings["epochs"] == 10
assert settings["pretrained"] is False
assert settings["seed"] == 0
assert settings["deterministic"] is True
assert settings["batch"] == 8
assert settings["imgsz"] == 640
assert settings["optimizer"] == "auto"
```

Parametrize forbidden drift for batch, image size, seed, optimizer, augmentations, AMP, and fraction. Add a resume test where `--epochs 100 --resume path/to/last.pt` is accepted while `--epochs 50` is rejected.

Add a callback test that writes one JSONL record with:

```text
epoch, map75, gates, residual_mean, residual_max, lpr_grad_norm, cuda_peak_mib
```

- [ ] **Step 2: Run and verify RED**

```bash
python -m pytest tests/test_lpr_training_cli.py -q
```

Expected: import failure for `scripts.train_rtdetr_lpr`.

- [ ] **Step 3: Implement CLI and callbacks**

The parser accepts only `--epochs {10,100}`, `--max-logit-delta`, `--resume`, `--project`, `--name`, `--device`, `--workers`, and the engineering-only `--smoke` flag. Scientific settings remain frozen. The default run name is `scratch-rtdetr-l-lpr-v1-10ep`; smoke changes only epochs to 1, fraction to 0.01, and the run-name suffix.

At `on_train_epoch_end`, collect detached gate and residual diagnostics from the model. At `on_fit_epoch_end`, obtain AP75 from `trainer.validator.metrics.box.map75` and atomically append the record to `lpr_diagnostics.jsonl`.

- [ ] **Step 4: Verify CLI and resume behavior**

```bash
python -m pytest tests/test_lpr_training_cli.py -q
python scripts/train_rtdetr_lpr.py --help
```

Expected: tests pass and help exits 0.

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/train_rtdetr_lpr.py tests/test_lpr_training_cli.py
git commit -m "feat: add frozen LPR training protocol"
```

### Task 5: Screen gate and optimization evidence

**Files:**
- Create: `scripts/evaluate_lpr_gate.py`
- Create: `tests/test_lpr_gate.py`

- [ ] **Step 1: Write failing decision tests**

Create pure-function tests for:

```python
assert evaluate_screen(map=0.04098, map50=0.08404, val_giou=1.2702, val_l1=0.19467, finite=True, gate_active=True).passed
assert not evaluate_screen(map=0.04097, map50=0.10, val_giou=1.0, val_l1=0.1, finite=True, gate_active=True).passed
assert evaluate_screen(map=0.0430, map50=0.0821, val_giou=1.20, val_l1=0.20, finite=True, gate_active=True).passed
assert not evaluate_screen(map=0.05, map50=0.10, val_giou=1.0, val_l1=0.1, finite=False, gate_active=True).passed
```

The returned report must list every individual gate, measured values, baseline values, decision, and recommended single-variable fallback.

- [ ] **Step 2: Run and verify RED**

```bash
python -m pytest tests/test_lpr_gate.py -q
```

- [ ] **Step 3: Implement CSV/JSONL parsing and gate report**

Use exact baseline constants from the design. Write JSON to the requested output path with an atomic rename. Never overwrite the training run.

- [ ] **Step 4: Run gate tests and a fixture-based CLI check**

```bash
python -m pytest tests/test_lpr_gate.py -q
```

- [ ] **Step 5: Commit Task 5**

```bash
git add scripts/evaluate_lpr_gate.py tests/test_lpr_gate.py
git commit -m "feat: evaluate LPR screening gate"
```

### Task 6: Overhead benchmark and full local verification

**Files:**
- Create: `scripts/benchmark_lpr.py`
- Modify: `tests/test_innovation_isolation.py`

- [ ] **Step 1: Add a failing isolation test**

Assert LPR contains no BTD-SE, VSF-RMR, IOQC-SA, NWD, P2 adapter, or D-FINE class and that only the decoder container gains new parameters.

- [ ] **Step 2: Implement benchmark script**

Instantiate stock and LPR models with the same weights. Report total/trainable parameters, percentage delta, GFLOPs at 640, warmup latency, 100-run CUDA-event latency, and percentage delta as JSON.

- [ ] **Step 3: Run the complete suite**

```bash
python -m pytest -q
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 4: Commit Task 6**

```bash
git add scripts/benchmark_lpr.py tests/test_innovation_isolation.py
git commit -m "test: verify LPR isolation and overhead reporting"
```

### Task 7: Deploy and run smoke validation

**Files:**
- Server checkout: `/data/uav/repo/uav-detection-baselines-lpr`
- Server logs: `/data/uav/logs/lpr`
- Server runs: `/data/uav/runs/lpr`

- [ ] **Step 1: Push the branch and create a clean server checkout**

```bash
git push -u origin codex/lpr-rtdetr
git clone --branch codex/lpr-rtdetr --single-branch \
  https://ghproxy.net/https://github.com/kkc236/uav-detection-baselines.git \
  /data/uav/repo/uav-detection-baselines-lpr
```

- [ ] **Step 2: Verify server code and tests**

```bash
source /data/uav/venvs/rtdetr-lpr/bin/activate
cd /data/uav/repo/uav-detection-baselines-lpr
python -m pytest -q
```

- [ ] **Step 3: Run a 1% one-epoch engineering smoke**

Use a dedicated smoke flag that changes only epochs, fraction, and name. Verify one checkpoint, one validation, finite losses, nonzero `alpha` gradient, and a readable resume checkpoint.

- [ ] **Step 4: Run overhead benchmark**

```bash
python scripts/benchmark_lpr.py \
  --device 0 \
  --output /data/uav/runs/lpr/lpr-v1-overhead.json
```

- [ ] **Step 5: Save smoke evidence and commit any evidence-only script correction separately**

Do not modify model behavior based only on smoke accuracy.

### Task 8: Run and evaluate the 10-epoch screen

- [ ] **Step 1: Launch in tmux**

```bash
tmux new-session -d -s lpr10-v1 \
  "cd /data/uav/repo/uav-detection-baselines-lpr && \
   export YOLO_CONFIG_DIR=/data/uav/config/ultralytics && \
   /data/uav/venvs/rtdetr-lpr/bin/python scripts/train_rtdetr_lpr.py \
     --epochs 10 --project /data/uav/runs/lpr --name scratch-rtdetr-l-lpr-v1-10ep \
     > /data/uav/logs/lpr/lpr-v1-10ep.log 2>&1"
```

- [ ] **Step 2: Monitor every epoch**

Check process state, GPU utilization, `results.csv`, diagnostics JSONL, checkpoint size, and the last 100 log lines. Preserve all failed evidence.

- [ ] **Step 3: Generate gate report**

```bash
python scripts/evaluate_lpr_gate.py \
  --run /data/uav/runs/lpr/scratch-rtdetr-l-lpr-v1-10ep \
  --output /data/uav/runs/lpr/lpr-v1-screen-report.json
```

- [ ] **Step 4: If failed, apply only the report's first supported fallback**

Write a failing regression/config test, verify RED, implement one change, verify GREEN, commit, deploy to a new clean checkout or commit, and launch a fresh run name. Never resume a failed scientific run with changed behavior.

- [ ] **Step 5: Repeat until one 10-epoch run passes or evidence rules out LPR**

If all three frozen fallbacks fail, stop and report LPR as rejected; do not hide the result by adding FDR or P2 features.

### Task 9: Resume the passing run to 100 epochs

- [ ] **Step 1: Verify the passing `last.pt`**

Check SHA-256, `torch.load`, completed epoch, branch commit, protocol manifest, and LPR state keys.

- [ ] **Step 2: Resume to total epoch 100**

```bash
tmux new-session -d -s lpr100 \
  "cd /data/uav/repo/uav-detection-baselines-lpr && \
   export YOLO_CONFIG_DIR=/data/uav/config/ultralytics && \
   /data/uav/venvs/rtdetr-lpr/bin/python scripts/train_rtdetr_lpr.py \
     --epochs 100 \
     --resume /data/uav/runs/lpr/PASSING_RUN/weights/last.pt \
     --project /data/uav/runs/lpr --name PASSING_RUN \
     > /data/uav/logs/lpr/lpr-100ep.log 2>&1"
```

- [ ] **Step 3: Monitor through epoch 100**

Validate each saved checkpoint and keep the run detached from the controlling SSH session.

- [ ] **Step 4: Run final baseline and LPR validation with AP75**

Evaluate `/data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt` and the LPR best/last checkpoints under one validator invocation and record `map`, `map50`, and `map75`.

- [ ] **Step 5: Produce final report and handoff**

Include best epoch, last three epoch means, baseline deltas, overhead, gate evolution, run paths, commit, checkpoint hashes, failures tried, and remaining risks.

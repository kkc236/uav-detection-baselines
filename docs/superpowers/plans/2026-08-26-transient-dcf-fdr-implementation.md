# Transient DCF-FDR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, deploy, and launch the frozen Formal100 Transient DCF-FDR arm whose feedback is learned through Epoch 66, frozen and withdrawn through Epoch 74, and absent from Epoch 75 onward and at inference.

**Architecture:** Keep the existing shared full-distribution DCF adapter and add a small pure schedule/controller boundary around it. The controller applies one exact Python-float scale to both live and EMA decoders, freezes only DCF parameters after two thirds, and makes scale zero skip the adapter call. A source-bound launcher records schedule evidence, resets best-checkpoint eligibility at Epoch 75, and refuses non-exact resume.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics 8.4.90, pytest, YAML, Git, remote Ubuntu/CUDA training.

---

## File Structure

- Create `src/transient_dcf.py`: pure schedule calculation, model traversal, live/EMA state application, and immutable per-epoch state representation.
- Modify `src/fdr_head.py`: expose a Python-float feedback scale, a setter, a frozen-state helper, and a scale-zero fast path around the existing adapter.
- Modify `src/rtdetr_fdr.py`: classify DCF parameters as private FDR gradient evidence.
- Create `configs/rtdetr-l-transient-dcf-fdr.yaml`: source-visible method configuration with the same model/loss values as persistent DCF.
- Create `scripts/train_transient_dcf_fdr.py`: frozen Formal100 launcher, schedule callback, EMA synchronization, Epoch-75 best reset, and schedule JSONL evidence.
- Create `src/transient_dcf_export.py`: state-dict stripping and zero-scale export validation helpers.
- Create `scripts/export_transient_dcf_fdr.py`: convert an eligible T-DCF checkpoint to Clean shape after the run.
- Modify `tests/test_fdr_head.py`: decoder scale, fast-path, full-path, and freeze tests.
- Create `tests/test_transient_dcf.py`: schedule, live/EMA synchronization, tail eligibility, and evidence tests.
- Modify `tests/test_train_dcf_fdr.py`: launcher/config/source-authority tests for the new arm.
- Create `tests/test_transient_dcf_export.py`: declared-key stripping and exact output-equivalence tests.

### Task 1: Decoder Scale and Exact Clean Fast Path

**Files:**
- Modify: `tests/test_fdr_head.py`
- Modify: `src/fdr_head.py:26-48, 208-356`

- [ ] **Step 1: Write failing scale-zero and scale-one tests**

Add tests that use the existing fake decoder fixtures:

```python
def test_dcf_scale_zero_skips_adapter_and_matches_no_adapter() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=3)
    for head in dist:
        head.layers[-1].weight.data.fill_(0.01)
    adapter = DistributionConditionedFeedback(16, private_seed=10_001)
    adapter.output.weight.data.fill_(0.1)
    with_adapter = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=deepcopy(pre), distribution_feedback=adapter
    )
    without_adapter = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=deepcopy(pre)
    )
    without_adapter.load_state_dict(
        {k: v for k, v in with_adapter.state_dict().items() if not k.startswith("distribution_feedback.")},
        strict=True,
    )
    calls = []
    hook = adapter.register_forward_hook(lambda *_: calls.append(True))
    with_adapter.set_distribution_feedback_scale(0.0)
    with_adapter.train()
    without_adapter.train()
    actual = with_adapter(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    expected = without_adapter(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    hook.remove()
    assert calls == []
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_dcf_scale_one_preserves_persistent_feedback_behavior() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=3)
    for head in dist:
        head.layers[-1].weight.data.fill_(0.01)
    adapter = DistributionConditionedFeedback(16, private_seed=10_001)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=pre, distribution_feedback=adapter
    )
    adapter.output.weight.data.fill_(0.1)
    decoder.set_distribution_feedback_scale(1.0)
    decoder.train()
    decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    assert torch.count_nonzero(decoder.last_corner_logits) > 0
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_fdr_head.py -k "scale_zero or scale_one"
```

Expected: FAIL because `set_distribution_feedback_scale` does not exist and the adapter is still always called.

- [ ] **Step 3: Implement the minimal decoder API and fast path**

In `FDRDeformableTransformerDecoder.__init__` add a non-buffer attribute:

```python
self.distribution_feedback_scale = 1.0
```

Add:

```python
def set_distribution_feedback_scale(self, value: float) -> None:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("distribution feedback scale must be finite and in [0, 1]")
    self.distribution_feedback_scale = value

def freeze_distribution_feedback(self) -> None:
    if self.distribution_feedback is not None:
        self.distribution_feedback.requires_grad_(False)
```

Change the feedback block to:

```python
if (
    index > 0
    and self.distribution_feedback is not None
    and self.distribution_feedback_scale != 0.0
):
    if not isinstance(cumulative_corners, Tensor):
        raise RuntimeError("DCF requires a preceding cumulative distribution")
    regression_input = regression_input + self.distribution_feedback_scale * self.distribution_feedback(
        cumulative_corners
    )
```

Ensure `__setstate__` supplies `distribution_feedback_scale = 1.0` for legacy checkpoints.

- [ ] **Step 4: Run the focused and complete FDR-head tests**

Run:

```powershell
python -m pytest -q tests/test_fdr_head.py
```

Expected: all tests PASS, including the prior persistent-DCF forward-call test.

- [ ] **Step 5: Commit Task 1**

```powershell
git add tests/test_fdr_head.py src/fdr_head.py
git commit -m "feat: add exact DCF feedback scaling"
```

### Task 2: Pure Schedule, Freeze Boundary, and EMA Synchronization

**Files:**
- Create: `tests/test_transient_dcf.py`
- Create: `src/transient_dcf.py`

- [ ] **Step 1: Write failing pure-schedule tests**

Create tests with the exact Formal100 boundaries:

```python
def test_formal100_schedule_has_frozen_boundaries() -> None:
    assert transient_dcf_state(66, 100).scale == 1.0
    middle = [transient_dcf_state(epoch, 100).scale for epoch in range(67, 75)]
    assert all(0.0 < value < 1.0 for value in middle)
    assert all(left > right for left, right in zip(middle, middle[1:]))
    assert transient_dcf_state(67, 100).frozen is True
    assert transient_dcf_state(74, 100).scale > 0.0
    assert transient_dcf_state(75, 100).scale == 0.0
    assert transient_dcf_state(75, 100).checkpoint_eligible is True


@pytest.mark.parametrize("epoch,total", [(0, 100), (101, 100), (1, 0)])
def test_schedule_rejects_invalid_epoch_domain(epoch: int, total: int) -> None:
    with pytest.raises(ValueError):
        transient_dcf_state(epoch, total)
```

- [ ] **Step 2: Run schedule tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_transient_dcf.py -k schedule
```

Expected: collection/import FAIL because `src.transient_dcf` does not exist.

- [ ] **Step 3: Implement the pure state function**

Create:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TransientDCFState:
    paper_epoch: int
    total_epochs: int
    ratio: float
    scale: float
    frozen: bool
    checkpoint_eligible: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def transient_dcf_state(paper_epoch: int, total_epochs: int) -> TransientDCFState:
    if total_epochs <= 0 or not 1 <= paper_epoch <= total_epochs:
        raise ValueError("paper epoch must be inside a positive training horizon")
    ratio = paper_epoch / total_epochs
    if ratio <= 2 / 3:
        scale = 1.0
    elif ratio >= 3 / 4:
        scale = 0.0
    else:
        phase = (ratio - 2 / 3) / (3 / 4 - 2 / 3)
        scale = 0.5 * (1.0 + math.cos(math.pi * phase))
    return TransientDCFState(
        paper_epoch=paper_epoch,
        total_epochs=total_epochs,
        ratio=ratio,
        scale=scale,
        frozen=ratio > 2 / 3,
        checkpoint_eligible=ratio >= 3 / 4,
    )
```

- [ ] **Step 4: Write failing live/EMA synchronization and freeze tests**

Use a small real module tree with one decoder in `model` and one deep-copied EMA:

```python
def test_apply_state_synchronizes_live_and_ema_and_freezes_only_dcf() -> None:
    live = _model_with_feedback_decoder()
    ema = deepcopy(live)
    state = transient_dcf_state(67, 100)
    apply_transient_dcf_state(live, ema, state)
    live_decoder = find_distribution_feedback_decoder(live)
    ema_decoder = find_distribution_feedback_decoder(ema)
    assert live_decoder.distribution_feedback_scale == state.scale
    assert ema_decoder.distribution_feedback_scale == state.scale
    assert all(not p.requires_grad for p in live_decoder.distribution_feedback.parameters())
    assert any(p.requires_grad for name, p in live.named_parameters() if "distribution_feedback" not in name)
```

- [ ] **Step 5: Run the synchronization test and verify RED**

Run:

```powershell
python -m pytest -q tests/test_transient_dcf.py -k synchronizes
```

Expected: FAIL because model traversal/application helpers are missing.

- [ ] **Step 6: Implement strict traversal and application**

Add helpers that require exactly one feedback-enabled decoder in each model:

```python
def find_distribution_feedback_decoder(model: nn.Module) -> FDRDeformableTransformerDecoder:
    found = [
        module
        for module in model.modules()
        if isinstance(module, FDRDeformableTransformerDecoder)
        and module.distribution_feedback is not None
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one DCF decoder, found {len(found)}")
    return found[0]


def apply_transient_dcf_state(
    live_model: nn.Module, ema_model: nn.Module, state: TransientDCFState
) -> None:
    live = find_distribution_feedback_decoder(live_model)
    ema = find_distribution_feedback_decoder(ema_model)
    live.set_distribution_feedback_scale(state.scale)
    ema.set_distribution_feedback_scale(state.scale)
    if state.frozen:
        live.freeze_distribution_feedback()
    if live.distribution_feedback_scale != ema.distribution_feedback_scale:
        raise RuntimeError("live/EMA DCF scales diverged")
```

- [ ] **Step 7: Run all transient schedule tests**

Run:

```powershell
python -m pytest -q tests/test_transient_dcf.py
```

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 2**

```powershell
git add tests/test_transient_dcf.py src/transient_dcf.py
git commit -m "feat: add frozen transient DCF schedule"
```

### Task 3: Source-Bound Launcher, Evidence, and Tail-Only Best Selection

**Files:**
- Create: `configs/rtdetr-l-transient-dcf-fdr.yaml`
- Create: `scripts/train_transient_dcf_fdr.py`
- Modify: `tests/test_train_dcf_fdr.py`

- [ ] **Step 1: Write failing launcher identity and callback tests**

Add tests for a distinct immutable method and a simulated trainer:

```python
def test_transient_launcher_binds_frozen_formal100_identity(tmp_path: Path) -> None:
    settings = transient.build_settings(
        data_yaml=tmp_path / "data.yaml", output_root=tmp_path / "runs"
    )
    assert Path(settings["model"]).name == "rtdetr-l-transient-dcf-fdr.yaml"
    assert settings["epochs"] == 100
    assert settings["seed"] == 0
    assert "resume" not in settings
    assert transient.build_schedule_record()["full_through_ratio"] == "2/3"
    assert transient.build_schedule_record()["off_from_ratio"] == "3/4"


def test_epoch75_resets_best_once_and_writes_eligible_evidence(tmp_path: Path) -> None:
    trainer = _fake_transient_trainer(paper_epoch=75, best_fitness=0.9)
    evidence = tmp_path / "transient-dcf-schedule.jsonl"
    transient.configure_transient_epoch(trainer, evidence)
    assert trainer.best_fitness is None
    assert trainer.transient_tail_best_reset is True
    with pytest.raises(ValueError, match="duplicate paper epoch"):
        transient.configure_transient_epoch(trainer, evidence)
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert rows[-1]["checkpoint_eligible"] is True
    assert rows[-1]["live_scale"] == rows[-1]["ema_scale"] == 0.0
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_train_dcf_fdr.py -k transient
```

Expected: import/attribute FAIL because the transient launcher does not exist.

- [ ] **Step 3: Add the distinct configuration**

Copy the existing DCF architecture values exactly into
`configs/rtdetr-l-transient-dcf-fdr.yaml`; change only its method comment. Keep:

```yaml
preliminary_box: false
distribution_feedback: true
supervise_pre_boxes: false
supervise_dn_fdr: false
edge_adaptive_fgl: false
```

- [ ] **Step 4: Implement the frozen launcher and callback**

Build on `scripts/train_dcf_fdr.py` imports but expose only one transient arm.
The schedule record must contain exact fractions as strings and the callback
must use `trainer.epoch + 1`:

```python
def build_schedule_record() -> dict[str, object]:
    return {
        "kind": "transient_dcf_v1",
        "full_through_ratio": "2/3",
        "off_from_ratio": "3/4",
        "formal_epochs": FORMAL_EPOCHS,
        "resume_policy": "restart_from_epoch_0",
    }


def configure_transient_epoch(trainer: Any, evidence_path: Path) -> None:
    state = transient_dcf_state(trainer.epoch + 1, trainer.epochs)
    apply_transient_dcf_state(trainer.model, trainer.ema.ema, state)
    if state.checkpoint_eligible and not getattr(trainer, "transient_tail_best_reset", False):
        trainer.best_fitness = None
        trainer.transient_tail_best_reset = True
    append_schedule_evidence(evidence_path, trainer, state)
```

`append_schedule_evidence` must write one canonical JSON object per epoch with
`paper_epoch`, `ratio`, `scale`, `live_scale`, `ema_scale`, `frozen`, and
`checkpoint_eligible`; reject duplicate paper epochs rather than append twice.

Register the callback before `trainer.train()`:

```python
trainer.add_callback(
    "on_train_epoch_start",
    lambda current: configure_transient_epoch(
        current, Path(current.save_dir) / "transient-dcf-schedule.jsonl"
    ),
)
```

Reject every `--resume` argument by omitting it from the parser and record the
restart-only policy in launch authority.

- [ ] **Step 5: Run launcher tests and dry-run**

Run:

```powershell
python -m pytest -q tests/test_train_dcf_fdr.py tests/test_transient_dcf.py
python scripts/train_transient_dcf_fdr.py --help
```

Expected: tests PASS and help exposes no resume option.

- [ ] **Step 6: Commit Task 3**

```powershell
git add configs/rtdetr-l-transient-dcf-fdr.yaml scripts/train_transient_dcf_fdr.py tests/test_train_dcf_fdr.py
git commit -m "feat: launch tail-eligible transient DCF Formal100"
```

### Task 4: Gradient Evidence and Clean Export Boundary

**Files:**
- Modify: `src/rtdetr_fdr.py:438-452`
- Create: `src/transient_dcf_export.py`
- Create: `scripts/export_transient_dcf_fdr.py`
- Create: `tests/test_transient_dcf_export.py`

- [ ] **Step 1: Write failing private-gradient classification test**

Add a small named-parameter model and call the unbound trainer method:

```python
def test_dcf_parameters_are_private_gradient_evidence() -> None:
    trainer = object.__new__(FDRTrainer)
    trainer.model = _GradientPartitionFixture()
    groups = trainer.gradient_parameter_groups()
    assert trainer.model.dcf.weight in groups["fdr_gradient_norm"]
    assert trainer.model.backbone.weight in groups["gradient_norm"]
```

The fixture must expose the DCF parameter name as
`model.28.decoder.distribution_feedback.output.weight`.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest -q tests/test_transient_dcf_export.py -k private_gradient
```

Expected: FAIL because DCF parameters are currently classified as common.

- [ ] **Step 3: Extend the private parameter predicate**

In `FDRTrainer.gradient_parameter_groups`, classify these prefixes as private:

```python
private_markers = (
    ".dec_bbox_head.",
    ".decoder.pre_bbox_head.",
    ".decoder.distribution_feedback.",
)
destination = private if any(marker in name for marker in private_markers) else common
```

- [ ] **Step 4: Write failing export-strip tests**

```python
def test_strip_feedback_keys_removes_only_declared_adapter_state() -> None:
    source = {
        "model.28.backbone.weight": torch.ones(1),
        "model.28.decoder.distribution_feedback.output.weight": torch.ones(1),
        "model.28.decoder.distribution_feedback.output.bias": torch.zeros(1),
    }
    clean, removed = strip_distribution_feedback_state(source)
    assert set(clean) == {"model.28.backbone.weight"}
    assert set(removed) == {
        "model.28.decoder.distribution_feedback.output.weight",
        "model.28.decoder.distribution_feedback.output.bias",
    }


def test_export_rejects_nonzero_feedback_scale() -> None:
    model = _model_with_feedback_decoder()
    find_distribution_feedback_decoder(model).set_distribution_feedback_scale(0.1)
    with pytest.raises(ValueError, match="scale zero"):
        require_zero_feedback_scale(model)
```

- [ ] **Step 5: Run and verify RED**

Run:

```powershell
python -m pytest -q tests/test_transient_dcf_export.py -k "strip_feedback or nonzero"
```

Expected: import/attribute FAIL because export helpers do not exist.

- [ ] **Step 6: Implement strict stripping and export checks**

Create:

```python
FEEDBACK_STATE_MARKER = ".decoder.distribution_feedback."


def strip_distribution_feedback_state(
    state: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    clean = {k: v for k, v in state.items() if FEEDBACK_STATE_MARKER not in k}
    removed = {k: v for k, v in state.items() if FEEDBACK_STATE_MARKER in k}
    if not removed:
        raise ValueError("checkpoint contains no declared DCF state")
    return clean, removed


def require_zero_feedback_scale(model: nn.Module) -> None:
    decoder = find_distribution_feedback_decoder(model)
    if decoder.distribution_feedback_scale != 0.0:
        raise ValueError("T-DCF export requires exact feedback scale zero")
```

The CLI must load the eligible checkpoint, reject a selected epoch outside
`[75, 100]`, instantiate the Clean FDR config, load all shared keys strictly,
save an export manifest, and compare fixed FP32 outputs with `rtol=0, atol=0`
before writing the Clean-shaped artifact.

- [ ] **Step 7: Run export and gradient tests**

Run:

```powershell
python -m pytest -q tests/test_transient_dcf_export.py tests/test_fdr_head.py
```

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 4**

```powershell
git add src/rtdetr_fdr.py src/transient_dcf_export.py scripts/export_transient_dcf_fdr.py tests/test_transient_dcf_export.py
git commit -m "feat: verify inference-free transient DCF export"
```

### Task 5: Full Local Verification and Source-Difference Authority

**Files:**
- Create: `docs/evidence/transient-dcf-preflight-20260826.json`

- [ ] **Step 1: Run the complete relevant test suite**

Run:

```powershell
python -m pytest -q tests/test_fdr_head.py tests/test_fdr_protocol.py tests/test_train_dcf_fdr.py tests/test_transient_dcf.py tests/test_transient_dcf_export.py tests/test_publish_dcf_fdr_results.py tests/test_dcf_fdr_publication.py
```

Expected: zero failures.

- [ ] **Step 2: Verify source boundaries and working tree**

Run:

```powershell
git diff --check f3dcf45e..HEAD
git diff --name-status f3dcf45e..HEAD
git status --short
```

Expected: no whitespace errors; only declared design, schedule, launcher,
export, config, and test files differ; worktree is clean.

- [ ] **Step 3: Produce a machine-readable preflight record**

Record commit SHA, source diff, config SHA-256, schedule formula/boundaries,
test command/results, old Clean release URL, old Clean source commit, and the
mandatory no-resume policy in
`docs/evidence/transient-dcf-preflight-20260826.json`. Do not record a PASS until
the commands in Steps 1-2 actually exit zero.

- [ ] **Step 4: Commit preflight evidence**

```powershell
git add docs/evidence/transient-dcf-preflight-20260826.json
git commit -m "docs: record transient DCF preflight authority"
```

### Task 6: Deploy, Verify on the Training Host, and Launch Formal100

**Files:**
- Remote source checkout under `/data/uav/source/`.
- Remote run root under `/data/uav/runs/`.

- [ ] **Step 1: Push the verified source branch**

Run:

```powershell
git push origin codex/ap-fdr-integrated-redesign
```

Expected: origin points to the preflight evidence commit.

- [ ] **Step 2: Create a fresh source-bound remote checkout**

Use a new explicit directory named with the short commit; do not overwrite the
old `ec4e2a46` source tree. Verify the resolved directory is under
`/data/uav/source/`, fetch the branch, and checkout the exact pushed commit.
Define these task-scoped shell variables from the checked-out commit:

```bash
TDCF_COMMIT="$(git rev-parse --short=8 HEAD)"
TDCF_SOURCE="/data/uav/source/uav-detection-baselines-${TDCF_COMMIT}"
TDCF_RUN_ROOT="/data/uav/runs/transient-dcf-fdr-${TDCF_COMMIT}"
```

- [ ] **Step 3: Run host-side tests and dry-run authority**

Run with `/data/uav/venvs/iber-be-v1/bin/python`:

```bash
python -m pytest -q tests/test_fdr_head.py tests/test_train_dcf_fdr.py tests/test_transient_dcf.py tests/test_transient_dcf_export.py
python scripts/train_transient_dcf_fdr.py \
  --dataset-root /data/uav/datasets/VisDrone \
  --initial-state /data/uav/protocols/fdr-d97e1eb7/initial-state.pt \
  --output-root "${TDCF_RUN_ROOT}" \
  --name formal-seed0-transient-dcf-fdr-v1 \
  --dry-run
```

Expected: tests exit zero; authority reports Formal100, seed 0, the frozen
schedule, and no resume field.

- [ ] **Step 4: Verify host readiness**

Confirm GPU is idle, the old training process is absent, dataset and initial
state hashes match authority, and `/data` has at least 5 GiB free. Abort on any
mismatch.

- [ ] **Step 5: Launch the uninterrupted Formal100 run**

Run the same command without `--dry-run` in a persistent hidden terminal/log
session. Capture PID, exact command, source commit, run directory, and log path.

- [ ] **Step 6: Verify the first epoch is genuinely running**

Confirm the process remains alive, GPU utilization/memory are nonzero, the
authority file exists, and the schedule evidence begins with paper Epoch 1,
`scale=1.0`, `frozen=false`, and `checkpoint_eligible=false`.

- [ ] **Step 7: Report launch state without claiming results**

Report only verified facts: commit, PID/session, run path, current paper epoch,
scale/frozen/eligibility state, disk free, and the next monitoring checkpoint.
Do not claim the method passes until an eligible Epoch 75-100 result meets the
frozen mAP gate.

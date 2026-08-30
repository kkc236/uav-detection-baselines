# AC-BPDD Feasible Full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh-start-only `LRS-FDR+AC-BPDD+FIA` Full method that removes BPDD/LRS target conflicts, guarantees positive decoded box extents, and can be launched on UAVDT from baseline-owned settings.

**Architecture:** Keep the stock matcher, LRS-FGL, FIA, optimizer, AMP invariant, and inference graph authority unchanged. Add one pure pairwise extent projection in the FDR decode path and one training-only assignment-consistent BPDD loss selected explicitly by the LRS G/I YAML files; a thin UAVDT launcher maps only to revised arm I and copies all non-identity training settings from the completed baseline `args.yaml`.

**Tech Stack:** Python 3.9+, PyTorch 2.5.1, Ultralytics 8.4.90, PyYAML, pytest, Git

---

## File map

- Modify `src/fdr_math.py`: pure straight-through pairwise extent projection.
- Modify `src/fdr_head.py`: use the projection for every FDR decode and retain non-persistent geometry evidence.
- Modify `src/bpdd_loss.py`: add assignment-consistent progressive distillation while retaining legacy BPDD for historical configs.
- Modify `src/rtdetr_fdr_bpdd.py`: parse explicit `assignment_mode` and reject ambiguous mode declarations.
- Modify `configs/rtdetr-l-lrs-fdr-bpdd.yaml`: bind AC-BPDD weight `0.15` for arm G.
- Modify `configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`: bind AC-BPDD weight `0.15` for Full arm I.
- Modify `scripts/train_visdrone_lrs_system.py`: version the G/I method identities.
- Create `scripts/train_uavdt_full.py`: baseline-authoritative UAVDT Full entrypoint.
- Create `tests/test_fdr_feasible_geometry.py`: projection, gradient, decode-path, and state-contract regressions.
- Modify `tests/test_bpdd_loss.py`: pure AC-BPDD behavior and gradient tests.
- Modify `tests/test_bpdd_fdr_criterion.py`: criterion-level assignment conflict and disabled-mode tests.
- Modify `tests/test_lrs_system_configs.py`: revised G/I loss contract.
- Modify `tests/test_lrs_system_launcher.py`: revised method identities.
- Create `tests/test_uavdt_full_launcher.py`: Full-only cross-server mapping and input validation.
- Create `docs/UAVDT_EXPERIMENT_HANDOFF_ZH.md`: fresh-start run and evidence commands.

### Task 1: Add a pairwise feasible-extent primitive

**Files:**
- Modify: `src/fdr_math.py`
- Create: `tests/test_fdr_feasible_geometry.py`

- [ ] **Step 1: Write failing pure-function tests**

```python
import pytest
import torch

from src.fdr_math import project_feasible_fdr_distances


def test_projection_makes_pair_extents_positive_and_preserves_center() -> None:
    raw = torch.tensor([[-3.0, -2.5, -2.0, -3.0]], requires_grad=True)
    safe, stats = project_feasible_fdr_distances(raw, reg_scale=4.0)
    assert torch.all(4.0 + safe[:, 0] + safe[:, 2] >= 1e-3)
    assert torch.all(4.0 + safe[:, 1] + safe[:, 3] >= 1e-3)
    torch.testing.assert_close(safe[:, 2] - safe[:, 0], raw[:, 2] - raw[:, 0])
    torch.testing.assert_close(safe[:, 3] - safe[:, 1], raw[:, 3] - raw[:, 1])
    assert stats["horizontal_infeasible"].item() == 1
    assert stats["vertical_infeasible"].item() == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_projection_is_exact_identity_for_feasible_values(dtype: torch.dtype) -> None:
    raw = torch.tensor([[-1.0, -0.5, 0.25, 0.75]], dtype=dtype)
    safe, stats = project_feasible_fdr_distances(raw, reg_scale=4.0)
    assert torch.equal(safe, raw)
    assert safe.dtype == raw.dtype
    assert stats["horizontal_infeasible"].item() == 0
    assert stats["vertical_infeasible"].item() == 0


def test_projection_keeps_nonzero_identity_gradient_when_extent_is_invalid() -> None:
    raw = torch.tensor([[-4.0, -4.0, -4.0, -4.0]], requires_grad=True)
    safe, _ = project_feasible_fdr_distances(raw, reg_scale=4.0)
    safe.sum().backward()
    torch.testing.assert_close(raw.grad, torch.ones_like(raw))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_feasible_geometry.py -q`

Expected: collection fails because `project_feasible_fdr_distances` is absent.

- [ ] **Step 3: Implement the minimal pure operator**

Add to `src/fdr_math.py` and export it from `__all__`:

```python
def project_feasible_fdr_distances(
    distance: Tensor,
    *,
    reg_scale: Tensor | Real = REG_SCALE,
    minimum_extent: float = 1e-3,
) -> tuple[Tensor, dict[str, Tensor]]:
    if distance.ndim == 0 or distance.shape[-1] != 4:
        raise ValueError("distance must end in four FDR edges")
    if not math.isfinite(float(minimum_extent)) or minimum_extent <= 0:
        raise ValueError("minimum_extent must be finite and positive")
    scale = torch.as_tensor(reg_scale, dtype=distance.dtype, device=distance.device).abs()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() == 0:
        raise ValueError("reg_scale must be one finite non-zero scalar")
    minimum = torch.as_tensor(minimum_extent, dtype=distance.dtype, device=distance.device)
    raw_x = scale + distance[..., 0] + distance[..., 2]
    raw_y = scale + distance[..., 1] + distance[..., 3]
    safe_x = raw_x + (raw_x.clamp_min(minimum) - raw_x).detach()
    safe_y = raw_y + (raw_y.clamp_min(minimum) - raw_y).detach()
    correction_x = (safe_x - raw_x) * 0.5
    correction_y = (safe_y - raw_y) * 0.5
    safe = torch.stack(
        (
            distance[..., 0] + correction_x,
            distance[..., 1] + correction_y,
            distance[..., 2] + correction_x,
            distance[..., 3] + correction_y,
        ),
        dim=-1,
    )
    stats = {
        "total": torch.tensor(raw_x.numel(), dtype=torch.long, device=distance.device),
        "horizontal_infeasible": (raw_x < minimum).sum().detach(),
        "vertical_infeasible": (raw_y < minimum).sum().detach(),
        "minimum_raw_horizontal": raw_x.detach().amin(),
        "minimum_raw_vertical": raw_y.detach().amin(),
        "minimum_extent": minimum.detach(),
    }
    return safe, stats
```

- [ ] **Step 4: Verify GREEN and edge validation**

Run: `python -m pytest tests/test_fdr_feasible_geometry.py -q`

Expected: all pure projection tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fdr_math.py tests/test_fdr_feasible_geometry.py
git commit -m "fix: project FDR distances to feasible extents"
```

### Task 2: Route every FDR decode through the projection

**Files:**
- Modify: `src/fdr_head.py`
- Modify: `tests/test_fdr_feasible_geometry.py`

- [ ] **Step 1: Add failing decoder integration tests**

Extend the geometry test with the existing fake decoder fixtures from `tests/test_fdr_head.py` and assert:

```python
def test_training_and_eval_decode_use_projection_without_new_state_keys() -> None:
    decoder = _build_invalid_extent_decoder()
    before = tuple(decoder.state_dict())
    decoder.train()
    train_boxes, _ = _run_decoder(decoder)
    assert torch.all(train_boxes[..., 2:] > 0)
    assert decoder.last_geometry_statistics["horizontal_infeasible"].item() > 0
    assert tuple(decoder.state_dict()) == before
    decoder.eval()
    with torch.no_grad():
        eval_boxes, _ = _run_decoder(decoder)
    assert torch.all(eval_boxes[..., 2:] > 0)
    assert tuple(decoder.state_dict()) == before
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_feasible_geometry.py -q`

Expected: decoder has no `last_geometry_statistics` and emits an invalid extent.

- [ ] **Step 3: Wire the operator and aggregate detached evidence**

In `src/fdr_head.py`, import the operator, initialize and clear a plain dictionary, then replace the decode input:

```python
raw_distance = self.integral(cumulative_corners)
safe_distance, geometry = project_feasible_fdr_distances(
    raw_distance,
    reg_scale=self.reg_scale,
)
refined = distance2bbox(initial_reference, safe_distance, self.reg_scale)
geometry_records.append(geometry)
```

At the end of `forward`, aggregate counts by sum and raw minima by minimum into
`self.last_geometry_statistics`. Do not register a buffer or parameter.

- [ ] **Step 4: Run geometry and existing head regressions**

Run: `python -m pytest tests/test_fdr_feasible_geometry.py tests/test_fdr_head.py tests/test_fdr_math.py -q`

Expected: all selected tests pass and state-dict keys are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/fdr_head.py tests/test_fdr_feasible_geometry.py
git commit -m "fix: enforce feasible geometry in every FDR decode"
```

### Task 3: Implement assignment-consistent BPDD

**Files:**
- Modify: `src/bpdd_loss.py`
- Modify: `tests/test_bpdd_loss.py`
- Modify: `tests/test_bpdd_fdr_criterion.py`

- [ ] **Step 1: Write failing pure AC-BPDD tests**

Use a two-layer, one-batch tensor with explicit layer match triples:

```python
def _triples(query: int, target: int):
    return (
        torch.tensor([0]),
        torch.tensor([query]),
        torch.tensor([target]),
    )


def test_assignment_consistent_bpdd_rejects_target_switch() -> None:
    result = assignment_consistent_bpdd_loss(
        corner_logits=_better_future_logits(),
        pre_boxes=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        gt_bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1]]),
        layer_matches=[_triples(0, 0), _triples(0, 1)],
        options=BPDDOptions(assignment_mode="consistent", margin=0.0),
    )
    assert result.loss.item() == 0.0
    assert result.statistics["stable_match_ratio"].item() == 0.0


def test_assignment_consistent_bpdd_trains_only_the_stable_student() -> None:
    logits = _better_future_logits().requires_grad_(True)
    result = assignment_consistent_bpdd_loss(
        logits,
        _pre_boxes(),
        _targets(),
        [_triples(0, 0), _triples(0, 0)],
        options=BPDDOptions(assignment_mode="consistent", weight=0.15, margin=0.0),
    )
    result.loss.backward()
    assert result.loss.item() > 0
    assert result.statistics["stable_match_ratio"].item() == 1.0
    assert logits.grad[0].abs().sum() > 0
    torch.testing.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]))
```

Also test match-order permutation, no future stable match, worse teacher, FP16 promotion, and empty assignments.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_bpdd_loss.py tests/test_bpdd_fdr_criterion.py -q`

Expected: tests fail because `assignment_mode` and `assignment_consistent_bpdd_loss` do not exist.

- [ ] **Step 3: Extend the options contract without changing legacy defaults**

```python
@dataclass(frozen=True)
class BPDDOptions:
    enabled: bool = True
    weight: float = 0.5
    temperature: float = 0.5
    margin: float = 0.02
    eps: float = 1e-6
    assignment_mode: str = "final"

    def __post_init__(self) -> None:
        # retain existing numerical checks
        if self.assignment_mode not in {"final", "consistent"}:
            raise ValueError("assignment_mode must be 'final' or 'consistent'")
```

- [ ] **Step 4: Implement the pure AC-BPDD loss**

Add this public signature:

```python
LayerMatchTriples = tuple[Tensor, Tensor, Tensor]


def assignment_consistent_bpdd_loss(
    corner_logits: Tensor,
    pre_boxes: Tensor,
    gt_bboxes: Tensor,
    layer_matches: Sequence[LayerMatchTriples],
    *,
    options: BPDDOptions,
) -> BPDDResult:
```

For every source layer, gather its `(batch, query, target)` triples. Build target
bins from that source target. For each future layer, construct a vectorized exact
triple-equality mask, gather the same query's detached distribution, rescore it
on the source target, compute masked softmin weights, and form a detached teacher.
Multiply KL by both the stable-match mask and existing improvement reliability.
Normalize over all source match edges. Return existing statistics plus
`stable_match_ratio`, `stable_source_matches`, and `candidate_source_matches`.

- [ ] **Step 5: Select AC-BPDD inside the criterion**

In `BPDDDetectionLoss.forward`, retain the current final-assignment path when
`assignment_mode == "final"`. For `consistent`, convert `assignments[1:]` into
triples with `self._get_index`, call the new function, and require the decoder
assignment count to equal the corner layer count.

- [ ] **Step 6: Verify GREEN and legacy compatibility**

Run: `python -m pytest tests/test_bpdd_loss.py tests/test_bpdd_fdr_criterion.py tests/test_bpdd_fdr_integration.py -q`

Expected: AC tests pass; legacy final-mode tests remain unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/bpdd_loss.py tests/test_bpdd_loss.py tests/test_bpdd_fdr_criterion.py
git commit -m "feat: align BPDD teachers with layer assignments"
```

### Task 4: Bind revised G/I identities and loss budget

**Files:**
- Modify: `src/rtdetr_fdr_bpdd.py`
- Modify: `configs/rtdetr-l-lrs-fdr-bpdd.yaml`
- Modify: `configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`
- Modify: `scripts/train_visdrone_lrs_system.py`
- Modify: `tests/test_lrs_system_configs.py`
- Modify: `tests/test_lrs_system_launcher.py`

- [ ] **Step 1: Write failing config/parser/identity tests**

```python
AC_BPDD_OPTIONS = {
    "enabled": True,
    "weight": 0.15,
    "temperature": 0.5,
    "margin": 0.02,
    "eps": 1.0e-6,
    "assignment_mode": "consistent",
    "include_dn": False,
}


def test_lrs_bpdd_arms_use_only_assignment_consistent_mode() -> None:
    assert _load(CONFIGS["g"])["bpdd_loss"] == AC_BPDD_OPTIONS
    assert _load(CONFIGS["i"])["bpdd_loss"] == AC_BPDD_OPTIONS
```

Require launcher identities `lrs_fdr_ac_bpdd` and
`lrs_fdr_ac_bpdd_fia`, and require a payload that declares both
`matched_layer` and `assignment_mode` to fail.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_lrs_system_configs.py tests/test_lrs_system_launcher.py -q`

Expected: old weight, mode, and method identities fail.

- [ ] **Step 3: Parse explicit modes safely**

Update `_BPDD_OPTION_KEYS` and `_parse_bpdd_options`:

```python
if "assignment_mode" in payload and "matched_layer" in payload:
    raise ValueError("BPDD assignment mode must have one authority")
mode = payload.get("assignment_mode")
if mode is None:
    if payload.get("matched_layer", "final") != "final":
        raise ValueError("legacy BPDD requires the final stock assignment")
    mode = "final"
if bool(payload.get("include_dn", False)):
    raise ValueError("BPDD excludes denoising queries")
return BPDDOptions(
    enabled=bool(payload.get("enabled", True)),
    weight=float(payload.get("weight", 0.5)),
    temperature=float(payload.get("temperature", 0.5)),
    margin=float(payload.get("margin", 0.02)),
    eps=float(payload.get("eps", 1e-6)),
    assignment_mode=str(mode),
)
```

- [ ] **Step 4: Update only LRS G/I configs and identities**

Replace `weight: 0.5` plus `matched_layer: final` with:

```yaml
  weight: 0.15
  assignment_mode: consistent
```

Update `ARM_METHODS`:

```python
ARM_METHODS = {
    "g": "lrs_fdr_ac_bpdd",
    "h": "lrs_fdr_fia",
    "i": "lrs_fdr_ac_bpdd_fia",
}
```

- [ ] **Step 5: Run LRS graph/model/launcher gates**

Run: `python -m pytest tests/test_lrs_system_configs.py tests/test_lrs_system_models.py tests/test_lrs_system_launcher.py -q`

Expected: all selected tests pass; H still differs from I only by BPDD options.

- [ ] **Step 6: Commit**

```bash
git add src/rtdetr_fdr_bpdd.py configs/rtdetr-l-lrs-fdr-bpdd.yaml configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml scripts/train_visdrone_lrs_system.py tests/test_lrs_system_configs.py tests/test_lrs_system_launcher.py
git commit -m "feat: version assignment-consistent LRS Full"
```

### Task 5: Add a Full-only UAVDT launcher and final audit

**Files:**
- Create: `scripts/train_uavdt_full.py`
- Create: `tests/test_uavdt_full_launcher.py`
- Create: `docs/UAVDT_EXPERIMENT_HANDOFF_ZH.md`

- [ ] **Step 1: Write failing launcher tests**

Test that the CLI exposes only `--data-yaml`, `--baseline-args`,
`--initial-state`, `--output-root`, `--name`, and `--dry-run`; validates ordinary
files and safe run names; derives `nc` from contiguous `names`; rejects a
conflicting `nc`; copies all baseline settings except `model`, `data`, `project`,
`name`, `save_dir`, and `resume`; maps only to arm I and method
`lrs_fdr_ac_bpdd_fia`; and dry-run never constructs a trainer.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_uavdt_full_launcher.py -q`

Expected: collection fails because `scripts/train_uavdt_full.py` is absent.

- [ ] **Step 3: Implement baseline-authoritative settings and validation**

Use these public constants and setting replacements:

```python
METHOD = "lrs_fdr_ac_bpdd_fia"
CONFIG = ARM_CONFIGS["i"]
TRAINER = TRAINER_TYPES["i"]
REPLACED_BASELINE_FIELDS = {"model", "data", "project", "name", "save_dir", "resume"}


def build_settings(baseline: Mapping[str, Any], *, data_yaml: Path,
                   output_root: Path, name: str) -> dict[str, Any]:
    settings = {k: v for k, v in baseline.items() if k not in REPLACED_BASELINE_FIELDS}
    settings.update({
        "model": str(CONFIG.resolve()),
        "data": str(data_yaml.resolve()),
        "project": str(output_root.resolve()),
        "name": validate_run_name(name),
        "exist_ok": False,
    })
    return settings
```

Load YAML with `yaml.safe_load`, require mappings, derive class count from list or
contiguous integer-key `names`, require non-empty `train` and `val`, validate the
initial artifact with `load_fdr_initial_state_artifact`, and hash every authority
input. The launch record includes source, config, baseline args, data YAML,
initial state, derived class count, method, and final settings.

- [ ] **Step 4: Implement dry-run and Full dispatch**

Write the immutable authority record before construction. Return on `--dry-run`;
otherwise call:

```python
trainer = TRAINER(
    overrides=settings,
    initial_state_path=initial_state,
    experiment_seed=int(settings.get("seed", 0)),
)
trainer.train()
```

- [ ] **Step 5: Document the exact cross-server command**

Add to `docs/UAVDT_EXPERIMENT_HANDOFF_ZH.md`:

```bash
python scripts/train_uavdt_full.py \
  --data-yaml /absolute/path/uavdt.yaml \
  --baseline-args /absolute/path/baseline/args.yaml \
  --initial-state /absolute/path/initial-state.pt \
  --output-root /absolute/path/runs \
  --dry-run
```

Then show the identical command without `--dry-run` and state that legacy Full
checkpoints must not be resumed.

- [ ] **Step 6: Run focused and full regression gates**

Run:

```bash
python -m pytest tests/test_fdr_feasible_geometry.py tests/test_bpdd_loss.py tests/test_bpdd_fdr_criterion.py tests/test_bpdd_fdr_integration.py tests/test_lrs_system_configs.py tests/test_lrs_system_models.py tests/test_lrs_system_launcher.py tests/test_uavdt_full_launcher.py -q
python -m pytest -q
git diff --check
```

Expected: focused and full suites pass with zero failures; diff check is clean.

- [ ] **Step 7: Perform the final adversarial source audit**

Run searches proving there is no legacy final-assignment mode in LRS G/I, no
per-edge coordinate clamp, no GIoU clamp, no registered geometry state, and no
UAVDT fallback to G/H or the stock trainer:

```bash
rg -n "matched_layer: final|weight: 0.5" configs/rtdetr-l-lrs-fdr-bpdd.yaml configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml
rg -n "clamp_min\(0|loss_giou.*clamp|register_buffer.*geometry" src
rg -n "TRAINER_TYPES\[[\"'](?:g|h)[\"']\]|FDRTrainer" scripts/train_uavdt_full.py
```

Expected: all searches return no matches.

- [ ] **Step 8: Commit**

```bash
git add scripts/train_uavdt_full.py tests/test_uavdt_full_launcher.py docs/UAVDT_EXPERIMENT_HANDOFF_ZH.md
git commit -m "feat: launch revised Full on UAVDT"
```

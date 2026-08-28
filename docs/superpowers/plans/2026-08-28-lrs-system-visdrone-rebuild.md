# VisDrone LRS System Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a directly runnable `--arm {g,h,i}` launcher for LRS-FDR+BPDD, LRS-FDR+FIA, and LRS-FDR+BPDD+FIA from the repository's already published module implementations.

**Architecture:** Start from the frozen LRS-FDR branch and keep its Formal100 settings as the sole training authority. Add the audited FIA primitive to the YAML parser, propagate LRS into BPDD's criterion, declare three explicit graphs, and dispatch three small Trainer specializations from one thin launcher. The launcher records resolved settings and hashes but does not require historical checkpoint bytes.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, YAML model graphs, pytest, Git

---

## File Map

- Create `src/fia.py`: audited P3 feature-refinement primitive.
- Modify `src/rtdetr_fdr.py`: register `FIA` with the Ultralytics YAML parser.
- Modify `src/rtdetr_fdr_bpdd.py`: pass LRS alpha into the BPDD criterion.
- Create `src/rtdetr_lrs_system.py`: G/H/I model loading and Trainer dispatch types.
- Create three `configs/rtdetr-l-lrs-fdr-*.yaml` files: explicit arm graphs.
- Create `scripts/train_visdrone_lrs_system.py`: argument parsing, authority, and dispatch.
- Create `tests/test_lrs_system_fia.py`: FIA primitive and parser tests.
- Create `tests/test_lrs_system_configs.py`: graph and loss-switch contracts.
- Create `tests/test_lrs_system_models.py`: model loading and Trainer isolation.
- Create `tests/test_lrs_system_launcher.py`: CLI, settings, authority, and dry-run behavior.
- Create `docs/VISDRONE_LRS_SYSTEM_RUNBOOK_ZH.md`: cross-server install and launch commands.

### Task 1: Port the audited FIA primitive and register it

**Files:**
- Create: `src/fia.py`
- Modify: `src/rtdetr_fdr.py`
- Create: `tests/test_lrs_system_fia.py`

- [ ] **Step 1: Write the failing FIA identity and parser-registration tests**

```python
import torch
from ultralytics.nn import tasks as ultralytics_tasks

from src.rtdetr_fdr import register_fdr_module


def test_fia_is_identity_at_initialization() -> None:
    from src.fia import FIA

    module = FIA(256)
    x = torch.randn(1, 256, 8, 8)
    torch.testing.assert_close(module(x), x, rtol=0, atol=0)
    assert module.residual_scale.item() == 0.0


def test_fdr_registration_exposes_fia_to_yaml_parser() -> None:
    from src.fia import FIA

    register_fdr_module()
    assert ultralytics_tasks.FIA is FIA
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_lrs_system_fia.py -q`

Expected: collection fails because `src.fia` does not exist or `FIA` is not registered.

- [ ] **Step 3: Add the audited FIA implementation**

Create `src/fia.py` with the published `FIABaseBlock`, `FIAAttention`, and `FIA`
implementation from `origin/codex/p3-only-fia-ablation:src/fia.py`. Preserve these
observable contracts exactly:

```python
class FIA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _validate_channels(channels)
        with torch.random.fork_rng(
            devices=_fia_construction_cuda_devices(), enabled=True
        ):
            self.refine = nn.Sequential(
                FIABaseBlock(self.channels),
                FIABaseBlock(self.channels),
                FIAAttention(self.channels),
            )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.channels)
        refined = self.refine(x)
        return x + self.residual_scale * (refined - x)
```

Modify `src/rtdetr_fdr.py`:

```python
from src.fia import FIA


def register_fdr_module() -> None:
    ultralytics_tasks.FDRRTDETRDecoder = FDRRTDETRDecoder
    ultralytics_tasks.FrequencyCM = FrequencyCM
    ultralytics_tasks.FIA = FIA
    ultralytics_tasks.IRA = IRA
    ultralytics_tasks.PRIRA = PRIRA
```

- [ ] **Step 4: Run FIA tests and existing FDR parser tests**

Run: `python -m pytest tests/test_lrs_system_fia.py tests/test_fdr_yaml_configs.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fia.py src/rtdetr_fdr.py tests/test_lrs_system_fia.py
git commit -m "feat: register audited FIA for LRS graphs"
```

### Task 2: Preserve LRS inside the BPDD criterion

**Files:**
- Modify: `src/rtdetr_fdr_bpdd.py`
- Create: `tests/test_lrs_system_models.py`

- [ ] **Step 1: Write a failing criterion test**

```python
from pathlib import Path

import yaml

from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_bpdd_criterion_preserves_lrs_alpha() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "rtdetr-l-lrs-fdr.yaml").read_text(encoding="utf-8")
    )
    payload["bpdd_loss"] = {
        "enabled": True,
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1.0e-6,
        "matched_layer": "final",
        "include_dn": False,
    }
    model = FDRBPDDDetectionModel(
        payload,
        nc=10,
        verbose=False,
    )
    criterion = model.init_criterion()
    assert criterion.reliability_shrinkage_alpha == 0.25
    assert criterion.supervise_dn_fdr is False
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_lrs_system_models.py::test_bpdd_criterion_preserves_lrs_alpha -q`

Expected: the assertion fails with `reliability_shrinkage_alpha == 0.0` before
the fix.

- [ ] **Step 3: Pass the existing YAML option to `BPDDDetectionLoss`**

Add this keyword inside `FDRBPDDDetectionModel.init_criterion`:

```python
reliability_shrinkage_alpha=float(
    self.fdr_loss_options.get("reliability_shrinkage_alpha", 0.0)
),
```

- [ ] **Step 4: Run focused BPDD and LRS loss tests**

Run: `python -m pytest tests/test_bpdd_fdr_integration.py tests/test_lrs_fgl_protocol.py tests/test_lrs_system_models.py::test_bpdd_criterion_preserves_lrs_alpha -q`

Expected: all selected tests pass; existing old BPDD continues to resolve alpha
`0.0`.

- [ ] **Step 5: Commit**

```bash
git add src/rtdetr_fdr_bpdd.py tests/test_lrs_system_models.py
git commit -m "fix: preserve LRS shrinkage in BPDD criterion"
```

### Task 3: Declare the three arm graphs

**Files:**
- Create: `configs/rtdetr-l-lrs-fdr-bpdd.yaml`
- Create: `configs/rtdetr-l-lrs-fdr-fia.yaml`
- Create: `configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`
- Create: `tests/test_lrs_system_configs.py`

- [ ] **Step 1: Write failing YAML contract tests**

```python
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "g": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd.yaml",
    "h": ROOT / "configs" / "rtdetr-l-lrs-fdr-fia.yaml",
    "i": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd-fia.yaml",
}


@pytest.mark.parametrize("arm", ["g", "h", "i"])
def test_all_arm_configs_keep_lrs_contract(arm: str) -> None:
    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8"))
    options = payload["head"][-1][3][-1]
    loss = payload["fdr_loss"]
    assert options["preliminary_box"] is False
    assert options["distribution_feedback"] is False
    assert loss == {
        "fgl_weight": 0.15,
        "supervise_pre_boxes": False,
        "supervise_dn_fdr": False,
        "edge_adaptive_fgl": False,
        "reliability_shrinkage_alpha": 0.25,
    }


def test_bpdd_is_present_only_in_g_and_i() -> None:
    payloads = {
        arm: yaml.safe_load(path.read_text(encoding="utf-8"))
        for arm, path in CONFIGS.items()
    }
    assert "bpdd_loss" in payloads["g"]
    assert "bpdd_loss" not in payloads["h"]
    assert "bpdd_loss" in payloads["i"]
    for arm in ("g", "i"):
        assert payloads[arm]["bpdd_loss"] == {
            "enabled": True,
            "weight": 0.5,
            "temperature": 0.5,
            "margin": 0.02,
            "eps": 1.0e-6,
            "matched_layer": "final",
            "include_dn": False,
        }


def test_fia_is_p3_only_in_h_and_i() -> None:
    for arm in ("h", "i"):
        payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8"))
        head = payload["head"]
        # Backbone has indices 0-9, so global model index 22 is head entry 12.
        assert head[12] == [21, 1, "FIA", [256]]
        assert head[13][0] == 21
        assert head[-1][0] == [22, 25, 28]
    assert all(layer[2] != "FIA" for layer in yaml.safe_load(
        CONFIGS["g"].read_text(encoding="utf-8")
    )["head"])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_lrs_system_configs.py -q`

Expected: fail because all three files are missing.

- [ ] **Step 3: Create G from the frozen LRS graph plus BPDD**

Copy `configs/rtdetr-l-lrs-fdr.yaml` byte-for-byte and append:

```yaml
bpdd_loss:
  enabled: true
  weight: 0.5
  temperature: 0.5
  margin: 0.02
  eps: 1.0e-6
  matched_layer: final
  include_dn: false
```

- [ ] **Step 4: Create H with only a P3 FIA insertion**

Keep the LRS decoder/loss options and replace the post-P3 graph tail with:

```yaml
  - [21, 1, FIA, [256]]
  - [21, 1, Conv, [256, 3, 2]]
  - [[23, 17], 1, Concat, [1]]
  - [-1, 3, RepC3, [256]]
  - [25, 1, Conv, [256, 3, 2]]
  - [[26, 12], 1, Concat, [1]]
  - [-1, 3, RepC3, [256]]
  - [[22, 25, 28], 1, FDRRTDETRDecoder,
     [nc, [256, 256, 256],
      {hidden_dim: 256, num_queries: 300, num_decoder_layers: 6,
       reg_max: 32, reg_scale: 4.0, up: 0.5, cumulative: true,
       preliminary_box: false, distribution_feedback: false, private_seed: 10000}]]
```

Do not add a `bpdd_loss` section to H.

- [ ] **Step 5: Create I as H plus the exact BPDD section from G**

Append the same `bpdd_loss` mapping shown in Step 3 to
`configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`; no other setting differs from H.

- [ ] **Step 6: Run YAML tests**

Run: `python -m pytest tests/test_lrs_system_configs.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add configs/rtdetr-l-lrs-fdr-*.yaml tests/test_lrs_system_configs.py
git commit -m "feat: declare VisDrone LRS system arms"
```

### Task 4: Add isolated G/H/I models and Trainers

**Files:**
- Create: `src/rtdetr_lrs_system.py`
- Modify: `tests/test_lrs_system_models.py`

- [ ] **Step 1: Add failing model identity tests**

```python
import torch

from src.bpdd_loss import BPDDDetectionLoss
from src.fdr_loss import FDRDetectionLoss
from src.fia import FIA
from src.rtdetr_lrs_system import ARM_CONFIGS, MODEL_TYPES


def test_g_h_i_select_expected_criterion_and_fia() -> None:
    expected = {
        "g": (BPDDDetectionLoss, False),
        "h": (FDRDetectionLoss, True),
        "i": (BPDDDetectionLoss, True),
    }
    for arm, (criterion_type, has_fia) in expected.items():
        model = MODEL_TYPES[arm](ARM_CONFIGS[arm], nc=10, verbose=False)
        assert isinstance(model.init_criterion(), criterion_type)
        assert any(isinstance(layer, FIA) for layer in model.model) is has_fia


def test_fia_gradient_group_is_disjoint() -> None:
    from src.rtdetr_lrs_system import LRSFDRFIATrainer

    trainer = LRSFDRFIATrainer.__new__(LRSFDRFIATrainer)
    trainer.model = MODEL_TYPES["h"](ARM_CONFIGS["h"], nc=10, verbose=False)
    groups = trainer.gradient_parameter_groups()
    identifiers = [{id(parameter) for parameter in group} for group in groups.values()]
    assert all(identifiers[index].isdisjoint(identifiers[other])
               for index in range(len(identifiers))
               for other in range(index + 1, len(identifiers)))
    assert "fia_gradient_norm" in groups
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_lrs_system_models.py -q`

Expected: fail because `src.rtdetr_lrs_system` does not exist.

- [ ] **Step 3: Implement shared FIA graph initialization and state remapping**

Create `src/rtdetr_lrs_system.py` with these constants and helpers:

```python
ROOT = Path(__file__).resolve().parents[1]
ARM_CONFIGS = {
    "g": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd.yaml",
    "h": ROOT / "configs" / "rtdetr-l-lrs-fdr-fia.yaml",
    "i": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd-fia.yaml",
}
FIA_MODEL_INDEX = 22
FIA_STATE_PREFIX = "model.22."


def remap_fia_shared_key(name: str) -> str:
    match = re.match(r"^model\.(\d+)\.(.+)$", name)
    if match is None or int(match.group(1)) < FIA_MODEL_INDEX:
        return name
    return f"model.{int(match.group(1)) + 1}.{match.group(2)}"


def initialize_fia_graph(model: nn.Module, private_seed: int) -> None:
    if len(model.model) != 30 or not isinstance(model.model[22], FIA):
        raise TypeError("FIA must be the standalone YAML layer at model index 22")
    if model.model[22].f != 21 or model.model[23].f != 21:
        raise ValueError("FIA must refine only P3 while P4 bypasses it")
    if model.model[-1].f != [22, 25, 28]:
        raise ValueError("decoder must consume FIA-P3 plus stock P4/P5")
    initialize_private_module(model.model[22], private_seed=private_seed)
    with torch.no_grad():
        model.model[22].residual_scale.zero_()
```

Implement `load_fia_initial_state` using `validate_fdr_initial_state`, map every
source model key at index 22 or later with `remap_fia_shared_key`, require every
unmapped target key to start with `model.22.`, and load with `strict=False` only
after checking shapes, dtypes, missing keys, and unexpected keys.

- [ ] **Step 4: Implement the three model types**

```python
class LRSFDRBPDDDetectionModel(FDRBPDDDetectionModel):
    pass


class LRSFDRFIADetectionModel(FDRRTDETRDetectionModel):
    def __init__(self, cfg=ARM_CONFIGS["h"], ch=3, nc=None, verbose=True,
                 *, private_seed=None, fia_private_seed=20_000):
        super().__init__(cfg, ch, nc, verbose, private_seed=private_seed)
        self.fia_private_seed = int(fia_private_seed)
        initialize_fia_graph(self, self.fia_private_seed)


class LRSFDRBPDDFIADetectionModel(FDRBPDDDetectionModel):
    def __init__(self, cfg=ARM_CONFIGS["i"], ch=3, nc=None, verbose=True,
                 *, private_seed=None, fia_private_seed=20_000):
        super().__init__(cfg, ch, nc, verbose, private_seed=private_seed)
        self.fia_private_seed = int(fia_private_seed)
        initialize_fia_graph(self, self.fia_private_seed)


MODEL_TYPES = {
    "g": LRSFDRBPDDDetectionModel,
    "h": LRSFDRFIADetectionModel,
    "i": LRSFDRBPDDFIADetectionModel,
}
```

- [ ] **Step 5: Implement Trainer specializations**

Define `_fia_gradient_parameter_groups(model)` so all trainable FIA parameters
go only to `fia_gradient_norm`, decoder distribution parameters go only to
`fdr_gradient_norm`, and all remaining parameters go to `gradient_norm`.

Define `LRSFDRBPDDTrainer(FDRBPDDTrainer)`,
`LRSFDRFIATrainer(FDRTrainer)`, and
`LRSFDRBPDDFIATrainer(FDRBPDDTrainer)`. Each `get_model` uses its matching
`ARM_CONFIGS` path, `private_seed=10000+experiment_seed`, and for FIA arms
`fia_private_seed=20000+experiment_seed`. G loads a validated FDR artifact with
`_load_initial_state(..., variant="fdr")`; H and I call
`load_fia_initial_state`. Export:

```python
TRAINER_TYPES = {
    "g": LRSFDRBPDDTrainer,
    "h": LRSFDRFIATrainer,
    "i": LRSFDRBPDDFIATrainer,
}
```

- [ ] **Step 6: Run model, BPDD, FIA, and LRS tests**

Run: `python -m pytest tests/test_lrs_system_models.py tests/test_bpdd_fdr_integration.py tests/test_lrs_system_fia.py tests/test_lrs_fgl_protocol.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/rtdetr_lrs_system.py tests/test_lrs_system_models.py src/rtdetr_fdr_bpdd.py
git commit -m "feat: integrate LRS BPDD and FIA trainers"
```

### Task 5: Add the unified `--arm` launcher

**Files:**
- Create: `scripts/train_visdrone_lrs_system.py`
- Create: `tests/test_lrs_system_launcher.py`

- [ ] **Step 1: Write failing CLI and settings tests**

```python
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_visdrone_lrs_system.py"


def test_cli_exposes_only_g_h_i_and_fixed_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for value in ("{g,h,i}", "--dataset-root", "--initial-state",
                  "--output-root", "--name", "--dry-run"):
        assert value in result.stdout
    for forbidden in ("--epochs", "--seed", "--batch", "--lr0",
                      "--bpdd-weight", "--fia-seed"):
        assert forbidden not in result.stdout


def test_arm_settings_differ_only_by_model_and_name(tmp_path: Path) -> None:
    from scripts.train_visdrone_lrs_system import build_settings

    settings = {
        arm: build_settings(arm, tmp_path / "data.yaml", tmp_path / "runs", None)
        for arm in ("g", "h", "i")
    }
    for arm, payload in settings.items():
        assert payload["epochs"] == 100
        assert payload["seed"] == 0
        assert payload["imgsz"] == 640
        assert Path(payload["model"]).name.startswith("rtdetr-l-lrs-fdr-")
    reference = {k: v for k, v in settings["g"].items()
                 if k not in {"model", "name"}}
    for arm in ("h", "i"):
        assert {k: v for k, v in settings[arm].items()
                if k not in {"model", "name"}} == reference
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_lrs_system_launcher.py -q`

Expected: fail because the launcher does not exist.

- [ ] **Step 3: Implement the thin launcher**

Use this public structure:

```python
ARM_METHODS = {
    "g": "lrs_fdr_bpdd",
    "h": "lrs_fdr_fia",
    "i": "lrs_fdr_bpdd_fia",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one VisDrone LRS system arm")
    parser.add_argument("--arm", choices=("g", "h", "i"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_settings(arm: str, data_yaml: Path, output_root: Path,
                   name: str | None) -> dict[str, Any]:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown LRS system arm: {arm}")
    return {
        **FROZEN_SETTINGS,
        "model": str(ARM_CONFIGS[arm].resolve()),
        "save_period": -1,
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": name or f"formal-seed0-{ARM_METHODS[arm]}-v1",
        "exist_ok": False,
    }
```

`main` calls `require_clean_tracked_worktree`, `prepare_data_yaml(...,
"formal", ...)`, `validate_initial_state_file`, `dataset_signature`, and
`current_source_identity`; writes one deterministic authority JSON using
`write_json_atomic`; prints the complete record; returns before Trainer creation
for `--dry-run`; otherwise constructs `TRAINER_TYPES[arm]` with the resolved
settings, initial state, and `experiment_seed=0`, then calls `train()`.

- [ ] **Step 4: Add authority conflict and dry-run tests**

Monkeypatch filesystem-heavy validators so `main` is exercised without a GPU.
Assert that dry-run writes `authority/<name>.json`, contains the arm, method,
source/config/initial-state hashes and full settings, never constructs a
Trainer, and rejects a pre-existing authority file with changed content.

- [ ] **Step 5: Run launcher and related CLI tests**

Run: `python -m pytest tests/test_lrs_system_launcher.py tests/test_train_rtdetr_bpdd_cli.py tests/test_lrs_fgl_protocol.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_visdrone_lrs_system.py tests/test_lrs_system_launcher.py
git commit -m "feat: add unified VisDrone LRS arm launcher"
```

### Task 6: Publish a cross-server runbook

**Files:**
- Create: `docs/VISDRONE_LRS_SYSTEM_RUNBOOK_ZH.md`

- [ ] **Step 1: Write the runbook with exact environment and commands**

Document:

```bash
git clone https://github.com/kkc236/uav-detection-baselines.git
cd uav-detection-baselines
git switch codex/lrs-system-visdrone-rebuild
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Add one dry-run and one training command for each arm, using shell variables
`VISDRONE_ROOT`, `INITIAL_STATE`, and `RUNS_ROOT`. State that the dataset and
initial-state artifact are external inputs and that the initial-state artifact
must pass structural validation but need not match the historical raw hash.

- [ ] **Step 2: Verify every documented path and option exists**

Run:

```bash
python scripts/train_visdrone_lrs_system.py --help
python -c "from src.rtdetr_lrs_system import ARM_CONFIGS; assert all(p.is_file() for p in ARM_CONFIGS.values())"
```

Expected: help exits 0 and every arm configuration exists.

- [ ] **Step 3: Commit**

```bash
git add docs/VISDRONE_LRS_SYSTEM_RUNBOOK_ZH.md
git commit -m "docs: add VisDrone LRS system runbook"
```

### Task 7: Final verification and GitHub publication

**Files:**
- Verify all files changed by Tasks 1-6.

- [ ] **Step 1: Run syntax and whitespace checks**

Run:

```bash
python -m compileall -q src scripts tests
git diff --check origin/codex/lrs-fgl...HEAD
```

Expected: both commands exit 0.

- [ ] **Step 2: Run the complete focused suite**

Run:

```bash
python -m pytest \
  tests/test_lrs_system_fia.py \
  tests/test_lrs_system_configs.py \
  tests/test_lrs_system_models.py \
  tests/test_lrs_system_launcher.py \
  tests/test_lrs_fgl_protocol.py \
  tests/test_bpdd_loss.py \
  tests/test_bpdd_fdr_integration.py \
  tests/test_fdr_yaml_configs.py -q
```

Expected: zero failures; hardware-dependent tests may report explicit skips.

- [ ] **Step 3: Inspect the final branch diff and status**

Run:

```bash
git status --short
git log --oneline origin/codex/lrs-fgl..HEAD
git diff --stat origin/codex/lrs-fgl...HEAD
```

Expected: clean status and only the design, plan, source, configuration, test,
and runbook commits described above.

- [ ] **Step 4: Push the source branch**

Run:

```bash
git push -u origin codex/lrs-system-visdrone-rebuild
```

Expected: GitHub creates or updates
`kkc236/uav-detection-baselines:codex/lrs-system-visdrone-rebuild`.

- [ ] **Step 5: Publish the handoff pointer to the material repository**

Update the material handoff to name the pushed source branch, head commit,
launcher, three YAML paths, runbook, and verification command. Commit and push
that update to `private-material/codex/lrs-fdr-material-freeze` without adding
weights, credentials, or datasets.

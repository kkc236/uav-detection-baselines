# Gradient-Decoupled Persistent DCF-FDR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, deploy, and launch a fresh Formal100 DCF-FDR arm that keeps the exact Epoch-1-66 gradient-decoupled DCF behavior enabled and trainable through Epoch 100.

**Architecture:** Add a dedicated persistent-gradient experiment identity around the existing DCF adapter and private FDR gradient partition. A read-only epoch audit asserts scale one and live DCF trainability without changing model state, while a detached milestone watcher atomically preserves full resumable checkpoints including Epoch 66. The transient controller, freezing, cosine withdrawal, tail reset, and Clean export are excluded.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics 8.4.90, YAML, pytest, Git, PowerShell, Bash, Ubuntu/CUDA.

---

## File Structure

- Create `src/persistent_dcf.py`: immutable all-on state and fail-closed live/EMA DCF audit.
- Create `configs/rtdetr-l-persistent-gradient-dcf-fdr.yaml`: dedicated method identity with model/loss bytes equivalent to the transient run's Epoch-1-66 configuration.
- Create `scripts/train_persistent_gradient_dcf_fdr.py`: Formal100 launcher, immutable authority, audit callback, and epoch JSONL evidence.
- Create `scripts/watch_persistent_dcf_checkpoints.py`: atomic, validated milestone snapshots from `last.pt`.
- Create `tests/test_persistent_dcf.py`: all-on schedule, live/EMA audit, and private-gradient membership tests.
- Modify `tests/test_train_dcf_fdr.py`: launcher identity, configuration equivalence, no-resume, and no-transient-control tests.
- Create `tests/test_persistent_dcf_checkpoint_watcher.py`: checkpoint validation, epoch mapping, atomic snapshot, and stale-source rejection tests.
- Create `docs/evidence/persistent-gradient-dcf-preflight-20260827.json`: sanitized local/server preflight authority after verification.

### Task 1: Pure Persistent DCF State and Audit

**Files:**
- Create: `tests/test_persistent_dcf.py`
- Create: `src/persistent_dcf.py`

- [ ] **Step 1: Write failing all-on state and audit tests**

Create `tests/test_persistent_dcf.py`:

```python
from copy import deepcopy

import pytest
from torch import nn

from src.fdr_head import (
    DistributionConditionedFeedback,
    FDRDeformableTransformerDecoder,
)
from src.persistent_dcf import (
    audit_persistent_dcf_state,
    persistent_dcf_state,
)
from src.rtdetr_fdr import FDRTrainer


class _Layer(nn.Module):
    pass


class _StockDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(6)])
        self.hidden_dim = 16
        self.num_layers = 6
        self.eval_idx = 5


def _model() -> nn.Module:
    model = nn.Module()
    model.backbone = nn.Linear(16, 16)
    model.decoder = FDRDeformableTransformerDecoder.from_stock(
        _StockDecoder(),
        pre_bbox_head=nn.Linear(16, 4),
        distribution_feedback=DistributionConditionedFeedback(
            16, private_seed=10_001
        ),
    )
    return model


def test_formal100_is_all_on_and_trainable_for_every_epoch() -> None:
    states = [persistent_dcf_state(epoch, 100) for epoch in range(1, 101)]
    assert all(state.scale == 1.0 for state in states)
    assert all(state.trainable for state in states)
    assert all(state.checkpoint_eligible for state in states)


@pytest.mark.parametrize("epoch,total", [(0, 100), (101, 100), (1, 0)])
def test_persistent_state_rejects_invalid_domain(epoch: int, total: int) -> None:
    with pytest.raises(ValueError, match="positive training horizon"):
        persistent_dcf_state(epoch, total)


def test_audit_requires_live_and_ema_scale_one_and_live_trainability() -> None:
    live = _model()
    ema = deepcopy(live)
    state = persistent_dcf_state(67, 100)
    record = audit_persistent_dcf_state(live, ema, state)
    assert record["live_scale"] == record["ema_scale"] == 1.0
    assert record["live_feedback_trainable"] is True

    live.decoder.set_distribution_feedback_scale(0.5)
    with pytest.raises(RuntimeError, match="scale must remain exactly 1.0"):
        audit_persistent_dcf_state(live, ema, state)


def test_dcf_parameters_are_exclusively_in_private_gradient_group() -> None:
    model = _model()
    holder = type("Holder", (), {"model": model})()
    groups = FDRTrainer.gradient_parameter_groups(holder)
    feedback = {id(p) for p in model.decoder.distribution_feedback.parameters()}
    private = {id(p) for p in groups["fdr_gradient_norm"]}
    common = {id(p) for p in groups["gradient_norm"]}
    assert feedback
    assert feedback <= private
    assert feedback.isdisjoint(common)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_persistent_dcf.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.persistent_dcf'`.

- [ ] **Step 3: Implement the immutable state and fail-closed audit**

Create `src/persistent_dcf.py`:

```python
"""All-on state authority for gradient-decoupled persistent DCF."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from src.transient_dcf import find_distribution_feedback_decoder


@dataclass(frozen=True)
class PersistentDCFState:
    paper_epoch: int
    total_epochs: int
    scale: float = 1.0
    trainable: bool = True
    checkpoint_eligible: bool = True

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def persistent_dcf_state(
    paper_epoch: int, total_epochs: int
) -> PersistentDCFState:
    if total_epochs <= 0 or not 1 <= paper_epoch <= total_epochs:
        raise ValueError("paper epoch must be inside a positive training horizon")
    return PersistentDCFState(paper_epoch=paper_epoch, total_epochs=total_epochs)


def audit_persistent_dcf_state(
    live_model: nn.Module,
    ema_model: nn.Module,
    state: PersistentDCFState,
) -> dict[str, int | float | bool]:
    live = find_distribution_feedback_decoder(live_model)
    ema = find_distribution_feedback_decoder(ema_model)
    live_scale = float(live.distribution_feedback_scale)
    ema_scale = float(ema.distribution_feedback_scale)
    if live_scale != 1.0 or ema_scale != 1.0 or state.scale != 1.0:
        raise RuntimeError("persistent DCF scale must remain exactly 1.0")
    trainable = all(
        parameter.requires_grad
        for parameter in live.distribution_feedback.parameters()
    )
    if not trainable or not state.trainable:
        raise RuntimeError("persistent DCF parameters must remain trainable")
    return {
        **state.to_dict(),
        "live_scale": live_scale,
        "ema_scale": ema_scale,
        "live_feedback_trainable": trainable,
    }


__all__ = [
    "PersistentDCFState",
    "audit_persistent_dcf_state",
    "persistent_dcf_state",
]
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest -q tests/test_persistent_dcf.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/persistent_dcf.py tests/test_persistent_dcf.py
git commit -m "feat: audit all-on persistent DCF state"
```

### Task 2: Dedicated Configuration and Formal100 Launcher

**Files:**
- Create: `configs/rtdetr-l-persistent-gradient-dcf-fdr.yaml`
- Create: `scripts/train_persistent_gradient_dcf_fdr.py`
- Modify: `tests/test_train_dcf_fdr.py`

- [ ] **Step 1: Write failing launcher and configuration tests**

Append to `tests/test_train_dcf_fdr.py`:

```python
from scripts import train_persistent_gradient_dcf_fdr as persistent_gradient
import yaml


def test_persistent_gradient_launcher_binds_all_on_formal100(tmp_path: Path) -> None:
    settings = persistent_gradient.build_settings(
        data_yaml=tmp_path / "data.yaml", output_root=tmp_path / "runs"
    )
    assert Path(settings["model"]).name == (
        "rtdetr-l-persistent-gradient-dcf-fdr.yaml"
    )
    assert settings["epochs"] == 100
    assert settings["seed"] == 0
    assert settings["save_period"] == -1
    assert "resume" not in settings
    assert "resume" not in persistent_gradient.build_parser().format_help()


def test_persistent_gradient_config_matches_transient_epoch1_to66() -> None:
    root = Path(__file__).resolve().parents[1]
    persistent_cfg = yaml.safe_load(
        (root / "configs/rtdetr-l-persistent-gradient-dcf-fdr.yaml").read_text()
    )
    transient_cfg = yaml.safe_load(
        (root / "configs/rtdetr-l-transient-dcf-fdr.yaml").read_text()
    )
    assert persistent_cfg == transient_cfg


def test_persistent_authority_excludes_every_transient_behavior() -> None:
    authority = persistent_gradient.build_method_record()
    assert authority == {
        "kind": "persistent_gradient_dcf_v1",
        "scale": "1.0_all_epochs",
        "trainable": "all_epochs",
        "checkpoint_eligible_from_epoch": 1,
        "resume_policy": "restart_from_epoch_0",
    }
    source = Path(persistent_gradient.__file__).read_text(encoding="utf-8")
    assert "configure_transient_epoch" not in source
    assert "freeze_distribution_feedback" not in source
    assert "best_fitness = None" not in source
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_train_dcf_fdr.py -k persistent_gradient
```

Expected: import fails because the dedicated launcher does not exist.

- [ ] **Step 3: Create the dedicated model configuration**

Create `configs/rtdetr-l-persistent-gradient-dcf-fdr.yaml` with the exact parsed
content of `configs/rtdetr-l-transient-dcf-fdr.yaml`; change only the leading
comment to:

```yaml
# RT-DETR-L Clean FDR with one all-on gradient-decoupled DCF adapter.
```

Do not change any parsed YAML value. In particular retain:

```yaml
{hidden_dim: 256, num_queries: 300, num_decoder_layers: 6,
 reg_max: 32, reg_scale: 4.0, up: 0.5, cumulative: true,
 preliminary_box: false, distribution_feedback: true, private_seed: 10000}

fdr_loss:
  fgl_weight: 0.15
  supervise_pre_boxes: false
  supervise_dn_fdr: false
  edge_adaptive_fgl: false
```

- [ ] **Step 4: Implement the dedicated launcher and epoch evidence**

Create `scripts/train_persistent_gradient_dcf_fdr.py`:

```python
"""Launch all-on gradient-decoupled Persistent DCF-FDR Formal100."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rtdetr-l-persistent-gradient-dcf-fdr.yaml"
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from scripts.train_ace_fdr import (  # noqa: E402
    require_clean_tracked_worktree,
    validate_initial_state_file,
)
from scripts.train_dcf_fdr import (  # noqa: E402
    build_launch_record as build_base_launch_record,
)
from scripts.train_rtdetr_fdr import (  # noqa: E402
    FORMAL_EPOCHS,
    FROZEN_SETTINGS,
    current_source_identity,
    prepare_data_yaml,
)
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.persistent_dcf import (  # noqa: E402
    audit_persistent_dcf_state,
    persistent_dcf_state,
)
from src.transient_dcf import find_distribution_feedback_decoder  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train all-on gradient-decoupled DCF-FDR Formal100 seed0."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_settings(
    *, data_yaml: Path, output_root: Path, name: str | None = None
) -> dict[str, Any]:
    return {
        **FROZEN_SETTINGS,
        "model": str(CONFIG.resolve()),
        "save_period": -1,
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": name or "formal-seed0-persistent-gradient-dcf-fdr-v1",
        "exist_ok": False,
    }


def build_method_record() -> dict[str, object]:
    return {
        "kind": "persistent_gradient_dcf_v1",
        "scale": "1.0_all_epochs",
        "trainable": "all_epochs",
        "checkpoint_eligible_from_epoch": 1,
        "resume_policy": "restart_from_epoch_0",
    }


def build_launch_record(
    *,
    source_identity: Mapping[str, Any],
    initial_state_path: Path,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    record = build_base_launch_record(
        source_identity=source_identity,
        initial_state_path=initial_state_path,
        dataset=dataset,
        settings=settings,
    )
    record["method"] = "persistent_gradient_dcf_fdr"
    record["persistent_dcf"] = build_method_record()
    return record


def append_epoch_evidence(path: Path, trainer: Any) -> None:
    state = persistent_dcf_state(trainer.epoch + 1, trainer.epochs)
    record = audit_persistent_dcf_state(trainer.model, trainer.ema.ema, state)
    decoder = find_distribution_feedback_decoder(trainer.model)
    feedback_ids = {
        id(parameter) for parameter in decoder.distribution_feedback.parameters()
    }
    groups = trainer.gradient_parameter_groups()
    private_ids = {id(parameter) for parameter in groups["fdr_gradient_norm"]}
    common_ids = {id(parameter) for parameter in groups["gradient_norm"]}
    if not feedback_ids or not feedback_ids <= private_ids:
        raise RuntimeError("DCF parameters are missing from private FDR gradient group")
    if not feedback_ids.isdisjoint(common_ids):
        raise RuntimeError("DCF parameters leaked into common gradient group")
    record["private_gradient_group"] = True
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(row["paper_epoch"] == state.paper_epoch for row in rows):
        raise ValueError(f"duplicate paper epoch: {state.paper_epoch}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_clean_tracked_worktree()
    output_root = args.output_root.resolve()
    authority_root = output_root / "authority"
    data_yaml = prepare_data_yaml(
        args.dataset_root.resolve(), "formal", authority_root / "data"
    )
    initial_state = validate_initial_state_file(args.initial_state)
    settings = build_settings(
        data_yaml=data_yaml, output_root=output_root, name=args.name
    )
    record = build_launch_record(
        source_identity=current_source_identity(),
        initial_state_path=initial_state,
        dataset=dataset_signature(args.dataset_root.resolve()),
        settings=settings,
    )
    record_path = authority_root / f"{settings['name']}.json"
    if record_path.exists():
        if json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"authority exists with different bytes: {record_path}")
    else:
        write_json_atomic(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    from src.rtdetr_fdr import FDRTrainer

    trainer = FDRTrainer(
        overrides=settings,
        initial_state_path=initial_state,
        experiment_seed=0,
    )
    trainer.add_callback(
        "on_train_epoch_start",
        lambda current: append_epoch_evidence(
            Path(current.save_dir) / "persistent-dcf-state.jsonl", current
        ),
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run launcher tests and full focused suite**

Run:

```powershell
python -m pytest -q tests/test_train_dcf_fdr.py -k persistent_gradient
python -m pytest -q tests/test_persistent_dcf.py tests/test_train_dcf_fdr.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add configs/rtdetr-l-persistent-gradient-dcf-fdr.yaml `
        scripts/train_persistent_gradient_dcf_fdr.py `
        tests/test_train_dcf_fdr.py
git commit -m "feat: launch persistent gradient-decoupled DCF"
```

### Task 3: Milestone Checkpoint Protection

**Files:**
- Create: `tests/test_persistent_dcf_checkpoint_watcher.py`
- Create: `scripts/watch_persistent_dcf_checkpoints.py`

- [ ] **Step 1: Write failing checkpoint validation tests**

Create `tests/test_persistent_dcf_checkpoint_watcher.py`:

```python
from pathlib import Path

import pytest
import torch

from scripts.watch_persistent_dcf_checkpoints import (
    checkpoint_summary,
    preserve_milestone,
)


def _checkpoint(path: Path, *, zero_based_epoch: int) -> None:
    torch.save(
        {
            "epoch": zero_based_epoch,
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {"scale": 128.0},
            "ema": {"weight": torch.tensor([1.0])},
        },
        path,
    )


def test_checkpoint_summary_maps_zero_based_epoch_to_paper_epoch(tmp_path: Path) -> None:
    source = tmp_path / "last.pt"
    _checkpoint(source, zero_based_epoch=65)
    assert checkpoint_summary(source)["paper_epoch"] == 66


@pytest.mark.parametrize("missing", ["optimizer", "scaler", "ema"])
def test_checkpoint_summary_requires_resumable_state(
    tmp_path: Path, missing: str
) -> None:
    source = tmp_path / "last.pt"
    payload = {
        "epoch": 65,
        "optimizer": {},
        "scaler": {},
        "ema": {},
    }
    payload[missing] = None
    torch.save(payload, source)
    with pytest.raises(RuntimeError, match=missing):
        checkpoint_summary(source)


def test_preserve_milestone_is_atomic_and_rejects_stale_source(tmp_path: Path) -> None:
    source = tmp_path / "last.pt"
    target = tmp_path / "milestones"
    _checkpoint(source, zero_based_epoch=65)
    saved = preserve_milestone(source, target, expected_paper_epoch=66)
    assert saved.name == "epoch0066.pt"
    assert checkpoint_summary(saved)["paper_epoch"] == 66
    assert not list(target.glob("*.tmp"))

    _checkpoint(source, zero_based_epoch=66)
    with pytest.raises(RuntimeError, match="expected paper epoch 66"):
        preserve_milestone(source, target / "other", expected_paper_epoch=66)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_persistent_dcf_checkpoint_watcher.py
```

Expected: collection fails because the watcher module does not exist.

- [ ] **Step 3: Implement validated atomic milestone preservation**

Create `scripts/watch_persistent_dcf_checkpoints.py` with:

```python
"""Preserve selected full Persistent DCF checkpoints without changing training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch

MILESTONES = (25, 50, 66, 75, 90, 100)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_summary(path: Path) -> dict[str, int | bool]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("epoch"), int):
        raise RuntimeError("checkpoint epoch is missing")
    for key in ("optimizer", "scaler", "ema"):
        if payload.get(key) is None:
            raise RuntimeError(f"checkpoint {key} is missing")
    return {
        "paper_epoch": int(payload["epoch"]) + 1,
        "optimizer": True,
        "scaler": True,
        "ema": True,
    }


def preserve_milestone(
    source: Path, target_root: Path, *, expected_paper_epoch: int
) -> Path:
    source = source.resolve()
    summary = checkpoint_summary(source)
    if summary["paper_epoch"] != expected_paper_epoch:
        raise RuntimeError(
            f"expected paper epoch {expected_paper_epoch}, "
            f"found {summary['paper_epoch']}"
        )
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"epoch{expected_paper_epoch:04d}.pt"
    temporary = target.with_suffix(".pt.tmp")
    shutil.copy2(source, temporary)
    copied = checkpoint_summary(temporary)
    if copied != summary:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("copied checkpoint summary changed")
    os.replace(temporary, target)
    manifest = {
        **summary,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "path": target.name,
    }
    manifest_tmp = target.with_suffix(".json.tmp")
    manifest_path = target.with_suffix(".json")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_tmp, manifest_path)
    return target


def completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return len(rows)


def watch(run_dir: Path, poll_seconds: float) -> int:
    run_dir = run_dir.resolve()
    source = run_dir / "weights" / "last.pt"
    target_root = run_dir / "milestone-checkpoints"
    results = run_dir / "results.csv"
    while True:
        completed = completed_epochs(results)
        missing = [
            epoch
            for epoch in MILESTONES
            if not (target_root / f"epoch{epoch:04d}.pt").exists()
        ]
        if missing and source.exists():
            current = checkpoint_summary(source)["paper_epoch"]
            expected = missing[0]
            if current == expected:
                preserve_milestone(
                    source, target_root, expected_paper_epoch=expected
                )
            elif current > expected:
                raise RuntimeError(
                    f"missed milestone {expected}; current checkpoint is {current}"
                )
        if completed >= 100 and not missing:
            return 0
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()
    return watch(args.run_dir, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run watcher tests and verify GREEN**

Run:

```powershell
python -m pytest -q tests/test_persistent_dcf_checkpoint_watcher.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add scripts/watch_persistent_dcf_checkpoints.py `
        tests/test_persistent_dcf_checkpoint_watcher.py
git commit -m "feat: preserve persistent DCF milestone checkpoints"
```

### Task 4: Regression Verification and Source Publication

**Files:**
- Create: `docs/evidence/persistent-gradient-dcf-preflight-20260827.json`

- [ ] **Step 1: Run focused and regression tests**

Run:

```powershell
python -m pytest -q `
  tests/test_persistent_dcf.py `
  tests/test_persistent_dcf_checkpoint_watcher.py `
  tests/test_train_dcf_fdr.py `
  tests/test_transient_dcf.py `
  tests/test_transient_dcf_export.py `
  tests/test_fdr_head.py `
  tests/test_rtdetr_fdr.py `
  tests/test_fdr_protocol.py
```

Expected: zero failures.

- [ ] **Step 2: Run repository integrity checks**

Run:

```powershell
git diff --check
git status --short
python -m compileall -q src scripts
```

Expected: no whitespace errors, only the planned preflight evidence remains
uncommitted before Step 4, and compileall exits zero.

- [ ] **Step 3: Record a sanitized preflight evidence file**

Write `docs/evidence/persistent-gradient-dcf-preflight-20260827.json` with this
schema and real verified values:

```json
{
  "design_commit": "29da6e69",
  "formal_epochs": 100,
  "initial_state_sha256": "51aab2eb3fb7d123501c69c7b8dc90ff3ea0b9344a108edeef2c7d6dcdbb742d",
  "method": "persistent_gradient_dcf_fdr",
  "scale_all_epochs": 1.0,
  "trainable_all_epochs": true,
  "transient_controls_present": false,
  "verification": {
    "focused_tests_passed": true,
    "config_equivalent_to_transient_epochs_1_66": true,
    "private_gradient_partition_verified": true
  }
}
```

Do not record host passwords, GitHub tokens, or secret paths.

- [ ] **Step 4: Commit verification evidence and push source branch**

```powershell
git add docs/evidence/persistent-gradient-dcf-preflight-20260827.json
git commit -m "docs: record persistent DCF preflight"
git push origin codex/ap-fdr-integrated-redesign
git ls-remote origin refs/heads/codex/ap-fdr-integrated-redesign
```

Expected: remote SHA equals local `HEAD`.

### Task 5: Server Deployment, Fail-Closed Preflight, and Formal Launch

**Files:**
- Deploy exact committed source into a new immutable server checkout.
- Create the run output from the exact deployed SHA as described below.

- [ ] **Step 1: Reconnect and verify server resources without mutation**

Run server checks:

```bash
date '+%F %T %Z'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
df -BG /data
pgrep -af 'train_.*dcf.*fdr' || true
sha256sum /data/uav/protocols/fdr-d97e1eb7/initial-state.pt
```

Expected: GPU idle, no trainer, at least 5 GiB free, and initial-state SHA-256
equals `51aab2eb...dbb742d`.

- [ ] **Step 2: Transfer and verify the exact source commit**

Create a Git bundle from local `HEAD`, transfer it into `/data/uav/source`, and
clone a new checkout named with the exact short SHA. Verify:

```bash
experiment_sha=$(git rev-parse HEAD)
experiment_short=${experiment_sha:0:8}
persistent_checkout=/data/uav/source/uav-detection-baselines-persistent-gradient-dcf-${experiment_short}
persistent_run_root=/data/uav/runs/persistent-gradient-dcf-fdr-${experiment_short}
git rev-parse HEAD
git status --short
```

Expected: exact local SHA and empty status. Do not modify or reuse the old
`50052b68` checkout.

- [ ] **Step 3: Run focused tests in the server environment**

```bash
/data/uav/venvs/iber-be-v1/bin/python -m pytest -q \
  tests/test_persistent_dcf.py \
  tests/test_persistent_dcf_checkpoint_watcher.py \
  tests/test_train_dcf_fdr.py -k 'persistent or persistent_gradient'
```

Expected: zero failures.

- [ ] **Step 4: Run dry-run authority generation and inspect exact semantics**

```bash
/data/uav/venvs/iber-be-v1/bin/python \
  scripts/train_persistent_gradient_dcf_fdr.py \
  --dataset-root /data/uav/datasets/VisDrone \
  --initial-state /data/uav/protocols/fdr-d97e1eb7/initial-state.pt \
  --output-root "$persistent_run_root" \
  --dry-run
```

Expected authority: Formal100, seed0, dedicated config, scale all epochs 1.0,
trainable all epochs, restart from Epoch 0, and no resume field.

- [ ] **Step 5: Launch training detached**

Use a hidden detached process and sanitized log:

```bash
setsid -f bash -c "cd '$persistent_checkout' && exec \
  /data/uav/venvs/iber-be-v1/bin/python \
  scripts/train_persistent_gradient_dcf_fdr.py \
  --dataset-root /data/uav/datasets/VisDrone \
  --initial-state /data/uav/protocols/fdr-d97e1eb7/initial-state.pt \
  --output-root '$persistent_run_root' \
  > '$persistent_run_root/train.log' 2>&1"
```

Record the detached PID after matching the exact output root.

- [ ] **Step 6: Launch the milestone watcher detached**

Wait until the run directory exists, then:

```bash
setsid -f bash -c "cd '$persistent_checkout' && exec \
  /data/uav/venvs/iber-be-v1/bin/python \
  scripts/watch_persistent_dcf_checkpoints.py \
  --run-dir '$persistent_run_root/formal-seed0-persistent-gradient-dcf-fdr-v1' \
  --poll-seconds 20 \
  > '$persistent_run_root/checkpoint-watcher.log' 2>&1"
```

Record its PID and verify PPID 1.

- [ ] **Step 7: Verify the first epoch is genuinely running**

Confirm:

- trainer process alive;
- GPU memory/utilization nonzero;
- output authority matches the exact source SHA;
- first `persistent-dcf-state.jsonl` row reports paper Epoch 1, live/EMA scale
  `1.0`, trainable true, private gradient group true;
- no `transient-dcf-schedule.jsonl` exists;
- logs contain no traceback, NaN, CUDA OOM, or resume message.

- [ ] **Step 8: Report verified launch state**

Report exact source commit, training PID, watcher PID, run directory, completed
epoch count, current mAP if available, scale/trainability audit state, GPU and
disk status. Do not claim scientific success before the Formal100 result exists.

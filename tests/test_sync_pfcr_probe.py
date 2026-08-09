from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path("scripts/sync_pfcr_probe.py")


def _module():
    assert SCRIPT.is_file()
    spec = importlib.util.spec_from_file_location("sync_pfcr_probe_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_epoch(root: Path, epoch: int) -> tuple[Path, Path]:
    checkpoints = root / "checkpoints"
    metrics = root / "metrics"
    checkpoints.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / f"epoch-{epoch:02d}.pt"
    torch.save({"format_version": 1, "epoch": epoch, "gate": {}, "optimizer": {}}, checkpoint)
    metric = metrics / f"epoch-{epoch:02d}.json"
    metric.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "rows": [
                    {"epoch": epoch, "split": "dev", "slots": slots}
                    for slots in (15, 30, 60)
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, metric


def test_discovery_requires_complete_contiguous_pairs(tmp_path: Path) -> None:
    module = _module()
    _write_epoch(tmp_path, 1)
    _write_epoch(tmp_path, 2)
    (tmp_path / "metrics" / "epoch-02.json").unlink()
    assert [item.epoch for item in module.discover_complete_epochs(tmp_path)] == [1]
    _write_epoch(tmp_path, 2)
    assert [item.epoch for item in module.discover_complete_epochs(tmp_path)] == [1, 2]


def test_epoch_validation_binds_checkpoint_metrics_and_hashes(tmp_path: Path) -> None:
    module = _module()
    checkpoint, metric = _write_epoch(tmp_path, 1)
    item = module.validate_epoch_pair(checkpoint, metric)
    assert item.epoch == 1
    assert len(item.checkpoint_sha256) == 64
    assert len(item.metrics_sha256) == 64
    payload = json.loads(metric.read_text("utf-8"))
    payload["rows"][0]["slots"] = 29
    metric.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="budget"):
        module.validate_epoch_pair(checkpoint, metric)


def test_status_is_atomic_and_never_contains_token(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "status.json"
    module.write_status(path, {"published_epochs": [1], "state": "running"})
    assert json.loads(path.read_text("utf-8"))["published_epochs"] == [1]
    assert "token" not in path.read_text("utf-8").lower()


def test_cli_freezes_publication_identity() -> None:
    module = _module()
    args = module.parse_args(
        [
            "--run-root", "run", "--report-root", "report",
            "--token-file", "secret", "--repo", "owner/repo",
            "--tag", "pfcr-live", "--status-file", "status.json",
        ]
    )
    assert args.interval == 30
    assert args.branch == "codex/fdr-yaml-module"
    with pytest.raises(SystemExit):
        module.parse_args([
            "--run-root", "run", "--report-root", "report",
            "--token-file", "secret", "--repo", "owner/repo",
            "--tag", "pfcr-live", "--status-file", "status.json",
            "--retain", "3",
        ])

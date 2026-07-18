from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from scripts.supervise_vsf_rmr import (
    EXIT_PLANNED_RESTART,
    build_child_command,
    build_child_environment,
    classify_child_exit,
    fixed_protocol_stop_code,
    parse_batch_levels,
    parse_device_indices,
    select_resume_checkpoint,
    variant_identity,
)


@pytest.mark.parametrize(
    ("returncode", "output", "expected"),
    [
        (EXIT_PLANNED_RESTART, "checkpoint saved", "planned_restart"),
        (1, "torch.OutOfMemoryError: CUDA out of memory", "oom"),
        (1, "FloatingPointError: NONFINITE_LOSS total", "numeric_failure"),
        (0, "100 epochs completed", "success"),
        (2, "filesystem unavailable", "failure"),
    ],
)
def test_child_exit_classification(returncode: int, output: str, expected: str):
    assert classify_child_exit(returncode, output) == expected


@pytest.mark.parametrize("variant", ["baseline", "vsf-rmr"])
def test_child_command_carries_variant_batch_amp_and_resume(tmp_path: Path, variant: str):
    command = build_child_command(
        python_executable="python",
        project=tmp_path / "runs",
        run_name=f"{variant}-run",
        state_path=tmp_path / "state.json",
        variant=variant,
        batch=8,
        amp_enabled=True,
        epochs=100,
        workers=8,
        device="0,1",
        save_period=1,
        optimizer="auto",
        lr0=0.01,
        momentum=0.937,
        resume=tmp_path / "last.pt",
        fixed_protocol=True,
    )

    assert Path(command[1]).name == "train_rtdetr_vsf_rmr.py"
    assert command[command.index("--variant") + 1] == variant
    assert command[command.index("--batch") + 1] == "8"
    assert command[command.index("--amp") + 1] == "true"
    assert command[command.index("--device") + 1] == "0,1"
    assert command[command.index("--resume") + 1].endswith("last.pt")
    assert "--fixed-protocol" in command


def test_fixed_protocol_stops_instead_of_changing_batch_or_amp():
    assert fixed_protocol_stop_code("oom") == 4
    assert fixed_protocol_stop_code("numeric_failure") == 5
    assert fixed_protocol_stop_code("failure") is None


def test_variants_have_disjoint_run_and_release_identities():
    baseline = variant_identity("baseline")
    vsf = variant_identity("vsf-rmr")

    assert baseline.run_name != vsf.run_name
    assert baseline.tag != vsf.tag
    assert baseline.asset_prefix != vsf.asset_prefix
    assert "ioqc" not in baseline.tag + vsf.tag
    assert "btdse" not in baseline.tag + vsf.tag


def test_device_and_batch_parsers_validate_server_inputs():
    assert parse_device_indices("0,1,2,3,4,5,6,7") == tuple(range(8))
    assert parse_device_indices("cuda:0, cuda:2") == (0, 2)
    assert parse_batch_levels("8,10,12") == (8, 10, 12)
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_batch_levels("8,12,10")


def test_child_environment_supports_ddp_and_fragmentation_control():
    environment = build_child_environment({"PYTHONPATH": "/existing"})

    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(Path(__file__).resolve().parents[1])
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def _save_checkpoint(path: Path, epoch: int) -> None:
    torch.save(
        {"epoch": epoch, "optimizer": {"state": {}, "param_groups": []}, "ema": {"weights": torch.ones(1)}},
        path,
    )


def test_resume_selection_falls_back_from_corrupt_last_to_latest_epoch(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"corrupt")
    _save_checkpoint(weights / "epoch3.pt", 3)
    _save_checkpoint(weights / "epoch7.pt", 7)

    assert select_resume_checkpoint(tmp_path) == (weights / "epoch7.pt").resolve()

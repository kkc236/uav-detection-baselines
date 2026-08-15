from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from scripts.prepare_lpr_g_protocol import create_initial_state_artifact
from src.lpr_g_protocol import (
    build_lpr_g_initial_state,
    load_lpr_g_initial_state,
    validate_lpr_g_initial_state,
)
from src.rtdetr_lpr_g import LPRGRTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_format_v2_initial_state_loads_exact_control_and_method_keys() -> None:
    artifact = create_initial_state_artifact(seed=0, nc=10, channels=3)
    control = RTDETRDetectionModel("rtdetr-l.yaml", nc=10, ch=3, verbose=False)
    method = LPRGRTDETRDetectionModel(
        "rtdetr-l.yaml",
        nc=10,
        ch=3,
        verbose=False,
        private_seed=10_000,
    )

    load_lpr_g_initial_state(control, artifact, variant="control")
    load_lpr_g_initial_state(method, artifact, variant="lprg")

    assert artifact["format_version"] == 2
    assert artifact["seed"] == 0
    assert artifact["lpr_g_state"]
    assert all("lpr_g_refiner." in name for name in artifact["lpr_g_state"])
    for name, value in control.state_dict().items():
        assert torch.equal(value, method.state_dict()[name])


def test_initial_state_validation_rejects_private_corruption() -> None:
    torch.manual_seed(0)
    control = RTDETRDetectionModel("rtdetr-l.yaml", nc=3, ch=3, verbose=False)
    torch.manual_seed(0)
    method = LPRGRTDETRDetectionModel("rtdetr-l.yaml", nc=3, ch=3, verbose=False)
    artifact = build_lpr_g_initial_state(control.state_dict(), method.state_dict(), seed=0)
    first_private = next(iter(artifact["lpr_g_state"]))
    artifact["lpr_g_state"][first_private].add_(1)

    with pytest.raises(ValueError, match="private fingerprint"):
        validate_lpr_g_initial_state(artifact, seed=0)


def test_prepare_protocol_script_runs_as_direct_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_lpr_g_protocol.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset-root" in result.stdout
    assert "--seed" in result.stdout

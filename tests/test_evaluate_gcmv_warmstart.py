from __future__ import annotations

import pytest
import torch

from scripts.evaluate_gcmv_plec import (
    finish_tensor_moments,
    update_tensor_moments,
)
from scripts.evaluate_gcmv_warmstart import advance_gate, build_parser


def test_parser_requires_the_two_checkpoint_endpoints(tmp_path):
    args = build_parser().parse_args(
        [
            "--control-checkpoint",
            "control.pt",
            "--method-checkpoint",
            "method.pt",
            "--data",
            "visdrone.yaml",
            "--output",
            str(tmp_path / "three-state.json"),
        ]
    )

    assert args.control_checkpoint == "control.pt"
    assert args.method_checkpoint == "method.pt"


def test_streaming_tensor_moments_detect_a_nonconstant_gate():
    moments = {}
    update_tensor_moments(moments, torch.tensor([0.25, 0.5]))
    update_tensor_moments(moments, torch.tensor([0.75]))

    result = finish_tensor_moments(moments)

    assert result["count"] == 3.0
    assert result["mean"] == pytest.approx(0.5)
    assert result["std"] > 0.0
    assert result["min"] == 0.25
    assert result["max"] == 0.75


def test_advance_gate_requires_total_and_direct_tiny_benefit():
    metrics = {
        "control": {
            "mAP50-95": 0.24,
            "AP-tiny-SBR": 0.10,
            "tiny_recall": 0.20,
            "AP-medium-SBR": 0.25,
            "AP-large-SBR": 0.30,
        },
        "method_off": {
            "mAP50-95": 0.241,
            "AP-tiny-SBR": 0.101,
            "tiny_recall": 0.201,
            "AP-medium-SBR": 0.25,
            "AP-large-SBR": 0.30,
        },
        "method_on": {
            "mAP50-95": 0.245,
            "AP-tiny-SBR": 0.11,
            "tiny_recall": 0.22,
            "AP-medium-SBR": 0.251,
            "AP-large-SBR": 0.301,
        },
    }
    runtime = {
        "method_on": {
            "checkpoint": {"gcmv_gamma": 0.02},
            "gcmv_diagnostics": {
                "peg_gate": {
                    "count": 100.0,
                    "std": 0.1,
                    "min": 0.2,
                    "max": 0.8,
                }
            },
        }
    }

    checks = advance_gate(metrics, runtime)

    assert checks["advance"] is True

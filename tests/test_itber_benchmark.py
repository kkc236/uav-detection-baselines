from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import nn

from scripts.benchmark_itber import (
    BENCHMARK_PROTOCOL,
    measurement_order,
    parameter_report,
    percentage_increase,
)


class _Method(nn.Module):
    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector
        self.refiner = nn.Linear(4, 2)


def test_benchmark_protocol_is_frozen_for_fp16_batch1_cuda() -> None:
    assert BENCHMARK_PROTOCOL == {
        "device": "cuda:0",
        "input": [1, 3, 640, 640],
        "dtype": "float16",
        "warmup": 50,
        "iterations": 200,
        "synchronized_cuda": True,
        "targets_nonblocking": {
            "parameters_percent": 1.0,
            "gflops_percent": 1.0,
            "latency_percent": 3.0,
        },
    }


def test_parameter_report_attributes_exact_private_overhead() -> None:
    baseline = nn.Linear(4, 4)
    method = _Method(nn.Linear(4, 4))
    method.detector.load_state_dict(baseline.state_dict())

    report = parameter_report(baseline, method)

    assert report["method_total"] - report["baseline_total"] == report["private_total"]
    assert report["private_total"] == sum(p.numel() for p in method.refiner.parameters())
    assert report["baseline_total"] > 0


def test_percentage_increase_rejects_nonpositive_baseline() -> None:
    assert percentage_increase(101.0, 100.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        percentage_increase(1.0, 0.0)


def test_measurement_order_rotates_control_stock_and_refined() -> None:
    assert measurement_order(0) == ("control", "stock", "refined")
    assert measurement_order(1) == ("stock", "refined", "control")
    assert measurement_order(2) == ("refined", "control", "stock")
    assert measurement_order(3) == measurement_order(0)


def test_cli_has_only_authority_paths_and_no_benchmark_overrides() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_itber.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for allowed in (
        "--baseline-checkpoint",
        "--private-checkpoint",
        "--gate1-cache-manifest",
        "--stage",
        "--output",
    ):
        assert allowed in result.stdout
    for forbidden in ("--imgsz", "--warmup", "--iterations", "--device", "--half", "--batch"):
        assert forbidden not in result.stdout

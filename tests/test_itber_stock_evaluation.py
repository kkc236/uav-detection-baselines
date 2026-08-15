from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_itber_stock import build_stock_authority_report
from src.itber_evaluation import EVALUATION_CONSTANTS
from src.itber_protocol import (
    BASELINE_REFERENCE_ENVIRONMENT,
    EXECUTION_ENVIRONMENT,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
)


METRICS = {
    "map": 0.241803,
    "ap50": 0.4102,
    "ap75": 0.2501,
    "ap_tiny": 0.102,
    "ap_small": 0.220,
}


def _report(**updates):
    values = {
        "repeats": [METRICS, dict(METRICS), dict(METRICS)],
        "baseline_path": Path("/data/uav/weights/matched_baseline.pt"),
        "baseline_bytes": 66262262,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "category_sha256": EXPECTED_CATEGORY_SHA256,
        "execution_environment": dict(EXECUTION_ENVIRONMENT),
        "source_commit": "A" * 40,
    }
    values.update(updates)
    return build_stock_authority_report(**values)


def test_stock_authority_records_exact_current_environment_identity() -> None:
    report = _report()

    assert report["status"] == "passed_with_runtime_amendment"
    assert report["stock"] == METRICS
    assert report["repeat_count"] == 3
    assert report["repeat_exact"] is True
    assert report["evaluation_constants"] == EVALUATION_CONSTANTS
    assert report["baseline_reference_environment"] == BASELINE_REFERENCE_ENVIRONMENT
    assert report["execution_environment"] == EXECUTION_ENVIRONMENT
    assert report["runtime_amendment"] == RUNTIME_AMENDMENT
    assert report["runtime_amendment_sha256"] == RUNTIME_AMENDMENT_SHA256


def test_stock_authority_rejects_repeat_or_environment_drift() -> None:
    changed = dict(METRICS, map=0.241804)
    with pytest.raises(ValueError, match="repeat 3"):
        _report(repeats=[METRICS, dict(METRICS), changed])
    drifted = dict(EXECUTION_ENVIRONMENT, driver="550.142")
    with pytest.raises(ValueError, match="execution environment"):
        _report(execution_environment=drifted)


def test_stock_authority_cli_exposes_only_artifact_paths() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_itber_stock.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for allowed in ("--baseline-checkpoint", "--dataset-root", "--output"):
        assert allowed in result.stdout
    for forbidden in ("--seed", "--batch", "--workers", "--imgsz", "--driver"):
        assert forbidden not in result.stdout

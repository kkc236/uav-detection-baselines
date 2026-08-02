from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_iber_stock import (
    BASELINE_REFERENCE_ENVIRONMENT,
    EVALUATION_CONSTANTS,
    build_stock_authority_report,
)
from src.iber_protocol import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    execution_environment,
)


EXPECTED_CATEGORY_SHA256 = (
    "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
)
METRICS = {
    "map": 0.24173427880487694,
    "ap50": 0.41453658664256154,
    "ap75": 0.2501,
    "ap_tiny": 0.102,
    "ap_small": 0.220,
}


def _report(**updates):
    values = {
        "repeats": [METRICS, dict(METRICS), dict(METRICS)],
        "baseline_path": Path("/data/uav/weights/matched-baseline.pt"),
        "baseline_bytes": 66262262,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "category_sha256": EXPECTED_CATEGORY_SHA256,
        "execution_environment": execution_environment(),
        "source_commit": "a" * 40,
    }
    values.update(updates)
    return build_stock_authority_report(**values)


def test_stock_evaluation_constants_are_frozen() -> None:
    assert EVALUATION_CONSTANTS == {
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "conf": 0.001,
        "max_det": 300,
        "nms": False,
        "half": False,
        "repeats": 3,
    }


def test_stock_authority_records_three_exact_runtime_amended_repeats() -> None:
    report = _report()
    assert report["design_version"] == "iber-be-v1.0"
    assert report["status"] == "passed_with_runtime_amendment"
    assert report["stock"] == METRICS
    assert report["repeat_count"] == 3
    assert report["repeat_exact"] is True
    assert report["evaluation_constants"] == EVALUATION_CONSTANTS
    assert report["baseline_reference_environment"] == BASELINE_REFERENCE_ENVIRONMENT
    assert report["execution_environment"] == execution_environment()
    assert report["runtime_amendment"] == dict(RUNTIME_AMENDMENT)
    assert report["runtime_amendment_sha256"] == RUNTIME_AMENDMENT_SHA256


def test_stock_authority_rejects_repeat_artifact_or_environment_drift() -> None:
    changed = dict(METRICS, map=METRICS["map"] + 1e-15)
    with pytest.raises(ValueError, match="repeat"):
        _report(repeats=[METRICS, dict(METRICS), changed])
    with pytest.raises(ValueError, match="exactly 3"):
        _report(repeats=[METRICS, dict(METRICS)])
    with pytest.raises(ValueError, match="artifact"):
        _report(dataset_sha256="0" * 64)
    with pytest.raises(ValueError, match="environment"):
        _report(execution_environment={**execution_environment(), "driver": "550.142"})


def test_stock_authority_cli_exposes_only_artifact_paths() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_iber_stock.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for allowed in ("--baseline-checkpoint", "--dataset-root", "--output"):
        assert allowed in result.stdout
    for forbidden in (
        "--seed",
        "--batch",
        "--workers",
        "--imgsz",
        "--driver",
        "--repeats",
    ):
        assert forbidden not in result.stdout


def test_stock_source_refuses_old_itber_identity() -> None:
    source = Path("scripts/evaluate_iber_stock.py").read_text(encoding="utf-8")
    assert "FrozenIBERAdapter" in source
    assert "rtdetr_itber" not in source
    assert "itber-v1.1" not in source

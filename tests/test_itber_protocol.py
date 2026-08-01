from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from src.itber_protocol import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ENVIRONMENT,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SUBSET_SHA256,
    ProtocolViolation,
    assert_detector_frozen,
    module_state_sha256,
    validate_authorities,
    write_immutable_report,
)


def _authority() -> dict:
    return {
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "category_sha256": EXPECTED_CATEGORY_SHA256,
        "source_sha256": dict(EXPECTED_SOURCE_SHA256),
        "environment": dict(EXPECTED_ENVIRONMENT),
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("baseline_sha256", "BAD"),
        ("dataset_sha256", "BAD"),
        ("subset_sha256", "BAD"),
        ("category_sha256", "BAD"),
    ],
)
def test_authority_rejects_changed_artifact(field: str, replacement: str) -> None:
    authority = _authority()
    authority[field] = replacement
    with pytest.raises(ProtocolViolation, match=field):
        validate_authorities(**authority)


def test_authority_rejects_source_package_and_gpu_drift() -> None:
    authority = _authority()
    authority["source_sha256"]["head.py"] = "BAD"
    authority["environment"]["torch"] = "2.6.0"
    authority["environment"]["gpu"] = "NVIDIA A100"

    with pytest.raises(ProtocolViolation) as captured:
        validate_authorities(**authority)

    assert set(captured.value.violations) >= {
        "source_sha256.head.py",
        "environment.torch",
        "environment.gpu",
    }


def test_frozen_detector_requires_eval_no_grad_and_stable_state() -> None:
    detector = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).eval()
    detector.requires_grad_(False)

    assert_detector_frozen(detector)
    first = module_state_sha256(detector)
    detector[0].weight.add_(1)
    second = module_state_sha256(detector)
    assert first != second

    detector[0].weight.requires_grad_(True)
    with pytest.raises(ProtocolViolation, match="requires_grad"):
        assert_detector_frozen(detector)


def test_immutable_report_is_exclusive_and_readable(tmp_path) -> None:
    path = tmp_path / "immutable" / "gate0.json"
    write_immutable_report(path, {"status": "passed", "value": 1})

    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "passed"
    with pytest.raises(FileExistsError):
        write_immutable_report(path, {"status": "changed"})


def test_exact_authority_is_accepted() -> None:
    report = validate_authorities(**_authority())
    assert report["status"] == "passed"


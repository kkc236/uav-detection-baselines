from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from src.itber_protocol import (
    BASELINE_TRAINING_CONTRACT,
    BASELINE_REFERENCE_ENVIRONMENT,
    EXECUTION_ENVIRONMENT,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SUBSET_SHA256,
    ProtocolViolation,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    assert_detector_frozen,
    module_state_sha256,
    validate_authorities,
    write_immutable_report,
)


def test_baseline_training_contract_is_exact_and_seed0_only() -> None:
    assert BASELINE_TRAINING_CONTRACT == {
        "base_model": "Ultralytics RT-DETR-L",
        "ultralytics": "8.4.90",
        "dataset": "VisDrone train/val",
        "train_images": 6471,
        "val_images": 548,
        "class_count": 10,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "screen_subset_images": 647,
        "screen_subset_sha256": EXPECTED_SUBSET_SHA256,
        "pretrained": False,
        "formal_epochs": 100,
        "seeds": [0],
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "amp": True,
        "amp_scale": 128.0,
        "deterministic": True,
        "cache": False,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 64,
        "cos_lr": False,
        "query_count": 300,
        "max_det": 300,
        "nms": False,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    }


def _authority() -> dict:
    return {
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "category_sha256": EXPECTED_CATEGORY_SHA256,
        "source_sha256": dict(EXPECTED_SOURCE_SHA256),
        "environment": dict(EXECUTION_ENVIRONMENT),
    }


def test_runtime_driver_amendment_preserves_baseline_reference() -> None:
    assert BASELINE_REFERENCE_ENVIRONMENT == {
        "gpu": "NVIDIA GeForce RTX 4090",
        "reported_memory": "24GB",
        "driver": "550.142",
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda": "12.1",
        "ultralytics": "8.4.90",
    }
    assert EXECUTION_ENVIRONMENT == {
        "gpu": "NVIDIA GeForce RTX 4090",
        "reported_memory_mib": 49140,
        "driver": "570.133.07",
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda": "12.1",
        "ultralytics": "8.4.90",
    }
    assert RUNTIME_AMENDMENT == {
        "amendment_id": "itber-v1.1-runtime-driver-2026-08-01",
        "approved_on": "2026-08-01",
        "baseline_driver": "550.142",
        "execution_driver": "570.133.07",
        "allowed_differences": ["driver", "reported_memory_mib"],
        "comparison": "same-checkpoint-stock-vs-refined",
    }
    assert len(RUNTIME_AMENDMENT_SHA256) == 64


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
    assert report["status"] == "passed_with_runtime_amendment"
    assert report["baseline_reference_environment"] == BASELINE_REFERENCE_ENVIRONMENT
    assert report["execution_environment"] == EXECUTION_ENVIRONMENT
    assert report["runtime_amendment"] == RUNTIME_AMENDMENT
    assert report["runtime_amendment_sha256"] == RUNTIME_AMENDMENT_SHA256


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("driver", "550.142"),
        ("reported_memory_mib", 24564),
        ("torch", "2.6.0"),
    ],
)
def test_authority_rejects_unapproved_execution_environment_drift(
    field: str, replacement: object
) -> None:
    authority = _authority()
    authority["environment"][field] = replacement
    with pytest.raises(ProtocolViolation, match=f"environment.{field}"):
        validate_authorities(**authority)

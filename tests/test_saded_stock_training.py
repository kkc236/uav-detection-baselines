from __future__ import annotations

from copy import deepcopy

from scripts.train_rtdetr_saded_stock import validate_runtime_summary
from src.tascv_protocol import FROZEN_OPTIMIZER_OBSERVATION


def _valid_summary() -> dict:
    return {
        "schema_version": "saded-stock-training-summary/v1",
        "stage": "FORMAL_100",
        "arm": "stock_control",
        "seed": 0,
        "completed_epochs": 100,
        "batch": 8,
        "workers": 8,
        "successful_batches": 80_900,
        "optimizer_attempts": 10_556,
        "expected_successful_batches": 80_900,
        "expected_optimizer_attempts": 10_556,
        "observed_tensor_batch_sizes": [7, 8],
        "batch_canaries": [
            {"epoch": 0, "batch": 1, "sha256": "a" * 64},
            {"epoch": 0, "batch": 2, "sha256": "b" * 64},
            {"epoch": 1, "batch": 810, "sha256": "c" * 64},
        ],
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "loader": {
            "trainer_batch_size": 8,
            "per_rank_batch_size": 8,
            "loader_batch_size": 8,
            "loader_num_workers": 8,
        },
        "optimizer": FROZEN_OPTIMIZER_OBSERVATION,
        "local_forward_calls": 0,
        "local_forward_call_histogram": {"1": 0, "2": 0},
        "local_bn_preserved_batches": 0,
        "auxiliary_non_tiny_pair_count": 0,
        "internal_validation_bypass_count": 1,
        "test_loader_is_none": True,
        "checkpoint": {
            "kind": "last.pt",
            "path": "/home/ubuntu/run/weights/last.pt",
            "sha256": "d" * 64,
            "loadable": True,
        },
        "source_unchanged": True,
        "data_unchanged": True,
        "initial_state_unchanged": True,
    }


def test_valid_runtime_summary_has_no_failures() -> None:
    assert validate_runtime_summary(_valid_summary()) == []


def test_runtime_summary_rejects_incomplete_or_drifted_training() -> None:
    summary = _valid_summary()
    summary["successful_batches"] -= 1
    summary["optimizer_attempts"] -= 1
    summary["completed_epochs"] = 99
    failures = validate_runtime_summary(summary)
    assert "completed epoch count drift" in failures
    assert "successful batch count drift" in failures
    assert "optimizer attempt count drift" in failures


def test_runtime_summary_rejects_amp_optimizer_and_eval_drift() -> None:
    summary = _valid_summary()
    summary["amp_scale_max"] = 256.0
    summary["optimizer"] = deepcopy(FROZEN_OPTIMIZER_OBSERVATION)
    summary["optimizer"]["class"] = "SGD"
    summary["test_loader_is_none"] = False
    summary["internal_validation_bypass_count"] = 2
    failures = validate_runtime_summary(summary)
    assert "AMP scale drift" in failures
    assert "optimizer observation drift" in failures
    assert "test loader was constructed" in failures
    assert "internal validation count drift" in failures


def test_runtime_summary_rejects_batch_canary_and_checkpoint_drift() -> None:
    summary = _valid_summary()
    summary["observed_tensor_batch_sizes"] = [8]
    summary["batch_canaries"][2]["batch"] = 809
    summary["checkpoint"]["loadable"] = False
    failures = validate_runtime_summary(summary)
    assert "observed tensor batch-size drift" in failures
    assert "batch canary position drift" in failures
    assert "last checkpoint is invalid" in failures

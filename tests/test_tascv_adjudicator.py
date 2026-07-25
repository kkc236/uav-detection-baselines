from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.tascv_adjudicator import (
    adjudicate_mechanism,
    adjudicate_preflight,
)
from src.tascv_protocol import (
    FROZEN_OPTIMIZER_OBSERVATION,
    sha256_file,
)


@pytest.fixture
def trusted_authority(monkeypatch):
    monkeypatch.setattr(
        "src.tascv_adjudicator._authority_failures",
        lambda summary: [],
    )


def _summary(tmp_path: Path, arm: str) -> dict:
    checkpoint = tmp_path / f"{arm}.pt"
    checkpoint.write_bytes(f"checkpoint-{arm}".encode())
    local = arm == "tascv"
    return {
        "schema_version": "tascv-training-summary/v1",
        "stage": "PREFLIGHT_1",
        "arm": arm,
        "protocol_manifest_sha256": "manifest",
        "protocol_source_commit": "commit",
        "source_repo_bundle_sha256": "repo",
        "source_upstream_bundle_sha256": "upstream",
        "approved_tascv_parent": {"commit": "parent"},
        "r0_evaluation_anchor_sha256": "r0",
        "control_slot": {"resolution": "RUN_FRESH"},
        "initial_state_sha256": "initial",
        "initial_state_common_fingerprint": "fingerprint",
        "data_sha256": "data",
        "subset_binding": {
            "count": 647,
            "semantic_sha256": "semantic",
            "file_sha256": "file",
        },
        "seed": 0,
        "batch": 8,
        "observed_tensor_batch_sizes": [8],
        "batch_canaries": [
            {"epoch": 0, "batch": 1, "sha256": "c" * 64}
        ],
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": 1,
        "optimizer_attempts": 1,
        "expected_successful_batches": 1,
        "expected_optimizer_attempts": 1,
        "workers": 8,
        "loader": {
            "trainer_batch_size": 8,
            "per_rank_batch_size": 8,
            "loader_batch_size": 8,
            "loader_num_workers": 8,
        },
        "optimizer": FROZEN_OPTIMIZER_OBSERVATION,
        "local_forward_calls": 2 if local else 0,
        "local_forward_call_histogram": (
            {"1": 0, "2": 1}
            if local
            else {"1": 0, "2": 0}
        ),
        "local_bn_preserved_batches": 1 if local else 0,
        "test_loader_is_none": True,
        "auxiliary_non_tiny_pair_count": 0,
        "internal_validation_bypass_count": 1,
        "checkpoint": {
            "kind": "last.pt",
            "path": checkpoint.as_posix(),
            "sha256": sha256_file(checkpoint),
        },
    }


def test_preflight_go_requires_exact_paired_runtime_and_rehashed_checkpoints(
    tmp_path: Path,
    trusted_authority,
) -> None:
    decision = adjudicate_preflight(
        {
            "control": _summary(tmp_path, "control"),
            "tascv": _summary(tmp_path, "tascv"),
        }
    )

    assert decision["decision"] == "TASCV_PREFLIGHT_GO"
    assert decision["failures"] == []


def test_preflight_runtime_drift_is_invalid_not_scientific_stop(
    tmp_path: Path,
    trusted_authority,
) -> None:
    summaries = {
        "control": _summary(tmp_path, "control"),
        "tascv": _summary(tmp_path, "tascv"),
    }
    summaries["tascv"]["amp_scale_min"] = 64.0

    decision = adjudicate_preflight(summaries)

    assert decision["decision"] == "INVALID"
    assert decision["failures"]


def test_preflight_rehashes_checkpoint_and_rejects_pairing_drift(
    tmp_path: Path,
    trusted_authority,
) -> None:
    summaries = {
        "control": _summary(tmp_path, "control"),
        "tascv": _summary(tmp_path, "tascv"),
    }
    tampered = copy.deepcopy(summaries)
    Path(tampered["control"]["checkpoint"]["path"]).write_bytes(b"changed")
    assert adjudicate_preflight(tampered)["decision"] == "INVALID"

    summaries = {
        "control": _summary(tmp_path, "control2").copy(),
        "tascv": _summary(tmp_path, "tascv").copy(),
    }
    summaries["control"]["arm"] = "control"
    summaries["control"]["protocol_manifest_sha256"] = "other"
    assert adjudicate_preflight(summaries)["decision"] == "INVALID"


def _mechanism_summary(
    tmp_path: Path,
    *,
    advantage: float = 1.0,
) -> dict:
    summary = _summary(tmp_path, "tascv")
    summary.update(
        {
            "stage": "TINY_MECHANISM_500",
            "seed": 1,
            "observed_tensor_batch_sizes": [7, 8],
            "successful_batches": 500,
            "optimizer_attempts": 106,
            "expected_successful_batches": 500,
            "expected_optimizer_attempts": 106,
            "local_forward_calls": 500,
            "local_forward_call_histogram": {"1": 500, "2": 0},
            "local_bn_preserved_batches": 500,
            "mechanism_records": {
                "path": "/sealed/records.jsonl",
                "sha256": "d" * 64,
                "count": 500,
            },
            "mechanism_summary": {
                "all": {
                    "batches": 500,
                    "matched_pairs": 500,
                    "tiny_pairs": 500,
                    "excluded_non_tiny_pairs": 0,
                    "auxiliary_non_tiny_pairs": 0,
                    "tiny_batches_with_pairs": 500,
                    "tiny_teacher_advantage_mean": advantage,
                    "tiny_teacher_win_rate": (
                        1.0 if advantage > 0 else 0.0
                    ),
                },
                "tail": {
                    "batches": 100,
                    "matched_pairs": 100,
                    "tiny_pairs": 100,
                    "excluded_non_tiny_pairs": 0,
                    "auxiliary_non_tiny_pairs": 0,
                    "tiny_batches_with_pairs": 100,
                    "tiny_teacher_advantage_mean": advantage,
                    "tiny_teacher_win_rate": (
                        1.0 if advantage > 0 else 0.0
                    ),
                },
                "tail_window": [401, 500],
            },
            "batch_canaries": [
                {"epoch": 0, "batch": 1, "sha256": "a" * 64},
                {"epoch": 0, "batch": 2, "sha256": "b" * 64},
                {"epoch": 1, "batch": 82, "sha256": "c" * 64},
            ],
        }
    )
    return summary


def _mechanism_records(*, advantage: float = 1.0) -> list[dict]:
    return [
        {
            "batch": index,
            "matched_pairs": 1,
            "auxiliary_tiny_pairs": 1,
            "excluded_non_tiny_pairs": 0,
            "auxiliary_non_tiny_pairs": 0,
            "tiny_teacher_advantage_sum": advantage,
            "tiny_teacher_win_count": int(advantage > 0),
        }
        for index in range(1, 501)
    ]


def test_mechanism_is_rebuilt_from_ordered_raw_records(
    tmp_path: Path,
    trusted_authority,
) -> None:
    decision = adjudicate_mechanism(
        _mechanism_summary(tmp_path),
        _mechanism_records(),
    )
    assert decision["decision"] == "TASCV_MECHANISM_GO"
    assert decision["tail"]["tiny_pairs"] == 100
    assert decision["tail_window"] == [401, 500]


def test_mechanism_complete_scientific_failure_is_stop_not_invalid(
    tmp_path: Path,
    trusted_authority,
) -> None:
    summary = _mechanism_summary(tmp_path, advantage=-1.0)
    decision = adjudicate_mechanism(
        summary,
        _mechanism_records(advantage=-1.0),
    )
    assert decision["decision"] == "TASCV_STOP"
    assert decision["failures"]


def test_mechanism_aggregate_raw_mismatch_is_invalid(
    tmp_path: Path,
    trusted_authority,
) -> None:
    summary = _mechanism_summary(tmp_path)
    decision = adjudicate_mechanism(
        summary,
        _mechanism_records(advantage=-1.0),
    )
    assert decision["decision"] == "INVALID"
    assert decision["failures"] == [
        "evidence:mechanism_summary_drift"
    ]


def test_mechanism_non_tiny_or_runtime_drift_is_invalid(
    tmp_path: Path,
    trusted_authority,
) -> None:
    records = _mechanism_records()
    records[450]["auxiliary_non_tiny_pairs"] = 1
    assert (
        adjudicate_mechanism(
            _mechanism_summary(tmp_path),
            records,
        )["decision"]
        == "INVALID"
    )


def test_fake_protocol_literals_cannot_receive_go(tmp_path: Path) -> None:
    decision = adjudicate_preflight(
        {
            "control": _summary(tmp_path, "control"),
            "tascv": _summary(tmp_path, "tascv"),
        }
    )
    assert decision["decision"] == "INVALID"
    assert any(
        failure.startswith("authority:")
        for failure in decision["failures"]
    )

    summary = _mechanism_summary(tmp_path)
    summary["optimizer_attempts"] = 105
    assert (
        adjudicate_mechanism(
            summary,
            _mechanism_records(),
        )["decision"]
        == "INVALID"
    )

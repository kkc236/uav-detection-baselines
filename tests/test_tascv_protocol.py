from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tascv_protocol import (
    APPROVED_TASCV_PARENT,
    CONTROL_SLOTS,
    FROZEN_OPTIMIZER_OBSERVATION,
    FROZEN_STAGE_CONTRACT,
    FROZEN_TRAINING_CONTRACT,
    R0_EVALUATION_ANCHOR_SHA256,
    R0_EVALUATION_ANCHOR_SHA256,
    R0_ROUTE_ANCHOR_SHA256,
    REPO_SOURCE_FILES,
    resolve_control_allowlist,
    sha256_file,
    validate_r0_authority,
    validate_runtime_manifest,
)


ROUTE_ANCHOR = {
    "input_manifest_sha256": "aa85a80d2f43bc0a72d6a083657aa2fe539746bb79f8cabbef71516dc014cbff",
    "predictions_sha256": "4c8e4998f0cbdbbc5963fecbf05ac4dc26d56db6b95d71a076fd129a66aa740e",
    "route_checksums_sha256": "6a5a4430dc53d4b196364ea5022ef88fcb3b5d165053db808db54689f7bf74fe",
    "route_manifest_sha256": "4256facf8b59bd92beb162a15ff025ce993499f1f42d5bc30043abe8698c645f",
    "schema_version": "sbr-saded-route-anchor/v1",
}
EVALUATION_ANCHOR = {
    "decision": "R0_GO",
    "evaluation_checksums_sha256": "7a9598773b7c4b32ffe0d1658f785d4131146438015cbd3a32a2c946cb1efc69",
    "evaluation_manifest_sha256": "793a6a541100a81a3133d3ebc9d8937446b266d23ce6ae61b3a315ad94e3dff6",
    "route_anchor_sha256": "e3c3a391496774412c60c921bf2db11cdbc2de908a562e5ad173123f36fb077c",
    "schema_version": "sbr-saded-r0-anchor/v1",
}
def _sealed(path: Path, value: dict) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_authoritative_r0_go_closure_is_accepted(tmp_path: Path) -> None:
    route = _sealed(tmp_path / "route.json", ROUTE_ANCHOR)
    evaluation = _sealed(
        tmp_path / "evaluation.json",
        EVALUATION_ANCHOR,
    )
    record = validate_r0_authority(
        route_anchor=route,
        evaluation_anchor=evaluation,
    )

    assert record["decision"] == "R0_GO"
    assert record["route_anchor_sha256"] == R0_ROUTE_ANCHOR_SHA256
    assert (
        record["evaluation_anchor_sha256"]
        == R0_EVALUATION_ANCHOR_SHA256
    )


@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    (
        ("evaluation", "decision", "STOP"),
        ("evaluation", "route_anchor_sha256", "0" * 64),
        ("route", "schema_version", "forged"),
    ),
)
def test_r0_authority_fails_closed_on_any_drift(
    tmp_path: Path,
    artifact: str,
    field: str,
    value,
) -> None:
    records = {
        "route": dict(ROUTE_ANCHOR),
        "evaluation": dict(EVALUATION_ANCHOR),
    }
    records[artifact][field] = value
    paths = {
        name: _sealed(tmp_path / f"{name}.json", record)
        for name, record in records.items()
    }

    with pytest.raises(ValueError, match="R0"):
        validate_r0_authority(
            route_anchor=paths["route"],
            evaluation_anchor=paths["evaluation"],
        )


def test_tascv_source_closure_excludes_stopped_trainer_and_adjudicator() -> None:
    assert "src/rtdetr_tascv.py" in REPO_SOURCE_FILES
    assert "src/tascv.py" in REPO_SOURCE_FILES
    assert "src/rtdetr_ascv_loc.py" not in REPO_SOURCE_FILES
    assert "src/ascv_loc_adjudicator.py" not in REPO_SOURCE_FILES


def test_runtime_manifest_rejects_nonfinal_identity_before_live_checks(
    tmp_path: Path,
) -> None:
    manifest = _sealed(
        tmp_path / "protocol_manifest.json",
        {
            "schema_version": "tascv-saded-protocol/v1",
            "protocol_id": "not-final",
            "runtime_source": {"commit": "a" * 40},
        },
    )
    with pytest.raises(ValueError, match="identity"):
        validate_runtime_manifest(manifest)


def test_runtime_manifest_rejects_test_dev_path_before_read(
    tmp_path: Path,
) -> None:
    forbidden = tmp_path / "test-dev" / "protocol_manifest.json"
    with pytest.raises(ValueError, match="test-dev"):
        validate_runtime_manifest(forbidden)


def test_runtime_manifest_allows_only_the_sealed_forbidden_data_declaration(
    tmp_path: Path,
) -> None:
    protocol_dir = tmp_path / "final-tascv-aaaaaaaa"
    protocol_dir.mkdir()
    manifest = _sealed(
        protocol_dir / "protocol_manifest.json",
        {
            "schema_version": "tascv-saded-protocol/v1",
            "protocol_id": "final-tascv-aaaaaaaa",
            "runtime_source": {"commit": "a" * 40},
            "environment": {
                "python": "3.10.12",
                "torch": "2.5.1+cu121",
                "ultralytics": "8.4.90",
                "cuda": "12.1",
                "gpu": "NVIDIA GeForce RTX 4090",
            },
            "forbidden_data": ["test-dev", "test_dev"],
        },
    )
    with pytest.raises(ValueError, match="environment|approved parent"):
        validate_runtime_manifest(manifest)


def test_runtime_manifest_still_rejects_test_dev_in_any_artifact_field(
    tmp_path: Path,
) -> None:
    protocol_dir = tmp_path / "final-tascv-aaaaaaaa"
    protocol_dir.mkdir()
    manifest = _sealed(
        protocol_dir / "protocol_manifest.json",
        {
            "schema_version": "tascv-saded-protocol/v1",
            "protocol_id": "final-tascv-aaaaaaaa",
            "runtime_source": {"commit": "a" * 40},
            "forbidden_data": ["test-dev", "test_dev"],
            "artifact": {"path": "/sealed/test-dev/predictions.json"},
        },
    )
    with pytest.raises(ValueError, match="test-dev"):
        validate_runtime_manifest(manifest)


def test_runtime_manifest_rejects_nested_forbidden_data_key(
    tmp_path: Path,
) -> None:
    protocol_dir = tmp_path / "final-tascv-aaaaaaaa"
    protocol_dir.mkdir()
    manifest = _sealed(
        protocol_dir / "protocol_manifest.json",
        {
            "schema_version": "tascv-saded-protocol/v1",
            "protocol_id": "final-tascv-aaaaaaaa",
            "runtime_source": {
                "commit": "a" * 40,
                "forbidden_data": [
                    "/sealed/test-dev/predictions.json"
                ],
            },
            "forbidden_data": ["test-dev", "test_dev"],
        },
    )
    with pytest.raises(ValueError, match="test-dev"):
        validate_runtime_manifest(manifest)


def test_approved_parent_and_stage_table_are_exactly_frozen() -> None:
    assert APPROVED_TASCV_PARENT["commit"] == (
        "c8fc52db0744177481c8e742b16871df76dd175a"
    )
    assert len(APPROVED_TASCV_PARENT["files"]) == 4
    assert FROZEN_STAGE_CONTRACT["PREFLIGHT_1"] == {
        "seeds": [0],
        "arms": ["control", "tascv"],
        "epochs": 100,
        "uses_hashed_subset": True,
        "max_train_batches": 1,
        "expected_successful_batches": 1,
        "expected_optimizer_attempts": 1,
        "allowed_observed_tensor_batch_sizes": [8],
    }
    assert FROZEN_STAGE_CONTRACT["TINY_MECHANISM_500"]["seeds"] == [1]
    assert FROZEN_STAGE_CONTRACT["TINY_MECHANISM_500"]["arms"] == [
        "tascv"
    ]
    assert FROZEN_STAGE_CONTRACT["FORMAL_100"]["epochs"] == 100
    assert len(CONTROL_SLOTS) == 7


def _requirements() -> dict:
    positions = {
        "PREFLIGHT_1": [[0, 1]],
        "SCREEN_10": [[0, 1], [0, 2], [1, 82]],
        "FORMAL_100": [[0, 1], [0, 2], [1, 810]],
    }
    return {
        "schema_version": "saded-control-requirements/v1",
        "slots": {
            slot: {
                "slot_id": slot,
                "provenance": {
                    "stage": slot.split(":")[1],
                    "seed": int(slot.split(":")[2]),
                    "model": "Ultralytics RT-DETR-L stock",
                    "runtime_source_commit": "c" * 40,
                    "repo_bundle_sha256": "r" * 64,
                    "upstream_bundle_sha256": "u" * 64,
                    "approved_tascv_parent": APPROVED_TASCV_PARENT,
                    "r0_evaluation_anchor_sha256": (
                        R0_EVALUATION_ANCHOR_SHA256
                    ),
                    "initial_state_sha256": "i" * 64,
                    "common_fingerprint": "f" * 64,
                    "dataset_sha256": "d" * 64,
                    "subset_sha256": "s" * 64,
                    "subset_binding": {
                        "count": 647,
                        "semantic_sha256": "s" * 64,
                        "file_sha256": "l" * 64,
                    },
                    "data_yaml_sha256": "y" * 64,
                    "training_contract": FROZEN_TRAINING_CONTRACT,
                    "stage_contract": FROZEN_STAGE_CONTRACT[
                        slot.split(":")[1]
                    ],
                    "batch_canary_contract": {
                        "digest_schema": FROZEN_TRAINING_CONTRACT[
                            "batch_digest_schema"
                        ],
                        "required_epoch_global_batch_positions": (
                            positions[slot.split(":")[1]]
                        ),
                    },
                    "endpoint_contract": {
                        "checkpoint_name": "last.pt",
                        "training_summary_name": (
                            "tascv_training_summary.json"
                        ),
                        "raw_predictions_binding_required": True,
                        "evaluator_binding_required": True,
                    },
                },
                "fresh_target": {
                    "project": f"/home/control/{slot}",
                    "name": "run",
                },
            }
            for slot in CONTROL_SLOTS
        },
    }


def _bound_candidate(
    tmp_path: Path,
    *,
    requirements: dict,
    slot: str,
) -> dict:
    provenance = requirements["slots"][slot]["provenance"]
    stage = provenance["stage"]
    seed = provenance["seed"]
    contract = provenance["stage_contract"]
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"stock-control")
    source_manifest = tmp_path / "source_protocol_manifest.json"
    source_manifest.write_text("{}", encoding="utf-8")
    raw = tmp_path / "raw_predictions.jsonl.gz"
    raw.write_bytes(b"sealed-raw")
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text("{}", encoding="utf-8")
    checkpoint_binding = {
        "kind": "last.pt",
        "path": checkpoint.as_posix(),
        "sha256": sha256_file(checkpoint),
    }
    canaries = [
        {
            "epoch": epoch,
            "batch": batch,
            "sha256": chr(ord("a") + index) * 64,
        }
        for index, (epoch, batch) in enumerate(
            provenance["batch_canary_contract"][
                "required_epoch_global_batch_positions"
            ]
        )
    ]
    summary_record = {
        "schema_version": "tascv-training-summary/v1",
        "stage": stage,
        "arm": "control",
        "seed": seed,
        "protocol_manifest": source_manifest.as_posix(),
        "protocol_manifest_sha256": sha256_file(source_manifest),
        "protocol_source_commit": provenance[
            "runtime_source_commit"
        ],
        "source_repo_bundle_sha256": provenance[
            "repo_bundle_sha256"
        ],
        "source_upstream_bundle_sha256": provenance[
            "upstream_bundle_sha256"
        ],
        "approved_tascv_parent": APPROVED_TASCV_PARENT,
        "r0_evaluation_anchor_sha256": (
            R0_EVALUATION_ANCHOR_SHA256
        ),
        "initial_state_sha256": provenance[
            "initial_state_sha256"
        ],
        "initial_state_common_fingerprint": provenance[
            "common_fingerprint"
        ],
        "data_sha256": provenance["data_yaml_sha256"],
        "subset_binding": provenance["subset_binding"],
        "batch": 8,
        "observed_tensor_batch_sizes": contract[
            "allowed_observed_tensor_batch_sizes"
        ],
        "batch_canaries": canaries,
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": contract[
            "expected_successful_batches"
        ],
        "optimizer_attempts": contract[
            "expected_optimizer_attempts"
        ],
        "expected_successful_batches": contract[
            "expected_successful_batches"
        ],
        "expected_optimizer_attempts": contract[
            "expected_optimizer_attempts"
        ],
        "workers": 8,
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
        "internal_validation_bypass_count": 1,
        "test_loader_is_none": True,
        "auxiliary_non_tiny_pair_count": 0,
        "checkpoint": checkpoint_binding,
    }
    summary = tmp_path / "training_summary.json"
    summary.write_text(json.dumps(summary_record), encoding="utf-8")
    return {
        "schema_version": "saded-stock-control-candidate/v1",
        "slot_id": slot,
        "provenance": provenance,
        "training_summary": {
            "path": summary.as_posix(),
            "sha256": sha256_file(summary),
        },
        "checkpoint": checkpoint_binding,
        "raw_predictions": {
            "path": raw.as_posix(),
            "sha256": sha256_file(raw),
        },
        "evaluator": {
            "path": evaluator.as_posix(),
            "sha256": sha256_file(evaluator),
        },
    }


def test_control_resolver_zero_one_and_multiple_matches(
    tmp_path: Path,
) -> None:
    requirements = _requirements()
    empty = resolve_control_allowlist(requirements, [])
    assert all(
        record["resolution"] == "RUN_FRESH"
        for record in empty["slots"].values()
    )

    slot = "B:SCREEN_10:0"
    candidate = _bound_candidate(
        tmp_path,
        requirements=requirements,
        slot=slot,
    )
    bound = resolve_control_allowlist(requirements, [candidate])
    assert bound["slots"][slot]["resolution"] == "BOUND"

    with pytest.raises(ValueError, match="multiple"):
        resolve_control_allowlist(
            requirements,
            [candidate, dict(candidate)],
        )


def test_control_resolver_rehashes_every_bound_artifact(
    tmp_path: Path,
) -> None:
    requirements = _requirements()
    slot = "B:SCREEN_10:0"
    candidate = _bound_candidate(
        tmp_path,
        requirements=requirements,
        slot=slot,
    )
    Path(candidate["raw_predictions"]["path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="bound artifact"):
        resolve_control_allowlist(requirements, [candidate])


def test_control_resolver_rejects_empty_self_reported_summary(
    tmp_path: Path,
) -> None:
    requirements = _requirements()
    slot = "B:SCREEN_10:0"
    candidate = _bound_candidate(
        tmp_path,
        requirements=requirements,
        slot=slot,
    )
    summary = Path(candidate["training_summary"]["path"])
    summary.write_text("{}", encoding="utf-8")
    candidate["training_summary"]["sha256"] = sha256_file(summary)

    with pytest.raises(ValueError, match="training summary drift"):
        resolve_control_allowlist(requirements, [candidate])


def test_control_resolver_rejects_best_checkpoint_masquerading_as_last(
    tmp_path: Path,
) -> None:
    requirements = _requirements()
    slot = "B:SCREEN_10:0"
    candidate = _bound_candidate(
        tmp_path,
        requirements=requirements,
        slot=slot,
    )
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")
    candidate["checkpoint"] = {
        "kind": "last.pt",
        "path": best.as_posix(),
        "sha256": sha256_file(best),
    }

    with pytest.raises(ValueError, match="last.pt"):
        resolve_control_allowlist(requirements, [candidate])


@pytest.mark.parametrize(
    "injection",
    (
        {"metrics": {"mAP": 0.5}},
        {"decision": "GO"},
        {"checkpoint": {"path": "/tmp/test-dev/last.pt"}},
        {"checkpoint": {"path": "/tmp/results/last.pt"}},
    ),
)
def test_control_resolver_rejects_performance_or_forbidden_data(
    injection,
) -> None:
    requirements = _requirements()
    slot = "B:SCREEN_10:0"
    candidate = {
        "schema_version": "saded-stock-control-candidate/v1",
        "slot_id": slot,
        "provenance": {"binding": slot},
        "checkpoint": {
            "kind": "last.pt",
            "path": "/home/control.pt",
            "sha256": "a" * 64,
        },
        **injection,
    }
    with pytest.raises(ValueError, match="forbidden"):
        resolve_control_allowlist(requirements, [candidate])

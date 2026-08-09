from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.fdr_protocol import FDR_PROTOCOL, FDR_PROTOCOL_SHA256
from src.bpdd_protocol import (
    BPDD_PROTOCOL,
    BPDD_PROTOCOL_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_resume_authority,
    write_create_only_manifest,
)


EXPECTED_FDR_PROTOCOL_SHA256 = (
    "2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302"
)
EXPECTED_FDR_SOURCE_COMMIT = "d97e1eb7f98414752a1c1f38287697db3f2a0679"
EXPECTED_INITIAL_STATE_SHA256 = (
    "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"
)
EXPECTED_DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)


def _source() -> dict[str, str]:
    return {
        "git_commit": "a" * 40,
        "tree_sha256": "A" * 64,
    }


def test_bpdd_protocol_is_independent_and_binds_the_exact_fdr_authority() -> None:
    assert FDR_PROTOCOL_SHA256 == EXPECTED_FDR_PROTOCOL_SHA256
    assert BPDD_PROTOCOL is not FDR_PROTOCOL
    assert BPDD_PROTOCOL["fdr_authority"] == {
        "protocol_sha256": EXPECTED_FDR_PROTOCOL_SHA256,
        "source_commit": EXPECTED_FDR_SOURCE_COMMIT,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
    }
    assert BPDD_PROTOCOL["dataset"] == FDR_PROTOCOL["dataset"]
    assert BPDD_PROTOCOL["dataset"]["sha256"] == EXPECTED_DATASET_SHA256
    assert BPDD_PROTOCOL["training"] == FDR_PROTOCOL["training"]
    assert BPDD_PROTOCOL["augmentation"] == FDR_PROTOCOL["augmentation"]
    assert BPDD_PROTOCOL["environment"] == FDR_PROTOCOL["environment"]
    assert BPDD_PROTOCOL_SHA256 == public_state_sha256(BPDD_PROTOCOL)
    assert BPDD_PROTOCOL_SHA256 != FDR_PROTOCOL_SHA256


def test_bpdd_candidate_and_two_arm_yaml_contract_are_frozen() -> None:
    assert BPDD_PROTOCOL["bpdd"] == {
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1e-6,
    }
    assert BPDD_PROTOCOL["arms"] == {
        "fdr": {
            "model_yaml": "configs/rtdetr-l-fdr.yaml",
            "bpdd_enabled": False,
        },
        "fdr_bpdd": {
            "model_yaml": "configs/rtdetr-l-fdr-bpdd.yaml",
            "bpdd_enabled": True,
        },
    }
    assert BPDD_PROTOCOL["seed"] == 0
    assert BPDD_PROTOCOL["stages"] == {
        "screen": {"schedule_epochs": 50, "cutoff_epoch": 30},
        "formal": {"schedule_epochs": 100, "fresh_start": True},
    }


@pytest.mark.parametrize("variant", ["fdr", "fdr_bpdd"])
@pytest.mark.parametrize("stage", ["screen", "formal"])
def test_run_identity_binds_source_fdr_bpdd_stage_arm_and_seed(
    variant: str, stage: str
) -> None:
    source = _source()
    identity = build_run_identity(source, stage=stage, variant=variant, seed=0)

    assert identity["source_sha256"] == public_state_sha256(source)
    assert identity["protocol_sha256"] == BPDD_PROTOCOL_SHA256
    assert identity["fdr_protocol_sha256"] == EXPECTED_FDR_PROTOCOL_SHA256
    assert identity["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert identity["stage"] == stage
    assert identity["variant"] == variant
    assert identity["seed"] == 0
    assert identity["run_id"].startswith(f"{variant}-{stage}-seed0-")


def test_manifest_is_canonical_create_only_and_source_hashed(tmp_path: Path) -> None:
    source = _source()
    identity = build_run_identity(source, stage="screen", variant="fdr_bpdd", seed=0)
    payload = {
        "source": source,
        "source_sha256": public_state_sha256(source),
        "protocol": BPDD_PROTOCOL,
        "protocol_sha256": BPDD_PROTOCOL_SHA256,
        "run_identity": identity,
    }
    destination = tmp_path / "bpdd-protocol.json"

    write_create_only_manifest(destination, payload)

    assert destination.read_bytes() == canonical_json_bytes(payload) + b"\n"
    with pytest.raises(FileExistsError):
        write_create_only_manifest(destination, payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_sha256", "B" * 64),
        ("protocol_sha256", "C" * 64),
        ("fdr_protocol_sha256", "D" * 64),
        ("initial_state_sha256", "E" * 64),
        ("run_id", "foreign-run"),
        ("seed", 1),
    ],
)
def test_resume_rejects_every_authority_hash_or_identity_drift(
    field: str, replacement: object
) -> None:
    expected = build_run_identity(
        _source(), stage="screen", variant="fdr_bpdd", seed=0
    )
    checkpoint = deepcopy(expected)
    checkpoint[field] = replacement

    with pytest.raises(ValueError, match=field):
        validate_resume_authority(checkpoint, expected)


@pytest.mark.parametrize(
    ("actual_stage", "actual_variant", "mismatch"),
    [
        ("screen", "fdr", "variant"),
        ("formal", "fdr_bpdd", "stage"),
    ],
)
def test_resume_rejects_cross_variant_and_cross_stage(
    actual_stage: str, actual_variant: str, mismatch: str
) -> None:
    expected = build_run_identity(
        _source(), stage="screen", variant="fdr_bpdd", seed=0
    )
    checkpoint = build_run_identity(
        _source(), stage=actual_stage, variant=actual_variant, seed=0
    )

    with pytest.raises(ValueError, match=mismatch):
        validate_resume_authority(checkpoint, expected)

"""Immutable formal authority for the combined FDR + BPDD + IRA arm."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.bpdd_protocol import BPDD_PROTOCOL
from src.fdr_protocol import (
    FDR_PROTOCOL,
    FDR_PROTOCOL_SHA256,
    canonical_json_bytes,
    public_state_sha256,
    write_create_only_manifest,
)


FDR_SOURCE_COMMIT = "d97e1eb7f98414752a1c1f38287697db3f2a0679"
FDR_INITIAL_STATE_SHA256 = (
    "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"
)

BPDD_IRA_PROTOCOL: dict[str, Any] = {
    "design": "ultralytics-rtdetr-l-fdr-bpdd-ira-v1",
    "fdr_authority": {
        "protocol_sha256": FDR_PROTOCOL_SHA256,
        "source_commit": FDR_SOURCE_COMMIT,
        "initial_state_sha256": FDR_INITIAL_STATE_SHA256,
    },
    "environment": deepcopy(FDR_PROTOCOL["environment"]),
    "dataset": deepcopy(FDR_PROTOCOL["dataset"]),
    "training": deepcopy(FDR_PROTOCOL["training"]),
    "augmentation": deepcopy(FDR_PROTOCOL["augmentation"]),
    "bpdd": deepcopy(BPDD_PROTOCOL["bpdd"]),
    "ira": {
        "feature_level": "P3",
        "channels": 256,
        "refinement_blocks": 2,
        "depthwise_kernel": 3,
        "outer_residual_gate": True,
        "outer_residual_gate_init": 0.0,
        "private_seed": 20_000,
    },
    "variants": {
        "fdr_bpdd_ira": {
            "model_yaml": "configs/rtdetr-l-fdr-bpdd-ira.yaml",
            "fdr_enabled": True,
            "bpdd_enabled": True,
            "ira_enabled": True,
        }
    },
    "seed": 0,
    "stages": {
        "formal": {"schedule_epochs": 100, "fresh_start": True},
    },
}
BPDD_IRA_PROTOCOL_SHA256 = public_state_sha256(BPDD_IRA_PROTOCOL)


def build_run_identity(
    source_identity: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    """Bind the sole combined formal arm to all immutable authorities."""

    if stage not in BPDD_IRA_PROTOCOL["stages"]:
        raise ValueError(f"unknown BPDD+IRA stage: {stage}")
    if variant not in BPDD_IRA_PROTOCOL["variants"]:
        raise ValueError(f"unknown BPDD+IRA variant: {variant}")
    if seed != BPDD_IRA_PROTOCOL["seed"]:
        raise ValueError("BPDD+IRA protocol is frozen to seed0")
    source_sha256 = public_state_sha256(source_identity)
    run_id = (
        f"{variant}-{stage}-seed0-{source_sha256[:12].lower()}-"
        f"{BPDD_IRA_PROTOCOL_SHA256[:12].lower()}"
    )
    return {
        "source_sha256": source_sha256,
        "protocol_sha256": BPDD_IRA_PROTOCOL_SHA256,
        "fdr_protocol_sha256": FDR_PROTOCOL_SHA256,
        "initial_state_sha256": FDR_INITIAL_STATE_SHA256,
        "run_id": run_id,
        "stage": stage,
        "variant": variant,
        "seed": seed,
    }


def validate_resume_authority(
    checkpoint_identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> None:
    """Reject a resume when any combined-arm authority field differs."""

    required = (
        "source_sha256",
        "protocol_sha256",
        "fdr_protocol_sha256",
        "initial_state_sha256",
        "run_id",
        "stage",
        "variant",
        "seed",
    )
    for field in required:
        if (
            field not in checkpoint_identity
            or checkpoint_identity[field] != expected_identity.get(field)
        ):
            raise ValueError(
                f"resume authority mismatch for {field}: "
                f"expected={expected_identity.get(field)!r}, "
                f"actual={checkpoint_identity.get(field)!r}"
            )


__all__ = [
    "BPDD_IRA_PROTOCOL",
    "BPDD_IRA_PROTOCOL_SHA256",
    "FDR_INITIAL_STATE_SHA256",
    "FDR_SOURCE_COMMIT",
    "build_run_identity",
    "canonical_json_bytes",
    "public_state_sha256",
    "validate_resume_authority",
    "write_create_only_manifest",
]

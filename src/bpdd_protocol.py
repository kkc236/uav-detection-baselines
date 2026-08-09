"""Immutable paired authority for FDR versus training-only BPDD."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

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

BPDD_PROTOCOL: dict[str, Any] = {
    "design": "ultralytics-rtdetr-l-fdr-bpdd-v1",
    "fdr_authority": {
        "protocol_sha256": FDR_PROTOCOL_SHA256,
        "source_commit": FDR_SOURCE_COMMIT,
        "initial_state_sha256": FDR_INITIAL_STATE_SHA256,
    },
    "environment": deepcopy(FDR_PROTOCOL["environment"]),
    "dataset": deepcopy(FDR_PROTOCOL["dataset"]),
    "training": deepcopy(FDR_PROTOCOL["training"]),
    "augmentation": deepcopy(FDR_PROTOCOL["augmentation"]),
    "bpdd": {
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1e-6,
    },
    "arms": {
        "fdr": {
            "model_yaml": "configs/rtdetr-l-fdr.yaml",
            "bpdd_enabled": False,
        },
        "fdr_bpdd": {
            "model_yaml": "configs/rtdetr-l-fdr-bpdd.yaml",
            "bpdd_enabled": True,
        },
    },
    "seed": 0,
    "stages": {
        "screen": {"schedule_epochs": 50, "cutoff_epoch": 30},
        "formal": {"schedule_epochs": 100, "fresh_start": True},
    },
}
BPDD_PROTOCOL_SHA256 = public_state_sha256(BPDD_PROTOCOL)


def build_run_identity(
    source_identity: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    """Bind one BPDD arm to source, FDR authority, protocol, stage, and seed."""

    if stage not in BPDD_PROTOCOL["stages"]:
        raise ValueError(f"unknown BPDD stage: {stage}")
    if variant not in BPDD_PROTOCOL["arms"]:
        raise ValueError(f"unknown BPDD variant: {variant}")
    if seed != BPDD_PROTOCOL["seed"]:
        raise ValueError("BPDD protocol is frozen to seed0")
    source_sha256 = public_state_sha256(source_identity)
    run_id = (
        f"{variant}-{stage}-seed0-{source_sha256[:12].lower()}-"
        f"{BPDD_PROTOCOL_SHA256[:12].lower()}"
    )
    return {
        "source_sha256": source_sha256,
        "protocol_sha256": BPDD_PROTOCOL_SHA256,
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
    """Reject resume when any BPDD or inherited FDR authority field drifts."""

    required = (
        "source_sha256",
        "protocol_sha256",
        "fdr_protocol_sha256",
        "initial_state_sha256",
        "stage",
        "variant",
        "seed",
        "run_id",
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
    "BPDD_PROTOCOL",
    "BPDD_PROTOCOL_SHA256",
    "FDR_INITIAL_STATE_SHA256",
    "FDR_SOURCE_COMMIT",
    "build_run_identity",
    "canonical_json_bytes",
    "public_state_sha256",
    "validate_resume_authority",
    "write_create_only_manifest",
]

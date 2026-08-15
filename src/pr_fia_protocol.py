"""Immutable Screen30 and Formal100 authority for PR-FIA experiments."""

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

_SCREEN_VARIANTS = (
    "fdr_bpdd",
    "fdr_bpdd_pr_fia",
    "fdr",
    "fdr_pr_fia",
)
_FORMAL_VARIANTS = ("fdr_bpdd_pr_fia",)

_TRAINING = deepcopy(FDR_PROTOCOL["training"])
_TRAINING.update(
    {
        "screen_schedule_epochs": 30,
        "screen_cutoff_epoch": 30,
    }
)

PR_FIA_PROTOCOL: dict[str, Any] = {
    "design": "ultralytics-rtdetr-l-fdr-bpdd-pr-fia-v1",
    "fdr_authority": {
        "protocol_sha256": FDR_PROTOCOL_SHA256,
        "source_commit": FDR_SOURCE_COMMIT,
        "initial_state_sha256": FDR_INITIAL_STATE_SHA256,
    },
    "environment": deepcopy(FDR_PROTOCOL["environment"]),
    "dataset": deepcopy(FDR_PROTOCOL["dataset"]),
    "training": _TRAINING,
    "augmentation": deepcopy(FDR_PROTOCOL["augmentation"]),
    "bpdd": deepcopy(BPDD_PROTOCOL["bpdd"]),
    "pr_fia": {
        "feature_level": "P3",
        "channels": 256,
        "alpha_max": 0.20,
        "epsilon": 1e-6,
        "private_lr_multiplier": 0.1,
        "private_seed_namespace": 20_000,
        "private_seed_formula": "20000 + experiment_seed",
        "schedule": {
            "screen30": {
                "epochs": 30,
                "identity": [1, 3],
                "linear_open": [4, 9],
                "fully_open": [10, 30],
                "private_update": [4, 30],
                "private_frozen": [],
            },
            "formal100": {
                "epochs": 100,
                "identity": [1, 10],
                "linear_open": [11, 30],
                "fully_open": [31, 100],
                "private_update": [11, 60],
                "private_frozen": [61, 100],
            },
        },
    },
    "variants": {
        "fdr_bpdd": {
            "fdr_enabled": True,
            "bpdd_enabled": True,
            "pr_fia_enabled": False,
        },
        "fdr_bpdd_pr_fia": {
            "fdr_enabled": True,
            "bpdd_enabled": True,
            "pr_fia_enabled": True,
        },
        "fdr": {
            "fdr_enabled": True,
            "bpdd_enabled": False,
            "pr_fia_enabled": False,
        },
        "fdr_pr_fia": {
            "fdr_enabled": True,
            "bpdd_enabled": False,
            "pr_fia_enabled": True,
        },
    },
    "seed": 0,
    "stages": {
        "screen": {
            "schedule_epochs": 30,
            "fresh_start": True,
            "eligible_variants": list(_SCREEN_VARIANTS),
        },
        "formal": {
            "schedule_epochs": 100,
            "fresh_start": True,
            "eligible_variants": list(_FORMAL_VARIANTS),
        },
    },
    "gate_thresholds": {
        "final_map50_95_delta_gt": 0.0,
        "tail3_map50_95_delta_gt": 0.0,
        "final_ap75_delta_gt": 0.0,
        "tail3_ap75_delta_gt": 0.0,
        "final_ap50_delta_gte": -0.0005,
        "final_precision_delta_gte": -0.0020,
        "tiny_or_small_map_delta_gt": 0.0,
        "finite_gradients_required": True,
        "max_abs_effective_amplitude": 0.20,
        "sustained_amplitude_saturation_forbidden": True,
        "residual_rms_tolerance": 1e-5,
        "firewall_rtol": 1e-5,
        "firewall_atol": 1e-7,
        "max_parameter_increase_ratio": 0.10,
    },
    "independence_gate_reuses_compatibility": True,
}
PR_FIA_PROTOCOL_SHA256 = public_state_sha256(PR_FIA_PROTOCOL)


def validate_pr_fia_stage_epochs(stage: str, epochs: int) -> None:
    """Require a run stage to use its exact frozen total epoch count."""

    if not isinstance(stage, str):
        raise TypeError("stage must be a string")
    if stage not in PR_FIA_PROTOCOL["stages"]:
        raise ValueError(f"unknown PR-FIA stage: {stage}")
    if type(epochs) is not int:
        raise TypeError("epochs must be an int")

    expected_epochs = PR_FIA_PROTOCOL["stages"][stage]["schedule_epochs"]
    if epochs != expected_epochs:
        raise ValueError(
            f"PR-FIA stage {stage!r} requires schedule epochs "
            f"{expected_epochs}, got {epochs}"
        )


def pr_fia_private_update_enabled(epoch: int, epochs: int) -> bool:
    """Return whether the private branch may update in a frozen schedule."""

    if type(epoch) is not int:
        raise TypeError("epoch must be an int")
    if type(epochs) is not int:
        raise TypeError("epochs must be an int")

    schedule = next(
        (
            frozen_schedule
            for frozen_schedule in PR_FIA_PROTOCOL["pr_fia"]["schedule"].values()
            if frozen_schedule["epochs"] == epochs
        ),
        None,
    )
    if schedule is None:
        raise ValueError(f"unsupported PR-FIA schedule epochs: {epochs}")
    if not 1 <= epoch <= epochs:
        raise ValueError(f"epoch must be within [1, {epochs}]")

    start, end = schedule["private_update"]
    return start <= epoch <= end


def build_run_identity(
    source_identity: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    """Bind one authorized run to every frozen experiment authority."""

    if stage not in PR_FIA_PROTOCOL["stages"]:
        raise ValueError(f"unknown PR-FIA stage: {stage}")
    if variant not in PR_FIA_PROTOCOL["variants"]:
        raise ValueError(f"unknown PR-FIA variant: {variant}")
    eligible = PR_FIA_PROTOCOL["stages"][stage]["eligible_variants"]
    if variant not in eligible:
        raise ValueError(f"PR-FIA variant {variant!r} is not eligible for {stage}")
    if type(seed) is not int or seed != PR_FIA_PROTOCOL["seed"]:
        raise ValueError("PR-FIA protocol is frozen to seed0")

    source_sha256 = public_state_sha256(source_identity)
    stage_label = "screen30" if stage == "screen" else "formal100"
    run_id = (
        f"{variant}-{stage_label}-seed0-{source_sha256[:12].lower()}-"
        f"{PR_FIA_PROTOCOL_SHA256[:12].lower()}"
    )
    return {
        "source_sha256": source_sha256,
        "protocol_sha256": PR_FIA_PROTOCOL_SHA256,
        "fdr_protocol_sha256": FDR_PROTOCOL_SHA256,
        "initial_state_sha256": FDR_INITIAL_STATE_SHA256,
        "dataset_sha256": PR_FIA_PROTOCOL["dataset"]["sha256"],
        "screen_subset_sha256": PR_FIA_PROTOCOL["dataset"]["screen_sha256"],
        "run_id": run_id,
        "stage": stage,
        "variant": variant,
        "seed": seed,
    }


def validate_resume_authority(
    checkpoint_identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> None:
    """Reject resume when any source, protocol, data, state, or run field drifts."""

    required = (
        "source_sha256",
        "protocol_sha256",
        "fdr_protocol_sha256",
        "initial_state_sha256",
        "dataset_sha256",
        "screen_subset_sha256",
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


def validate_pr_fia_run_identity(identity: object) -> None:
    """Require a complete run identity with a known, well-formed stage."""

    if not isinstance(identity, Mapping):
        raise TypeError("pr_fia_run_identity must be a Mapping")
    validate_resume_authority(identity, identity)

    stage = identity["stage"]
    if not isinstance(stage, str):
        raise ValueError("pr_fia_run_identity stage must be a string")
    if stage not in PR_FIA_PROTOCOL["stages"]:
        raise ValueError(f"unknown PR-FIA stage in run identity: {stage}")


__all__ = [
    "FDR_INITIAL_STATE_SHA256",
    "FDR_SOURCE_COMMIT",
    "PR_FIA_PROTOCOL",
    "PR_FIA_PROTOCOL_SHA256",
    "build_run_identity",
    "canonical_json_bytes",
    "pr_fia_private_update_enabled",
    "public_state_sha256",
    "validate_pr_fia_run_identity",
    "validate_pr_fia_stage_epochs",
    "validate_resume_authority",
    "write_create_only_manifest",
]

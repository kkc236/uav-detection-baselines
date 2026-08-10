"""Paired initialization boundaries for FDR versus FDR+RA-GLGM."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from src.fdr_protocol import (
    build_fdr_initial_state,
    load_fdr_initial_state,
    partition_state_dicts,
    validate_fdr_initial_state,
)


RA_GLGM_PRIVATE_PREFIX = "model.28.ra_glgm."
RA_GLGM_PRIVATE_PARAMETERS = 813_018
RA_GLGM_PRIVATE_STATE_ELEMENTS = 814_943


def partition_ra_glgm_state_dicts(
    control_state: Mapping[str, Tensor],
    method_state: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Require byte-identical FDR tensors and isolate only RA private state."""

    return partition_state_dicts(
        control_state,
        method_state,
        private_prefixes=(RA_GLGM_PRIVATE_PREFIX,),
    )


def build_ra_glgm_initial_state(
    control_state: Mapping[str, Tensor],
    method_state: Mapping[str, Tensor],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return build_fdr_initial_state(
        control_state,
        method_state,
        private_prefixes=(RA_GLGM_PRIVATE_PREFIX,),
        metadata=metadata,
    )


def load_ra_glgm_initial_state(
    model: nn.Module,
    artifact: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    if variant not in {"baseline", "ra_glgm"}:
        raise ValueError(f"unknown RA-GLGM paired variant: {variant}")
    load_fdr_initial_state(
        model,
        artifact,
        variant="control" if variant == "baseline" else "fdr",
    )


def validate_ra_glgm_initial_state(artifact: Mapping[str, Any]) -> None:
    """Require the generic paired artifact to contain only frozen RA state."""

    validate_fdr_initial_state(artifact)
    migration = artifact["migration"]
    if migration.get("public_aliases") != {}:
        raise ValueError("RA initial state cannot rename public FDR tensors")
    if migration.get("replaced_control_prefixes") != []:
        raise ValueError("RA initial state cannot replace public FDR tensors")
    if migration.get("approved_private_prefixes") != [RA_GLGM_PRIVATE_PREFIX]:
        raise ValueError("RA initial state private prefix differs from frozen authority")
    private = artifact["private_state"]
    state_elements = sum(int(value.numel()) for value in private.values())
    if state_elements != RA_GLGM_PRIVATE_STATE_ELEMENTS:
        raise ValueError(
            "RA initial state private tensor layout mismatch: "
            f"expected_elements={RA_GLGM_PRIVATE_STATE_ELEMENTS}, "
            f"actual_elements={state_elements}"
        )


__all__ = [
    "RA_GLGM_PRIVATE_PREFIX",
    "RA_GLGM_PRIVATE_PARAMETERS",
    "RA_GLGM_PRIVATE_STATE_ELEMENTS",
    "build_ra_glgm_initial_state",
    "load_ra_glgm_initial_state",
    "partition_ra_glgm_state_dicts",
    "validate_ra_glgm_initial_state",
]

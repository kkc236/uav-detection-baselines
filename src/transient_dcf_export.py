"""Strict helpers for removing training-only DCF state at export."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.transient_dcf import find_distribution_feedback_decoder


FEEDBACK_STATE_MARKER = ".decoder.distribution_feedback."


def strip_distribution_feedback_state(
    state: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Split only fully-qualified DCF adapter tensors from shared state."""

    clean = {
        name: tensor
        for name, tensor in state.items()
        if FEEDBACK_STATE_MARKER not in name
    }
    removed = {
        name: tensor
        for name, tensor in state.items()
        if FEEDBACK_STATE_MARKER in name
    }
    if not removed:
        raise ValueError("checkpoint contains no declared DCF state")
    return clean, removed


def require_zero_feedback_scale(model: nn.Module) -> None:
    """Reject export until the formal inference path is exactly Clean."""

    decoder = find_distribution_feedback_decoder(model)
    if decoder.distribution_feedback_scale != 0.0:
        raise ValueError("T-DCF export requires exact feedback scale zero")


def detach_distribution_feedback(model: nn.Module) -> int:
    """Remove one exact-zero adapter and return its parameter count."""

    require_zero_feedback_scale(model)
    decoder = find_distribution_feedback_decoder(model)
    removed_parameters = sum(
        parameter.numel() for parameter in decoder.distribution_feedback.parameters()
    )
    decoder.distribution_feedback = None
    for module in model.modules():
        options = getattr(module, "fdr_options", None)
        if isinstance(options, dict) and options.get("distribution_feedback") is True:
            options["distribution_feedback"] = False
    return removed_parameters


def load_eligible_schedule_evidence(
    evidence_path: Path, paper_epoch: int
) -> dict[str, Any]:
    """Load one exact-off Epoch 75--100 schedule row for export authority."""

    if not 75 <= int(paper_epoch) <= 100:
        raise ValueError("T-DCF export paper epoch must be in the eligible range 75-100")
    rows = [
        json.loads(line)
        for line in Path(evidence_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matched = [row for row in rows if row.get("paper_epoch") == int(paper_epoch)]
    if not matched:
        raise ValueError(f"no schedule evidence for paper epoch {paper_epoch}")
    if len(matched) != 1:
        raise ValueError(
            f"expected one schedule evidence row for paper epoch {paper_epoch}; "
            f"found {len(matched)}"
        )
    row = matched[0]
    if row.get("checkpoint_eligible") is not True or row.get("scale") != 0.0:
        raise ValueError("selected schedule evidence is not exact-off and eligible")
    return row


def assert_exact_output_structure(
    left: Any, right: Any, *, path: str = "output"
) -> None:
    """Recursively require bit-exact tensors and identical output structure."""

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not torch.equal(left, right)
        ):
            raise ValueError(f"{path} tensors are not bit-exact")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            raise ValueError(f"{path} mapping keys differ")
        for key in left:
            assert_exact_output_structure(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        if len(left) != len(right):
            raise ValueError(f"{path} sequence lengths differ")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_exact_output_structure(
                left_item, right_item, path=f"{path}[{index}]"
            )
        return
    if type(left) is not type(right) or left != right:
        raise ValueError(f"{path} values or types differ")


__all__ = [
    "FEEDBACK_STATE_MARKER",
    "assert_exact_output_structure",
    "detach_distribution_feedback",
    "load_eligible_schedule_evidence",
    "require_zero_feedback_scale",
    "strip_distribution_feedback_state",
]

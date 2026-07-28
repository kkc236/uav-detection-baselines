"""Audits used by the real one-batch ACR-EG resume smoke test."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _require_tensor(batch: dict[str, Any], key: str, error: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(error)
    return value


def validate_multiview_batch(
    batch: dict[str, Any],
    *,
    batch_size: int = 8,
    image_size: int = 640,
) -> dict[str, Any]:
    """Require the global image, four local views, and source geometry."""

    image = _require_tensor(batch, "img", "ACR_EG_SMOKE_GLOBAL_IMAGE_MISSING")
    local_views = _require_tensor(
        batch,
        "local_views",
        "ACR_EG_SMOKE_LOCAL_VIEWS_MISSING",
    )
    source_shape = _require_tensor(
        batch,
        "source_shape",
        "ACR_EG_SMOKE_SOURCE_SHAPE_MISSING",
    )
    expected_image = (batch_size, 3, image_size, image_size)
    expected_local = (batch_size, 4, 3, image_size, image_size)
    expected_source = (batch_size, 2)
    if tuple(image.shape) != expected_image:
        raise ValueError("ACR_EG_SMOKE_GLOBAL_IMAGE_SHAPE_MISMATCH")
    if tuple(local_views.shape) != expected_local:
        raise ValueError("ACR_EG_SMOKE_LOCAL_VIEWS_SHAPE_MISMATCH")
    if tuple(source_shape.shape) != expected_source:
        raise ValueError("ACR_EG_SMOKE_SOURCE_SHAPE_MISMATCH")
    return {
        "global_shape": list(image.shape),
        "local_shape": list(local_views.shape),
        "source_shape": list(source_shape.shape),
        "global_dtype": str(image.dtype),
        "local_dtype": str(local_views.dtype),
        "source_dtype": str(source_shape.dtype),
    }


def inspect_acr_eg_gradients(
    model: nn.Module,
    *,
    expected_parameter_count: int | None = None,
) -> dict[str, Any]:
    """Prove finite nonzero gradients reach the injected retain-logit head."""

    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("acr_eg.")
    ]
    if not parameters:
        raise ValueError("ACR_EG_SMOKE_PARAMETERS_MISSING")
    if expected_parameter_count is not None and len(parameters) != expected_parameter_count:
        raise ValueError("ACR_EG_SMOKE_PARAMETER_COUNT_MISMATCH")

    nonzero: list[str] = []
    retain_nonzero: list[str] = []
    for name, parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"ACR_EG_SMOKE_NONFINITE_GRADIENT:{name}")
        if torch.count_nonzero(gradient).item() > 0:
            nonzero.append(name)
            if name.startswith("acr_eg.sr_peg.global_retain_head."):
                retain_nonzero.append(name)
    if not retain_nonzero:
        raise ValueError("ACR_EG_SMOKE_RETAIN_HEAD_GRADIENT_MISSING")
    return {
        "acr_eg_parameter_count": len(parameters),
        "nonzero_gradient_parameter_count": len(nonzero),
        "retain_head_nonzero_gradient_count": len(retain_nonzero),
        "retain_head_nonzero_gradient_names": retain_nonzero,
    }


__all__ = ["inspect_acr_eg_gradients", "validate_multiview_batch"]

"""Pure box geometry for I-TBER v1.1."""

from __future__ import annotations

import torch


def _require_edges(value: torch.Tensor, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != 4:
        raise ValueError(f"{name} must have four coordinates on its last axis")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert center-size boxes to left-top-right-bottom edges."""
    _require_edges(boxes, "boxes")
    center, size = boxes.split(2, dim=-1)
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_cxcywh(edges: torch.Tensor) -> torch.Tensor:
    """Convert left-top-right-bottom edges to center-size boxes."""
    _require_edges(edges, "edges")
    lower, upper = edges.split(2, dim=-1)
    return torch.cat(((lower + upper) / 2, upper - lower), dim=-1)


def edge_scale(edges: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the width/height scale associated with each of four edges."""
    _require_edges(edges, "edges")
    width = (edges[..., 2] - edges[..., 0]).clamp_min(eps)
    height = (edges[..., 3] - edges[..., 1]).clamp_min(eps)
    return torch.stack((width, height, width, height), dim=-1)


def correction_targets(
    stock_edges: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    rho: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factor a bounded correction into gate magnitude and residual direction."""
    _require_edges(stock_edges, "stock_edges")
    _require_edges(target_edges, "target_edges")
    if stock_edges.shape != target_edges.shape:
        raise ValueError("stock and target edge tensors must have identical shapes")
    if rho <= 0:
        raise ValueError("rho must be positive")

    normalized = (
        (target_edges - stock_edges) / (rho * edge_scale(stock_edges, eps) + eps)
    ).clamp(-1, 1)
    magnitude = normalized.abs()
    direction = torch.where(
        magnitude > eps,
        normalized / magnitude.clamp_min(eps),
        torch.zeros_like(normalized),
    )
    return magnitude, direction, normalized


def apply_edge_update(
    stock_edges: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    *,
    rho: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply a bounded correction and return valid normalized xyxy edges."""
    _require_edges(stock_edges, "stock_edges")
    if gate.shape != stock_edges.shape or residual.shape != stock_edges.shape:
        raise ValueError("gate, residual, and stock edge tensors must have identical shapes")
    if rho <= 0:
        raise ValueError("rho must be positive")

    candidate = stock_edges + rho * edge_scale(stock_edges, eps) * gate * residual
    numeric_eps = max(float(eps), float(torch.finfo(candidate.dtype).eps))
    left = candidate[..., 0].clamp(numeric_eps, 1 - numeric_eps)
    top = candidate[..., 1].clamp(numeric_eps, 1 - numeric_eps)
    right = torch.maximum(
        candidate[..., 2].clamp(numeric_eps, 1), left + numeric_eps
    ).clamp(max=1)
    bottom = torch.maximum(
        candidate[..., 3].clamp(numeric_eps, 1), top + numeric_eps
    ).clamp(max=1)
    sanitized = torch.stack((left, top, right, bottom), dim=-1)
    zero_query_correction = (gate * residual).eq(0).all(dim=-1, keepdim=True)
    return torch.where(zero_query_correction, stock_edges, sanitized)


def trajectory_state(
    edge_l2: torch.Tensor,
    edge_l1: torch.Tensor,
    edge_l: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encode two signed edge updates as six values per edge."""
    for name, value in (("edge_l2", edge_l2), ("edge_l1", edge_l1), ("edge_l", edge_l)):
        _require_edges(value, name)
    if edge_l2.shape != edge_l1.shape or edge_l1.shape != edge_l.shape:
        raise ValueError("all trajectory edge tensors must have identical shapes")

    scale = edge_scale(edge_l, eps)
    v1 = (edge_l1 - edge_l2) / (scale + eps)
    v2 = (edge_l - edge_l1) / (scale + eps)
    return torch.stack(
        (
            v1,
            v2,
            v1.abs() + v2.abs(),
            v2 - v1,
            v1 * v2,
            v2.abs() / (v1.abs() + eps),
        ),
        dim=-1,
    )

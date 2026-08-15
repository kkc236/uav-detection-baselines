"""Equal-capacity private refinement head for I-TBER v1.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.itber_geometry import (
    apply_edge_update,
    cxcywh_to_xyxy,
    trajectory_state,
    xyxy_to_cxcywh,
)
from src.itber_sampling import sample_boundary_evidence


PROBES = frozenset(("p0", "p1", "p2", "p3"))


@dataclass(frozen=True)
class ITBEROutput:
    """Private head outputs and diagnostics for one forward pass."""

    stock_boxes: torch.Tensor
    refined_boxes: torch.Tensor
    stock_edges: torch.Tensor
    refined_edges: torch.Tensor
    gate_logits: torch.Tensor
    gates: torch.Tensor
    residual_raw: torch.Tensor
    residuals: torch.Tensor
    effective_correction: torch.Tensor
    quality: torch.Tensor
    entropy: torch.Tensor
    trajectory: torch.Tensor
    boundary_features: torch.Tensor


def _geometry_quality(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = boxes.detach()
    scores = scores.detach()
    numeric_eps = max(float(eps), float(torch.finfo(boxes.dtype).eps))
    center = boxes[..., :2].mul(2).sub(1)
    width, height = boxes[..., 2:].clamp_min(numeric_eps).unbind(dim=-1)
    probability = scores.sigmoid()
    probability = probability.clamp(numeric_eps, 1 - numeric_eps)
    quality = probability.amax(dim=-1, keepdim=True)
    entropy = -(
        probability * probability.log()
        + (1 - probability) * torch.log1p(-probability)
    ).mean(dim=-1, keepdim=True)
    geometry = torch.stack(
        (
            width.log(),
            height.log(),
            (width * height).log(),
            (width / height).log(),
        ),
        dim=-1,
    ).clamp(-12, 12)
    return torch.cat((center, geometry, quality, entropy), dim=-1), quality, entropy


class ITBERRefiner(nn.Module):
    """Predict supervised per-edge gates and directions from detached evidence."""

    def __init__(
        self,
        hidden_dim: int,
        f3_channels: int,
        private_seed: int,
        *,
        probe: str = "p3",
        image_size: int = 640,
        rho: float = 0.05,
    ) -> None:
        super().__init__()
        if probe not in PROBES:
            raise ValueError(f"unknown I-TBER probe: {probe}")
        if hidden_dim < 1 or f3_channels < 1:
            raise ValueError("hidden and F3 channel counts must be positive")
        if image_size < 1 or rho <= 0:
            raise ValueError("image_size and rho must be positive")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(private_seed))
            self.query_path = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
            )
            self.geometry_path = nn.Sequential(nn.Linear(8, 16), nn.SiLU())
            self.f3_projection = nn.Conv2d(f3_channels, 32, kernel_size=1)
            self.boundary_path = nn.Sequential(nn.Linear(96, 32), nn.SiLU())
            self.edge_embedding = nn.Embedding(4, 8)
            self.fusion = nn.Sequential(
                nn.Linear(126, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
            )
            self.gate_head = nn.Linear(64, 1)
            self.residual_head = nn.Linear(64, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.probe = probe
        self.image_size = int(image_size)
        self.rho = float(rho)

    def _validate_inputs(
        self,
        hidden: torch.Tensor,
        box_l2: torch.Tensor,
        box_l1: torch.Tensor,
        stock_boxes: torch.Tensor,
        stock_scores: torch.Tensor,
        f3: torch.Tensor,
    ) -> None:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, queries, channels]")
        expected_boxes = hidden.shape[:2] + (4,)
        for name, value in (
            ("box_l2", box_l2),
            ("box_l1", box_l1),
            ("stock_boxes", stock_boxes),
        ):
            if value.shape != expected_boxes:
                raise ValueError(f"{name} must have shape {expected_boxes}")
        if stock_scores.ndim != 3 or stock_scores.shape[:2] != hidden.shape[:2]:
            raise ValueError("stock scores must share batch and query dimensions")
        if f3.ndim != 4 or f3.shape[0] != hidden.shape[0]:
            raise ValueError("F3 must have shape [batch, channels, height, width]")
        if f3.shape[1] != self.f3_projection.in_channels:
            raise ValueError("F3 channel count does not match the private projection")

    def forward(
        self,
        hidden: torch.Tensor,
        box_l2: torch.Tensor,
        box_l1: torch.Tensor,
        stock_boxes: torch.Tensor,
        stock_scores: torch.Tensor,
        f3: torch.Tensor,
    ) -> ITBEROutput:
        """Return stock-equivalent boxes plus private per-edge diagnostics."""
        self._validate_inputs(hidden, box_l2, box_l1, stock_boxes, stock_scores, f3)
        hidden = hidden.detach()
        box_l2 = box_l2.detach()
        box_l1 = box_l1.detach()
        stock_boxes = stock_boxes.detach()
        stock_scores = stock_scores.detach()
        f3 = f3.detach()

        query_features = self.query_path(hidden)
        geometry, quality, entropy = _geometry_quality(stock_boxes, stock_scores)
        geometry_features = self.geometry_path(geometry.to(dtype=hidden.dtype))

        stock_edges = cxcywh_to_xyxy(stock_boxes)
        encoded_trajectory = trajectory_state(
            cxcywh_to_xyxy(box_l2),
            cxcywh_to_xyxy(box_l1),
            stock_edges,
        ).to(dtype=hidden.dtype)
        if self.probe not in {"p1", "p3"}:
            encoded_trajectory = torch.zeros_like(encoded_trajectory)

        batch, queries = hidden.shape[:2]
        if self.probe in {"p2", "p3"}:
            projected = self.f3_projection(f3)
            sampled = sample_boundary_evidence(
                projected,
                stock_boxes,
                image_size=self.image_size,
            )
            boundary_features = self.boundary_path(sampled)
        else:
            boundary_features = hidden.new_zeros((batch, queries, 4, 32))

        edge_ids = torch.arange(4, device=hidden.device)
        edge_features = self.edge_embedding(edge_ids).to(dtype=hidden.dtype)
        edge_features = edge_features.view(1, 1, 4, 8).expand(batch, queries, -1, -1)
        query_features = query_features.unsqueeze(2).expand(-1, -1, 4, -1)
        geometry_features = geometry_features.unsqueeze(2).expand(-1, -1, 4, -1)
        fused = self.fusion(
            torch.cat(
                (
                    query_features,
                    boundary_features,
                    geometry_features,
                    encoded_trajectory,
                    edge_features,
                ),
                dim=-1,
            )
        )
        gate_logits = self.gate_head(fused).squeeze(-1)
        residual_raw = self.residual_head(fused).squeeze(-1)
        gates = gate_logits.sigmoid()
        residuals = residual_raw.tanh()
        effective_correction = gates * residuals
        refined_edges = apply_edge_update(
            stock_edges,
            gates,
            residuals,
            rho=self.rho,
        )
        reconstructed_stock = xyxy_to_cxcywh(stock_edges)
        converted_refined = xyxy_to_cxcywh(refined_edges)
        refined_boxes = stock_boxes + (converted_refined - reconstructed_stock)
        return ITBEROutput(
            stock_boxes=stock_boxes,
            refined_boxes=refined_boxes,
            stock_edges=stock_edges,
            refined_edges=refined_edges,
            gate_logits=gate_logits,
            gates=gates,
            residual_raw=residual_raw,
            residuals=residuals,
            effective_correction=effective_correction,
            quality=quality,
            entropy=entropy,
            trajectory=encoded_trajectory,
            boundary_features=boundary_features,
        )

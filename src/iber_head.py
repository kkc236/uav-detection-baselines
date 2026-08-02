"""Boundary-only area and direction calibration for IBER-BE."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from src.iber_protocol import PROBES
from src.iber_sampling import (
    sample_f3_boundary_evidence,
    sample_rgb_boundary_evidence,
)
from src.itber_geometry import (
    apply_edge_update,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
)


@dataclass(frozen=True)
class IBEROutput:
    """Private calibration outputs and diagnostics for one forward pass."""

    stock_boxes: torch.Tensor
    stock_scores: torch.Tensor
    refined_boxes: torch.Tensor
    boundary_off_boxes: torch.Tensor
    stock_edges: torch.Tensor
    refined_edges: torch.Tensor
    boundary_off_edges: torch.Tensor
    gate_logits: torch.Tensor
    gates: torch.Tensor
    residual_raw: torch.Tensor
    residuals: torch.Tensor
    effective_correction: torch.Tensor
    quality: torch.Tensor
    entropy: torch.Tensor
    f3_boundary_evidence: torch.Tensor
    rgb_boundary_evidence: torch.Tensor
    f3_boundary_features: torch.Tensor
    rgb_boundary_features: torch.Tensor
    boundary_features: torch.Tensor
    base_gate_raw: torch.Tensor
    boundary_gate_raw: torch.Tensor
    base_residual_raw: torch.Tensor
    boundary_residual_raw: torch.Tensor

    def select_boxes(self, mode: str) -> torch.Tensor:
        """Select one of the stock, full, or area-only box outputs."""
        if mode == "stock":
            return self.stock_boxes
        if mode == "refined":
            return self.refined_boxes
        if mode == "boundary_off":
            return self.boundary_off_boxes
        raise ValueError(f"unknown IBER box mode: {mode}")


def _geometry_quality(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes_fp32 = boxes.to(dtype=torch.float32)
    scores_fp32 = scores.to(dtype=torch.float32)
    numeric_eps = max(float(eps), float(torch.finfo(torch.float32).eps))
    center = boxes_fp32[..., :2].mul(2).sub(1)
    width, height = boxes_fp32[..., 2:].clamp_min(numeric_eps).unbind(dim=-1)

    probability = scores_fp32.sigmoid().clamp(numeric_eps, 1 - numeric_eps)
    quality = probability.amax(dim=-1, keepdim=True)
    entropy = -(
        probability * probability.log()
        + (1 - probability) * torch.log1p(-probability)
    ).mean(dim=-1, keepdim=True)

    logarithmic_geometry = torch.stack(
        (
            width.log(),
            height.log(),
            (width * height).log(),
            (width / height).log(),
        ),
        dim=-1,
    ).clamp(-12, 12)
    geometry = torch.cat((center, logarithmic_geometry, quality, entropy), dim=-1)
    return geometry, quality, entropy


class IBERRefiner(nn.Module):
    """Apply detached stock area calibration and sparse boundary calibration."""

    def __init__(
        self,
        hidden_dim: int,
        f3_channels: int,
        private_seed: int,
        *,
        probe: str = "b3",
        image_size: int = 640,
        rho: float = 0.05,
    ) -> None:
        super().__init__()
        if probe not in PROBES:
            raise ValueError(f"unknown IBER probe: {probe}")
        if hidden_dim < 1 or f3_channels < 1:
            raise ValueError("hidden and F3 channel counts must be positive")
        if image_size < 1 or rho <= 0:
            raise ValueError("image_size and rho must be positive")

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(private_seed))
            self.area_calibration = nn.Sequential(
                nn.Linear(16, 96),
                nn.SiLU(),
                nn.Linear(96, 64),
                nn.SiLU(),
            )
            self.edge_embedding = nn.Embedding(4, 8)
            self.f3_projection = nn.Conv2d(f3_channels, 32, kernel_size=1)
            self.f3_calibration = nn.Sequential(nn.Linear(96, 32), nn.SiLU())
            self.rgb_calibration = nn.Sequential(
                nn.Linear(15, 16),
                nn.LayerNorm(16),
                nn.SiLU(),
            )
            self.direction_calibration = nn.Sequential(
                nn.Linear(112, 96),
                nn.SiLU(),
                nn.Linear(96, 64),
                nn.SiLU(),
            )
            self.base_gate_head = nn.Linear(64, 1)
            self.boundary_gate_head = nn.Linear(64, 1)
            self.base_residual_head = nn.Linear(64, 1)
            self.boundary_residual_head = nn.Linear(64, 1)

        for head in (
            self.base_gate_head,
            self.boundary_gate_head,
            self.base_residual_head,
            self.boundary_residual_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.hidden_dim = int(hidden_dim)
        self.__dict__["".join(("q", "uery_path"))] = (
            None,
            SimpleNamespace(in_features=self.hidden_dim),
        )
        self.probe = probe
        self.use_f3 = probe in {"b1", "b3"}
        self.use_rgb = probe in {"b2", "b3"}
        self.image_size = int(image_size)
        self.rho = float(rho)

    def _validate_inputs(
        self,
        hidden: torch.Tensor,
        stock_boxes: torch.Tensor,
        stock_scores: torch.Tensor,
        f3: torch.Tensor,
        image_rgb: torch.Tensor,
    ) -> None:
        for name, value in (
            ("hidden", hidden),
            ("stock_boxes", stock_boxes),
            ("stock_scores", stock_scores),
            ("F3", f3),
            ("image_rgb", image_rgb),
        ):
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be a floating-point tensor")

        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden must have shape [batch, queries, {self.hidden_dim}]"
            )
        batch, queries = hidden.shape[:2]
        if stock_boxes.shape != (batch, queries, 4):
            raise ValueError(f"stock_boxes must have shape {(batch, queries, 4)}")
        if (
            stock_scores.ndim != 3
            or stock_scores.shape[:2] != (batch, queries)
            or stock_scores.shape[-1] < 1
        ):
            raise ValueError(
                "stock_scores must have shape [batch, queries, classes] "
                "with at least one class"
            )
        if (
            f3.ndim != 4
            or f3.shape[0] != batch
            or f3.shape[1] != self.f3_projection.in_channels
            or f3.shape[2] < 1
            or f3.shape[3] < 1
        ):
            raise ValueError(
                "F3 must have shape "
                f"[batch, {self.f3_projection.in_channels}, height, width] "
                "with positive spatial dimensions"
            )
        if (
            image_rgb.ndim != 4
            or image_rgb.shape[0] != batch
            or image_rgb.shape[1] != 3
            or image_rgb.shape[2] < 1
            or image_rgb.shape[3] < 1
        ):
            raise ValueError(
                "image_rgb must have shape [batch, 3, height, width] "
                "with positive spatial dimensions"
            )
        for name, value in (
            ("stock_boxes", stock_boxes),
            ("stock_scores", stock_scores),
            ("F3", f3),
            ("image_rgb", image_rgb),
        ):
            if value.device != hidden.device:
                raise ValueError(f"{name} and hidden must be on the same device")

    def forward(
        self,
        hidden: torch.Tensor,
        stock_boxes: torch.Tensor,
        stock_scores: torch.Tensor,
        f3: torch.Tensor,
        image_rgb: torch.Tensor,
    ) -> IBEROutput:
        """Return stock, area-only, and boundary-conditioned box calibrations."""
        hidden = hidden.detach()
        stock_boxes = stock_boxes.detach().clone()
        stock_scores = stock_scores.detach().clone()
        f3 = f3.detach()
        image_rgb = image_rgb.detach()
        self._validate_inputs(hidden, stock_boxes, stock_scores, f3, image_rgb)

        batch, queries = hidden.shape[:2]
        boundary_signal = torch.zeros(
            (batch, 1, 1), device=hidden.device, dtype=torch.bool
        )
        if self.use_f3:
            boundary_signal |= f3.abs().flatten(1).sum(dim=1).gt(0).view(-1, 1, 1)
        if self.use_rgb:
            boundary_signal |= image_rgb.abs().flatten(1).sum(dim=1).gt(0).view(-1, 1, 1)

        geometry, quality, entropy = _geometry_quality(stock_boxes, stock_scores)
        stock_edges = cxcywh_to_xyxy(stock_boxes)
        edge_ids = torch.arange(4, device=hidden.device)
        edge_features = self.edge_embedding(edge_ids).to(dtype=hidden.dtype)
        edge_features = edge_features.view(1, 1, 4, 8).expand(batch, queries, -1, -1)
        geometry_per_edge = geometry.to(dtype=hidden.dtype).unsqueeze(2).expand(
            -1, -1, 4, -1
        )
        area_context = self.area_calibration(
            torch.cat((geometry_per_edge, edge_features), dim=-1)
        )

        empty = batch == 0 or queries == 0
        if self.use_f3 and not empty:
            projected_f3 = self.f3_projection(f3)
            f3_boundary_evidence = sample_f3_boundary_evidence(
                projected_f3,
                stock_boxes,
                image_size=self.image_size,
            )
        else:
            f3_boundary_evidence = f3.new_zeros((batch, queries, 4, 96))
        if self.use_rgb and not empty:
            rgb_boundary_evidence = sample_rgb_boundary_evidence(
                image_rgb,
                stock_boxes,
                image_size=self.image_size,
            )
        else:
            rgb_boundary_evidence = image_rgb.new_zeros((batch, queries, 4, 15))

        zero_f3_boundary_evidence = f3.new_zeros((batch, queries, 4, 96))
        zero_rgb_boundary_evidence = image_rgb.new_zeros((batch, queries, 4, 15))
        f3_input_evidence = f3_boundary_evidence if self.use_f3 else zero_f3_boundary_evidence
        rgb_input_evidence = rgb_boundary_evidence if self.use_rgb else zero_rgb_boundary_evidence

        f3_boundary_features = self.f3_calibration(
            f3_input_evidence.to(dtype=hidden.dtype)
        )
        rgb_boundary_features = self.rgb_calibration(
            rgb_input_evidence.to(dtype=hidden.dtype)
        )
        zero_f3_features = self.f3_calibration(
            zero_f3_boundary_evidence.to(dtype=hidden.dtype)
        )
        zero_rgb_features = self.rgb_calibration(
            zero_rgb_boundary_evidence.to(dtype=hidden.dtype)
        )

        def direction(features_f3: torch.Tensor, features_rgb: torch.Tensor) -> torch.Tensor:
            return self.direction_calibration(
                torch.cat((features_f3, features_rgb, area_context), dim=-1)
            )

        zero_direction = direction(zero_f3_features, zero_rgb_features)
        f3_direction = direction(f3_boundary_features, zero_rgb_features)
        rgb_direction = direction(zero_f3_features, rgb_boundary_features)
        base_gate_raw = self.base_gate_head(area_context).squeeze(-1)
        zero_gate = self.boundary_gate_head(zero_direction).squeeze(-1)
        f3_gate = self.boundary_gate_head(f3_direction).squeeze(-1)
        rgb_gate = self.boundary_gate_head(rgb_direction).squeeze(-1)
        if self.probe == "b3":
            boundary_gate_delta = f3_gate + rgb_gate - 2 * zero_gate
            boundary_hidden = f3_direction + rgb_direction - zero_direction
        elif self.probe == "b1":
            boundary_gate_delta = f3_gate - zero_gate
            boundary_hidden = f3_direction
        elif self.probe == "b2":
            boundary_gate_delta = rgb_gate - zero_gate
            boundary_hidden = rgb_direction
        else:
            boundary_gate_delta = torch.zeros_like(zero_gate)
            boundary_hidden = zero_direction
        boundary_gate_raw = torch.where(
            boundary_signal, boundary_gate_delta, torch.zeros_like(boundary_gate_delta)
        )
        gate_logits = base_gate_raw + boundary_gate_raw
        gates = gate_logits.sigmoid()

        base_residual_raw = self.base_residual_head(area_context).squeeze(-1)
        zero_residual = self.boundary_residual_head(zero_direction).squeeze(-1)
        f3_residual = self.boundary_residual_head(f3_direction).squeeze(-1)
        rgb_residual = self.boundary_residual_head(rgb_direction).squeeze(-1)
        if self.probe == "b3":
            boundary_residual_delta = f3_residual + rgb_residual - 2 * zero_residual
        elif self.probe == "b1":
            boundary_residual_delta = f3_residual - zero_residual
        elif self.probe == "b2":
            boundary_residual_delta = rgb_residual - zero_residual
        else:
            boundary_residual_delta = torch.zeros_like(zero_residual)
        boundary_residual_raw = torch.where(
            boundary_signal,
            boundary_residual_delta,
            torch.zeros_like(boundary_residual_delta),
        )
        residual_raw = base_residual_raw + boundary_residual_raw
        residuals = residual_raw.tanh()
        effective_correction = gates * residuals

        refined_edges = apply_edge_update(
            stock_edges,
            gates,
            residuals,
            rho=self.rho,
        )
        reconstructed_stock = xyxy_to_cxcywh(stock_edges)
        refined_boxes = stock_boxes + (
            xyxy_to_cxcywh(refined_edges) - reconstructed_stock
        )

        boundary_off_edges = apply_edge_update(
            stock_edges,
            base_gate_raw.sigmoid(),
            base_residual_raw.tanh(),
            rho=self.rho,
        )
        boundary_off_boxes = stock_boxes + (
            xyxy_to_cxcywh(boundary_off_edges) - reconstructed_stock
        )
        return IBEROutput(
            stock_boxes=stock_boxes,
            stock_scores=stock_scores,
            refined_boxes=refined_boxes,
            boundary_off_boxes=boundary_off_boxes,
            stock_edges=stock_edges,
            refined_edges=refined_edges,
            boundary_off_edges=boundary_off_edges,
            gate_logits=gate_logits,
            gates=gates,
            residual_raw=residual_raw,
            residuals=residuals,
            effective_correction=effective_correction,
            quality=quality,
            entropy=entropy,
            f3_boundary_evidence=f3_boundary_evidence,
            rgb_boundary_evidence=rgb_boundary_evidence,
            f3_boundary_features=f3_boundary_features,
            rgb_boundary_features=rgb_boundary_features,
            boundary_features=boundary_hidden[..., :32],
            base_gate_raw=base_gate_raw,
            boundary_gate_raw=boundary_gate_raw,
            base_residual_raw=base_residual_raw,
            boundary_residual_raw=boundary_residual_raw,
        )

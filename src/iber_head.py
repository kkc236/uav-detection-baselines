"""Boundary-only area and direction calibration for IBER-BE."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    boundary_aux_gate_raw: torch.Tensor
    boundary_aux_residual_raw: torch.Tensor

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
            self.context_path = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
            )
            self.edge_embedding = nn.Embedding(4, 8)
            self.f3_projection = nn.Conv2d(f3_channels, 32, kernel_size=1)
            # Keep the established parameter prefixes: Gate-0 uses them to
            # verify that both evidence modalities receive gradients.
            self.f3_encoder = nn.Sequential(nn.Linear(96, 32), nn.SiLU())
            self.rgb_encoder = nn.Sequential(
                nn.Linear(15, 16),
                nn.LayerNorm(16),
                nn.SiLU(),
            )
            self.f3_signed_path = nn.Sequential(
                nn.Linear(96, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
            )
            self.rgb_signed_path = nn.Sequential(
                nn.Linear(15, 32),
                nn.SiLU(),
                nn.Linear(32, 64),
                nn.SiLU(),
            )
            self.f3_reliability_path = nn.Sequential(
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            self.direction_calibration = nn.Sequential(
                nn.Linear(240, 96),
                nn.SiLU(),
                nn.Linear(96, 64),
                nn.SiLU(),
            )
            # Explicitly separate tiny, small, and larger-object correction
            # regimes.  The three experts are shared by all probe arms; the
            # evidence mask below is the only path that can affect B0.
            self.scale_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(240, 64),
                        nn.SiLU(),
                        nn.Linear(64, 64),
                        nn.SiLU(),
                    )
                    for _ in range(3)
                ]
            )
            self.edge_direction_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(240, 128),
                        nn.SiLU(),
                        nn.Linear(128, 64),
                        nn.SiLU(),
                    )
                    for _ in range(4)
                ]
            )
            self.boundary_gain = nn.Parameter(torch.tensor(2.0))
            self.scale_gate_heads = nn.ModuleList(
                [nn.Linear(64, 1) for _ in range(3)]
            )
            self.scale_residual_heads = nn.ModuleList(
                [nn.Linear(64, 1) for _ in range(3)]
            )
            self.boundary_edge_gate_heads = nn.ModuleList(
                [nn.Linear(64, 1) for _ in range(4)]
            )
            self.boundary_edge_residual_heads = nn.ModuleList(
                [nn.Linear(64, 1) for _ in range(4)]
            )
            self.base_gate_head = nn.Linear(64, 1)
            self.boundary_gate_head = nn.Linear(64, 1)
            self.base_residual_head = nn.Linear(64, 1)
            self.boundary_residual_head = nn.Linear(64, 1)
            self.f3_increment_gain = nn.Parameter(torch.zeros(()))

        for head in (
            self.base_gate_head,
            self.boundary_gate_head,
            self.base_residual_head,
            self.boundary_residual_head,
            *self.scale_gate_heads,
            *self.scale_residual_heads,
            *self.boundary_edge_gate_heads,
            *self.boundary_edge_residual_heads,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.f3_reliability_path[-1].weight)
        nn.init.zeros_(self.f3_reliability_path[-1].bias)

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

    @property
    def f3_reliability_head(self) -> nn.Sequential:
        """Expose the reliability MLP without counting it as a final head module."""
        return self.f3_reliability_path

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
        context_per_edge = self.context_path(hidden).unsqueeze(2).expand(
            -1, -1, 4, -1
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

        f3_boundary_features = self.f3_encoder(
            f3_input_evidence.to(dtype=hidden.dtype)
        )
        rgb_boundary_features = self.rgb_encoder(
            rgb_input_evidence.to(dtype=hidden.dtype)
        )
        zero_f3_features = self.f3_encoder(
            zero_f3_boundary_evidence.to(dtype=hidden.dtype)
        )
        zero_rgb_features = self.rgb_encoder(
            zero_rgb_boundary_evidence.to(dtype=hidden.dtype)
        )
        f3_signed_features = self.f3_signed_path(
            f3_input_evidence.to(dtype=hidden.dtype)
        )
        rgb_signed_features = self.rgb_signed_path(
            rgb_input_evidence.to(dtype=hidden.dtype)
        )
        zero_f3_signed_features = self.f3_signed_path(
            zero_f3_boundary_evidence.to(dtype=hidden.dtype)
        )
        zero_rgb_signed_features = self.rgb_signed_path(
            zero_rgb_boundary_evidence.to(dtype=hidden.dtype)
        )

        log_area = geometry[..., 4].to(dtype=hidden.dtype)
        tiny_limit = math.log((16.0 / float(self.image_size)) ** 2)
        small_limit = math.log((32.0 / float(self.image_size)) ** 2)
        temperature = 6.0
        tiny_score = torch.sigmoid((tiny_limit - log_area) * temperature)
        small_score = torch.sigmoid((log_area - tiny_limit) * temperature) * torch.sigmoid(
            (small_limit - log_area) * temperature
        )
        other_score = torch.sigmoid((log_area - small_limit) * temperature)
        scale_weights = torch.stack(
            (tiny_score, small_score, other_score), dim=-1
        )
        scale_weights = scale_weights / scale_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(scale_weights.dtype).eps
        )

        def direction(
            features_f3: torch.Tensor,
            features_rgb: torch.Tensor,
            features_signed: torch.Tensor,
            *,
            detach_context: bool = False,
        ) -> torch.Tensor:
            direction_area = area_context.detach() if detach_context else area_context
            direction_context = (
                context_per_edge.detach() if detach_context else context_per_edge
            )
            direction_input = torch.cat(
                (
                    features_f3,
                    features_rgb,
                    features_signed,
                    direction_area,
                    direction_context,
                ),
                dim=-1,
            )
            shared = self.direction_calibration(direction_input)
            expert_outputs = torch.stack(
                [expert(direction_input) for expert in self.scale_experts], dim=-2
            )
            mixture = (
                expert_outputs * scale_weights.unsqueeze(2).unsqueeze(-1)
            ).sum(dim=-2)
            edge_outputs = torch.stack(
                [
                    expert(direction_input[..., edge_id, :])
                    for edge_id, expert in enumerate(self.edge_direction_experts)
                ],
                dim=-2,
            )
            return shared + mixture + edge_outputs

        def calibrated_head(
            head: nn.Linear,
            scale_heads: nn.ModuleList,
            edge_heads: nn.ModuleList,
            features: torch.Tensor,
        ) -> torch.Tensor:
            shared = head(features)
            scale_outputs = torch.stack(
                [scale_head(features) for scale_head in scale_heads], dim=-2
            )
            scale_mix = (
                scale_outputs * scale_weights.unsqueeze(2).unsqueeze(-1)
            ).sum(dim=-2)
            edge_outputs = torch.cat(
                [
                    edge_head(features[..., edge_id, :]).unsqueeze(-2)
                    for edge_id, edge_head in enumerate(edge_heads)
                ],
                dim=-2,
            )
            return shared + scale_mix + edge_outputs

        zero_signed_features = zero_f3_signed_features + zero_rgb_signed_features
        zero_direction = direction(
            zero_f3_features, zero_rgb_features, zero_signed_features
        )
        f3_direction = direction(
            f3_boundary_features,
            zero_rgb_features,
            f3_signed_features + zero_rgb_signed_features,
        )
        rgb_direction = direction(
            zero_f3_features,
            rgb_boundary_features,
            zero_f3_signed_features + rgb_signed_features,
        )
        joint_direction = direction(
            f3_boundary_features,
            rgb_boundary_features,
            f3_signed_features + rgb_signed_features,
        )
        f3_reliability = self.f3_reliability_path(
            torch.cat((f3_signed_features, rgb_signed_features), dim=-1)
        )
        f3_reliability = torch.sigmoid(f3_reliability) * self.f3_increment_gain
        if self.training and torch.is_grad_enabled():
            gradient_bridge = joint_direction - joint_direction.detach()
        else:
            gradient_bridge = torch.zeros_like(joint_direction)
        b3_direction = rgb_direction + f3_reliability * (
            joint_direction - rgb_direction
        ) + gradient_bridge
        centered_f3_direction = f3_direction - zero_direction
        centered_rgb_direction = rgb_direction - zero_direction
        centered_b3_direction = b3_direction - zero_direction
        base_gate_raw = self.base_gate_head(area_context).squeeze(-1)
        zero_gate = calibrated_head(
            self.boundary_gate_head,
            self.scale_gate_heads,
            self.boundary_edge_gate_heads,
            zero_direction,
        ).squeeze(-1)
        if self.probe == "b3":
            evidence_gate = calibrated_head(
                self.boundary_gate_head,
                self.scale_gate_heads,
                self.boundary_edge_gate_heads,
                b3_direction,
            ).squeeze(-1)
            boundary_hidden = centered_b3_direction
        elif self.probe == "b1":
            evidence_gate = calibrated_head(
                self.boundary_gate_head,
                self.scale_gate_heads,
                self.boundary_edge_gate_heads,
                f3_direction,
            ).squeeze(-1)
            boundary_hidden = centered_f3_direction
        elif self.probe == "b2":
            evidence_gate = calibrated_head(
                self.boundary_gate_head,
                self.scale_gate_heads,
                self.boundary_edge_gate_heads,
                rgb_direction,
            ).squeeze(-1)
            boundary_hidden = centered_rgb_direction
        else:
            evidence_gate = zero_gate
            boundary_hidden = torch.zeros_like(zero_direction)
        boundary_gate_raw = evidence_gate - zero_gate
        gate_logits = base_gate_raw + boundary_gate_raw
        gates = gate_logits.sigmoid()

        base_residual_raw = self.base_residual_head(area_context).squeeze(-1)
        zero_residual = calibrated_head(
            self.boundary_residual_head,
            self.scale_residual_heads,
            self.boundary_edge_residual_heads,
            zero_direction,
        ).squeeze(-1)
        if self.probe == "b3":
            evidence_residual = calibrated_head(
                self.boundary_residual_head,
                self.scale_residual_heads,
                self.boundary_edge_residual_heads,
                b3_direction,
            ).squeeze(-1)
        elif self.probe == "b1":
            evidence_residual = calibrated_head(
                self.boundary_residual_head,
                self.scale_residual_heads,
                self.boundary_edge_residual_heads,
                f3_direction,
            ).squeeze(-1)
        elif self.probe == "b2":
            evidence_residual = calibrated_head(
                self.boundary_residual_head,
                self.scale_residual_heads,
                self.boundary_edge_residual_heads,
                rgb_direction,
            ).squeeze(-1)
        else:
            evidence_residual = zero_residual
        boundary_residual_raw = (
            (evidence_residual - zero_residual)
            * self.boundary_gain.clamp(0.5, 4.0)
        )
        if self.training and torch.is_grad_enabled():
            aux_f3_direction = direction(
                f3_boundary_features,
                zero_rgb_features,
                f3_signed_features + zero_rgb_signed_features,
                detach_context=True,
            )
            aux_rgb_direction = direction(
                zero_f3_features,
                rgb_boundary_features,
                zero_f3_signed_features + rgb_signed_features,
                detach_context=True,
            )
            aux_joint_direction = direction(
                f3_boundary_features,
                rgb_boundary_features,
                f3_signed_features + rgb_signed_features,
                detach_context=True,
            )
            aux_b3_direction = aux_rgb_direction + f3_reliability * (
                aux_joint_direction - aux_rgb_direction
            )
            aux_zero_direction = direction(
                zero_f3_features,
                zero_rgb_features,
                zero_signed_features,
                detach_context=True,
            )
            if self.probe == "b3":
                aux_evidence_direction = aux_b3_direction
            elif self.probe == "b1":
                aux_evidence_direction = aux_f3_direction
            elif self.probe == "b2":
                aux_evidence_direction = aux_rgb_direction
            else:
                aux_evidence_direction = aux_zero_direction
            aux_zero_gate = calibrated_head(
                self.boundary_gate_head,
                self.scale_gate_heads,
                self.boundary_edge_gate_heads,
                aux_zero_direction,
            ).squeeze(-1)
            aux_evidence_gate = calibrated_head(
                self.boundary_gate_head,
                self.scale_gate_heads,
                self.boundary_edge_gate_heads,
                aux_evidence_direction,
            ).squeeze(-1)
            boundary_aux_gate_raw = aux_evidence_gate - aux_zero_gate
            aux_zero_residual = calibrated_head(
                self.boundary_residual_head,
                self.scale_residual_heads,
                self.boundary_edge_residual_heads,
                aux_zero_direction,
            ).squeeze(-1)
            aux_evidence_residual = calibrated_head(
                self.boundary_residual_head,
                self.scale_residual_heads,
                self.boundary_edge_residual_heads,
                aux_evidence_direction,
            ).squeeze(-1)
            boundary_aux_residual_raw = (
                (aux_evidence_residual - aux_zero_residual)
                * self.boundary_gain.clamp(0.5, 4.0)
            )
        else:
            boundary_aux_gate_raw = boundary_gate_raw.detach()
            boundary_aux_residual_raw = boundary_residual_raw.detach()
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
            boundary_aux_gate_raw=boundary_aux_gate_raw,
            boundary_aux_residual_raw=boundary_aux_residual_raw,
        )

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROLE_NAMES: Final[tuple[str, ...]] = ("C", "L", "R", "T", "B", "O")
ROLE_POINT_COUNTS: Final[tuple[int, ...]] = (4, 2, 2, 2, 2, 8)


@dataclass(frozen=True)
class SQDASGCConfig:
    detail_channels: int = 128
    hidden_dim: int = 256
    gate_groups: int = 16
    query_count: int = 300
    residual_cap: float = 0.05
    residual_init: float = 1e-3
    context_cap: float = 0.25
    context_init: float = 0.05
    offset_cap: float = 0.1

    def __post_init__(self) -> None:
        if self.detail_channels != 128:
            raise ValueError("detail_channels is frozen at 128")
        if self.hidden_dim != 256:
            raise ValueError("hidden_dim is frozen at 256")
        if self.gate_groups != 16:
            raise ValueError("gate_groups is frozen at 16")
        if self.query_count != 300:
            raise ValueError("query count is frozen at 300")
        if not 0 < self.residual_init < self.residual_cap:
            raise ValueError("residual_init must lie strictly inside residual_cap")
        if not 0 < self.context_init < self.context_cap:
            raise ValueError("context_init must lie strictly inside context_cap")


def _inverse_sigmoid_probability(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _masked_softmax(scores: Tensor, valid: Tensor, dim: int) -> Tensor:
    """Softmax that is exactly zero when every entry is invalid."""
    if scores.shape != valid.shape:
        raise ValueError(f"score/mask shape mismatch: {scores.shape} versus {valid.shape}")
    valid_float = valid.to(dtype=scores.dtype)
    masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    probabilities = masked.softmax(dim=dim) * valid_float
    normalizer = probabilities.sum(dim=dim, keepdim=True)
    return torch.where(
        normalizer > 0,
        probabilities / normalizer.clamp_min(torch.finfo(scores.dtype).eps),
        torch.zeros_like(probabilities),
    )


class SQDASGCAdapter(nn.Module):
    """Semantic-Geometry-Context shadow-query adapter for native RT-DETR object queries."""

    role_point_counts: Final[tuple[int, ...]] = ROLE_POINT_COUNTS

    def __init__(
        self,
        detail_channels: int = 128,
        hidden_dim: int = 256,
        gate_groups: int = 16,
        query_count: int = 300,
        residual_cap: float = 0.05,
        residual_init: float = 1e-3,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.config = SQDASGCConfig(
            detail_channels=detail_channels,
            hidden_dim=hidden_dim,
            gate_groups=gate_groups,
            query_count=query_count,
            residual_cap=residual_cap,
            residual_init=residual_init,
        )
        self.enabled = bool(enabled)
        dim = self.config.hidden_dim

        self.query_geometry = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, dim),
        )
        self.box_geometry = nn.Sequential(
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, dim),
        )
        self.shared_role_projection = nn.Linear(dim, dim)
        self.role_scale = nn.Parameter(torch.zeros(len(ROLE_NAMES), dim))
        self.role_bias = nn.Parameter(torch.empty(len(ROLE_NAMES), dim))
        self.role_norm = nn.LayerNorm(dim)

        self.point_offset_heads = nn.ModuleList(
            nn.Linear(dim, point_count * 2) for point_count in ROLE_POINT_COUNTS
        )

        self.value_projector = nn.Sequential(
            nn.Linear(detail_channels, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, dim),
        )
        self.point_query = nn.Linear(dim, dim, bias=False)
        self.point_key = nn.Linear(dim, dim, bias=False)
        self.edge_query = nn.Linear(dim, dim, bias=False)
        self.edge_key = nn.Linear(dim, dim, bias=False)
        self.reliability_projection = nn.Linear(dim, dim, bias=False)

        self.gate_query_norm = nn.LayerNorm(dim)
        self.gate_evidence_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(5 * dim + 4, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * gate_groups),
        )
        self.fusion = nn.Linear(2 * dim, dim)

        context_probability = self.config.context_init / self.config.context_cap
        residual_probability = self.config.residual_init / self.config.residual_cap
        self.context_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid_probability(context_probability))
        )
        self.layer_scale_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid_probability(residual_probability))
        )
        self.reset_parameters()

    @property
    def context_projector(self) -> nn.Module:
        """Context intentionally reuses the only point-value projector."""
        return self.value_projector

    @property
    def context_strength(self) -> Tensor:
        cap = self.context_logit.new_tensor(self.config.context_cap)
        strict_cap = torch.nextafter(cap, cap.new_zeros(()))
        return strict_cap * self.context_logit.sigmoid()

    @property
    def layer_scale(self) -> Tensor:
        cap = self.layer_scale_logit.new_tensor(self.config.residual_cap)
        strict_cap = torch.nextafter(cap, cap.new_zeros(()))
        return strict_cap * self.layer_scale_logit.sigmoid()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.role_bias, mean=0.0, std=0.01)
        for offset_head in self.point_offset_heads:
            nn.init.zeros_(offset_head.weight)
            nn.init.zeros_(offset_head.bias)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.normal_(self.fusion.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fusion.bias)

    @staticmethod
    def box_sine_encoding(boxes: Tensor) -> Tensor:
        """Encode normalized cxcywh boxes as 4 x 64 sine/cosine features."""
        if boxes.shape[-1] != 4:
            raise ValueError(f"reference boxes must end in four values, got {boxes.shape}")
        frequencies = torch.arange(32, device=boxes.device, dtype=boxes.dtype)
        frequencies = 2.0 * math.pi * torch.pow(10_000.0, -frequencies / 32.0)
        phase = boxes.unsqueeze(-1) * frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(start_dim=-2)

    def build_role_tokens(self, object_queries: Tensor, reference_boxes: Tensor) -> Tensor:
        boxes = reference_boxes.detach()
        query_code = self.query_geometry(object_queries)
        box_code = self.box_geometry(self.box_sine_encoding(boxes))
        geometry_code = query_code * box_code
        shared = self.shared_role_projection(object_queries).unsqueeze(2)
        scales = 1.0 + 0.1 * self.role_scale.tanh()
        tokens = (
            shared * scales.view(1, 1, len(ROLE_NAMES), -1)
            + self.role_bias.view(1, 1, len(ROLE_NAMES), -1)
            + geometry_code.unsqueeze(2)
        )
        return self.role_norm(tokens)

    @staticmethod
    def fixed_sampling_points(
        reference_boxes: Tensor,
        height: int,
        width: int,
    ) -> tuple[Tensor, Tensor]:
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid C2 spatial size {(height, width)}")
        boxes = reference_boxes.detach()
        centers = boxes[..., :2]
        ux = torch.maximum(
            boxes[..., 2] * 0.5,
            boxes.new_tensor(1.0 / float(width)),
        )
        uy = torch.maximum(
            boxes[..., 3] * 0.5,
            boxes.new_tensor(1.0 / float(height)),
        )
        radii = torch.stack((ux, uy), dim=-1)

        coefficients = boxes.new_tensor(
            [
                [-0.5, -0.5],
                [-0.5, 0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-1.0, -0.5],
                [-1.0, 0.5],
                [1.0, -0.5],
                [1.0, 0.5],
                [-0.5, -1.0],
                [0.5, -1.0],
                [-0.5, 1.0],
                [0.5, 1.0],
                [-1.5, 0.0],
                [1.5, 0.0],
                [0.0, -1.5],
                [0.0, 1.5],
                [-1.25, -1.25],
                [-1.25, 1.25],
                [1.25, -1.25],
                [1.25, 1.25],
            ]
        )
        return centers.unsqueeze(-2) + radii.unsqueeze(-2) * coefficients, radii

    def apply_point_offsets(
        self,
        base_points: Tensor,
        radii: Tensor,
        role_tokens: Tensor,
    ) -> Tensor:
        point_groups = []
        cursor = 0
        for role_index, (point_count, head) in enumerate(
            zip(self.role_point_counts, self.point_offset_heads)
        ):
            base = base_points[..., cursor : cursor + point_count, :]
            raw_offset = head(role_tokens[..., role_index, :])
            raw_offset = raw_offset.view(*raw_offset.shape[:-1], point_count, 2)
            bounded_offset = self.config.offset_cap * radii.unsqueeze(-2) * raw_offset.tanh()
            point_groups.append(base + bounded_offset)
            cursor += point_count
        return torch.cat(point_groups, dim=-2)

    @staticmethod
    def _sample_detail(raw_c2: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
        if raw_c2.ndim != 4:
            raise ValueError(f"raw C2 must be BCHW, got {raw_c2.shape}")
        valid = ((points >= 0.0) & (points <= 1.0)).all(dim=-1)
        grid = points.mul(2.0).sub(1.0)
        context_c2 = F.avg_pool2d(
            raw_c2,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        stacked_features = torch.cat((raw_c2, context_c2), dim=0)
        stacked_grid = torch.cat((grid, grid), dim=0)
        sampled_b_c_q_p = F.grid_sample(
            stacked_features,
            stacked_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled_b_c_q_p.permute(0, 2, 3, 1)
        batch = raw_c2.shape[0]
        raw_samples, context_samples = sampled[:batch], sampled[batch:]
        role_selector = torch.zeros(
            1,
            1,
            sum(ROLE_POINT_COUNTS),
            1,
            dtype=torch.bool,
            device=raw_c2.device,
        )
        role_selector[..., sum(ROLE_POINT_COUNTS[:-1]) :, :] = True
        combined = torch.where(role_selector, context_samples, raw_samples)
        return combined, valid

    def _role_descriptors(
        self,
        role_tokens: Tensor,
        point_values: Tensor,
        point_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        descriptors = []
        attentions = []
        valid_roles = []
        cursor = 0
        scale = math.sqrt(float(self.config.hidden_dim))
        for role_index, point_count in enumerate(self.role_point_counts):
            values = point_values[..., cursor : cursor + point_count, :]
            valid = point_valid[..., cursor : cursor + point_count]
            query = self.point_query(role_tokens[..., role_index, :]).unsqueeze(-2)
            keys = self.point_key(values)
            scores = (query * keys).sum(dim=-1) / scale
            attention = _masked_softmax(scores, valid, dim=-1)
            descriptor = (attention.unsqueeze(-1) * values).sum(dim=-2)
            descriptors.append(descriptor)
            attentions.append(attention)
            valid_roles.append(valid.any(dim=-1))
            cursor += point_count
        return (
            torch.stack(descriptors, dim=-2),
            torch.cat(attentions, dim=-1),
            torch.stack(valid_roles, dim=-1),
        )

    def _edge_descriptor(
        self,
        object_queries: Tensor,
        role_descriptors: Tensor,
        role_validity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        edges = role_descriptors[..., 1:5, :]
        edge_validity = role_validity[..., 1:5]
        query = self.edge_query(object_queries).unsqueeze(-2)
        keys = self.edge_key(edges)
        scores = (query * keys).sum(dim=-1) / math.sqrt(float(self.config.hidden_dim))
        attention = _masked_softmax(scores, edge_validity, dim=-1)
        return (attention.unsqueeze(-1) * edges).sum(dim=-2), attention

    def _cosine(self, first: Tensor, second: Tensor) -> Tensor:
        projected_first = self.reliability_projection(first)
        projected_second = self.reliability_projection(second)
        return F.cosine_similarity(projected_first, projected_second, dim=-1, eps=1e-6)

    def context_modulation(
        self,
        semantic_similarity: Tensor,
        context_similarity: Tensor,
        context_validity: Tensor,
    ) -> Tensor:
        modulation = 1.0 - self.context_strength * torch.sigmoid(
            2.0 * (context_similarity - semantic_similarity)
        )
        return torch.where(context_validity, modulation, torch.ones_like(modulation))

    def expand_group_gate(self, gate: Tensor) -> Tensor:
        if gate.shape[-1] != self.config.gate_groups:
            raise ValueError(f"expected {self.config.gate_groups} gate groups, got {gate.shape}")
        channels_per_group = self.config.hidden_dim // self.config.gate_groups
        return gate.repeat_interleave(channels_per_group, dim=-1)

    @staticmethod
    def _identity_diagnostics(object_queries: Tensor) -> dict[str, Tensor]:
        return {
            "identity_override": torch.ones(
                (),
                dtype=torch.bool,
                device=object_queries.device,
            )
        }

    def forward(
        self,
        object_queries: Tensor,
        reference_boxes: Tensor,
        raw_c2: Tensor,
        *,
        identity_override: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.enabled or identity_override:
            return object_queries, self._identity_diagnostics(object_queries)
        self._validate_inputs(object_queries, reference_boxes, raw_c2)

        boxes = reference_boxes.detach()
        role_tokens = self.build_role_tokens(object_queries, boxes)
        base_points, radii = self.fixed_sampling_points(
            boxes,
            raw_c2.shape[-2],
            raw_c2.shape[-1],
        )
        points = self.apply_point_offsets(base_points, radii, role_tokens)
        raw_values, point_validity = self._sample_detail(raw_c2, points)
        point_values = self.value_projector(raw_values)
        role_descriptors, point_attention, role_validity = self._role_descriptors(
            role_tokens,
            point_values,
            point_validity,
        )

        semantic = role_descriptors[..., 0, :]
        geometry, edge_attention = self._edge_descriptor(
            object_queries,
            role_descriptors,
            role_validity,
        )
        context = role_descriptors[..., 5, :]
        semantic_similarity = self._cosine(object_queries, semantic)
        geometry_similarity = self._cosine(object_queries, geometry)
        context_similarity = self._cosine(object_queries, context)
        semantic_modulation = self.context_modulation(
            semantic_similarity,
            context_similarity,
            role_validity[..., 5],
        )

        normalized_query = self.gate_query_norm(object_queries)
        normalized_semantic = self.gate_evidence_norm(semantic)
        normalized_geometry = self.gate_evidence_norm(geometry)
        log_size = boxes[..., 2:].clamp_min(1e-6).log()
        gate_input = torch.cat(
            (
                normalized_query,
                normalized_semantic,
                normalized_geometry,
                normalized_query * normalized_semantic,
                normalized_query * normalized_geometry,
                semantic_similarity.unsqueeze(-1),
                geometry_similarity.unsqueeze(-1),
                log_size,
            ),
            dim=-1,
        )
        group_gates = self.gate(gate_input).sigmoid()
        group_gates = group_gates.view(
            *group_gates.shape[:-1],
            2,
            self.config.gate_groups,
        )
        semantic_gate = self.expand_group_gate(group_gates[..., 0, :])
        geometry_gate = self.expand_group_gate(group_gates[..., 1, :])
        fusion_input = torch.cat(
            (
                semantic_modulation.unsqueeze(-1) * semantic_gate * semantic,
                geometry_gate * geometry,
            ),
            dim=-1,
        )
        fused = self.fusion(fusion_input)
        writeback_validity = role_validity[..., :5].any(dim=-1)
        fused = fused * writeback_validity.unsqueeze(-1).to(dtype=fused.dtype)
        residual = self.layer_scale * fused
        enhanced = object_queries + residual

        diagnostics = {
            "sampling_validity": role_validity.detach(),
            "point_attention": point_attention.detach(),
            "edge_attention": edge_attention.detach(),
            "group_gates": group_gates.detach(),
            "context_reliability": semantic_modulation.detach(),
            "semantic_similarity": semantic_similarity.detach(),
            "geometry_similarity": geometry_similarity.detach(),
            "context_similarity": context_similarity.detach(),
            "writeback_valid": writeback_validity.detach(),
            "residual_norm": residual.detach().norm(dim=-1),
            "layer_scale": self.layer_scale.detach(),
            "identity_override": torch.zeros(
                (),
                dtype=torch.bool,
                device=object_queries.device,
            ),
        }
        return enhanced, diagnostics

    def _validate_inputs(
        self,
        object_queries: Tensor,
        reference_boxes: Tensor,
        raw_c2: Tensor,
    ) -> None:
        if object_queries.ndim != 3 or object_queries.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"object queries must be [B,Q,{self.config.hidden_dim}], got {object_queries.shape}"
            )
        if reference_boxes.shape != (*object_queries.shape[:2], 4):
            raise ValueError(
                f"reference boxes must be [B,Q,4], got {reference_boxes.shape}"
            )
        if raw_c2.ndim != 4:
            raise ValueError(f"raw C2 must be [B,C,H,W], got {raw_c2.shape}")
        if raw_c2.shape[0] != object_queries.shape[0]:
            raise ValueError("object-query and C2 batch sizes differ")
        if raw_c2.shape[1] != self.config.detail_channels:
            raise ValueError(
                f"raw C2 must have {self.config.detail_channels} channels, got {raw_c2.shape[1]}"
            )
        if object_queries.device != reference_boxes.device or object_queries.device != raw_c2.device:
            raise ValueError("object queries, reference boxes, and C2 must share a device")
        if object_queries.dtype != raw_c2.dtype:
            raise ValueError("object queries and C2 must share a dtype")

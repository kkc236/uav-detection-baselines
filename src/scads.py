"""Scale-conditioned adaptive support primitives for FDR.

SCADS keeps the pinned D-FINE logits and box transform intact. It learns a
query-level convex combination of fixed non-uniform supports and uses that
same support throughout all six decoder layers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.fdr_math import REG_MAX, REG_SCALE, UP, weighting_function


DEFAULT_SUPPORT_UPS = (0.25, UP, 1.0)
DEFAULT_ROUTER_HIDDEN = 64
DEFAULT_ROUTER_TEMPERATURE = 1.0
DEFAULT_ROUTER_SEED = 20_000
DEFAULT_ROUTE_BIAS = (-4.0, 4.0, -4.0)


def build_support_projects(
    support_ups: Sequence[float] = DEFAULT_SUPPORT_UPS,
    *,
    reg_max: int = REG_MAX,
    reg_scale: float = REG_SCALE,
) -> Tensor:
    """Build ordered fixed support projects without trainable parameters."""

    values = tuple(float(value) for value in support_ups)
    if len(values) < 2 or any(value <= 0 for value in values):
        raise ValueError("SCADS requires at least two positive support up values")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("SCADS support up values must be strictly increasing")
    if not any(value == float(UP) for value in values):
        raise ValueError(f"SCADS supports must include the FDR base up={UP}")
    projects = torch.stack(
        [
            weighting_function(
                reg_max,
                torch.tensor([value], dtype=torch.float32),
                torch.tensor([reg_scale], dtype=torch.float32),
            )
            for value in values
        ]
    )
    if projects.shape != (len(values), reg_max + 1):
        raise RuntimeError("SCADS support project shape is invalid")
    if not torch.all(projects[:, 1:] >= projects[:, :-1]):
        raise RuntimeError("SCADS support projects must be monotonic")
    return projects


class ScaleConditionedSupportRouter(nn.Module):
    """Predict one support mixture per query from detached scale evidence."""

    def __init__(
        self,
        hidden_dim: int,
        router_hidden: int = DEFAULT_ROUTER_HIDDEN,
        num_supports: int = len(DEFAULT_SUPPORT_UPS),
        *,
        temperature: float = DEFAULT_ROUTER_TEMPERATURE,
        private_seed: int = DEFAULT_ROUTER_SEED,
        base_support_index: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or router_hidden <= 0:
            raise ValueError("SCADS router dimensions must be positive")
        if num_supports < 2:
            raise ValueError("SCADS router requires at least two supports")
        if temperature <= 0:
            raise ValueError("SCADS router temperature must be positive")
        if not 0 <= base_support_index < num_supports:
            raise ValueError("SCADS base support index is out of range")

        self.hidden_dim = int(hidden_dim)
        self.router_hidden = int(router_hidden)
        self.num_supports = int(num_supports)
        self.temperature = float(temperature)
        self.base_support_index = int(base_support_index)

        with torch.device("meta"):
            hidden_norm = nn.LayerNorm(self.hidden_dim)
            input_layer = nn.Linear(self.hidden_dim + 4, self.router_hidden)
            output_layer = nn.Linear(self.router_hidden, self.num_supports)
        self.hidden_norm = hidden_norm.to_empty(device=torch.device("cpu"))
        self.input_layer = input_layer.to_empty(device=torch.device("cpu"))
        self.output_layer = output_layer.to_empty(device=torch.device("cpu"))
        self._initialize(private_seed)

    def _initialize(self, private_seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(private_seed))
        nn.init.ones_(self.hidden_norm.weight)
        nn.init.zeros_(self.hidden_norm.bias)
        nn.init.kaiming_uniform_(
            self.input_layer.weight,
            a=math.sqrt(5),
            generator=generator,
        )
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.input_layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(
            self.input_layer.bias,
            -bound,
            bound,
            generator=generator,
        )
        nn.init.zeros_(self.output_layer.weight)
        with torch.no_grad():
            self.output_layer.bias.fill_(-4.0)
            self.output_layer.bias[self.base_support_index] = 4.0

    @staticmethod
    def geometry_features(pre_boxes: Tensor, eps: float = 1e-6) -> Tensor:
        """Return absolute scale/aspect evidence without scale-normalizing it."""

        if pre_boxes.ndim < 2 or pre_boxes.shape[-1] != 4:
            raise ValueError("pre_boxes must end in [cx,cy,w,h]")
        width = pre_boxes[..., 2].detach().clamp_min(eps)
        height = pre_boxes[..., 3].detach().clamp_min(eps)
        log_width = width.log()
        log_height = height.log()
        geometry = torch.stack(
            [
                log_width,
                log_height,
                log_width + log_height,
                log_width - log_height,
            ],
            dim=-1,
        )
        return geometry.clamp(min=-12.0, max=4.0)

    def forward(self, hidden: Tensor, pre_boxes: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.shape[:-1] != pre_boxes.shape[:-1]:
            raise ValueError("SCADS hidden and preliminary boxes must align")
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError("SCADS hidden dimension does not match the router")
        hidden_evidence = self.hidden_norm(hidden.detach())
        geometry = self.geometry_features(pre_boxes)
        features = torch.cat([hidden_evidence, geometry], dim=-1)
        logits = self.output_layer(F.silu(self.input_layer(features)))
        weights = F.softmax(logits / self.temperature, dim=-1)
        return logits, weights


class AdaptiveIntegral(nn.Module):
    """Decode FDR logits with one convex support project per query."""

    def __init__(
        self,
        support_ups: Sequence[float] = DEFAULT_SUPPORT_UPS,
        *,
        reg_max: int = REG_MAX,
        reg_scale: float = REG_SCALE,
    ) -> None:
        super().__init__()
        self.reg_max = int(reg_max)
        self.support_ups = tuple(float(value) for value in support_ups)
        self.register_buffer(
            "projects",
            build_support_projects(
                self.support_ups,
                reg_max=self.reg_max,
                reg_scale=reg_scale,
            ),
        )

    def effective_project(self, route_weights: Tensor) -> Tensor:
        if route_weights.ndim < 1 or route_weights.shape[-1] != len(self.support_ups):
            raise ValueError("SCADS route weights have the wrong support dimension")
        if not torch.isfinite(route_weights).all():
            raise ValueError("SCADS route weights must be finite")
        return torch.einsum(
            "...k,kn->...n",
            route_weights,
            self.projects.to(device=route_weights.device, dtype=route_weights.dtype),
        )

    def forward(self, logits: Tensor, route_weights: Tensor) -> Tensor:
        expected = 4 * (self.reg_max + 1)
        if logits.ndim == 0 or logits.shape[-1] != expected:
            raise ValueError(f"SCADS logits must end in {expected} values")
        if route_weights.shape[:-1] != logits.shape[:-1]:
            raise ValueError("SCADS route weights must align with logits")
        probabilities = F.softmax(
            logits.reshape(*logits.shape[:-1], 4, self.reg_max + 1),
            dim=-1,
        )
        project = self.effective_project(route_weights)
        return torch.einsum("...en,...n->...e", probabilities, project)


def continuous_edge_offsets(
    points: Tensor,
    targets_xyxy: Tensor,
    *,
    reg_scale: float | Tensor = REG_SCALE,
) -> Tensor:
    """Encode GT edges as continuous offsets before discrete binning."""

    if points.shape != targets_xyxy.shape or points.ndim != 2 or points.shape[-1] != 4:
        raise ValueError("points and targets_xyxy must both have shape [N,4]")
    scale = torch.as_tensor(reg_scale, dtype=points.dtype, device=points.device).abs()
    width_step = points[:, 2] / scale + 1e-16
    height_step = points[:, 3] / scale + 1e-16
    left = (points[:, 0] - targets_xyxy[:, 0]) / width_step - 0.5 * scale
    top = (points[:, 1] - targets_xyxy[:, 1]) / height_step - 0.5 * scale
    right = (targets_xyxy[:, 2] - points[:, 0]) / width_step - 0.5 * scale
    bottom = (targets_xyxy[:, 3] - points[:, 1]) / height_step - 0.5 * scale
    return torch.stack([left, top, right, bottom], dim=-1)


def translate_with_project(
    values: Tensor,
    projects: Tensor,
    *,
    reg_max: int = REG_MAX,
    eps: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode four offsets against detached query-specific monotonic projects."""

    if values.ndim != 2 or values.shape[-1] != 4:
        raise ValueError("SCADS offset values must have shape [N,4]")
    if projects.shape != (values.shape[0], reg_max + 1):
        raise ValueError("SCADS projects must have shape [N,reg_max+1]")
    project = projects.detach().to(device=values.device, dtype=values.dtype)
    if not torch.all(project[:, 1:] >= project[:, :-1]):
        raise ValueError("SCADS effective projects must be monotonic")

    flattened = values.detach().reshape(-1)
    expanded = (
        project[:, None, :]
        .expand(-1, 4, -1)
        .reshape(-1, reg_max + 1)
        .contiguous()
    )
    insertion = torch.searchsorted(
        expanded,
        flattened.unsqueeze(-1).contiguous(),
        right=True,
    ).squeeze(-1)
    left_index = insertion - 1
    indices = left_index.to(flattened.dtype)
    weight_right = torch.zeros_like(flattened)
    weight_left = torch.zeros_like(flattened)

    valid = (left_index >= 0) & (left_index < reg_max)
    safe_left = left_index.clamp(min=0, max=reg_max - 1)
    left_value = expanded.gather(1, safe_left.unsqueeze(1)).squeeze(1)
    right_value = expanded.gather(1, (safe_left + 1).unsqueeze(1)).squeeze(1)
    denominator = (right_value - left_value).abs().clamp_min(1e-16)
    weight_right[valid] = (
        (flattened[valid] - left_value[valid]).abs() / denominator[valid]
    )
    weight_left[valid] = 1.0 - weight_right[valid]

    below = left_index < 0
    weight_left[below] = 1.0
    indices[below] = 0.0
    above = left_index >= reg_max
    weight_right[above] = 1.0
    indices[above] = reg_max - eps
    return indices.detach(), weight_right.detach(), weight_left.detach()


def smallest_covering_support(
    offsets: Tensor,
    projects: Tensor,
    *,
    margin_ratio: float = 0.02,
) -> tuple[Tensor, Tensor]:
    """Choose the narrowest support that covers all four target offsets."""

    if offsets.ndim != 2 or offsets.shape[-1] != 4:
        raise ValueError("SCADS offsets must have shape [N,4]")
    if projects.ndim != 2 or projects.shape[1] < 2:
        raise ValueError("SCADS project bank must have shape [K,bins]")
    if not 0 <= margin_ratio < 0.5:
        raise ValueError("SCADS margin ratio must be in [0,0.5)")
    bank = projects.detach().to(device=offsets.device, dtype=offsets.dtype)
    span = bank[:, -1] - bank[:, 0]
    lower = bank[:, 0] + margin_ratio * span
    upper = bank[:, -1] - margin_ratio * span
    fits = (
        (offsets[:, None, :] >= lower[None, :, None])
        & (offsets[:, None, :] <= upper[None, :, None])
    ).all(dim=-1)
    has_fit = fits.any(dim=1)
    target = fits.to(torch.int64).argmax(dim=1)
    target = torch.where(
        has_fit,
        target,
        torch.full_like(target, bank.shape[0] - 1),
    )
    overflow = ~fits[:, -1]
    return target.detach(), overflow.detach()


__all__ = [
    "AdaptiveIntegral",
    "DEFAULT_ROUTER_HIDDEN",
    "DEFAULT_ROUTER_SEED",
    "DEFAULT_ROUTER_TEMPERATURE",
    "DEFAULT_SUPPORT_UPS",
    "ScaleConditionedSupportRouter",
    "build_support_projects",
    "continuous_edge_offsets",
    "smallest_covering_support",
    "translate_with_project",
]

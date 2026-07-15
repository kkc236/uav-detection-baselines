from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class P3SamplingStatistics:
    center: torch.Tensor
    extent: torch.Tensor
    p3_mass: torch.Tensor
    valid: torch.Tensor
    p3_shape: tuple[int, int]


class P3SamplingProbe:
    """Observe final-decoder P3 deformable sampling without changing attention output."""

    def __init__(self, cross_attention: nn.Module | None = None, *, epsilon: float = 1e-6) -> None:
        self.cross_attention = cross_attention
        self.epsilon = float(epsilon)
        self.last_statistics: P3SamplingStatistics | None = None
        self._hook: torch.utils.hooks.RemovableHandle | None = None

    def attach(self, decoder: nn.Module) -> None:
        self.remove()
        self.cross_attention = decoder.layers[-1].cross_attn
        self._hook = self.cross_attention.register_forward_pre_hook(self._capture_hook)

    def remove(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def clear(self) -> None:
        self.last_statistics = None

    def _capture_hook(self, module: nn.Module, args: tuple[object, ...]) -> None:
        if module.training:
            self.capture(*args)

    def capture(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list | torch.Tensor,
        value_mask: torch.Tensor | None = None,
    ) -> None:
        del value, value_mask
        attention = self.cross_attention
        if attention is None:
            raise RuntimeError("P3SamplingProbe must be attached to a cross-attention module before capture")
        if not attention.training:
            return

        batch_size, query_count = query.shape[:2]
        n_heads = int(attention.n_heads)
        n_levels = int(attention.n_levels)
        n_points = int(attention.n_points)
        total_points = n_levels * n_points

        with torch.autocast(device_type=query.device.type, enabled=False):
            query_fp32 = query.float()
            references_fp32 = reference_boxes.float()
            offsets = attention.sampling_offsets(query_fp32).view(
                batch_size, query_count, n_heads, total_points, 2
            )
            logits = attention.attention_weights(query_fp32).view(
                batch_size, query_count, n_heads, total_points
            )
            weights = torch.softmax(logits, dim=-1)

            if references_fp32.shape[-1] == 2:
                shapes = torch.as_tensor(value_shapes, dtype=torch.float32, device=query.device)
                normalizer = shapes.flip(-1)[:, None, :].expand(-1, n_points, -1).reshape(total_points, 2)
                locations = references_fp32[:, :, None, :, :] + offsets / normalizer
            elif references_fp32.shape[-1] == 4:
                locations = (
                    references_fp32[:, :, None, :, :2]
                    + offsets / n_points * references_fp32[:, :, None, :, 2:] * 0.5
                )
            else:
                raise ValueError("reference_boxes must end in either 2 or 4 coordinates")

            p3_locations = locations[..., :n_points, :]
            p3_weights = weights[..., :n_points]
            p3_mass = p3_weights.sum(dim=(-1, -2))
            denominator = p3_mass.clamp_min(self.epsilon)
            normalized = p3_weights / denominator[:, :, None, None]
            center = (normalized[..., None] * p3_locations).sum(dim=(-2, -3))
            variance = (
                normalized[..., None] * (p3_locations - center[:, :, None, None, :]).square()
            ).sum(dim=(-2, -3))
            extent = torch.sqrt(variance.clamp_min(0.0) + self.epsilon)

            finite = (
                torch.isfinite(p3_mass)
                & torch.isfinite(center).all(dim=-1)
                & torch.isfinite(extent).all(dim=-1)
            )
            valid = finite & (p3_mass > self.epsilon)
            p3_shape = tuple(int(x) for x in value_shapes[0])

        self.last_statistics = P3SamplingStatistics(
            center=center,
            extent=extent,
            p3_mass=p3_mass,
            valid=valid,
            p3_shape=(p3_shape[0], p3_shape[1]),
        )

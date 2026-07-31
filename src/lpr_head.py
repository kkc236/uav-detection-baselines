"""Localization-prior refinement components for RT-DETR.

The refiner is identity-initialized so adding it to a stock decoder does not
change any predicted box before training. Geometry is intentionally detached:
it conditions refinement without creating a second gradient path through the
stock box regression head.
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules.utils import inverse_sigmoid


def box_geometry_prior(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encode normalized ``(cx, cy, width, height)`` boxes into six priors."""
    detached = boxes.detach()
    center = detached[..., :2].mul(2).sub(1)
    width, height = detached[..., 2:].clamp_min(eps).unbind(-1)
    scale = torch.stack(
        (width.log(), height.log(), (width * height).log(), (width / height).log()),
        dim=-1,
    ).clamp(-12, 12)
    return torch.cat((center, scale), dim=-1)


class LocalizationPriorRefiner(nn.Module):
    """Predict a bounded geometry-aware residual around a stock decoder box."""

    def __init__(self, hidden_dim: int, seed: int, max_logit_delta: float = 0.5) -> None:
        super().__init__()
        # Deterministic private initialization must not perturb the parent
        # model's global parameter-initialization stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.query_path = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
            )
            self.geometry_path = nn.Sequential(nn.Linear(6, 16), nn.SiLU())
            self.residual_head = nn.Linear(80, 4)

        self.alpha = nn.Parameter(torch.zeros(()))
        self.max_logit_delta = float(max_logit_delta)

    def forward(self, hidden: torch.Tensor, stock_boxes: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry_path(box_geometry_prior(stock_boxes).to(hidden.dtype))
        features = torch.cat((self.query_path(hidden), geometry), dim=-1)
        residual = torch.tanh(self.residual_head(features))
        stock_logits = torch.logit(stock_boxes.clamp(1e-6, 1 - 1e-6))
        candidate = torch.sigmoid(stock_logits + self.max_logit_delta * residual)
        gate = 0.5 * torch.tanh(self.alpha)
        return (stock_boxes + gate * (candidate - stock_boxes)).clamp(1e-6, 1 - 1e-6)


class LPRDeformableTransformerDecoder(nn.Module):
    """Stock RT-DETR decoder trajectory with LPR applied only to output boxes."""

    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        eval_idx: int,
        max_logit_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx
        self.lpr_refiners = nn.ModuleList(
            LocalizationPriorRefiner(
                hidden_dim,
                seed=3407 + index,
                max_logit_delta=max_logit_delta,
            )
            for index in range(num_layers)
        )

    @classmethod
    def from_stock(
        cls,
        stock: nn.Module,
        max_logit_delta: float = 0.5,
    ) -> "LPRDeformableTransformerDecoder":
        """Wrap an existing decoder while reusing its exact decoder layers."""
        return cls(
            layers=stock.layers,
            hidden_dim=stock.hidden_dim,
            num_layers=stock.num_layers,
            eval_idx=stock.eval_idx,
            max_logit_delta=max_logit_delta,
        )

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode while keeping every internal reference on the stock path."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()

        for index, layer in enumerate(self.layers):
            output = layer(
                output,
                refer_bbox,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(refer_bbox),
            )
            bbox_delta = bbox_head[index](output)
            refined_bbox = torch.sigmoid(bbox_delta + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[index](output))
                stock_output_bbox = (
                    refined_bbox
                    if index == 0
                    else torch.sigmoid(bbox_delta + inverse_sigmoid(last_refined_bbox))
                )
                dec_bboxes.append(self.lpr_refiners[index](output, stock_output_bbox))
            elif index == self.eval_idx:
                dec_cls.append(score_head[index](output))
                dec_bboxes.append(self.lpr_refiners[index](output, refined_bbox))
                break

            # The next layer always receives the unrefined stock trajectory.
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

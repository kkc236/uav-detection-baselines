"""Detached quality-gated last-layer box refinement for RT-DETR."""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules.utils import inverse_sigmoid


def box_geometry_prior(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encode detached normalized boxes as stable center, scale, and aspect priors."""
    detached = boxes.detach()
    center = detached[..., :2].mul(2).sub(1)
    width, height = detached[..., 2:].clamp_min(eps).unbind(-1)
    scale = torch.stack(
        (width.log(), height.log(), (width * height).log(), (width / height).log()),
        dim=-1,
    ).clamp(-12, 12)
    return torch.cat((center, scale), dim=-1)


def detached_vfl_quality(stock_scores: torch.Tensor) -> torch.Tensor:
    """Return one detached Varifocal-score quality prior per query."""
    return stock_scores.detach().sigmoid().amax(dim=-1, keepdim=True)


class QualityGatedRefiner(nn.Module):
    """Predict a bounded per-query box residual without gradients to stock paths."""

    def __init__(self, hidden_dim: int, private_seed: int, max_logit_delta: float = 0.5) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(private_seed)
            self.query_path = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
            )
            self.geometry_path = nn.Sequential(nn.Linear(6, 16), nn.SiLU())
            self.gate_head = nn.Linear(80, 1)
            self.residual_head = nn.Linear(80, 4)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.max_logit_delta = float(max_logit_delta)
        self.last_quality: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        stock_boxes: torch.Tensor,
        stock_scores: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden.detach()
        stock_boxes = stock_boxes.detach()
        quality = detached_vfl_quality(stock_scores)
        geometry = box_geometry_prior(stock_boxes).to(dtype=hidden.dtype)
        features = torch.cat((self.query_path(hidden), self.geometry_path(geometry)), dim=-1)
        learned_gate = torch.sigmoid(self.gate_head(features))
        gate = (1 - quality.to(dtype=learned_gate.dtype)) * learned_gate
        residual = self.max_logit_delta * torch.tanh(self.residual_head(features))
        stock_logits = inverse_sigmoid(stock_boxes.clamp(1e-6, 1 - 1e-6))
        reconstructed_stock = torch.sigmoid(stock_logits)
        candidate = torch.sigmoid(stock_logits + gate * residual)
        refined = (stock_boxes + (candidate - reconstructed_stock)).clamp(1e-6, 1 - 1e-6)
        self.last_quality = quality.detach()
        self.last_gate = gate.detach()
        self.last_residual = residual.detach()
        return refined


class LPRGDeformableTransformerDecoder(nn.Module):
    """Preserve the stock decoder trajectory and expose one private side output."""

    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        eval_idx: int,
        private_seed: int,
        max_logit_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx
        self.lpr_g_refiner = QualityGatedRefiner(
            hidden_dim,
            private_seed=private_seed,
            max_logit_delta=max_logit_delta,
        )
        self.output_mode = "refined"
        self.last_stock_bboxes: torch.Tensor | None = None
        self.last_stock_scores: torch.Tensor | None = None
        self.last_refined_bboxes: torch.Tensor | None = None

    @classmethod
    def from_stock(
        cls,
        stock: nn.Module,
        private_seed: int = 10_000,
        max_logit_delta: float = 0.5,
    ) -> "LPRGDeformableTransformerDecoder":
        """Wrap a stock decoder while reusing its exact decoder layers."""
        return cls(
            layers=stock.layers,
            hidden_dim=stock.hidden_dim,
            num_layers=stock.num_layers,
            eval_idx=stock.eval_idx,
            private_seed=private_seed,
            max_logit_delta=max_logit_delta,
        )

    def set_output_mode(self, mode: str) -> None:
        """Choose stock or refined boxes for evaluation output."""
        if mode not in {"stock", "refined"}:
            raise ValueError(f"unsupported LPR-G output mode: {mode}")
        self.output_mode = mode

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
        """Decode with stock outputs and one detached last-layer refinement branch."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        self.last_stock_bboxes = None
        self.last_stock_scores = None
        self.last_refined_bboxes = None

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
                stock_score = score_head[index](output)
                stock_output_bbox = (
                    refined_bbox
                    if index == 0
                    else torch.sigmoid(bbox_delta + inverse_sigmoid(last_refined_bbox))
                )
                dec_cls.append(stock_score)
                dec_bboxes.append(stock_output_bbox)
                if index == self.num_layers - 1:
                    self.last_stock_bboxes = stock_output_bbox
                    self.last_stock_scores = stock_score
                    self.last_refined_bboxes = self.lpr_g_refiner(
                        output,
                        stock_output_bbox,
                        stock_score,
                    )
            elif index == self.eval_idx:
                stock_score = score_head[index](output)
                self.last_stock_bboxes = refined_bbox
                self.last_stock_scores = stock_score
                self.last_refined_bboxes = self.lpr_g_refiner(
                    output,
                    refined_bbox,
                    stock_score,
                )
                dec_cls.append(stock_score)
                dec_bboxes.append(
                    refined_bbox if self.output_mode == "stock" else self.last_refined_bboxes
                )
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

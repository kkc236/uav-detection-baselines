"""CSHC network primitives for pre-query high-resolution candidate generation.

The module intentionally changes neither RT-DETR's final decoder nor its final
post-processing.  It only supplies additional, learned C2 proposals to the
stock encoder proposal pool before the existing fixed Top-300 selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import RTDETRDecoder


@dataclass(frozen=True)
class SparseCandidates:
    """Learned C2 proposals, expressed in the same logit-box space as RT-DETR anchors."""

    tokens: Tensor
    anchor_logits: Tensor
    class_logits: Tensor
    objectness_logits: Tensor
    indices: Tensor


class C2CandidateFusion(nn.Module):
    """Small depthwise C2 fusion block used before sparse candidate selection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(out_channels)

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4:
            raise ValueError(f"C2CandidateFusion expects BCHW input, got {tuple(feature.shape)}")
        feature = F.silu(self.norm(self.depthwise(feature)))
        return F.silu(self.out_norm(self.pointwise(feature)))


class DySample(nn.Module):
    """Point-sampling dynamic upsampler using only regular PyTorch operators.

    Its zero-offset state is bilinear interpolation with ``align_corners=False``.
    Learned, group-wise offsets perturb this sampling lattice, so it stays an
    ordinary differentiable network layer and does not introduce custom CUDA ops.
    """

    def __init__(self, channels: int, scale: int = 2, groups: int = 4) -> None:
        super().__init__()
        if channels <= 0 or scale <= 0 or groups <= 0 or channels % groups:
            raise ValueError("channels must be positive and divisible by positive groups; scale must be positive")
        self.channels = int(channels)
        self.scale = int(scale)
        self.groups = int(groups)
        self.offset = nn.Conv2d(channels, 2 * groups * scale * scale, kernel_size=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4:
            raise ValueError(f"DySample expects BCHW input, got {tuple(feature.shape)}")
        batch, channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")

        out_height, out_width = height * self.scale, width * self.scale
        offsets = F.pixel_shuffle(self.offset(feature), self.scale)
        offsets = offsets.reshape(batch, self.groups, 2, out_height, out_width)
        offsets = 0.5 * torch.tanh(offsets)

        y = torch.arange(out_height, device=feature.device, dtype=feature.dtype)
        x = torch.arange(out_width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        base_x = (2.0 * (grid_x + 0.5) / out_width - 1.0).unsqueeze(0).unsqueeze(0)
        base_y = (2.0 * (grid_y + 0.5) / out_height - 1.0).unsqueeze(0).unsqueeze(0)
        sample_x = base_x + 2.0 * offsets[:, :, 0] / width
        sample_y = base_y + 2.0 * offsets[:, :, 1] / height
        grid = torch.stack((sample_x, sample_y), dim=-1).reshape(batch * self.groups, out_height, out_width, 2)

        grouped = feature.reshape(batch * self.groups, channels // self.groups, height, width)
        sampled = F.grid_sample(grouped, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.reshape(batch, channels, out_height, out_width)


class SparseC2CandidateGenerator(nn.Module):
    """Turns the strongest C2 locations into query tokens and tiny reference boxes."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        candidates: int,
        anchor_size: float,
        nc: int = 10,
    ) -> None:
        super().__init__()
        if min(channels, hidden_dim, candidates, nc) <= 0 or not 0.0 < anchor_size < 1.0:
            raise ValueError("channels, hidden_dim, candidates and nc must be positive; anchor_size must be in (0, 1)")
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.candidates = int(candidates)
        self.anchor_size = float(anchor_size)
        self.nc = int(nc)
        self.objectness = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.token_projection = nn.Sequential(nn.Linear(channels, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.class_head = nn.Linear(hidden_dim, nc)
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.objectness.bias, log(0.01 / 0.99))
        class_prior = log(0.01 / 0.99) / 80.0 * self.nc
        nn.init.constant_(self.class_head.bias, class_prior)
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    @staticmethod
    def _inverse_sigmoid(value: Tensor, eps: float = 1e-4) -> Tensor:
        value = value.clamp(min=eps, max=1.0 - eps)
        return torch.log(value / (1.0 - value))

    def forward(self, feature: Tensor) -> SparseCandidates:
        if feature.ndim != 4:
            raise ValueError(f"SparseC2CandidateGenerator expects BCHW input, got {tuple(feature.shape)}")
        batch, channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        cells = height * width
        if cells < self.candidates:
            raise ValueError(f"C2 map has {cells} cells but requires at least {self.candidates} candidates")

        objectness_logits = self.objectness(feature)
        _, indices = objectness_logits.flatten(1).topk(self.candidates, dim=1)
        flattened = feature.flatten(2).transpose(1, 2)
        selected = flattened.gather(1, indices.unsqueeze(-1).expand(-1, -1, channels))
        tokens = self.token_projection(selected)

        selected_objectness = objectness_logits.flatten(1).gather(1, indices).unsqueeze(-1)
        class_logits = self.class_head(tokens) + selected_objectness
        box_delta = self.box_head(tokens)

        center_x = (indices.remainder(width).to(feature.dtype) + 0.5) / width
        center_y = (torch.div(indices, width, rounding_mode="floor").to(feature.dtype) + 0.5) / height
        anchor = torch.stack(
            (
                center_x,
                center_y,
                torch.full_like(center_x, self.anchor_size),
                torch.full_like(center_y, self.anchor_size),
            ),
            dim=-1,
        )
        anchor_logits = self._inverse_sigmoid(anchor) + box_delta
        return SparseCandidates(tokens, anchor_logits, class_logits, objectness_logits, indices)


class CSHCRTDDETRDecoder(RTDETRDecoder):
    """RT-DETR decoder that adds sparse C2 proposals before its unchanged Top-300 query selection."""

    def __init__(
        self,
        nc: int = 80,
        ch: tuple[int, int, int, int] | list[int] = (64, 512, 1024, 2048),
        candidates: int = 512,
        anchor_size: float = 0.025,
        **kwargs,
    ) -> None:
        if len(ch) != 4:
            raise ValueError("CSHCRTDDETRDecoder requires [C2, F3, F4, F5] channel dimensions")
        self.c2_channels = int(ch[0])
        super().__init__(nc=nc, ch=tuple(ch[1:]), **kwargs)
        self.c2_candidates = SparseC2CandidateGenerator(
            channels=self.c2_channels,
            hidden_dim=self.hidden_dim,
            candidates=candidates,
            anchor_size=anchor_size,
            nc=nc,
        )
        self.last_candidates: SparseCandidates | None = None

    @staticmethod
    def _gather(values: Tensor, indices: Tensor) -> Tensor:
        return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

    def _get_cshc_decoder_input(
        self,
        c2_feature: Tensor,
        stock_features: list[Tensor],
        dn_embed: Tensor | None = None,
        dn_bbox: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[list[int]], Tensor]:
        """Build decoder inputs, selecting from stock and new C2 proposals together."""
        feats, shapes = self._get_encoder_input(stock_features)
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        stock_encoded = self.enc_output(self.valid_mask * feats)
        stock_scores = self.enc_score_head(stock_encoded)
        candidates = self.c2_candidates(c2_feature)
        self.last_candidates = candidates
        all_features = torch.cat((stock_encoded, candidates.tokens), dim=1)
        all_scores = torch.cat((stock_scores, candidates.class_logits), dim=1)
        all_anchors = torch.cat((self.anchors.expand(feats.shape[0], -1, -1), candidates.anchor_logits), dim=1)

        topk_indices = torch.topk(all_scores.max(-1).values, self.num_queries, dim=1).indices
        top_features = self._gather(all_features, topk_indices)
        top_anchors = self._gather(all_anchors, topk_indices)
        refer_bbox = self.enc_bbox_head(top_features) + top_anchors
        enc_bboxes = refer_bbox.sigmoid()
        enc_scores = self._gather(all_scores, topk_indices)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(feats.shape[0], 1, 1) if self.learnt_init_query else top_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_bbox is not None:
            refer_bbox = torch.cat((dn_bbox, refer_bbox), dim=1)
        if dn_embed is not None:
            embeddings = torch.cat((dn_embed, embeddings), dim=1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores, shapes, feats

    def forward(self, x: list[Tensor], batch: dict | None = None) -> tuple | Tensor:
        """Run normal RT-DETR decoding after adding C2 candidates to encoder proposal selection."""
        from ultralytics.models.utils.ops import get_cdn_group

        if len(x) != 4:
            raise ValueError(f"CSHCRTDDETRDecoder expects [C2, F3, F4, F5], got {len(x)} levels")
        c2_feature, *stock_features = x
        if self.training and batch is not None and "gt_groups" not in batch:
            # The production dataloader supplies this field.  Deriving it here keeps
            # direct module use and zero-GT unit tests on the same RT-DETR code path.
            batch = dict(batch)
            batch_indices = batch["batch_idx"].to(device=c2_feature.device, dtype=torch.long)
            batch["gt_groups"] = torch.bincount(batch_indices, minlength=c2_feature.shape[0]).tolist()
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        embed, refer_bbox, enc_bboxes, enc_scores, shapes, feats = self._get_cshc_decoder_input(
            c2_feature, stock_features, dn_embed, dn_bbox
        )
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        if self.training and dn_meta is None:
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        raw = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return raw
        prediction = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return prediction if self.export else (prediction, raw)

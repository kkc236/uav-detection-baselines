from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import RTDETRDecoder

from src.ebc_qp_config import EBCQPConfig
from src.ebc_qp_loss import P2Targets, compute_ebc_loss, compute_sparse_p2_loss, differentiable_zero
from src.ebc_qp_matching import assign_local_p2, match_centers_inside_boxes
from src.ebc_qp_queries import QuerySet, compete_queries, gather_query_set, stable_rank_indices


@dataclass
class EBCQPForwardState:
    stock_topk_indices: torch.Tensor
    p2_topk_indices: torch.Tensor
    final_sources: torch.Tensor
    p2_loss: torch.Tensor
    ebc_loss: torch.Tensor
    p2_entry_count: int
    ordinary_query_count: int
    encoder_aux_source_is_stock: bool
    competition_active: bool
    ebc_active: bool
    stock_boundary: torch.Tensor
    assigned_pairs: list[torch.Tensor]
    uncovered: list[torch.Tensor]
    stock_loss: torch.Tensor | None = None


class EBCQPDecoder(RTDETRDecoder):
    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (),
        *args,
        ebc_config: EBCQPConfig | None = None,
        **kwargs,
    ):
        if len(ch) < 2:
            raise ValueError("EBCQPDecoder requires C2 followed by stock decoder inputs")
        c2_channels, *stock_channels = ch
        super().__init__(nc=nc, ch=tuple(stock_channels), *args, **kwargs)
        self.ebc_config = ebc_config or EBCQPConfig()
        self.p2_adapter = nn.Sequential(
            nn.Conv2d(c2_channels, self.hidden_dim, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
        )
        self.p2_bbox_head = deepcopy(self.enc_bbox_head)
        self.ebc_epoch = 0
        self.ebc_enabled = True
        self.last_state: EBCQPForwardState | None = None

    @property
    def competition_active(self) -> bool:
        return self.ebc_enabled and self.ebc_epoch >= self.ebc_config.warmup_epochs

    def set_progress(self, epoch: int) -> None:
        self.ebc_epoch = int(epoch)

    def forward_with_state(self, x: list[torch.Tensor], batch: dict | None = None) -> EBCQPForwardState:
        self.forward(x, batch)
        if self.last_state is None:
            raise RuntimeError("EBC-QP forward state was not populated")
        return self.last_state

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        from ultralytics.models.utils.ops import get_cdn_group

        self.last_state = None
        if len(x) != 4:
            raise ValueError("EBCQPDecoder expects C2, P3, P4, and P5 inputs")

        feats, shapes, projected_p3 = self._project_stock_inputs(x[1:])
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
        stock, stock_indices = self._stock_query_set(feats, shapes)
        stock_boundary = stock.ranking_score[:, -1].detach()

        if self.ebc_enabled:
            p2_all, p2_top, p2_indices, p2_valid_mask, p2_shape = self._p2_query_sets(x[0], projected_p3)
            p2_loss, raw_ebc_loss, assigned_pairs, uncovered = self._training_losses(
                batch,
                stock,
                p2_all,
                p2_indices,
                p2_valid_mask,
                p2_shape,
                stock_boundary,
            )
        else:
            batch_size = feats.shape[0]
            p2_indices = torch.empty((batch_size, 0), dtype=torch.long, device=feats.device)
            p2_loss = differentiable_zero(feats)
            raw_ebc_loss = differentiable_zero(feats)
            assigned_pairs = [torch.empty((0, 2), dtype=torch.long, device=feats.device) for _ in range(batch_size)]
            uncovered = [torch.empty(0, dtype=torch.bool, device=feats.device) for _ in range(batch_size)]
            p2_top = None

        active = self.competition_active
        final_queries = compete_queries(stock, p2_top, budget=self.num_queries) if active else stock
        ebc_loss = raw_ebc_loss if active else differentiable_zero(raw_ebc_loss)
        p2_entry_count = int((final_queries.source == 1).sum())

        embeddings = (
            self.tgt_embed.weight.unsqueeze(0).repeat(feats.shape[0], 1, 1)
            if self.learnt_init_query
            else final_queries.features
        )
        refer_bbox = final_queries.reference_logits
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_bbox is not None:
            refer_bbox = torch.cat((dn_bbox, refer_bbox), dim=1)
        if dn_embed is not None:
            embeddings = torch.cat((dn_embed, embeddings), dim=1)

        dec_bboxes, dec_scores = self.decoder(
            embeddings,
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

        self.last_state = EBCQPForwardState(
            stock_topk_indices=stock_indices.detach(),
            p2_topk_indices=p2_indices.detach(),
            final_sources=final_queries.source.detach(),
            p2_loss=p2_loss,
            ebc_loss=ebc_loss,
            p2_entry_count=p2_entry_count,
            ordinary_query_count=final_queries.features.shape[1],
            encoder_aux_source_is_stock=True,
            competition_active=active,
            ebc_active=active,
            stock_boundary=stock_boundary,
            assigned_pairs=assigned_pairs,
            uncovered=uncovered,
        )

        output = dec_bboxes, dec_scores, stock.boxes, stock.logits, dn_meta
        if self.training:
            return output
        predictions = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return predictions if self.export else (predictions, output)

    def _project_stock_inputs(
        self,
        stock_inputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
        projected = [self.input_proj[index](feature) for index, feature in enumerate(stock_inputs)]
        shapes = [[feature.shape[2], feature.shape[3]] for feature in projected]
        flattened = [feature.flatten(2).permute(0, 2, 1) for feature in projected]
        return torch.cat(flattened, dim=1), shapes, projected[0]

    def _stock_query_set(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
    ) -> tuple[QuerySet, torch.Tensor]:
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        features = self.enc_output(self.valid_mask * feats)
        all_scores = self.enc_score_head(features)
        topk_indices = torch.topk(all_scores.max(-1).values, self.num_queries, dim=1).indices
        top_features = _gather_tensor(features, topk_indices)
        anchors = self.anchors.expand(feats.shape[0], -1, -1)
        top_anchors = _gather_tensor(anchors, topk_indices)
        reference_logits = self.enc_bbox_head(top_features) + top_anchors
        scores = _gather_tensor(all_scores, topk_indices)
        centers = _gather_tensor(anchors.sigmoid()[..., :2], topk_indices)
        levels = torch.cat(
            [torch.full((height * width,), level, dtype=torch.long, device=feats.device) for level, (height, width) in enumerate(shapes)]
        ).unsqueeze(0).expand(feats.shape[0], -1)
        return (
            QuerySet(
                features=top_features,
                reference_logits=reference_logits,
                boxes=reference_logits.sigmoid(),
                logits=scores,
                ranking_score=scores.max(-1).values,
                centers=centers,
                source=torch.zeros_like(topk_indices),
                source_level=torch.gather(levels, 1, topk_indices),
                source_index=topk_indices,
            ),
            topk_indices,
        )

    def _p2_query_sets(
        self,
        c2: torch.Tensor,
        projected_p3: torch.Tensor,
    ) -> tuple[QuerySet, QuerySet, torch.Tensor, torch.Tensor, tuple[int, int]]:
        p2_map = self._p2_features(c2, projected_p3)
        height, width = p2_map.shape[-2:]
        tokens = p2_map.flatten(2).permute(0, 2, 1)
        anchors, valid_mask = self._generate_anchors(
            [[height, width]],
            grid_size=self.ebc_config.p2_anchor_size,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        transformed = self._detached_stock_transform(valid_mask * tokens)
        logits = self._detached_stock_scores(transformed)
        reference_logits = self.p2_bbox_head(transformed) + anchors
        ranking_score = logits.max(-1).values.masked_fill(~valid_mask.squeeze(-1), -torch.inf)
        batch_size, candidate_count = ranking_score.shape
        source_index = torch.arange(candidate_count, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        all_queries = QuerySet(
            features=transformed,
            reference_logits=reference_logits,
            boxes=reference_logits.sigmoid(),
            logits=logits,
            ranking_score=ranking_score,
            centers=anchors.sigmoid()[..., :2].expand(batch_size, -1, -1),
            source=torch.ones_like(source_index),
            source_level=torch.full_like(source_index, 2),
            source_index=source_index,
        )
        valid_count = int(valid_mask.sum())
        topk_count = min(self.ebc_config.p2_candidates, valid_count)
        topk_indices = stable_rank_indices(ranking_score, all_queries.source, source_index, topk_count)
        return all_queries, gather_query_set(all_queries, topk_indices), topk_indices, valid_mask, (height, width)

    def _training_losses(
        self,
        batch: dict | None,
        stock: QuerySet,
        p2_all: QuerySet,
        p2_topk_indices: torch.Tensor,
        p2_valid_mask: torch.Tensor,
        p2_shape: tuple[int, int],
        stock_boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        if not self.training or batch is None:
            batch_size = p2_all.logits.shape[0]
            empty_pairs = [torch.empty((0, 2), dtype=torch.long, device=p2_all.logits.device) for _ in range(batch_size)]
            empty_uncovered = [torch.empty(0, dtype=torch.bool, device=p2_all.logits.device) for _ in range(batch_size)]
            zero = differentiable_zero(p2_all.logits, p2_all.boxes)
            return zero, differentiable_zero(p2_all.logits), empty_pairs, empty_uncovered

        boxes_by_image, classes_by_image = self._split_batch_targets(batch, p2_all.logits.shape[0], p2_all.logits.device)
        assigned_pairs = []
        uncovered = []
        height, width = p2_shape
        valid_mask = p2_valid_mask.reshape(-1)
        image_height, image_width = batch["img"].shape[-2:]

        for image_index, boxes in enumerate(boxes_by_image):
            radius = ((boxes[:, 2] * image_width) * (boxes[:, 3] * image_height)).clamp_min(0).sqrt()
            tiny_mask = radius <= self.ebc_config.tiny_radius
            tiny_indices = torch.where(tiny_mask)[0]
            tiny_boxes = boxes[tiny_mask]

            stock_match = match_centers_inside_boxes(stock.centers[image_index], tiny_boxes)
            image_uncovered = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
            if stock_match.unassigned_gt.numel():
                image_uncovered[tiny_indices[stock_match.unassigned_gt]] = True
            uncovered.append(image_uncovered)

            local = assign_local_p2(
                height=height,
                width=width,
                boxes=tiny_boxes,
                valid_mask=valid_mask,
                radius=self.ebc_config.local_radius,
            )
            pairs = local.pairs.clone()
            if pairs.numel():
                pairs[:, 0] = tiny_indices[pairs[:, 0]]
            assigned_pairs.append(pairs)

        targets = P2Targets(
            gt_boxes=boxes_by_image,
            gt_classes=classes_by_image,
            assigned_pairs=assigned_pairs,
            topk_indices=p2_topk_indices,
            anchor_centers=p2_all.centers[0],
        )
        p2_loss = compute_sparse_p2_loss(p2_all.logits, p2_all.boxes, targets).total
        ebc_loss = compute_ebc_loss(
            p2_logits=p2_all.logits,
            assigned_pairs=assigned_pairs,
            gt_classes=classes_by_image,
            uncovered=uncovered,
            stock_boundary=stock_boundary,
        )
        return p2_loss, ebc_loss, assigned_pairs, uncovered

    @staticmethod
    def _split_batch_targets(
        batch: dict,
        batch_size: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_index = batch["batch_idx"].reshape(-1).to(device=device, dtype=torch.long)
        boxes = batch["bboxes"].reshape(-1, 4).to(device=device, dtype=torch.float32)
        classes = batch["cls"].reshape(-1).to(device=device, dtype=torch.long)
        return (
            [boxes[batch_index == image_index] for image_index in range(batch_size)],
            [classes[batch_index == image_index] for image_index in range(batch_size)],
        )

    def _p2_features(self, c2: torch.Tensor, projected_p3: torch.Tensor) -> torch.Tensor:
        lateral = self.p2_adapter(c2.detach())
        context = F.interpolate(projected_p3.detach(), size=lateral.shape[-2:], mode="nearest")
        return F.silu(lateral + context)

    def _detached_stock_transform(self, p2_tokens: torch.Tensor) -> torch.Tensor:
        linear, norm = self.enc_output
        value = F.linear(p2_tokens, linear.weight.detach(), linear.bias.detach())
        return F.layer_norm(value, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)

    def _detached_stock_scores(self, p2_embed: torch.Tensor) -> torch.Tensor:
        return F.linear(p2_embed, self.enc_score_head.weight.detach(), self.enc_score_head.bias.detach())


def register_ebc_qp_decoder() -> None:
    import ultralytics.nn.tasks as ultralytics_tasks

    ultralytics_tasks.RTDETRDecoder = EBCQPDecoder


def _gather_tensor(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, dim=1, index=gather_indices)

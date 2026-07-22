from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from src.ebc_qp_matching import match_centers_inside_boxes


@dataclass(frozen=True)
class QuerySet:
    features: torch.Tensor
    reference_logits: torch.Tensor
    boxes: torch.Tensor
    logits: torch.Tensor
    ranking_score: torch.Tensor
    centers: torch.Tensor
    source: torch.Tensor
    source_level: torch.Tensor
    source_index: torch.Tensor


@dataclass(frozen=True)
class ReplacementStats:
    n_gain: int
    n_loss: int
    value: int


@dataclass(frozen=True)
class P2DiversityStats:
    foreground_at_50: int
    unique_gt_at_50: int
    duplicate_rate_at_50: float
    background_rate_at_50: float


def stable_rank_indices(
    scores: torch.Tensor,
    source: torch.Tensor,
    source_index: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if scores.shape != source.shape or scores.shape != source_index.shape:
        raise ValueError("scores, source, and source_index must have the same shape")
    if scores.ndim != 2:
        raise ValueError("ranking inputs must have shape [batch, candidates]")

    order = torch.argsort(source_index, dim=1, stable=True)
    ordered_source = torch.gather(source, 1, order)
    by_source = torch.argsort(ordered_source, dim=1, stable=True)
    order = torch.gather(order, 1, by_source)
    ordered_scores = torch.gather(scores.detach(), 1, order)
    by_score = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)
    return torch.gather(order, 1, by_score)[:, :k]


def compete_queries(stock: QuerySet, p2: QuerySet, budget: int = 300) -> QuerySet:
    merged = concatenate_query_sets(stock, p2)
    if merged.ranking_score.shape[1] < budget:
        raise ValueError("query competition cannot satisfy the requested budget")
    indices = stable_rank_indices(
        merged.ranking_score,
        merged.source,
        merged.source_index,
        budget,
    )
    return gather_query_set(merged, indices)


def concatenate_query_sets(first: QuerySet, second: QuerySet) -> QuerySet:
    values = {}
    for field in fields(QuerySet):
        first_value = getattr(first, field.name)
        second_value = getattr(second, field.name)
        if first_value.shape[0] != second_value.shape[0]:
            raise ValueError("query sets must share a batch size")
        values[field.name] = torch.cat((first_value, second_value), dim=1)
    return QuerySet(**values)


def gather_query_set(queries: QuerySet, indices: torch.Tensor) -> QuerySet:
    return QuerySet(
        **{
            field.name: _gather_candidates(getattr(queries, field.name), indices)
            for field in fields(QuerySet)
        }
    )


def replacement_statistics(
    stock_centers: torch.Tensor,
    final_centers: torch.Tensor,
    gt_boxes: torch.Tensor,
    tiny_mask: torch.Tensor,
) -> ReplacementStats:
    stock_gt = set(match_centers_inside_boxes(stock_centers.detach(), gt_boxes.detach()).covered_gt.tolist())
    final_gt = set(match_centers_inside_boxes(final_centers.detach(), gt_boxes.detach()).covered_gt.tolist())
    tiny_gt = set(torch.where(tiny_mask.detach().bool())[0].tolist())
    gain = len((final_gt - stock_gt) & tiny_gt)
    loss = len(stock_gt - final_gt)
    return ReplacementStats(n_gain=gain, n_loss=loss, value=gain - loss)


def p2_diversity_statistics(
    p2_centers: torch.Tensor,
    tiny_boxes: torch.Tensor,
) -> P2DiversityStats:
    association = nearest_containing_gt(p2_centers.detach(), tiny_boxes.detach())
    foreground = association >= 0
    foreground_count = int(foreground.sum())
    unique_count = int(association[foreground].unique().numel())
    duplicate_rate = 0.0 if foreground_count == 0 else 1.0 - unique_count / foreground_count
    background_rate = 1.0 - foreground_count / max(len(p2_centers), 1)
    return P2DiversityStats(
        foreground_at_50=foreground_count,
        unique_gt_at_50=unique_count,
        duplicate_rate_at_50=duplicate_rate,
        background_rate_at_50=background_rate,
    )


def nearest_containing_gt(centers: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if len(centers) == 0:
        return torch.empty(0, dtype=torch.long, device=centers.device)
    if len(boxes) == 0:
        return torch.full((len(centers),), -1, dtype=torch.long, device=centers.device)

    centers_fp32 = centers.float()
    boxes_fp32 = boxes.float()
    half = boxes_fp32[None, :, 2:] / 2
    delta = (centers_fp32[:, None] - boxes_fp32[None, :, :2]).abs()
    legal = (delta <= half).all(-1)
    distances = torch.cdist(centers_fp32, boxes_fp32[:, :2])
    distances = distances.masked_fill(~legal, torch.inf)
    association = distances.argmin(dim=1)
    association[~legal.any(dim=1)] = -1
    return association


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_indices = indices
    for _ in range(values.ndim - 2):
        gather_indices = gather_indices.unsqueeze(-1)
    gather_indices = gather_indices.expand(*indices.shape, *values.shape[2:])
    return torch.gather(values, dim=1, index=gather_indices)

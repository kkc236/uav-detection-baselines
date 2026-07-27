"""Five-view evidence reshaping and Hungarian identity propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.gcte_types import QueryEvidence, ViewGeometry
from src.gcte_views import (
    LOCAL_VIEWS,
    build_frozen_view_geometry,
    transform_xywh_homography,
)
from src.rtdetr_gcqf import DecoderEvidenceExtraction


@dataclass(frozen=True)
class FiveViewEvidence:
    global_evidence: QueryEvidence
    local_evidence: QueryEvidence
    geometry: ViewGeometry
    postprocessed: torch.Tensor
    selected_query_indices: torch.Tensor


def _slice_evidence(
    evidence: QueryEvidence,
    selection: slice,
) -> QueryEvidence:
    return QueryEvidence(
        queries=evidence.queries[selection],
        logits=evidence.logits[selection],
        boxes=evidence.boxes[selection],
        quality=evidence.quality[selection],
    )


def split_five_view_extraction(
    extraction: DecoderEvidenceExtraction,
    *,
    source_shape: tuple[int, int],
    queries_per_view: int = 300,
) -> FiveViewEvidence:
    """Split ``[global, TL, TR, BL, BR]`` and flatten local queries view-major."""

    evidence = extraction.evidence
    if evidence.batch_size != 1 + LOCAL_VIEWS:
        raise ValueError("five-view extraction must contain exactly five views")
    if evidence.query_count != queries_per_view:
        raise ValueError("per-view decoder query count drift")
    if extraction.postprocessed.shape[:2] != (
        1 + LOCAL_VIEWS,
        queries_per_view,
    ):
        raise ValueError("five-view postprocessed shape drift")
    if extraction.selected_query_indices.shape != (
        1 + LOCAL_VIEWS,
        queries_per_view,
    ):
        raise ValueError("five-view query-index shape drift")
    global_evidence = _slice_evidence(evidence, slice(0, 1))
    local = _slice_evidence(evidence, slice(1, None))
    local_evidence = QueryEvidence(
        queries=local.queries.reshape(
            1,
            LOCAL_VIEWS * queries_per_view,
            local.query_dim,
        ),
        logits=local.logits.reshape(
            1,
            LOCAL_VIEWS * queries_per_view,
            local.num_classes,
        ),
        boxes=local.boxes.reshape(
            1,
            LOCAL_VIEWS * queries_per_view,
            4,
        ),
        quality=local.quality.reshape(
            1,
            LOCAL_VIEWS * queries_per_view,
            1,
        ),
    )
    geometry = build_frozen_view_geometry(
        source_shapes=[source_shape],
        queries_per_view=queries_per_view,
    )
    return FiveViewEvidence(
        global_evidence=global_evidence,
        local_evidence=local_evidence,
        geometry=geometry,
        postprocessed=extraction.postprocessed,
        selected_query_indices=extraction.selected_query_indices,
    )


def _matcher_pair(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, torch.Tensor) for item in value)
    ):
        raise RuntimeError("RT-DETR matcher output contract drift")
    return value[0].to(torch.long), value[1].to(torch.long)


def build_local_match_assignments(
    *,
    matcher: Any,
    local_evidence: QueryEvidence,
    geometry: ViewGeometry,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    queries_per_view: int = 300,
) -> torch.Tensor:
    """Run the stock matcher per crop and retain each original global GT id."""

    expected_queries = LOCAL_VIEWS * queries_per_view
    if (
        local_evidence.batch_size != 1
        or local_evidence.query_count != expected_queries
    ):
        raise ValueError("local evidence query layout drift")
    if geometry.batch_size != 1 or geometry.query_count != expected_queries:
        raise ValueError("local geometry query layout drift")
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError("gt_boxes must be [N,4]")
    if (
        gt_classes.ndim != 1
        or gt_classes.shape[0] != gt_boxes.shape[0]
        or gt_classes.dtype != torch.long
    ):
        raise ValueError("gt_classes must be long [N]")
    assignments = torch.full(
        (expected_queries,),
        -1,
        dtype=torch.long,
        device=local_evidence.boxes.device,
    )
    gt_boxes = gt_boxes.to(local_evidence.boxes.device)
    gt_classes = gt_classes.to(local_evidence.boxes.device)
    for view_index in range(LOCAL_VIEWS):
        start = view_index * queries_per_view
        stop = start + queries_per_view
        matrix = geometry.homography[0, start].to(
            local_evidence.boxes.device
        )
        inverse = torch.linalg.inv(matrix.to(torch.float64)).to(
            local_evidence.boxes.dtype
        )
        inverse_batch = inverse.reshape(1, 1, 3, 3).repeat(
            1,
            gt_boxes.shape[0],
            1,
            1,
        )
        if gt_boxes.numel():
            untrimmed = transform_xywh_homography(
                gt_boxes.reshape(1, -1, 4),
                inverse_batch,
                clip=False,
            )[0]
            local_targets = transform_xywh_homography(
                gt_boxes.reshape(1, -1, 4),
                inverse_batch,
                clip=True,
            )[0]
            visible = (
                (untrimmed[:, 0] >= 0.0)
                & (untrimmed[:, 0] <= 1.0)
                & (untrimmed[:, 1] >= 0.0)
                & (untrimmed[:, 1] <= 1.0)
                & (local_targets[:, 2] > 0.0)
                & (local_targets[:, 3] > 0.0)
            )
        else:
            local_targets = gt_boxes
            visible = torch.zeros(0, dtype=torch.bool, device=gt_boxes.device)
        original_indices = torch.nonzero(visible, as_tuple=False).flatten()
        visible_boxes = local_targets[visible].float().contiguous()
        visible_classes = gt_classes[visible]
        matched_queries, matched_targets = _matcher_pair(
            matcher(
                local_evidence.boxes[:, start:stop].float().contiguous(),
                local_evidence.logits[:, start:stop].float().contiguous(),
                visible_boxes,
                visible_classes,
                [int(visible.sum())],
            )
        )
        if matched_queries.numel() != matched_targets.numel():
            raise RuntimeError("RT-DETR matcher pair length drift")
        if matched_queries.numel():
            assignments[start + matched_queries] = original_indices[
                matched_targets
            ]
    return assignments


__all__ = [
    "FiveViewEvidence",
    "build_local_match_assignments",
    "split_five_view_extraction",
]

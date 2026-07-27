"""Rebuild fixed/learned SADED routes from sealed GCQF evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from src.gcqf_cache import GCQFEvidenceRecord
from src.saded import ExpertCandidate, route_saded_image
from src.sbr_fusion import Detection
from src.sbr_g0 import (
    FrozenSBRProtocol,
    RawViewRecord,
    assemble_paired_arms,
    build_arm_views,
)


VIEW_ORDER = ("global", "TL", "TR", "BL", "BR")
RESIDUAL_ETA = 0.2


@dataclass(frozen=True)
class GCQFRouteResult:
    control: tuple[Detection, ...]
    raw_union: tuple[Detection, ...]
    output: tuple[Detection, ...]
    invariants: dict[str, bool]
    coverage: dict[str, int]


def rescore_postprocessed(
    record: GCQFEvidenceRecord,
    *,
    score_residual: torch.Tensor | None,
) -> torch.Tensor:
    """Apply one query residual to all selected classes from that query."""

    payload = record.fixed_anchor_payload
    postprocessed = payload.get("postprocessed")
    selected = payload.get("selected_query_indices")
    if (
        not isinstance(postprocessed, torch.Tensor)
        or postprocessed.shape != (5, 300, 6)
        or not isinstance(selected, torch.Tensor)
        or selected.shape != (5, 300)
    ):
        raise ValueError("fixed anchor postprocess payload drift")
    if score_residual is None:
        # Explicit bypass: no clone, cast, sigmoid, multiplication, or sort.
        return postprocessed
    residual = score_residual.detach().to(
        device=postprocessed.device,
        dtype=postprocessed.dtype,
    )
    if residual.shape != (1, 1200, 1):
        raise ValueError("score_residual must be [1,1200,1]")
    output = postprocessed.clone()
    for view_index in range(1, 5):
        decoder_queries = selected[view_index].to(torch.long)
        if bool(
            ((decoder_queries < 0) | (decoder_queries >= 300)).any()
        ):
            raise ValueError("selected decoder query index drift")
        local_indices = (view_index - 1) * 300 + decoder_queries
        delta = residual[0, local_indices, 0]
        output[view_index, :, 4] = torch.minimum(
            torch.ones_like(output[view_index, :, 4]),
            output[view_index, :, 4]
            * torch.exp(RESIDUAL_ETA * delta),
        )
    if not torch.equal(output[0], postprocessed[0]):
        raise RuntimeError("GCQF residual modified global predictions")
    return output


def _raw_detection(record: RawViewRecord) -> Detection:
    return Detection(
        box=record.global_xyxy,
        global_xyxy=record.global_xyxy,
        score=record.score,
        class_id=record.class_id,
        source_order=record.source_order,
        query_index=record.query_index,
        view_xyxy=record.view_xyxy,
        network_xyxy=record.network_xyxy,
        tile_bounds=record.tile_bounds,
        transform=record.transform,
        tile_index=(
            record.source_order - 1
            if record.tile_bounds is not None
            else None
        ),
    )


def _raw_records(
    record: GCQFEvidenceRecord,
    postprocessed: torch.Tensor,
) -> tuple[tuple[RawViewRecord, ...], list[dict[str, Any]]]:
    payload = record.fixed_anchor_payload
    if tuple(payload.get("view_order", ())) != VIEW_ORDER:
        raise ValueError("fixed anchor view order drift")
    source_shape = payload.get("source_shape")
    if (
        not isinstance(source_shape, (list, tuple))
        or len(source_shape) != 2
    ):
        raise ValueError("fixed anchor source shape drift")
    height, width = (int(value) for value in source_shape)
    views = build_arm_views("C", width, height)
    raw: list[RawViewRecord] = []
    manifest: list[dict[str, Any]] = []
    for view_position, view in enumerate(views):
        prediction = postprocessed[view_position].detach().float().cpu()
        candidates = []
        for row_index, row in enumerate(prediction):
            center_x, center_y, box_width, box_height, score, class_id = (
                float(value) for value in row.tolist()
            )
            if (
                not all(
                    math.isfinite(value)
                    for value in (
                        center_x,
                        center_y,
                        box_width,
                        box_height,
                        score,
                        class_id,
                    )
                )
                or score < FrozenSBRProtocol().conf
                or box_width <= 0.0
                or box_height <= 0.0
            ):
                continue
            network_xyxy = (
                (center_x - box_width * 0.5) * view.imgsz,
                (center_y - box_height * 0.5) * view.imgsz,
                (center_x + box_width * 0.5) * view.imgsz,
                (center_y + box_height * 0.5) * view.imgsz,
            )
            candidates.append(
                (
                    -score,
                    row_index,
                    network_xyxy,
                    score,
                    int(class_id),
                )
            )
        candidates.sort()
        manifest.append(
            {
                "view_id": view.view_id,
                "source_order": view.source_order,
                "executed": True,
            }
        )
        for (
            _negative_score,
            row_index,
            network_xyxy,
            score,
            class_id,
        ) in candidates[: FrozenSBRProtocol().max_det]:
            try:
                raw.append(
                    RawViewRecord.from_prediction(
                        view,
                        network_xyxy,
                        score,
                        class_id,
                        row_index,
                        width,
                        height,
                        image_id=record.image_id,
                    )
                )
            except ValueError as error:
                if str(error) != "prediction lies outside source frame":
                    raise
    return tuple(raw), manifest


def route_gcqf_record(
    record: GCQFEvidenceRecord,
    *,
    score_residual: torch.Tensor | None,
) -> GCQFRouteResult:
    """Route one cached image through Control, raw union, and SADED anchor."""

    full, raw_union = decode_gcqf_record(
        record,
        score_residual=score_residual,
    )
    height, width = (
        int(value)
        for value in record.fixed_anchor_payload["source_shape"]
    )
    routed = route_saded_image(
        image_id=record.image_id,
        width=width,
        height=height,
        baseline=tuple(
            ExpertCandidate(
                detection=detection,
                image_id=record.image_id,
                original_index=index,
            )
            for index, detection in enumerate(full)
        ),
        local_fused=tuple(
            ExpertCandidate(
                detection=detection,
                image_id=record.image_id,
                original_index=index,
            )
            for index, detection in enumerate(raw_union)
        ),
    )
    return GCQFRouteResult(
        control=full,
        raw_union=raw_union,
        output=routed.predictions,
        invariants=dict(routed.invariants),
        coverage=dict(routed.coverage),
    )


def decode_gcqf_record(
    record: GCQFEvidenceRecord,
    *,
    score_residual: torch.Tensor | None,
) -> tuple[tuple[Detection, ...], tuple[Detection, ...]]:
    """Decode sealed five-view evidence without applying a routing policy."""

    postprocessed = rescore_postprocessed(
        record,
        score_residual=score_residual,
    )
    raw, manifest = _raw_records(record, postprocessed)
    height, width = (
        int(value)
        for value in record.fixed_anchor_payload["source_shape"]
    )
    full = tuple(
        _raw_detection(value)
        for value in raw
        if value.source_order == 0
    )
    raw_union = tuple(
        assemble_paired_arms(
            raw,
            width=width,
            height=height,
            view_manifest=manifest,
        )["C"]["predictions"]
    )
    return full, raw_union


__all__ = [
    "GCQFRouteResult",
    "decode_gcqf_record",
    "rescore_postprocessed",
    "route_gcqf_record",
]

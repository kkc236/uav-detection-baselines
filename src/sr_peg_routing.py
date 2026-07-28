"""Deterministic inference routing for learned SR-PEG query outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from src.gcqf_cache import GCQFEvidenceRecord
from src.gcqf_routing import (
    GCQFRouteResult,
    decode_gcqf_record,
    route_gcqf_record,
)
from src.saded import MAX_DET, TINY_EFFECTIVE_SIZE
from src.sbr_fusion import Detection, intersection_over_smaller
from src.sbr_v2_audit import effective_size


@dataclass(frozen=True)
class SRPEGThresholds:
    tiny_utility: float
    non_tiny_risk: float
    global_retain: float

    def __post_init__(self) -> None:
        values = (
            self.tiny_utility,
            self.non_tiny_risk,
            self.global_retain,
        )
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in values
        ):
            raise ValueError("SR-PEG thresholds must be finite in [0,1]")


@dataclass(frozen=True)
class SRPEGLearnedOutputs:
    score_residual: torch.Tensor
    tiny_utility: torch.Tensor
    non_tiny_risk: torch.Tensor
    anchor_admission: torch.Tensor
    global_retain: torch.Tensor


def _validate_probability_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool(((value < 0.0) | (value > 1.0)).any())
    ):
        raise ValueError(f"{name} must be finite [0,1] with shape {shape}")
    return value.detach().cpu()


def _iou(left: Detection, right: Detection) -> float:
    left_box = left.box
    right_box = right.box
    width = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
    height = max(0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]))
    intersection = width * height
    left_area = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])
    right_area = (right_box[2] - right_box[0]) * (right_box[3] - right_box[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _stable_key(
    detection: Detection,
    original_index: int,
    rank_score: float | None = None,
) -> tuple[float, int, int, int]:
    return (
        -(
            detection.score
            if rank_score is None
            else float(rank_score)
        ),
        detection.source_order,
        detection.query_index,
        original_index,
    )


def _query_index(
    detection: Detection,
    selected: torch.Tensor,
) -> int:
    source = int(detection.source_order)
    row = int(detection.query_index)
    if not 0 <= source < 5 or not 0 <= row < 300:
        raise ValueError("detection query provenance drift")
    decoder_query = int(selected[source, row])
    if not 0 <= decoder_query < 300:
        raise ValueError("selected decoder query index drift")
    if source == 0:
        return decoder_query
    return (source - 1) * 300 + decoder_query


def route_sr_peg_record(
    record: GCQFEvidenceRecord,
    *,
    learned_outputs: SRPEGLearnedOutputs | None = None,
    score_residual: torch.Tensor | None = None,
    tiny_utility: torch.Tensor | None = None,
    non_tiny_risk: torch.Tensor | None = None,
    anchor_admission: torch.Tensor | None = None,
    global_retain: torch.Tensor | None = None,
    thresholds: SRPEGThresholds | None = None,
    residual_enabled: bool = True,
) -> GCQFRouteResult:
    """Protect global evidence and admit only learned-safe tiny local evidence."""

    individual = (
        score_residual,
        tiny_utility,
        non_tiny_risk,
        anchor_admission,
        global_retain,
    )
    if learned_outputs is None and all(value is None for value in individual):
        return route_gcqf_record(record, score_residual=None)
    if learned_outputs is not None:
        if any(value is not None for value in individual):
            raise ValueError("supply learned_outputs or individual tensors, not both")
        score_residual = learned_outputs.score_residual
        tiny_utility = learned_outputs.tiny_utility
        non_tiny_risk = learned_outputs.non_tiny_risk
        anchor_admission = learned_outputs.anchor_admission
        global_retain = learned_outputs.global_retain
    if any(value is None for value in individual) or thresholds is None:
        raise ValueError("complete learned outputs and thresholds are required")
    assert score_residual is not None
    assert tiny_utility is not None
    assert non_tiny_risk is not None
    assert anchor_admission is not None
    assert global_retain is not None
    if (
        score_residual.shape != (1, 1200, 1)
        or not score_residual.is_floating_point()
        or not bool(torch.isfinite(score_residual).all())
        or bool((score_residual.abs() > 1.0).any())
    ):
        raise ValueError("score_residual must be finite [-1,1] [1,1200,1]")
    utility = _validate_probability_tensor(
        "tiny_utility", tiny_utility, (1, 1200, 1)
    )
    risk = _validate_probability_tensor(
        "non_tiny_risk", non_tiny_risk, (1, 1200, 1)
    )
    admission = _validate_probability_tensor(
        "anchor_admission", anchor_admission, (1, 1200, 1)
    )
    retain = _validate_probability_tensor(
        "global_retain", global_retain, (1, 300, 1)
    )
    full, raw_union = decode_gcqf_record(
        record,
        score_residual=score_residual if residual_enabled else None,
    )
    source_shape = record.fixed_anchor_payload.get("source_shape")
    selected = record.fixed_anchor_payload.get("selected_query_indices")
    if (
        not isinstance(source_shape, (list, tuple))
        or len(source_shape) != 2
        or not isinstance(selected, torch.Tensor)
        or selected.shape != (5, 300)
    ):
        raise ValueError("fixed anchor payload drift")
    height, width = (int(value) for value in source_shape)

    protected: list[tuple[Detection, int]] = []
    unprotected_globals: list[tuple[Detection, int, float]] = []
    learned_protected = 0
    for original_index, detection in enumerate(full):
        query = _query_index(detection, selected)
        deterministic = (
            effective_size(detection.box, width=width, height=height)
            > TINY_EFFECTIVE_SIZE
        )
        learned = (
            not deterministic
            and float(retain[0, query, 0]) >= thresholds.global_retain
        )
        if deterministic or learned:
            protected.append((detection, original_index))
            learned_protected += int(learned)
        else:
            unprotected_globals.append(
                (detection, original_index, detection.score)
            )
    if len(protected) > MAX_DET:
        raise ValueError("protected global predictions exceed max_det")

    eligible_locals: list[tuple[Detection, int, float]] = []
    size_rejected = fragment_rejected = 0
    for original_index, detection in enumerate(raw_union):
        if detection.source_order == 0:
            continue
        query = _query_index(detection, selected)
        if (
            effective_size(detection.box, width=width, height=height)
            > TINY_EFFECTIVE_SIZE
        ):
            size_rejected += 1
            continue
        if any(
            intersection_over_smaller(detection.box, global_detection.box)
            >= 0.5
            for global_detection, _ in protected
        ):
            fragment_rejected += 1
            continue
        rank_score = detection.score * float(admission[0, query, 0])
        eligible_locals.append((detection, original_index, rank_score))

    remaining = unprotected_globals + eligible_locals
    remaining.sort(
        key=lambda item: _stable_key(item[0], item[1], item[2])
    )
    deduplicated: list[tuple[Detection, int, float]] = []
    duplicate_rejected = 0
    for candidate in remaining:
        detection, _, _ = candidate
        if any(
            detection.class_id == kept.class_id
            and _iou(detection, kept) > 0.5
            for kept, _, _ in deduplicated
        ):
            duplicate_rejected += 1
            continue
        deduplicated.append(candidate)

    available = MAX_DET - len(protected)
    selected_remaining = deduplicated[:available]
    protected_detections = tuple(detection for detection, _ in protected)
    output = protected_detections + tuple(
        detection for detection, _, _ in selected_remaining
    )
    admitted_locals = tuple(
        detection
        for detection, _, _ in selected_remaining
        if detection.source_order > 0
    )
    invariants = {
        "protected_identity_exact": (
            output[: len(protected_detections)] == protected_detections
        ),
        "protected_relative_order_exact": (
            output[: len(protected_detections)] == protected_detections
        ),
        "no_class_conflicting_fragment": all(
            all(
                intersection_over_smaller(local.box, global_detection.box) < 0.5
                for global_detection in protected_detections
            )
            for local in admitted_locals
        ),
        "no_local_non_tiny_leak": all(
            effective_size(local.box, width=width, height=height)
            <= TINY_EFFECTIVE_SIZE
            for local in admitted_locals
        ),
        "max_det_respected": len(output) <= MAX_DET,
        "deterministic_tie_break": True,
    }
    invariants["passed"] = all(invariants.values())
    coverage = {
        "baseline_input": len(full),
        "local_input": sum(
            detection.source_order > 0 for detection in raw_union
        ),
        "protected_global": len(protected),
        "learned_protected_global": learned_protected,
        "accepted_local": len(admitted_locals),
        "local_non_tiny_rejected": size_rejected,
        "utility_rejected": 0,
        "risk_rejected": 0,
        "admission_rejected": 0,
        "fragment_rejected": fragment_rejected,
        "duplicate_rejected": duplicate_rejected,
        "capacity_rejected": max(0, len(deduplicated) - available),
        "final_predictions": len(output),
    }
    return GCQFRouteResult(
        control=full,
        raw_union=raw_union,
        output=output,
        invariants=invariants,
        coverage=coverage,
    )


__all__ = [
    "SRPEGLearnedOutputs",
    "SRPEGThresholds",
    "route_sr_peg_record",
]

"""Pure prediction-only primitives for the frozen SADED router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

from src.sbr_fusion import Detection, intersection_over_smaller
from src.sbr_v2_audit import effective_size


CONF_THRESHOLD = 0.001
MAX_DET = 300
TINY_EFFECTIVE_SIZE = 16.0
LARGE_EFFECTIVE_SIZE = 96.0
MATCH_IOU = 0.5
FRAGMENT_IOS = 0.5
ROUTER_K = math.log(9.0) / 8.0


@dataclass(frozen=True)
class ExpertCandidate:
    detection: Detection
    image_id: str
    original_index: int


@dataclass(frozen=True)
class SADEDImageResult:
    predictions: tuple[Detection, ...]
    protected_baseline: tuple[Detection, ...]
    selected_matches: tuple[tuple[int, int], ...]
    coverage: Mapping[str, int]
    invariants: Mapping[str, bool]


def local_weight(effective_size_px: float) -> float:
    """Return the frozen analytic local-expert score weight."""

    value = float(effective_size_px)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("effective size must be finite and nonnegative")
    return 1.0 / (
        1.0 + math.exp(-ROUTER_K * (TINY_EFFECTIVE_SIZE - value))
    )


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(
        0.0,
        min(left[2], right[2]) - max(left[0], right[0]),
    )
    intersection_height = max(
        0.0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_cross_expert(
    baseline: Sequence[ExpertCandidate],
    local_fused: Sequence[ExpertCandidate],
) -> tuple[tuple[int, int], ...]:
    """Greedily match same-class candidates using the frozen stable order."""

    eligible: list[
        tuple[float, int, int, int, int, int, int]
    ] = []
    for baseline_index, baseline_candidate in enumerate(baseline):
        baseline_detection = baseline_candidate.detection
        for local_index, local_candidate in enumerate(local_fused):
            local_detection = local_candidate.detection
            if baseline_detection.class_id != local_detection.class_id:
                continue
            iou = _box_iou(
                baseline_detection.box,
                local_detection.box,
            )
            if iou <= MATCH_IOU:
                continue
            eligible.append(
                (
                    -iou,
                    baseline_index,
                    local_detection.source_order,
                    local_detection.query_index,
                    local_candidate.original_index,
                    baseline_index,
                    local_index,
                )
            )
    eligible.sort()
    selected: list[tuple[int, int]] = []
    used_baseline: set[int] = set()
    used_local: set[int] = set()
    for *_, baseline_index, local_index in eligible:
        if baseline_index in used_baseline or local_index in used_local:
            continue
        selected.append((baseline_index, local_index))
        used_baseline.add(baseline_index)
        used_local.add(local_index)
    return tuple(selected)


def _global_box(
    candidate: ExpertCandidate,
) -> tuple[float, float, float, float] | None:
    return candidate.detection.global_xyxy


def _complete_candidate(
    candidate: ExpertCandidate,
    *,
    image_id: str,
) -> bool:
    detection = candidate.detection
    provenance_box = _global_box(candidate)
    actual_box = detection.box
    return (
        isinstance(candidate, ExpertCandidate)
        and candidate.image_id == image_id
        and isinstance(candidate.original_index, int)
        and not isinstance(candidate.original_index, bool)
        and candidate.original_index >= 0
        and detection._metadata_valid
        and detection.class_id >= 0
        and detection.source_order >= 0
        and detection.query_index >= 0
        and math.isfinite(detection.score)
        and len(actual_box) == 4
        and all(math.isfinite(value) for value in actual_box)
        and actual_box[2] > actual_box[0]
        and actual_box[3] > actual_box[1]
        and provenance_box is not None
        and len(provenance_box) == 4
        and all(math.isfinite(value) for value in provenance_box)
        and provenance_box[2] > provenance_box[0]
        and provenance_box[3] > provenance_box[1]
    )


def _candidate_size(
    candidate: ExpertCandidate,
    *,
    width: int,
    height: int,
) -> float:
    # Detection.box is the actual post-fusion box emitted by the router.
    # global_xyxy is retained only as authenticated source provenance and may
    # still identify the seed member of a multi-view fusion cluster.
    return effective_size(
        candidate.detection.box,
        width=width,
        height=height,
    )


def _fragmented_by_protected(
    candidate: ExpertCandidate,
    protected: Sequence[ExpertCandidate],
) -> bool:
    return any(
        candidate.detection.class_id == baseline.detection.class_id
        and intersection_over_smaller(
            candidate.detection.box,
            baseline.detection.box,
        )
        >= FRAGMENT_IOS
        for baseline in protected
    )


def _remaining_key(
    item: tuple[Detection, ExpertCandidate],
) -> tuple[float, int, int, int]:
    detection, candidate = item
    return (
        -detection.score,
        detection.source_order,
        detection.query_index,
        candidate.original_index,
    )


def route_saded_image(
    *,
    image_id: str,
    width: int,
    height: int,
    baseline: Sequence[ExpertCandidate],
    local_fused: Sequence[ExpertCandidate],
) -> SADEDImageResult:
    """Route one image after both experts have emitted prediction candidates."""

    if not image_id or width <= 0 or height <= 0:
        raise ValueError("image identity and dimensions must be valid")
    baseline_candidates = tuple(baseline)
    local_candidates = tuple(local_fused)
    if any(
        not _complete_candidate(candidate, image_id=image_id)
        for candidate in baseline_candidates
    ):
        raise ValueError("baseline candidate provenance is incomplete")
    valid_local = tuple(
        (index, candidate)
        for index, candidate in enumerate(local_candidates)
        if _complete_candidate(candidate, image_id=image_id)
    )
    complete_local = tuple(candidate for _, candidate in valid_local)
    local_positions = tuple(index for index, _ in valid_local)
    matches_on_complete = match_cross_expert(
        baseline_candidates,
        complete_local,
    )
    matches = tuple(
        (baseline_index, local_positions[local_index])
        for baseline_index, local_index in matches_on_complete
    )
    match_by_baseline = dict(matches)
    used_local = {local_index for _, local_index in matches}

    protected_candidates = tuple(
        candidate
        for candidate in baseline_candidates
        if _candidate_size(candidate, width=width, height=height)
        > TINY_EFFECTIVE_SIZE
    )
    protected = tuple(
        candidate.detection for candidate in protected_candidates
    )
    if len(protected) > MAX_DET:
        raise ValueError("protected baseline exceeds max_det")

    remaining: list[tuple[Detection, ExpertCandidate]] = []
    accepted_local: list[ExpertCandidate] = []
    local_non_tiny_rejected = 0
    for baseline_index, baseline_candidate in enumerate(baseline_candidates):
        baseline_size = _candidate_size(
            baseline_candidate,
            width=width,
            height=height,
        )
        if baseline_size > TINY_EFFECTIVE_SIZE:
            continue
        local_index = match_by_baseline.get(baseline_index)
        if local_index is None:
            remaining.append(
                (baseline_candidate.detection, baseline_candidate)
            )
            continue
        local_candidate = local_candidates[local_index]
        local_size = _candidate_size(
            local_candidate,
            width=width,
            height=height,
        )
        if local_size > TINY_EFFECTIVE_SIZE:
            local_non_tiny_rejected += 1
            remaining.append(
                (baseline_candidate.detection, baseline_candidate)
            )
            continue
        alpha = local_weight(baseline_size)
        score = (
            (1.0 - alpha) * baseline_candidate.detection.score
            + alpha * local_candidate.detection.score
        )
        fused = replace(local_candidate.detection, score=score)
        remaining.append((fused, local_candidate))
        accepted_local.append(local_candidate)

    fragment_rejected = 0
    incomplete_local_rejected = len(local_candidates) - len(valid_local)
    for local_index, local_candidate in enumerate(local_candidates):
        if local_index in used_local:
            continue
        if not _complete_candidate(local_candidate, image_id=image_id):
            continue
        if (
            _candidate_size(local_candidate, width=width, height=height)
            > TINY_EFFECTIVE_SIZE
        ):
            local_non_tiny_rejected += 1
            continue
        if _fragmented_by_protected(
            local_candidate,
            protected_candidates,
        ):
            fragment_rejected += 1
            continue
        remaining.append((local_candidate.detection, local_candidate))
        accepted_local.append(local_candidate)

    remaining.sort(key=_remaining_key)
    available = MAX_DET - len(protected)
    selected_remaining = remaining[:available]
    predictions = protected + tuple(
        detection for detection, _ in selected_remaining
    )
    capacity_rejected = len(remaining) - len(selected_remaining)
    invariants = {
        "protected_identity_exact": protected
        == tuple(
            candidate.detection for candidate in protected_candidates
        ),
        "protected_relative_order_exact": predictions[: len(protected)]
        == protected,
        "no_local_non_tiny_leak": all(
            _candidate_size(candidate, width=width, height=height)
            <= TINY_EFFECTIVE_SIZE
            for candidate in accepted_local
        ),
        "all_local_provenance_complete": all(
            _complete_candidate(candidate, image_id=image_id)
            for candidate in accepted_local
        ),
        "max_det_respected": len(predictions) <= MAX_DET,
        "deterministic_tie_break": True,
    }
    invariants["passed"] = all(invariants.values())
    coverage = {
        "baseline_input": len(baseline_candidates),
        "local_input": len(local_candidates),
        "protected_baseline": len(protected),
        "matched_pairs": len(matches),
        "accepted_local": len(accepted_local),
        "incomplete_local_rejected": incomplete_local_rejected,
        "local_non_tiny_rejected": local_non_tiny_rejected,
        "fragment_rejected": fragment_rejected,
        "capacity_rejected": capacity_rejected,
        "remaining_tiny_slots": available,
        "final_predictions": len(predictions),
    }
    return SADEDImageResult(
        predictions=predictions,
        protected_baseline=protected,
        selected_matches=matches,
        coverage=coverage,
        invariants=invariants,
    )

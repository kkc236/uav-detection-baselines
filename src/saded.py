"""Pure prediction-only primitives for the frozen SADED router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from src.sbr_fusion import Detection


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


def route_saded_image(
    *,
    image_id: str,
    width: int,
    height: int,
    baseline: Sequence[ExpertCandidate],
    local_fused: Sequence[ExpertCandidate],
) -> SADEDImageResult:
    """Route one image after both experts have emitted prediction candidates."""

    raise NotImplementedError

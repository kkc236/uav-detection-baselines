"""Frozen Scale-Partitioned Prefix-Preserved Additive Fusion primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Any

from src.sbr_fusion import Detection, intersection_over_smaller
from src.sbr_v2_audit import (
    AuditRawDetection,
    CClusterReconstruction,
    effective_size,
    reconstruct_c_clusters,
)


CONF_THRESHOLD = 0.001
MAX_DET = 300
LARGE_EFFECTIVE_SIZE = 96.0
FRAGMENT_IOS = 0.5
A_FLOOR = 0.01706760562956333
C_CEILING = 0.008533802814781666
SCORE_LOW = math.nextafter(CONF_THRESHOLD, math.inf)
SCORE_HIGH = math.nextafter(C_CEILING, -math.inf)

ARM_NAMES = ("A", "All-A", "P1", "P2", "P3")
GATE_THRESHOLDS = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}

IdentityKey = tuple[str, int, int, int]


@dataclass(frozen=True)
class TailCandidate:
    """One eligible Arm-C pre-cap cluster and its immutable provenance."""

    cluster_rank: int
    prediction: Detection
    original_score: float
    member_indices: tuple[int, ...]
    member_identities: frozenset[IdentityKey]
    tile_only: bool


@dataclass(frozen=True)
class PPAFImageResult:
    """All frozen SP-PPAF arms for one image."""

    arms: Mapping[str, tuple[Detection, ...]]
    coverage: Mapping[str, Mapping[str, int]]
    invariants: Mapping[str, bool]
    eligible_tail: tuple[TailCandidate, ...]
    selected_cluster_ranks: Mapping[str, tuple[int, ...]]


def map_tail_score(score: object) -> float:
    """Map one legal Arm-C score into the frozen low-priority band."""

    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("tail score must be a finite real in [conf, 1]")
    value = float(score)
    if not math.isfinite(value) or not CONF_THRESHOLD <= value <= 1.0:
        raise ValueError("tail score must be a finite real in [conf, 1]")
    mapped = SCORE_LOW + (SCORE_HIGH - SCORE_LOW) * (
        (value - CONF_THRESHOLD) / (1.0 - CONF_THRESHOLD)
    )
    if not CONF_THRESHOLD < mapped < C_CEILING:
        raise ValueError("mapped tail score escaped the frozen band")
    return mapped


def verify_tail_score_domain(scores: Sequence[object]) -> dict[str, Any]:
    """Verify float64 order and collision properties over a complete score set."""

    values: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("tail score domain must contain finite real values")
        value = float(score)
        if not math.isfinite(value) or not CONF_THRESHOLD <= value <= 1.0:
            raise ValueError("tail score domain escaped [conf, 1]")
        values.append(value)
    distinct = tuple(sorted(set(values)))
    mapped = tuple(map_tail_score(value) for value in distinct)
    collision_free = len(set(mapped)) == len(mapped)
    strictly_monotone = all(
        left < right for left, right in zip(mapped, mapped[1:])
    )
    in_band = all(CONF_THRESHOLD < value < C_CEILING for value in mapped)
    equal_inputs_equal = all(
        map_tail_score(left) == map_tail_score(right)
        for left, right in zip(values, values)
    )
    return {
        "input_count": len(values),
        "distinct_input_count": len(distinct),
        "mapped_distinct_count": len(set(mapped)),
        "collision_free": collision_free,
        "strictly_monotone": strictly_monotone,
        "in_band": in_band,
        "equal_inputs_equal": equal_inputs_equal,
        "passed": (
            collision_free
            and strictly_monotone
            and in_band
            and equal_inputs_equal
        ),
    }


def verify_a_floor(scores: Sequence[object]) -> dict[str, Any]:
    """Require the complete sealed Arm-A population to match the frozen floor."""

    values: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("Arm A score population must contain finite reals")
        value = float(score)
        if not math.isfinite(value) or not CONF_THRESHOLD <= value <= 1.0:
            raise ValueError("Arm A score population escaped [conf, 1]")
        values.append(value)
    if not values:
        raise ValueError("Arm A score population must not be empty")
    actual = min(values)
    return {
        "actual_a_min": actual,
        "expected_a_floor": A_FLOOR,
        "exact_equal": actual == A_FLOOR,
        "ceiling_below_actual_a_min": C_CEILING < actual,
        "passed": actual == A_FLOOR and C_CEILING < actual,
    }


def _identity(image_id: str, detection: Detection) -> IdentityKey:
    return (
        image_id,
        int(detection.class_id),
        int(detection.source_order),
        int(detection.query_index),
    )


def _validate_dimensions(image_id: object, width: object, height: object) -> None:
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a nonempty string")
    for name, value in (("width", width), ("height", height)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")


def _validate_a_final(a_final: Sequence[Detection]) -> tuple[Detection, ...]:
    a = tuple(a_final)
    if len(a) > MAX_DET:
        raise ValueError("sealed Arm A exceeds max_det")
    for prediction in a:
        if not isinstance(prediction, Detection):
            raise ValueError("Arm A values must be Detection instances")
        if prediction.source_order != 0:
            raise ValueError("Arm A final detections must be full-view")
        if (
            prediction.global_xyxy is None
            or not math.isfinite(float(prediction.score))
            or not CONF_THRESHOLD <= float(prediction.score) <= 1.0
        ):
            raise ValueError("invalid sealed Arm A detection")
        effective_size(
            prediction.global_xyxy,
            width=1,
            height=1,
        )
    return a


def _fill(
    prefix: tuple[Detection, ...],
    candidates: tuple[TailCandidate, ...],
) -> tuple[Detection, ...]:
    remaining = MAX_DET - len(prefix)
    if remaining < 0:
        raise ValueError("prefix exceeds max_det")
    return prefix + tuple(item.prediction for item in candidates[:remaining])


def _selected_ranks(
    prefix: tuple[Detection, ...],
    candidates: tuple[TailCandidate, ...],
) -> tuple[int, ...]:
    return tuple(
        item.cluster_rank for item in candidates[: MAX_DET - len(prefix)]
    )


def _coverage(
    *,
    prefix: int,
    raw_candidates: int,
    source_scale_eligible: int,
    provenance_rejected: int,
    fragment_rejected: int,
    output: int,
) -> dict[str, int]:
    return {
        "prefix": prefix,
        "remaining": MAX_DET - prefix,
        "raw_candidates": raw_candidates,
        "source_scale_eligible": source_scale_eligible,
        "provenance_rejected": provenance_rejected,
        "fragment_rejected": fragment_rejected,
        "capacity_rejected": max(
            0,
            source_scale_eligible
            - provenance_rejected
            - fragment_rejected
            - (output - prefix),
        ),
        "appended": output - prefix,
        "output": output,
    }


def _fragmented_by_large(
    candidate: TailCandidate,
    anchors: tuple[Detection, ...],
) -> bool:
    return candidate.tile_only and any(
        candidate.prediction.class_id == anchor.class_id
        and intersection_over_smaller(
            candidate.prediction.box,
            anchor.global_xyxy,
        )
        >= FRAGMENT_IOS
        for anchor in anchors
    )


def build_ppaf_arms(
    *,
    image_id: str,
    width: int,
    height: int,
    a_final: Sequence[Detection],
    c_reconstruction: CClusterReconstruction,
    c_raw: Sequence[AuditRawDetection],
) -> PPAFImageResult:
    """Build all predeclared arms without accepting ground truth."""

    _validate_dimensions(image_id, width, height)
    a = _validate_a_final(a_final)
    raw = tuple(c_raw)
    for item in raw:
        if (
            not isinstance(item, AuditRawDetection)
            or item.arm != "C"
            or item.image_id != image_id
            or item.width != width
            or item.height != height
        ):
            raise ValueError("invalid Arm C raw provenance")
    rebuilt = reconstruct_c_clusters(raw)
    if rebuilt != c_reconstruction:
        raise ValueError("Arm C reconstruction does not match raw provenance")

    raw_by_index = {item.original_index: item for item in raw}
    if len(raw_by_index) != len(raw):
        raise ValueError("Arm C raw indices must be unique")

    a_large = tuple(
        prediction
        for prediction in a
        if effective_size(
            prediction.global_xyxy,
            width=width,
            height=height,
        )
        > LARGE_EFFECTIVE_SIZE
    )
    a_large_ids = frozenset(_identity(image_id, item) for item in a_large)
    all_a_ids = frozenset(_identity(image_id, item) for item in a)

    eligible_rows: list[TailCandidate] = []
    for rank, (prediction, member_indices) in enumerate(
        zip(
            c_reconstruction.pre_cap_predictions,
            c_reconstruction.cluster_members,
            strict=True,
        )
    ):
        try:
            members = tuple(raw_by_index[index] for index in member_indices)
        except KeyError as exc:
            raise ValueError(
                "cluster references missing raw provenance"
            ) from exc
        if not members:
            raise ValueError("empty Arm C cluster provenance")
        if not any(member.source_order > 0 for member in members):
            continue
        if prediction.global_xyxy is None:
            raise ValueError("Arm C seed is missing global_xyxy")
        if (
            effective_size(
                prediction.global_xyxy,
                width=width,
                height=height,
            )
            > LARGE_EFFECTIVE_SIZE
        ):
            continue
        if (
            not math.isfinite(float(prediction.score))
            or float(prediction.score) < CONF_THRESHOLD
        ):
            raise ValueError("eligible Arm C score escaped the frozen domain")
        mapped = replace(
            prediction,
            score=map_tail_score(prediction.score),
        )
        eligible_rows.append(
            TailCandidate(
                cluster_rank=rank,
                prediction=mapped,
                original_score=float(prediction.score),
                member_indices=tuple(member_indices),
                member_identities=frozenset(
                    member.identity_key for member in members
                ),
                tile_only=all(member.source_order > 0 for member in members),
            )
        )
    eligible = tuple(eligible_rows)
    score_domain = verify_tail_score_domain(
        tuple(item.original_score for item in eligible)
    )
    if score_domain["passed"] is not True:
        raise ValueError("Arm C mapped score domain is invalid")

    p2_tail = tuple(
        item
        for item in eligible
        if item.member_identities.isdisjoint(a_large_ids)
    )
    p3_tail = tuple(
        item
        for item in p2_tail
        if not _fragmented_by_large(item, a_large)
    )
    all_a_p2_tail = tuple(
        item
        for item in eligible
        if item.member_identities.isdisjoint(all_a_ids)
    )
    all_a_p3_tail = tuple(
        item
        for item in all_a_p2_tail
        if not _fragmented_by_large(item, a_large)
    )

    arms = {
        "A": a,
        "All-A": _fill(a, all_a_p3_tail),
        "P1": _fill(a_large, eligible),
        "P2": _fill(a_large, p2_tail),
        "P3": _fill(a_large, p3_tail),
    }
    selected_cluster_ranks = {
        "A": (),
        "All-A": _selected_ranks(a, all_a_p3_tail),
        "P1": _selected_ranks(a_large, eligible),
        "P2": _selected_ranks(a_large, p2_tail),
        "P3": _selected_ranks(a_large, p3_tail),
    }
    raw_candidate_count = len(c_reconstruction.pre_cap_predictions)
    coverage = {
        "A": _coverage(
            prefix=len(a),
            raw_candidates=0,
            source_scale_eligible=0,
            provenance_rejected=0,
            fragment_rejected=0,
            output=len(a),
        ),
        "All-A": _coverage(
            prefix=len(a),
            raw_candidates=raw_candidate_count,
            source_scale_eligible=len(eligible),
            provenance_rejected=len(eligible) - len(all_a_p2_tail),
            fragment_rejected=len(all_a_p2_tail) - len(all_a_p3_tail),
            output=len(arms["All-A"]),
        ),
        "P1": _coverage(
            prefix=len(a_large),
            raw_candidates=raw_candidate_count,
            source_scale_eligible=len(eligible),
            provenance_rejected=0,
            fragment_rejected=0,
            output=len(arms["P1"]),
        ),
        "P2": _coverage(
            prefix=len(a_large),
            raw_candidates=raw_candidate_count,
            source_scale_eligible=len(eligible),
            provenance_rejected=len(eligible) - len(p2_tail),
            fragment_rejected=0,
            output=len(arms["P2"]),
        ),
        "P3": _coverage(
            prefix=len(a_large),
            raw_candidates=raw_candidate_count,
            source_scale_eligible=len(eligible),
            provenance_rejected=len(eligible) - len(p2_tail),
            fragment_rejected=len(p2_tail) - len(p3_tail),
            output=len(arms["P3"]),
        ),
    }
    invariants = {
        "a_identity": arms["A"] == a,
        "a_large_prefix_identity": all(
            arms[arm][: len(a_large)] == a_large for arm in ("P1", "P2", "P3")
        ),
        "all_a_prefix_identity": arms["All-A"][: len(a)] == a,
        "c_fused_boxes_preserved": all(
            item.prediction.box
            == c_reconstruction.pre_cap_predictions[item.cluster_rank].box
            for item in eligible
        ),
        "score_band": all(
            CONF_THRESHOLD < item.prediction.score < C_CEILING
            for item in eligible
        ),
        "score_domain": score_domain["passed"] is True,
        "cluster_rank_order": tuple(
            item.cluster_rank for item in eligible
        )
        == tuple(sorted(item.cluster_rank for item in eligible)),
        "p2_exact_difference": {
            item.cluster_rank for item in eligible
        }
        - {item.cluster_rank for item in p2_tail}
        == {
            item.cluster_rank
            for item in eligible
            if not item.member_identities.isdisjoint(a_large_ids)
        },
        "p3_exact_difference": {
            item.cluster_rank for item in p2_tail
        }
        - {item.cluster_rank for item in p3_tail}
        == {
            item.cluster_rank
            for item in p2_tail
            if _fragmented_by_large(item, a_large)
        },
        "max_det": all(len(value) <= MAX_DET for value in arms.values()),
    }
    invariants["passed"] = all(invariants.values())
    return PPAFImageResult(
        arms=arms,
        coverage=coverage,
        invariants=invariants,
        eligible_tail=eligible,
        selected_cluster_ranks=selected_cluster_ranks,
    )


def metric_deltas(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, float]:
    """Compute exact unrounded gate deltas."""

    deltas: dict[str, float] = {}
    for key in GATE_THRESHOLDS:
        left = candidate.get(key)
        right = baseline.get(key)
        if (
            isinstance(left, bool)
            or not isinstance(left, Real)
            or isinstance(right, bool)
            or not isinstance(right, Real)
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
        ):
            raise ValueError(f"invalid metric: {key}")
        deltas[key] = float(left) - float(right)
    return deltas


def decide_ppaf(
    a_metrics: Mapping[str, Any],
    p3_metrics: Mapping[str, Any],
    fallback_metrics: Mapping[str, Any],
    *,
    invariants_passed: bool,
) -> dict[str, Any]:
    """Apply the frozen P3, then All-A, then STOP state machine."""

    if invariants_passed is not True:
        return {
            "status": "SP_PPAF_INVALID",
            "selected_arm": "none",
            "p3_delta": None,
            "fallback_delta": None,
            "p3_gates": None,
            "fallback_gates": None,
            "invariants_passed": False,
        }
    try:
        p3_delta = metric_deltas(p3_metrics, a_metrics)
        fallback_delta = metric_deltas(fallback_metrics, a_metrics)
    except ValueError as exc:
        return {
            "status": "SP_PPAF_INVALID",
            "selected_arm": "none",
            "p3_delta": None,
            "fallback_delta": None,
            "p3_gates": None,
            "fallback_gates": None,
            "invariants_passed": True,
            "error": str(exc),
        }
    p3_gates = {
        key: p3_delta[key] >= threshold
        for key, threshold in GATE_THRESHOLDS.items()
    }
    fallback_gates = {
        key: fallback_delta[key] >= threshold
        for key, threshold in GATE_THRESHOLDS.items()
    }
    if all(p3_gates.values()):
        status = "SP_PPAF_PASS"
        selected_arm = "P3"
    elif all(fallback_gates.values()):
        status = "SP_PPAF_FALLBACK_PASS"
        selected_arm = "All-A"
    else:
        status = "SP_PPAF_STOP"
        selected_arm = "none"
    return {
        "status": status,
        "selected_arm": selected_arm,
        "p3_delta": p3_delta,
        "fallback_delta": fallback_delta,
        "p3_gates": p3_gates,
        "fallback_gates": fallback_gates,
        "invariants_passed": invariants_passed is True,
    }

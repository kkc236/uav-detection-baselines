"""Frozen score-only causal-oracle primitives for SBR-RTDETR."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import struct
from typing import Any, Iterable, Mapping

import numpy as np

from .sbr_metrics import (
    _evaluate_threshold,
    _in_bin,
    _ioa_prediction_ignore,
    _prepare_predictions,
    _sqrt_effective_area,
    _validate,
    box_iou,
)
from .sbr_v2_audit import (
    AuditRawDetection,
    CClusterReconstruction,
    reconstruct_c_clusters,
)


CONF = 0.001
MAX_DET = 300
IOS = 0.5
THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
SIZE_BINS = ("tiny", "small", "medium", "large")
GATES = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class AggressorGroup:
    image_id: str
    unit_id: str
    stock_cluster_position: int
    stock_member_indices: tuple[int, ...]
    full_anchor_index: int
    aggressor_indices: tuple[int, ...]
    anchor_score: float


@dataclass(frozen=True)
class ScorePatch:
    image_id: str
    unit_id: str
    original_index: int
    full_anchor_index: int
    old_score: float
    new_score: float


@dataclass(frozen=True)
class OverlayReplay:
    retained_raw: tuple[AuditRawDetection, ...]
    overlaid_raw: tuple[AuditRawDetection, ...]
    active_raw: tuple[AuditRawDetection, ...]
    patches: tuple[ScorePatch, ...]
    reconstruction: CClusterReconstruction


@dataclass(frozen=True)
class OracleImage:
    image_id: str
    width: int
    height: int
    gt_boxes: tuple[tuple[float, float, float, float], ...]
    gt_classes: tuple[int, ...]
    ignore_boxes: tuple[tuple[float, float, float, float], ...]
    a_raw: tuple[AuditRawDetection, ...]
    c_raw: tuple[AuditRawDetection, ...]


@dataclass(frozen=True)
class GroupEvent:
    unit_id: str
    selected: bool
    reason: str
    before_profile: Mapping[
        str, Mapping[str, Mapping[str, int]]
    ]
    after_profile: Mapping[
        str, Mapping[str, Mapping[str, int]]
    ]
    tp_delta: Mapping[str, Mapping[str, int]]
    fp_delta: Mapping[str, Mapping[str, int]]
    group: AggressorGroup
    patches: tuple[ScorePatch, ...]


@dataclass(frozen=True)
class OracleImageResult:
    image_id: str
    groups: tuple[AggressorGroup, ...]
    events: tuple[GroupEvent, ...]
    stock: OverlayReplay
    joint: OverlayReplay
    selection_rounds: int
    stock_profile: Mapping[
        str, Mapping[str, Mapping[str, int]]
    ]
    joint_profile: Mapping[
        str, Mapping[str, Mapping[str, int]]
    ]


@dataclass(frozen=True)
class OracleGate:
    status: str
    deltas: Mapping[str, float]
    gates: Mapping[str, bool]


def _rank(
    record: AuditRawDetection,
) -> tuple[float, int, int, int]:
    return (
        -float(record.score),
        int(record.source_order),
        int(record.query_index),
        int(record.original_index),
    )


def find_aggressor_groups(
    retained_raw: Iterable[AuditRawDetection],
) -> tuple[AggressorGroup, ...]:
    raw = tuple(retained_raw)
    stock = reconstruct_c_clusters(raw)
    by_index = {record.original_index: record for record in raw}
    groups: list[AggressorGroup] = []
    for position, member_indices in enumerate(stock.cluster_members):
        members = tuple(by_index[index] for index in member_indices)
        full = tuple(
            sorted(
                (
                    member
                    for member in members
                    if member.source_order == 0
                ),
                key=_rank,
            )
        )
        if not full or not any(
            member.source_order > 0 for member in members
        ):
            continue
        anchor = full[0]
        aggressors = tuple(
            sorted(
                (
                    member
                    for member in members
                    if member.source_order > 0
                    and float(member.score) > float(anchor.score)
                ),
                key=_rank,
            )
        )
        if not aggressors:
            continue
        payload = {
            "image_id": anchor.image_id,
            "members": list(member_indices),
            "anchor": anchor.original_index,
            "aggressors": [
                member.original_index for member in aggressors
            ],
        }
        unit_id = (
            f"{anchor.image_id}:"
            f"{hashlib.sha256(_canonical(payload)).hexdigest()[:24]}"
        )
        groups.append(
            AggressorGroup(
                image_id=anchor.image_id,
                unit_id=unit_id,
                stock_cluster_position=position,
                stock_member_indices=tuple(member_indices),
                full_anchor_index=anchor.original_index,
                aggressor_indices=tuple(
                    member.original_index for member in aggressors
                ),
                anchor_score=float(anchor.score),
            )
        )
    return tuple(groups)


def apply_group_overlay(
    retained_raw: Iterable[AuditRawDetection],
    groups: Iterable[AggressorGroup],
) -> tuple[
    tuple[AuditRawDetection, ...],
    tuple[ScorePatch, ...],
]:
    raw = tuple(retained_raw)
    requested_groups = tuple(groups)
    if requested_groups:
        eligible = {
            group.unit_id: group
            for group in find_aggressor_groups(raw)
        }
        if any(
            eligible.get(group.unit_id) != group
            for group in requested_groups
        ):
            raise ValueError(
                "overlay group is not an exact stock eligible group"
            )
    by_index = {record.original_index: record for record in raw}
    if len(by_index) != len(raw):
        raise ValueError("retained raw identities must be unique")
    replacements: dict[int, tuple[float, AggressorGroup]] = {}
    for group in requested_groups:
        new_score = math.nextafter(
            float(group.anchor_score), -math.inf
        )
        if (
            not math.isfinite(new_score)
            or new_score >= group.anchor_score
        ):
            raise ValueError("invalid frozen predecessor score")
        anchor = by_index.get(group.full_anchor_index)
        if anchor is None or anchor.source_order != 0:
            raise ValueError(
                "group full anchor is missing or not full-view"
            )
        if float(anchor.score) != float(group.anchor_score):
            raise ValueError(
                "group anchor score disagrees with retained raw"
            )
        for original_index in group.aggressor_indices:
            record = by_index.get(original_index)
            if (
                record is None
                or record.source_order == 0
                or float(record.score) <= float(group.anchor_score)
                or original_index in replacements
            ):
                raise ValueError("invalid or duplicated aggressor")
            replacements[original_index] = (new_score, group)
    overlaid: list[AuditRawDetection] = []
    patches: list[ScorePatch] = []
    for record in raw:
        change = replacements.get(record.original_index)
        if change is None:
            overlaid.append(record)
            continue
        new_score, group = change
        overlaid.append(replace(record, score=new_score))
        patches.append(
            ScorePatch(
                image_id=record.image_id,
                unit_id=group.unit_id,
                original_index=record.original_index,
                full_anchor_index=group.full_anchor_index,
                old_score=float(record.score),
                new_score=float(new_score),
            )
        )
    return tuple(overlaid), tuple(patches)


def replay_overlay(
    retained_raw: Iterable[AuditRawDetection],
    groups: Iterable[AggressorGroup],
) -> OverlayReplay:
    raw = tuple(retained_raw)
    overlaid, patches = apply_group_overlay(raw, groups)
    active = tuple(
        record for record in overlaid if record.score >= CONF
    )
    reconstruction = reconstruct_c_clusters(active)
    return OverlayReplay(
        retained_raw=raw,
        overlaid_raw=overlaid,
        active_raw=active,
        patches=patches,
        reconstruction=reconstruction,
    )


def tp_fp_profile(
    image: OracleImage,
    replay: OverlayReplay,
) -> dict[str, dict[str, dict[str, int]]]:
    predictions = replay.reconstruction.standard_predictions
    boxes = [
        tuple(float(value) for value in prediction.global_xyxy)
        for prediction in predictions
    ]
    scores = [
        float(prediction.score) for prediction in predictions
    ]
    classes = [
        int(prediction.class_id) for prediction in predictions
    ]
    sources = [
        int(prediction.source_order) for prediction in predictions
    ]
    queries = [
        int(prediction.query_index) for prediction in predictions
    ]
    pb, ps, pc, gb, gc, ign, src, qry = _validate(
        boxes,
        scores,
        classes,
        image.gt_boxes,
        image.gt_classes,
        image.ignore_boxes,
        sources,
        queries,
        CONF,
    )
    pb, ps, pc, src, qry, _ = _prepare_predictions(
        pb,
        ps,
        pc,
        src,
        qry,
        CONF,
        MAX_DET,
    )
    neutral = _ioa_prediction_ignore(pb, ign)
    iou = box_iou(pb, gb)
    gain = min(
        640.0 / float(image.width),
        640.0 / float(image.height),
        1.0,
    )
    radius = _sqrt_effective_area(gb, gain)
    profile: dict[str, dict[str, dict[str, int]]] = {}
    for threshold in THRESHOLDS:
        masks = {"all": np.ones(len(gc), dtype=bool)}
        masks.update(
            {
                name: _in_bin(radius, name)
                for name in SIZE_BINS
            }
        )
        key = f"{threshold:.2f}"
        profile[key] = {}
        for name, selected in masks.items():
            counts, _ = _evaluate_threshold(
                pb,
                ps,
                pc,
                neutral,
                gb,
                gc,
                selected,
                iou,
                threshold,
            )
            profile[key][name] = {
                count: int(value)
                for count, value in counts.items()
            }
            profile[key][name]["gt"] = int(
                np.count_nonzero(selected)
            )
    return profile


def _count_delta(
    before: Mapping[str, Mapping[str, Mapping[str, int]]],
    after: Mapping[str, Mapping[str, Mapping[str, int]]],
    field: str,
) -> dict[str, dict[str, int]]:
    return {
        threshold: {
            name: int(
                after[threshold][name][field]
                - before[threshold][name][field]
            )
            for name in ("all", *SIZE_BINS)
        }
        for threshold in (
            f"{value:.2f}" for value in THRESHOLDS
        )
    }


def _selected(
    delta: Mapping[str, Mapping[str, int]],
) -> tuple[bool, str]:
    protected = ("all", "tiny", "large")
    safe = all(
        delta[f"{threshold:.2f}"][name] >= 0
        for threshold in THRESHOLDS
        for name in protected
    )
    large_gain = (
        sum(
            delta[f"{threshold:.2f}"]["large"]
            for threshold in THRESHOLDS
        )
        > 0
    )
    if safe and large_gain:
        return True, "SAFE_LARGE_GAIN"
    if not safe:
        return False, "TP_SAFETY_FAIL"
    return False, "NO_LARGE_GAIN"


def evaluate_oracle_image(
    image: OracleImage,
) -> OracleImageResult:
    groups = find_aggressor_groups(image.c_raw)
    stock = replay_overlay(image.c_raw, ())
    stock_profile = tp_fp_profile(image, stock)
    events: list[GroupEvent] = []
    selected: list[AggressorGroup] = []
    for group in groups:
        single = replay_overlay(image.c_raw, (group,))
        profile = tp_fp_profile(image, single)
        tp_delta = _count_delta(
            stock_profile, profile, "tp"
        )
        fp_delta = _count_delta(
            stock_profile, profile, "fp"
        )
        take, reason = _selected(tp_delta)
        if take:
            selected.append(group)
        events.append(
            GroupEvent(
                unit_id=group.unit_id,
                selected=take,
                reason=reason,
                before_profile=stock_profile,
                after_profile=profile,
                tp_delta=tp_delta,
                fp_delta=fp_delta,
                group=group,
                patches=single.patches,
            )
        )
    joint = replay_overlay(image.c_raw, tuple(selected))
    return OracleImageResult(
        image_id=image.image_id,
        groups=groups,
        events=tuple(events),
        stock=stock,
        joint=joint,
        selection_rounds=1,
        stock_profile=stock_profile,
        joint_profile=tp_fp_profile(image, joint),
    )


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", float(value))


def _raw_non_score_signature(
    record: AuditRawDetection,
) -> tuple[Any, ...]:
    def box_bits(
        box: tuple[float, float, float, float],
    ) -> tuple[bytes, ...]:
        return tuple(_float_bits(value) for value in box)

    return (
        record.image_id,
        record.arm,
        record.width,
        record.height,
        record.source_order,
        record.query_index,
        record.class_id,
        box_bits(record.network_xyxy),
        box_bits(record.view_xyxy),
        box_bits(record.global_xyxy),
        record.tile_bounds,
        record.original_index,
    )


def _raw_population_digest(
    records: Iterable[AuditRawDetection],
) -> str:
    payload = [
        {
            "non_score": [
                item.hex() if isinstance(item, bytes) else item
                for item in _flatten_signature(
                    _raw_non_score_signature(record)
                )
            ],
            "score_bits": _float_bits(record.score).hex(),
        }
        for record in records
    ]
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _flatten_signature(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_signature(item))
        return tuple(flattened)
    return (value,)


def _raw_population_is_legal(
    records: Iterable[AuditRawDetection],
) -> bool:
    for record in records:
        if (
            not math.isfinite(record.score)
            or not 0.0 <= record.score <= 1.0
            or record.width <= 0
            or record.height <= 0
        ):
            return False
        for box in (
            record.network_xyxy,
            record.view_xyxy,
            record.global_xyxy,
        ):
            if (
                len(box) != 4
                or not all(math.isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                return False
    return True


def _expected_oracle_components(
    image: OracleImage,
) -> tuple[
    tuple[AggressorGroup, ...],
    OverlayReplay,
    Mapping[str, Mapping[str, Mapping[str, int]]],
    tuple[GroupEvent, ...],
    OverlayReplay,
]:
    groups = find_aggressor_groups(image.c_raw)
    stock = replay_overlay(image.c_raw, ())
    stock_profile = tp_fp_profile(image, stock)
    events: list[GroupEvent] = []
    selected: list[AggressorGroup] = []
    for group in groups:
        single = replay_overlay(image.c_raw, (group,))
        after_profile = tp_fp_profile(image, single)
        tp_delta = _count_delta(
            stock_profile, after_profile, "tp"
        )
        fp_delta = _count_delta(
            stock_profile, after_profile, "fp"
        )
        take, reason = _selected(tp_delta)
        if take:
            selected.append(group)
        events.append(
            GroupEvent(
                unit_id=group.unit_id,
                selected=take,
                reason=reason,
                before_profile=stock_profile,
                after_profile=after_profile,
                tp_delta=tp_delta,
                fp_delta=fp_delta,
                group=group,
                patches=single.patches,
            )
        )
    joint = replay_overlay(image.c_raw, selected)
    return groups, stock, stock_profile, tuple(events), joint


def verify_oracle_image_invariants(
    image: OracleImage,
    result: OracleImageResult,
) -> dict[str, bool]:
    """Independently replay and verify every frozen score-only invariant."""

    (
        expected_groups,
        expected_stock,
        expected_stock_profile,
        expected_events,
        expected_joint,
    ) = _expected_oracle_components(image)
    expected_selected = tuple(
        event.group for event in expected_events if event.selected
    )
    selected_indices = {
        index
        for group in expected_selected
        for index in group.aggressor_indices
    }
    original_by_index = {
        record.original_index: record for record in image.c_raw
    }
    joint_by_index = {
        record.original_index: record
        for record in result.joint.overlaid_raw
    }
    stock_by_index = {
        record.original_index: record
        for record in result.stock.overlaid_raw
    }
    original_order = tuple(
        record.original_index for record in image.c_raw
    )
    retained_identity_count_unchanged = (
        len(original_by_index) == len(image.c_raw)
        and tuple(
            record.original_index
            for record in result.stock.retained_raw
        )
        == original_order
        and tuple(
            record.original_index
            for record in result.stock.overlaid_raw
        )
        == original_order
        and tuple(
            record.original_index
            for record in result.joint.retained_raw
        )
        == original_order
        and tuple(
            record.original_index
            for record in result.joint.overlaid_raw
        )
        == original_order
    )
    active_exclusions_exact = all(
        replay.active_raw
        == tuple(
            record
            for record in replay.overlaid_raw
            if record.score >= CONF
        )
        for replay in (result.stock, result.joint)
    )
    actual_modified = {
        index
        for index, original in original_by_index.items()
        if index in joint_by_index
        and _float_bits(joint_by_index[index].score)
        != _float_bits(original.score)
    }
    modified_records_exact_selected_aggressors = (
        actual_modified == selected_indices
        and all(
            original_by_index[index].source_order > 0
            for index in actual_modified
        )
    )
    modified_scores_exact_predecessor = all(
        index in joint_by_index
        and _float_bits(joint_by_index[index].score)
        == _float_bits(
            math.nextafter(group.anchor_score, -math.inf)
        )
        and joint_by_index[index].score < group.anchor_score
        for group in expected_selected
        for index in group.aggressor_indices
    )
    unselected_scores_bit_identical = all(
        index in joint_by_index
        and _float_bits(joint_by_index[index].score)
        == _float_bits(original.score)
        for index, original in original_by_index.items()
        if index not in selected_indices
    )
    non_score_fields_bit_identical = all(
        index in stock_by_index
        and index in joint_by_index
        and _raw_non_score_signature(stock_by_index[index])
        == _raw_non_score_signature(original)
        and _raw_non_score_signature(joint_by_index[index])
        == _raw_non_score_signature(original)
        for index, original in original_by_index.items()
    )
    complete_single_replays = (
        len(result.events) == len(expected_events)
        and result.events == expected_events
    )
    complete_joint_replay = result.joint == expected_joint
    no_direct_prediction_editing = all(
        replay.reconstruction
        == reconstruct_c_clusters(replay.active_raw)
        for replay in (result.stock, result.joint)
    )
    no_op_digest_exact = (
        _raw_population_digest(result.stock.retained_raw)
        == _raw_population_digest(image.c_raw)
        and _raw_population_digest(result.stock.overlaid_raw)
        == _raw_population_digest(image.c_raw)
        and result.stock == expected_stock
    )
    patches_exact = result.joint.patches == expected_joint.patches
    finite_and_legal = (
        _raw_population_is_legal(image.c_raw)
        and _raw_population_is_legal(result.stock.retained_raw)
        and _raw_population_is_legal(result.stock.overlaid_raw)
        and _raw_population_is_legal(result.stock.active_raw)
        and _raw_population_is_legal(result.joint.retained_raw)
        and _raw_population_is_legal(result.joint.overlaid_raw)
        and _raw_population_is_legal(result.joint.active_raw)
        and all(
            math.isfinite(patch.old_score)
            and math.isfinite(patch.new_score)
            and 0.0 <= patch.old_score <= 1.0
            and 0.0 <= patch.new_score <= 1.0
            for patch in (
                *result.stock.patches,
                *result.joint.patches,
            )
        )
    )
    selection_is_one_frozen_round = (
        result.selection_rounds == 1
        and result.groups == expected_groups
        and result.events == expected_events
    )
    absolute_profiles_complete = (
        result.stock_profile == expected_stock_profile
        and result.joint_profile
        == tp_fp_profile(image, expected_joint)
        and all(
            event.before_profile == expected_stock_profile
            and event.after_profile
            == expected_events[index].after_profile
            for index, event in enumerate(result.events)
        )
        if len(result.events) == len(expected_events)
        else False
    )
    checks = {
        "retained_identity_count_unchanged": (
            retained_identity_count_unchanged
        ),
        "active_exclusions_exact": active_exclusions_exact,
        "modified_records_exact_selected_aggressors": (
            modified_records_exact_selected_aggressors
        ),
        "modified_scores_exact_predecessor": (
            modified_scores_exact_predecessor
        ),
        "unselected_scores_bit_identical": (
            unselected_scores_bit_identical
        ),
        "non_score_fields_bit_identical": (
            non_score_fields_bit_identical
        ),
        "complete_single_replays": complete_single_replays,
        "complete_joint_replay": complete_joint_replay,
        "no_direct_prediction_editing": no_direct_prediction_editing,
        "no_op_digest_exact": no_op_digest_exact,
        "patches_exact": patches_exact,
        "finite_and_legal": finite_and_legal,
        "selection_is_one_frozen_round": (
            selection_is_one_frozen_round
        ),
        "absolute_profiles_complete": absolute_profiles_complete,
    }
    return {
        **checks,
        "passed": all(checks.values()),
    }


def gate_oracle_metrics(
    a_metrics: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any],
    *,
    selected_count: int,
) -> OracleGate:
    deltas = {
        name: float(oracle_metrics[name])
        - float(a_metrics[name])
        for name in GATES
    }
    if not all(math.isfinite(value) for value in deltas.values()):
        raise ValueError("oracle gate metrics must be finite")
    gates = {
        name: deltas[name] >= threshold
        for name, threshold in GATES.items()
    }
    status = (
        "SBR_SCORE_ORACLE_GO"
        if selected_count > 0 and all(gates.values())
        else "SBR_SCORE_ORACLE_STOP"
    )
    return OracleGate(
        status=status,
        deltas=deltas,
        gates=gates,
    )


__all__ = [
    "AggressorGroup",
    "CONF",
    "GATES",
    "GroupEvent",
    "IOS",
    "MAX_DET",
    "OracleGate",
    "OracleImage",
    "OracleImageResult",
    "OverlayReplay",
    "ScorePatch",
    "SIZE_BINS",
    "THRESHOLDS",
    "apply_group_overlay",
    "evaluate_oracle_image",
    "find_aggressor_groups",
    "gate_oracle_metrics",
    "replay_overlay",
    "tp_fp_profile",
    "verify_oracle_image_invariants",
]

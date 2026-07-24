"""Frozen score-only causal-oracle primitives for SBR-RTDETR."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Iterable

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
    active_raw: tuple[AuditRawDetection, ...]
    patches: tuple[ScorePatch, ...]
    reconstruction: CClusterReconstruction


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
    by_index = {record.original_index: record for record in raw}
    if len(by_index) != len(raw):
        raise ValueError("retained raw identities must be unique")
    replacements: dict[int, tuple[float, AggressorGroup]] = {}
    for group in groups:
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
        active_raw=active,
        patches=patches,
        reconstruction=reconstruction,
    )


__all__ = [
    "AggressorGroup",
    "CONF",
    "GATES",
    "IOS",
    "MAX_DET",
    "OverlayReplay",
    "ScorePatch",
    "SIZE_BINS",
    "THRESHOLDS",
    "apply_group_overlay",
    "find_aggressor_groups",
    "replay_overlay",
]

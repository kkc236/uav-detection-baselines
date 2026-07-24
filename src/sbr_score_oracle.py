"""Frozen score-only causal-oracle primitives for SBR-RTDETR."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from .sbr_v2_audit import AuditRawDetection, reconstruct_c_clusters


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


__all__ = [
    "AggressorGroup",
    "CONF",
    "GATES",
    "IOS",
    "MAX_DET",
    "SIZE_BINS",
    "THRESHOLDS",
    "find_aggressor_groups",
]

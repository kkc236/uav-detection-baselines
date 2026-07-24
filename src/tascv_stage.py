"""Frozen stage policy for the independent T-ASCV expert."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class TASCVStage(str, Enum):
    PREFLIGHT_1 = "PREFLIGHT_1"
    TINY_MECHANISM_500 = "TINY_MECHANISM_500"
    SCREEN_10 = "SCREEN_10"
    FORMAL_100 = "FORMAL_100"


@dataclass(frozen=True)
class TASCVStagePolicy:
    epochs: int
    uses_hashed_subset: bool
    builds_val_loader: bool
    validates_each_epoch: bool
    runs_final_eval: bool
    max_train_batches: int | None
    expected_successful_batches: int
    expected_optimizer_attempts: int


_POLICIES = MappingProxyType({
    TASCVStage.PREFLIGHT_1: TASCVStagePolicy(
        100, True, False, False, False, 1, 1, 1
    ),
    TASCVStage.TINY_MECHANISM_500: TASCVStagePolicy(
        100, True, False, False, False, 500, 500, 106
    ),
    TASCVStage.SCREEN_10: TASCVStagePolicy(
        10, True, False, False, False, None, 810, 145
    ),
    TASCVStage.FORMAL_100: TASCVStagePolicy(
        100, False, False, False, False, None, 80_900, 10_556
    ),
})


def _stage(stage: TASCVStage | str) -> TASCVStage:
    try:
        return stage if isinstance(stage, TASCVStage) else TASCVStage(stage)
    except ValueError as error:
        raise ValueError(f"unknown T-ASCV stage: {stage!r}") from error


def stage_policy(stage: TASCVStage | str) -> TASCVStagePolicy:
    return _POLICIES[_stage(stage)]


def allowed_seeds(stage: TASCVStage | str) -> frozenset[int]:
    resolved = _stage(stage)
    if resolved is TASCVStage.PREFLIGHT_1:
        return frozenset({0})
    if resolved is TASCVStage.TINY_MECHANISM_500:
        return frozenset({1})
    return frozenset({0, 1, 2})


def allowed_observed_tensor_batch_sizes(
    stage: TASCVStage | str,
) -> frozenset[int]:
    return (
        frozenset({8})
        if _stage(stage) is TASCVStage.PREFLIGHT_1
        else frozenset({7, 8})
    )


__all__ = [
    "TASCVStage",
    "TASCVStagePolicy",
    "allowed_observed_tensor_batch_sizes",
    "allowed_seeds",
    "stage_policy",
]

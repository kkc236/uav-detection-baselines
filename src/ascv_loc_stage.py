from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ASCVStage(str, Enum):
    PREFLIGHT_1 = "PREFLIGHT_1"
    MECHANISM_500 = "MECHANISM_500"
    SCREEN_10 = "SCREEN_10"
    SEED0_100 = "SEED0_100"
    SEED1_100 = "SEED1_100"
    SEED2_100 = "SEED2_100"


@dataclass(frozen=True)
class ASCVStagePolicy:
    epochs: int
    uses_hashed_subset: bool
    builds_val_loader: bool
    validates_each_epoch: bool
    runs_final_eval: bool
    max_train_batches: int | None
    expected_successful_batches: int
    expected_optimizer_attempts: int


_POLICIES = {
    ASCVStage.PREFLIGHT_1: ASCVStagePolicy(100, True, False, False, False, 1, 1, 1),
    ASCVStage.MECHANISM_500: ASCVStagePolicy(100, True, False, False, False, 500, 500, 106),
    ASCVStage.SCREEN_10: ASCVStagePolicy(10, True, False, False, False, None, 810, 145),
    # Fixed epoch-100 is the paper checkpoint. There is no best.pt selection
    # and no per-epoch val inspection in any ASCV-Loc stage.
    ASCVStage.SEED0_100: ASCVStagePolicy(100, False, False, False, False, None, 80_900, 10_556),
    ASCVStage.SEED1_100: ASCVStagePolicy(100, False, False, False, False, None, 80_900, 10_556),
    ASCVStage.SEED2_100: ASCVStagePolicy(100, False, False, False, False, None, 80_900, 10_556),
}


def stage_policy(stage: ASCVStage | str) -> ASCVStagePolicy:
    try:
        normalized = stage if isinstance(stage, ASCVStage) else ASCVStage(stage)
    except ValueError as error:
        raise ValueError(f"unknown ASCV-Loc stage: {stage!r}") from error
    return _POLICIES[normalized]


def allowed_seeds(stage: ASCVStage | str) -> frozenset[int]:
    normalized = stage if isinstance(stage, ASCVStage) else ASCVStage(stage)
    if normalized in {ASCVStage.PREFLIGHT_1, ASCVStage.MECHANISM_500, ASCVStage.SEED0_100}:
        return frozenset({0})
    if normalized is ASCVStage.SCREEN_10:
        return frozenset({0, 1, 2})
    if normalized is ASCVStage.SEED1_100:
        return frozenset({1})
    if normalized is ASCVStage.SEED2_100:
        return frozenset({2})
    raise AssertionError(f"unmapped ASCV stage: {normalized}")

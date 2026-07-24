from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ASCVStage(str, Enum):
    MECHANISM_500 = "MECHANISM_500"
    SCREEN_6 = "SCREEN_6"
    FULL_20 = "FULL_20"
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


_POLICIES = {
    ASCVStage.MECHANISM_500: ASCVStagePolicy(100, True, False, False, False, 500),
    ASCVStage.SCREEN_6: ASCVStagePolicy(6, True, False, False, False, None),
    ASCVStage.FULL_20: ASCVStagePolicy(20, False, False, False, False, None),
    # Fixed epoch-100 is the paper checkpoint. There is no best.pt selection
    # and no per-epoch val inspection in any ASCV-Loc stage.
    ASCVStage.SEED0_100: ASCVStagePolicy(100, False, False, False, False, None),
    ASCVStage.SEED1_100: ASCVStagePolicy(100, False, False, False, False, None),
    ASCVStage.SEED2_100: ASCVStagePolicy(100, False, False, False, False, None),
}


def stage_policy(stage: ASCVStage | str) -> ASCVStagePolicy:
    try:
        normalized = stage if isinstance(stage, ASCVStage) else ASCVStage(stage)
    except ValueError as error:
        raise ValueError(f"unknown ASCV-Loc stage: {stage!r}") from error
    return _POLICIES[normalized]

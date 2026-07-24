"""Runtime and mechanism diagnostics for T-ASCV."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

from src.tascv import TASCVLossResult
from src.tascv_stage import TASCVStage


@dataclass(frozen=True)
class TASCVMechanismGate:
    successful_batches: int = 500
    scientific_tail_window: tuple[int, int] = (401, 500)
    minimum_tiny_pairs: int = 100
    minimum_tiny_batches: int = 80
    advantage_strictly_positive: bool = True
    win_rate_strictly_greater_than: float = 0.5


FROZEN_TASCV_MECHANISM_GATE = TASCVMechanismGate()


def _strict_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def validate_tascv_checkpoint_runtime(
    *,
    stage: TASCVStage | str,
    calls: int,
    batchnorm_preserved: bool,
) -> None:
    resolved = TASCVStage(stage)
    call_count = _strict_count("calls", calls)
    if type(batchnorm_preserved) is not bool:
        raise TypeError("batchnorm_preserved must be a bool")
    if not batchnorm_preserved:
        raise RuntimeError("TASCV_LOCAL_BRANCH_BATCHNORM_DRIFT")
    allowed_calls = (
        {2}
        if resolved is TASCVStage.PREFLIGHT_1
        else {1, 2}
    )
    if call_count not in allowed_calls:
        raise RuntimeError(
            "TASCV_CHECKPOINT_RECOMPUTE_INVALID: "
            f"stage={resolved.value}, calls={call_count}, "
            f"allowed={sorted(allowed_calls)}"
        )


class TASCVMechanismAccumulator:
    """Accumulate immutable records; all public counters are derived."""

    def __init__(self) -> None:
        self._results: list[dict[str, int | float]] = []

    @property
    def batches(self) -> int:
        return len(self._results)

    @property
    def matched_pairs(self) -> int:
        return sum(int(item["matched_pairs"]) for item in self._results)

    @property
    def auxiliary_tiny_pairs(self) -> int:
        return sum(
            int(item["auxiliary_tiny_pairs"]) for item in self._results
        )

    @property
    def excluded_non_tiny_pairs(self) -> int:
        return sum(
            int(item["excluded_non_tiny_pairs"])
            for item in self._results
        )

    @property
    def auxiliary_non_tiny_pairs(self) -> int:
        return 0

    @property
    def tiny_advantage_sum(self) -> float:
        return sum(
            float(item["tiny_teacher_advantage_sum"])
            for item in self._results
        )

    @property
    def tiny_wins(self) -> int:
        return sum(
            int(item["tiny_teacher_win_count"])
            for item in self._results
        )

    def record(
        self,
        result: TASCVLossResult,
        *,
        auxiliary_non_tiny_pair_count: int = 0,
    ) -> None:
        forbidden = _strict_count(
            "auxiliary_non_tiny_pair_count",
            auxiliary_non_tiny_pair_count,
        )
        if forbidden:
            raise RuntimeError(
                "TASCV_NON_TINY_AUXILIARY_CONTRIBUTION_INVALID"
            )
        tiny = _strict_count(
            "auxiliary_tiny_pair_count",
            result.auxiliary_tiny_pair_count,
        )
        excluded = _strict_count(
            "excluded_non_tiny_pair_count",
            result.excluded_non_tiny_pair_count,
        )
        matched = _strict_count(
            "matched_pair_count",
            result.matched_pair_count,
        )
        wins = _strict_count(
            "tiny_teacher_win_count",
            result.tiny_teacher_win_count,
        )
        raw_advantage = result.tiny_teacher_advantage_sum
        if (
            not hasattr(raw_advantage, "detach")
            or raw_advantage.numel() != 1
        ):
            raise TypeError(
                "tiny_teacher_advantage_sum must be a scalar tensor"
            )
        advantage = float(raw_advantage.detach().cpu())
        if (
            matched != tiny + excluded
            or wins > tiny
            or not math.isfinite(advantage)
            or (tiny == 0 and (wins != 0 or advantage != 0.0))
        ):
            raise RuntimeError("TASCV_MECHANISM_RECORD_INVALID")
        self._results.append(
            {
                "matched_pairs": matched,
                "auxiliary_tiny_pairs": tiny,
                "excluded_non_tiny_pairs": excluded,
                "tiny_teacher_advantage_sum": advantage,
                "tiny_teacher_win_count": wins,
            }
        )

    @staticmethod
    def _summary(
        results: list[dict[str, int | float]],
    ) -> dict[str, int | float | None]:
        tiny_pairs = sum(
            int(item["auxiliary_tiny_pairs"])
            for item in results
        )
        advantage = sum(
            float(item["tiny_teacher_advantage_sum"])
            for item in results
        )
        wins = sum(
            int(item["tiny_teacher_win_count"])
            for item in results
        )
        if not math.isfinite(advantage):
            raise RuntimeError("TASCV_MECHANISM_SUMMARY_NONFINITE")
        return {
            "batches": len(results),
            "matched_pairs": sum(
                int(item["matched_pairs"]) for item in results
            ),
            "tiny_pairs": tiny_pairs,
            "excluded_non_tiny_pairs": sum(
                int(item["excluded_non_tiny_pairs"])
                for item in results
            ),
            "auxiliary_non_tiny_pairs": 0,
            "tiny_batches_with_pairs": sum(
                int(item["auxiliary_tiny_pairs"]) > 0
                for item in results
            ),
            "tiny_teacher_advantage_mean": (
                advantage / tiny_pairs if tiny_pairs else None
            ),
            "tiny_teacher_win_rate": (
                wins / tiny_pairs if tiny_pairs else None
            ),
        }

    def summary(self) -> dict:
        gate = FROZEN_TASCV_MECHANISM_GATE
        tail_size = (
            gate.scientific_tail_window[1]
            - gate.scientific_tail_window[0]
            + 1
        )
        tail_results = self._results[-tail_size:]
        tail_start = (
            self.batches - len(tail_results) + 1
            if tail_results
            else 0
        )
        return {
            "all": self._summary(self._results),
            "tail": self._summary(tail_results),
            "tail_window": [tail_start, self.batches],
        }

    def mechanism_gate(self) -> tuple[bool, list[str]]:
        gate = FROZEN_TASCV_MECHANISM_GATE
        failures: list[str] = []
        summary = self.summary()
        tail = summary["tail"]
        if self.batches != gate.successful_batches:
            failures.append(
                f"successful_batches={self.batches}, "
                f"expected={gate.successful_batches}"
            )
        expected_window = list(gate.scientific_tail_window)
        if summary["tail_window"] != expected_window:
            failures.append(
                f"tail_window={summary['tail_window']}, "
                f"expected={expected_window}"
            )
        if int(tail["tiny_batches_with_pairs"]) < gate.minimum_tiny_batches:
            failures.append(
                "tail_tiny_batches_with_pairs"
                f"<{gate.minimum_tiny_batches}"
            )
        if int(tail["tiny_pairs"]) < gate.minimum_tiny_pairs:
            failures.append(f"tail_tiny_pairs<{gate.minimum_tiny_pairs}")
        if int(tail["tiny_pairs"]) > 0:
            if float(tail["tiny_teacher_advantage_mean"]) <= 0:
                failures.append("tiny_teacher_advantage_mean<=0")
            if (
                float(tail["tiny_teacher_win_rate"])
                <= gate.win_rate_strictly_greater_than
            ):
                failures.append(
                    "tiny_teacher_win_rate"
                    f"<={gate.win_rate_strictly_greater_than}"
                )
        return not failures, failures


__all__ = [
    "FROZEN_TASCV_MECHANISM_GATE",
    "TASCVMechanismAccumulator",
    "TASCVMechanismGate",
    "validate_tascv_checkpoint_runtime",
]

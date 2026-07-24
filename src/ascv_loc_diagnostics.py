from __future__ import annotations

from dataclasses import dataclass, field

from src.ascv_loc import ASCVLocLossResult
from src.ascv_loc_protocol import FROZEN_MECHANISM_GATE
from src.ascv_loc_stage import ASCVStage


def validate_local_checkpoint_runtime(
    *,
    stage: ASCVStage | str,
    calls: int,
    batchnorm_preserved: bool,
    non_tiny_pair_count: int,
) -> None:
    """Fail closed on impossible local-forward behavior without rejecting teacher-only batches."""

    resolved_stage = ASCVStage(stage)
    if not batchnorm_preserved:
        raise RuntimeError("ASCV_LOC_LOCAL_BRANCH_BATCHNORM_DRIFT")
    requires_recompute = (
        resolved_stage is ASCVStage.PREFLIGHT_1 or int(non_tiny_pair_count) > 0
    )
    allowed_calls = {2} if requires_recompute else {1, 2}
    if int(calls) not in allowed_calls:
        raise RuntimeError(
            "ASCV_LOC_CHECKPOINT_RECOMPUTE_INVALID: "
            f"stage={resolved_stage.value}, calls={calls}, allowed={sorted(allowed_calls)}"
        )


@dataclass
class ASCVMechanismAccumulator:
    batches: int = 0
    pairs: int = 0
    tiny_pairs: int = 0
    non_tiny_pairs: int = 0
    tiny_advantage_sum: float = 0.0
    tiny_wins: int = 0
    non_tiny_advantage_sum: float = 0.0
    non_tiny_wins: int = 0
    _results: list[dict[str, int | float]] = field(default_factory=list, repr=False)

    def record(self, result: ASCVLocLossResult) -> None:
        self._results.append(
            {
                "pair_count": int(result.pair_count),
                "tiny_pair_count": int(result.tiny_pair_count),
                "non_tiny_pair_count": int(result.non_tiny_pair_count),
                "tiny_teacher_advantage_sum": float(
                    result.tiny_teacher_advantage_sum.detach().cpu()
                ),
                "tiny_teacher_win_count": int(result.tiny_teacher_win_count),
                "non_tiny_teacher_advantage_sum": float(
                    result.non_tiny_teacher_advantage_sum.detach().cpu()
                ),
                "non_tiny_teacher_win_count": int(result.non_tiny_teacher_win_count),
            }
        )
        self.batches += 1
        self.pairs += result.pair_count
        self.tiny_pairs += result.tiny_pair_count
        self.non_tiny_pairs += result.non_tiny_pair_count
        self.tiny_advantage_sum += float(result.tiny_teacher_advantage_sum.cpu())
        self.tiny_wins += result.tiny_teacher_win_count
        self.non_tiny_advantage_sum += float(result.non_tiny_teacher_advantage_sum.cpu())
        self.non_tiny_wins += result.non_tiny_teacher_win_count

    @staticmethod
    def _scale_summary(results: list[dict[str, int | float]]) -> dict:
        tiny_pairs = sum(int(result["tiny_pair_count"]) for result in results)
        non_tiny_pairs = sum(int(result["non_tiny_pair_count"]) for result in results)
        tiny_advantage = sum(float(result["tiny_teacher_advantage_sum"]) for result in results)
        non_tiny_advantage = sum(float(result["non_tiny_teacher_advantage_sum"]) for result in results)
        tiny_wins = sum(int(result["tiny_teacher_win_count"]) for result in results)
        non_tiny_wins = sum(int(result["non_tiny_teacher_win_count"]) for result in results)
        return {
            "batches": len(results),
            "pairs": sum(int(result["pair_count"]) for result in results),
            "tiny_pairs": tiny_pairs,
            "non_tiny_pairs": non_tiny_pairs,
            "tiny_batches_with_pairs": sum(int(result["tiny_pair_count"]) > 0 for result in results),
            "non_tiny_batches_with_pairs": sum(int(result["non_tiny_pair_count"]) > 0 for result in results),
            "tiny_teacher_advantage_mean": tiny_advantage / tiny_pairs if tiny_pairs else None,
            "tiny_teacher_win_rate": tiny_wins / tiny_pairs if tiny_pairs else None,
            "non_tiny_teacher_advantage_mean": non_tiny_advantage / non_tiny_pairs if non_tiny_pairs else None,
            "non_tiny_teacher_win_rate": non_tiny_wins / non_tiny_pairs if non_tiny_pairs else None,
        }

    def summary(self) -> dict:
        all_summary = self._scale_summary(self._results)
        tail_size = (
            FROZEN_MECHANISM_GATE["scientific_tail_window"][1]
            - FROZEN_MECHANISM_GATE["scientific_tail_window"][0]
            + 1
        )
        tail_results = self._results[-tail_size:]
        tail_summary = self._scale_summary(tail_results)
        tail_start = self.batches - len(tail_results) + 1 if tail_results else 0
        return {
            **tail_summary,
            "all": all_summary,
            "tail": tail_summary,
            "tail_window": [tail_start, self.batches],
        }

    def mechanism_gate(
        self,
        expected_batches: int = FROZEN_MECHANISM_GATE["successful_batches"],
    ) -> tuple[bool, list[str]]:
        summary = self.summary()
        failures = []
        if self.batches != expected_batches:
            failures.append(f"successful_batches={self.batches}, expected={expected_batches}")
        all_summary = summary["all"]
        tail = summary["tail"]
        if all_summary["pairs"] <= 0:
            failures.append("shared_pairs=0")
        for scale in ("tiny", "non_tiny"):
            if all_summary[f"{scale}_pairs"] <= 0:
                failures.append(f"{scale}_pairs=0")
                continue
            tail_pairs = tail[f"{scale}_pairs"]
            tail_batches = tail[f"{scale}_batches_with_pairs"]
            minimum_batches = FROZEN_MECHANISM_GATE["minimum_batches_per_direction"]
            minimum_pairs = FROZEN_MECHANISM_GATE["minimum_pairs_per_direction"]
            if tail_batches < minimum_batches:
                failures.append(
                    f"tail_{scale}_batches_with_pairs={tail_batches}, required>={minimum_batches}"
                )
            if tail_pairs < minimum_pairs:
                failures.append(f"tail_{scale}_pairs={tail_pairs}, required>={minimum_pairs}")
            if not tail_pairs:
                continue
            if tail[f"{scale}_teacher_advantage_mean"] <= 0:
                failures.append(f"{scale}_teacher_advantage_mean<=0")
            if (
                tail[f"{scale}_teacher_win_rate"]
                <= FROZEN_MECHANISM_GATE["win_rate_strictly_greater_than"]
            ):
                failures.append(f"{scale}_teacher_win_rate<=0.5")
        return not failures, failures

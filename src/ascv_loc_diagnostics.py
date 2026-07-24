from __future__ import annotations

from dataclasses import dataclass

from src.ascv_loc import ASCVLocLossResult


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

    def record(self, result: ASCVLocLossResult) -> None:
        self.batches += 1
        self.pairs += result.pair_count
        self.tiny_pairs += result.tiny_pair_count
        self.non_tiny_pairs += result.non_tiny_pair_count
        self.tiny_advantage_sum += float(result.tiny_teacher_advantage_sum.cpu())
        self.tiny_wins += result.tiny_teacher_win_count
        self.non_tiny_advantage_sum += float(result.non_tiny_teacher_advantage_sum.cpu())
        self.non_tiny_wins += result.non_tiny_teacher_win_count

    def summary(self) -> dict:
        tiny_mean = self.tiny_advantage_sum / self.tiny_pairs if self.tiny_pairs else None
        non_tiny_mean = self.non_tiny_advantage_sum / self.non_tiny_pairs if self.non_tiny_pairs else None
        tiny_win_rate = self.tiny_wins / self.tiny_pairs if self.tiny_pairs else None
        non_tiny_win_rate = self.non_tiny_wins / self.non_tiny_pairs if self.non_tiny_pairs else None
        return {
            "batches": self.batches,
            "pairs": self.pairs,
            "tiny_pairs": self.tiny_pairs,
            "non_tiny_pairs": self.non_tiny_pairs,
            "tiny_teacher_advantage_mean": tiny_mean,
            "tiny_teacher_win_rate": tiny_win_rate,
            "non_tiny_teacher_advantage_mean": non_tiny_mean,
            "non_tiny_teacher_win_rate": non_tiny_win_rate,
        }

    def mechanism_gate(self, expected_batches: int = 500) -> tuple[bool, list[str]]:
        summary = self.summary()
        failures = []
        if self.batches != expected_batches:
            failures.append(f"successful_batches={self.batches}, expected={expected_batches}")
        if self.pairs <= 0:
            failures.append("shared_pairs=0")
        for scale in ("tiny", "non_tiny"):
            if summary[f"{scale}_pairs"] <= 0:
                failures.append(f"{scale}_pairs=0")
                continue
            if summary[f"{scale}_teacher_advantage_mean"] <= 0:
                failures.append(f"{scale}_teacher_advantage_mean<=0")
            if summary[f"{scale}_teacher_win_rate"] <= 0.5:
                failures.append(f"{scale}_teacher_win_rate<=0.5")
        return not failures, failures

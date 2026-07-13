from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


BATCH_LEVELS = (10, 12, 14, 16, 18)
PROMOTION_PEAK_GIB = {10: 22.0, 12: 24.0, 14: 26.0, 16: 28.0}
OOM_COOLDOWNS = (5, 10, 20)


@dataclass
class AdaptiveBatchState:
    current_batch: int = 16
    completed_epoch: int = 3
    cooldown_remaining: int = 0
    oom_count: int = 0
    stable_epochs: int = 0
    last_peak_gib: float = 0.0
    checkpoint: str = ""
    last_event: str = "start"
    unexpected_failures: int = 0

    def _lower_batch(self) -> int:
        index = BATCH_LEVELS.index(self.current_batch)
        return BATCH_LEVELS[max(0, index - 1)]

    def _higher_batch(self) -> int:
        index = BATCH_LEVELS.index(self.current_batch)
        return BATCH_LEVELS[min(len(BATCH_LEVELS) - 1, index + 1)]

    def record_oom(self) -> int:
        self.oom_count += 1
        self.current_batch = self._lower_batch()
        self.cooldown_remaining = OOM_COOLDOWNS[min(self.oom_count - 1, len(OOM_COOLDOWNS) - 1)]
        self.stable_epochs = 0
        self.last_event = "oom_demote"
        return self.current_batch

    def record_epoch(self, *, peak_gib: float, completed_epoch: int | None = None) -> int:
        if completed_epoch is None:
            self.completed_epoch += 1
        else:
            self.completed_epoch = max(self.completed_epoch, completed_epoch)
        self.last_peak_gib = peak_gib

        if self.current_batch == 18 and peak_gib >= 29.0:
            self.current_batch = self._lower_batch()
            self.cooldown_remaining = max(self.cooldown_remaining, OOM_COOLDOWNS[0])
            self.stable_epochs = 0
            self.last_event = "peak_demote"
            return self.current_batch

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.stable_epochs = 0
            self.last_event = "cooldown"
            return self.current_batch

        threshold = PROMOTION_PEAK_GIB.get(self.current_batch)
        if threshold is not None and peak_gib < threshold:
            self.stable_epochs += 1
            self.last_event = "stable"
        else:
            self.stable_epochs = 0
            self.last_event = "hold"

        if self.stable_epochs >= 3:
            self.current_batch = self._higher_batch()
            self.stable_epochs = 0
            self.last_event = "promote"

        return self.current_batch


def save_state(path: Path, state: AdaptiveBatchState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_state(path: Path) -> AdaptiveBatchState:
    return AdaptiveBatchState(**json.loads(path.read_text(encoding="utf-8")))

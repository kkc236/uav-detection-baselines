from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchPolicy:
    levels: tuple[int, ...]
    initial_batch: int
    reason: str = "vram_default"


@dataclass(frozen=True)
class GPUProfile:
    name: str
    total_gib: float
    free_gib: float
    cuda_version: str
    driver_version: str


@dataclass
class AdaptiveTrainingState:
    levels: tuple[int, ...]
    current_batch: int
    amp_enabled: bool = True
    completed_epoch: int = 0
    cooldown_remaining: int = 0
    stable_epochs: int = 0
    oom_count: int = 0
    numeric_failure_count: int = 0
    unexpected_failures: int = 0
    last_peak_gib: float = 0.0
    checkpoint: str = ""
    last_event: str = "start"

    def _lower(self) -> int:
        index = self.levels.index(self.current_batch)
        return self.levels[max(0, index - 1)]

    def _higher(self) -> int:
        index = self.levels.index(self.current_batch)
        return self.levels[min(len(self.levels) - 1, index + 1)]

    def record_epoch(self, *, completed_epoch: int, peak_gib: float, total_gib: float) -> int:
        self.completed_epoch = max(self.completed_epoch, int(completed_epoch))
        self.last_peak_gib = float(peak_gib)
        peak_ratio = peak_gib / max(total_gib, 1e-6)

        if peak_ratio >= 0.94:
            self.current_batch = self._lower()
            self.cooldown_remaining = 5
            self.stable_epochs = 0
            self.last_event = "peak_demote"
            return self.current_batch

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.stable_epochs = 0
            self.last_event = "cooldown"
            return self.current_batch

        if peak_ratio < 0.82:
            self.stable_epochs += 1
            self.last_event = "stable"
        else:
            self.stable_epochs = 0
            self.last_event = "hold"

        if self.stable_epochs >= 3:
            self.current_batch = self._higher()
            self.stable_epochs = 0
            self.last_event = "promote"
        return self.current_batch

    def record_oom(self) -> int:
        self.oom_count += 1
        self.current_batch = self._lower()
        self.cooldown_remaining = 5
        self.stable_epochs = 0
        self.unexpected_failures = 0
        self.last_event = "oom_demote"
        return self.current_batch

    def record_numeric_failure(self) -> int:
        self.numeric_failure_count += 1
        self.amp_enabled = False
        self.current_batch = self._lower()
        self.cooldown_remaining = 5
        self.stable_epochs = 0
        self.unexpected_failures = 0
        self.last_event = "numeric_fp32_demote"
        return self.current_batch


def batch_policy_for_vram(*, total_gib: float, free_gib: float) -> BatchPolicy:
    if total_gib < 20:
        raise ValueError(f"RT-DETR-L server training requires at least 20 GiB VRAM, found {total_gib:.1f}")
    if total_gib < 27:
        levels, initial = (2, 4, 6, 8), 6
    elif total_gib < 36:
        levels, initial = (4, 6, 8, 10, 12), 8
    elif total_gib < 55:
        levels, initial = (6, 8, 12, 16, 20), 12
    else:
        levels, initial = (8, 12, 16, 20, 24, 28), 16

    free_ratio = free_gib / max(total_gib, 1e-6)
    if free_ratio < 0.85:
        safe_capacity = initial * free_ratio / 0.85
        initial = max(level for level in levels if level <= max(levels[0], safe_capacity))
        return BatchPolicy(levels=levels, initial_batch=initial, reason="startup_free_memory")
    return BatchPolicy(levels=levels, initial_batch=initial)


def scale_batch_policy(policy: BatchPolicy, *, world_size: int) -> BatchPolicy:
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return BatchPolicy(
        levels=tuple(level * world_size for level in policy.levels),
        initial_batch=policy.initial_batch * world_size,
        reason=f"{policy.reason}_x{world_size}",
    )


def detect_gpu_profile(device: int = 0) -> GPUProfile:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    name = torch.cuda.get_device_name(device)
    driver = "unknown"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(device)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return GPUProfile(
        name=name,
        total_gib=total_bytes / 1024**3,
        free_gib=free_bytes / 1024**3,
        cuda_version=str(torch.version.cuda),
        driver_version=driver,
    )


def save_adaptive_state(path: Path, state: AdaptiveTrainingState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    payload["levels"] = list(state.levels)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_adaptive_state(path: Path) -> AdaptiveTrainingState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["levels"] = tuple(payload["levels"])
    return AdaptiveTrainingState(**payload)

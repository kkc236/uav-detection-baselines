"""Frozen-protocol RT-DETR trainer used by the ACR-EG formal run."""

from __future__ import annotations

from numbers import Integral

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer


MATCHED_AMP_SCALE = 128.0
MATCHED_AMP_GROWTH_INTERVAL = 2**31 - 1
MATCHED_BATCH_SIZE = 8


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"GCTE_FORMAL_{name.upper()}_TYPE")
    return int(value)


class GCTEFormalTrainer(RTDETRTrainer):
    """RT-DETR trainer with the frozen MuSGD and AMP-scale contract."""

    def __init__(self, *args, **kwargs) -> None:
        overrides = kwargs.get("overrides")
        if not isinstance(overrides, dict):
            raise ValueError("GCTE_FORMAL_OVERRIDES_REQUIRED")
        if _strict_int("batch", overrides.get("batch")) != MATCHED_BATCH_SIZE:
            raise ValueError("GCTE_FORMAL_BATCH_DRIFT")
        if overrides.get("optimizer") != "MuSGD":
            raise ValueError("GCTE_FORMAL_OPTIMIZER_DRIFT")
        if float(overrides.get("amp_scale", MATCHED_AMP_SCALE)) != MATCHED_AMP_SCALE:
            raise ValueError("GCTE_FORMAL_AMP_SCALE_DRIFT")
        self.gcte_amp_scale_min = MATCHED_AMP_SCALE
        self.gcte_amp_scale_max = MATCHED_AMP_SCALE
        self.gcte_optimizer_attempts = 0
        super().__init__(*args, **kwargs)

    def _setup_train(self):
        # Let Ultralytics construct the scaler-dependent training state, then
        # replace it with a fixed-scale scaler for the frozen protocol. Resume
        # restores optimizer/epoch state in super(); the integrated trainer then
        # audits that reconstruction stayed at the exact frozen scale.
        self.args.amp = False
        super()._setup_train()
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("GCTE_FORMAL_AMP_REQUIRES_CUDA")
        self.args.amp = True
        self.amp = True
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=MATCHED_AMP_SCALE,
            growth_interval=MATCHED_AMP_GROWTH_INTERVAL,
        )

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        if name not in {"auto", "MuSGD"}:
            raise ValueError("GCTE_FORMAL_OPTIMIZER_DRIFT")
        optimizer = super().build_optimizer(
            model,
            name="MuSGD",
            lr=0.01,
            momentum=0.937,
            decay=decay,
            iterations=iterations,
        )
        if type(optimizer).__name__ != "MuSGD":
            raise RuntimeError("GCTE_FORMAL_OPTIMIZER_CLASS_DRIFT")
        if any(
            float(group.get("momentum", 0.0)) != 0.937
            for group in optimizer.param_groups
        ):
            raise RuntimeError("GCTE_FORMAL_MOMENTUM_DRIFT")
        return optimizer

    def optimizer_step(self):
        before = float(self.scaler.get_scale())
        if before != MATCHED_AMP_SCALE:
            raise RuntimeError(f"GCTE_FORMAL_AMP_SCALE_BEFORE_STEP:{before}")
        super().optimizer_step()
        after = float(self.scaler.get_scale())
        self.gcte_optimizer_attempts += 1
        self.gcte_amp_scale_min = min(self.gcte_amp_scale_min, before, after)
        self.gcte_amp_scale_max = max(self.gcte_amp_scale_max, before, after)
        if after != MATCHED_AMP_SCALE:
            raise FloatingPointError(f"GCTE_FORMAL_AMP_SCALE_AFTER_STEP:{after}")


__all__ = [
    "GCTEFormalTrainer",
    "MATCHED_AMP_GROWTH_INTERVAL",
    "MATCHED_AMP_SCALE",
    "MATCHED_BATCH_SIZE",
]

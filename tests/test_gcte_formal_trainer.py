from __future__ import annotations

from types import SimpleNamespace

import torch


class _FixedScaler:
    def get_scale(self) -> float:
        return 128.0

    def state_dict(self) -> dict:
        return {"scale": 128.0, "growth_interval": 2**31 - 1}


def test_integrated_resume_restores_fixed_scaler_optimizer_and_epoch() -> None:
    from src.rtdetr_acr_eg import assert_acr_eg_resume_continuity

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01, momentum=0.937)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    trainer = SimpleNamespace(
        scaler=_FixedScaler(),
        optimizer=optimizer,
        start_epoch=9,
    )
    checkpoint = {
        "epoch": 8,
        "scaler": {"scale": 128.0, "growth_interval": 2**31 - 1},
    }

    assert_acr_eg_resume_continuity(trainer, checkpoint)

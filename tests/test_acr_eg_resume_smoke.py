from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.smoke_acr_eg_resume import build_parser
from src.acr_eg_smoke import (
    inspect_acr_eg_gradients,
    validate_multiview_batch,
)


class FakeSmokeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.acr_eg = nn.Module()
        self.acr_eg.sr_peg = nn.Module()
        self.acr_eg.sr_peg.global_retain_head = nn.Linear(2, 1)
        self.acr_eg.other = nn.Linear(2, 2)


def test_smoke_cli_requires_resume_and_source_identity(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--resume",
            str(tmp_path / "epoch8.pt"),
            "--baseline-checkpoint",
            str(tmp_path / "baseline.pt"),
            "--project",
            str(tmp_path / "smoke"),
            "--source-commit",
            "a" * 40,
        ]
    )

    assert args.batch == 8
    assert args.workers == 8
    assert args.device == "0"
    assert args.expected_start_epoch == 9


def test_multiview_batch_contract_requires_all_real_training_inputs() -> None:
    batch = {
        "img": torch.zeros(2, 3, 4, 4),
        "local_views": torch.zeros(2, 4, 3, 4, 4),
        "source_shape": torch.zeros(2, 2),
    }

    evidence = validate_multiview_batch(batch, batch_size=2, image_size=4)

    assert evidence["global_shape"] == [2, 3, 4, 4]
    assert evidence["local_shape"] == [2, 4, 3, 4, 4]
    assert evidence["source_shape"] == [2, 2]

    with pytest.raises(ValueError, match="ACR_EG_SMOKE_LOCAL_VIEWS_MISSING"):
        validate_multiview_batch(
            {key: value for key, value in batch.items() if key != "local_views"},
            batch_size=2,
            image_size=4,
        )


def test_smoke_requires_nonzero_finite_gradient_on_logit_injection_head() -> None:
    model = FakeSmokeModel()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    evidence = inspect_acr_eg_gradients(model)

    assert evidence["acr_eg_parameter_count"] == 4
    assert evidence["nonzero_gradient_parameter_count"] == 4
    assert evidence["retain_head_nonzero_gradient_count"] == 2

    for name, parameter in model.named_parameters():
        if "global_retain_head" in name:
            parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(ValueError, match="ACR_EG_SMOKE_RETAIN_HEAD_GRADIENT_MISSING"):
        inspect_acr_eg_gradients(model)

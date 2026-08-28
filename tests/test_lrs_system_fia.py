from __future__ import annotations

import torch
from ultralytics.nn import tasks as ultralytics_tasks

from src.fia import FIA
from src.rtdetr_fdr import register_fdr_module


def test_fia_is_bit_exact_identity_at_initialization() -> None:
    torch.manual_seed(17)
    module = FIA(8)
    inputs = torch.randn(2, 8, 11, 13)

    outputs = module(inputs)

    torch.testing.assert_close(outputs, inputs, rtol=0, atol=0)
    assert module.residual_scale.ndim == 0
    assert module.residual_scale.item() == 0.0


def test_register_fdr_module_exposes_exact_fia_class() -> None:
    register_fdr_module()

    assert ultralytics_tasks.FIA is FIA

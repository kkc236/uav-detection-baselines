from __future__ import annotations

import pytest
import torch

from scripts.preflight_gcmv_plec import (
    build_parser,
    require_nonzero_gradient_families,
    tensor_tree_equal,
)


def test_preflight_parser_requires_real_artifact_paths(tmp_path):
    args = build_parser().parse_args(
        [
            "--pretrained-weights",
            "baseline.pt",
            "--data",
            "visdrone.yaml",
            "--output",
            str(tmp_path / "preflight.json"),
        ]
    )

    assert args.batch == 1
    assert args.device == "0"
    assert args.pretrained_weights == "baseline.pt"


def test_tensor_tree_equal_requires_bitwise_tensor_identity():
    first = (torch.tensor([1.0]), [None, torch.tensor([2.0])])
    second = (torch.tensor([1.0]), [None, torch.tensor([2.0])])
    drift = (torch.tensor([1.0]), [None, torch.tensor([2.0001])])

    assert tensor_tree_equal(first, second)
    assert not tensor_tree_equal(first, drift)


def test_gradient_family_gate_rejects_zero_or_missing_gradients():
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    module[0].weight.grad = torch.ones_like(module[0].weight)
    module[0].bias.grad = torch.ones_like(module[0].bias)

    with pytest.raises(RuntimeError, match="family=1"):
        require_nonzero_gradient_families(module, prefixes=("0", "1"))


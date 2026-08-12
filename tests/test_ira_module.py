from __future__ import annotations

import pytest
import torch
from torch import nn

from src.ira import IRA, IRAAttention, IRABaseBlock


def _state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def test_ira_starts_as_bit_exact_identity_for_256_channels() -> None:
    module = IRA(256)
    x = torch.randn(2, 256, 16, 16)

    output = module(x)

    assert module.residual_scale.ndim == 0
    assert module.residual_scale.item() == 0.0
    torch.testing.assert_close(output, x, rtol=0, atol=0)


def test_ira_contains_two_depthwise_dual_residual_blocks_and_joint_attention() -> None:
    module = IRA(256)

    assert len(module.refine) == 3
    assert isinstance(module.refine[0], IRABaseBlock)
    assert isinstance(module.refine[1], IRABaseBlock)
    assert isinstance(module.refine[2], IRAAttention)
    for block in module.refine[:2]:
        assert block.depthwise.in_channels == 256
        assert block.depthwise.out_channels == 256
        assert block.depthwise.groups == 256
        assert block.internal_residuals == 2
    assert module.refine[2].uses_spatial_attention is True
    assert module.refine[2].uses_channel_attention is True


@pytest.mark.parametrize("channels", [0, -1, 1.5, True])
def test_ira_rejects_invalid_channel_configuration(channels: object) -> None:
    with pytest.raises((TypeError, ValueError), match="channels"):
        IRA(channels)  # type: ignore[arg-type]


def test_ira_rejects_non_bchw_and_wrong_channel_inputs() -> None:
    module = IRA(256)

    with pytest.raises(ValueError, match="BCHW"):
        module(torch.randn(256, 8, 8))
    with pytest.raises(ValueError, match="256 channels"):
        module(torch.randn(1, 128, 8, 8))
    with pytest.raises(TypeError, match="Tensor"):
        module([[1.0]])  # type: ignore[arg-type]


def test_ira_private_path_reaches_every_parameter_when_gate_is_open() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(2901)
        module = IRA(256)
        with torch.no_grad():
            module.residual_scale.fill_(0.1)
        x = torch.randn(2, 256, 8, 8, requires_grad=True)

    module(x).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
        assert torch.count_nonzero(parameter.grad) > 0, f"zero gradient for {name}"


def test_ira_construction_is_deterministic_and_fork_rng_isolated() -> None:
    torch.manual_seed(71)
    global_state = torch.random.get_rng_state().clone()

    with torch.random.fork_rng():
        torch.manual_seed(20000)
        first = _state_dict(IRA(256))
    torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)

    with torch.random.fork_rng():
        torch.manual_seed(20000)
        second = _state_dict(IRA(256))

    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_ira_state_dict_round_trip_preserves_exact_output() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(19)
        source = IRA(256)
        with torch.no_grad():
            source.residual_scale.fill_(0.25)
        x = torch.randn(1, 256, 9, 7)

    restored = IRA(256)
    restored.load_state_dict(source.state_dict(), strict=True)

    assert source.state_dict().keys() == restored.state_dict().keys()
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(restored(x), source(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ira_preserves_cpu_dtype_and_device(dtype: torch.dtype) -> None:
    module = IRA(256).to(dtype=dtype, device="cpu")
    x = torch.randn(1, 256, 5, 7, dtype=dtype, device="cpu")

    output = module(x)

    assert output.dtype == dtype
    assert output.device == x.device
    torch.testing.assert_close(output, x, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_ira_preserves_cuda_dtype_and_device() -> None:
    module = IRA(256).to(dtype=torch.float16, device="cuda")
    x = torch.randn(1, 256, 8, 8, dtype=torch.float16, device="cuda")

    output = module(x)

    assert output.dtype == torch.float16
    assert output.device == x.device
    torch.testing.assert_close(output, x, rtol=0, atol=0)

from __future__ import annotations

import torch

from src.frequency_cm import FrequencyCM


def test_frequency_cm_preserves_shape_and_is_exact_identity_at_initialization() -> None:
    module = FrequencyCM(16, private_seed=20_000).eval()
    value = torch.randn(2, 16, 20, 20)

    output = module(value)

    assert output.shape == value.shape
    torch.testing.assert_close(output, value, rtol=0, atol=0)


def test_frequency_cm_private_initialization_does_not_advance_public_rng() -> None:
    torch.manual_seed(1234)
    expected = torch.rand(5)
    torch.manual_seed(1234)

    FrequencyCM(8, private_seed=20_000)
    actual = torch.rand(5)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_frequency_cm_private_seed_is_deterministic() -> None:
    first = FrequencyCM(8, private_seed=20_000).state_dict()
    second = FrequencyCM(8, private_seed=20_000).state_dict()

    assert set(first) == set(second)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_frequency_cm_fft_path_is_finite_for_twenty_by_twenty_feature() -> None:
    module = FrequencyCM(8, private_seed=20_000).eval()
    with torch.no_grad():
        module.gamma.fill_(1.0)
    value = torch.randn(2, 8, 20, 20)

    output = module(value)

    assert output.dtype == value.dtype
    assert torch.isfinite(output).all()


def test_frequency_cm_first_backward_reaches_zero_initialized_gates() -> None:
    module = FrequencyCM(8, private_seed=20_000)
    value = torch.randn(2, 8, 20, 20, requires_grad=True)

    module(value).square().mean().backward()

    assert module.gamma.grad is not None
    assert module.beta.grad is not None
    assert torch.isfinite(module.gamma.grad).all()
    assert torch.isfinite(module.beta.grad).all()


def test_frequency_cm_branches_receive_gradients_after_gates_open() -> None:
    module = FrequencyCM(8, private_seed=20_000)
    with torch.no_grad():
        module.gamma.fill_(0.25)
        module.beta.fill_(0.25)
    value = torch.randn(2, 8, 20, 20, requires_grad=True)

    module(value).square().mean().backward()

    frequency_gradients = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name.startswith("frequency.")
    ]
    spatial_gradients = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name.startswith("spatial_")
    ]
    assert frequency_gradients and all(gradient is not None for gradient in frequency_gradients)
    assert spatial_gradients and all(gradient is not None for gradient in spatial_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in frequency_gradients + spatial_gradients)

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.pr_ira import PRIRA, relative_open_ratio


def _state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _assert_floating_bits_equal(
    actual: torch.Tensor, expected: torch.Tensor
) -> None:
    integer_dtype = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }[expected.dtype]
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(integer_dtype), expected.view(integer_dtype))


def _stable_spatial_rms(x: torch.Tensor) -> torch.Tensor:
    spatial_size = x.shape[-2] * x.shape[-1]
    scale = x.abs().amax(dim=(-2, -1), keepdim=True)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.linalg.vector_norm(
        x / safe_scale,
        ord=2,
        dim=(-2, -1),
        keepdim=True,
    ) / math.sqrt(spatial_size) * scale


@pytest.mark.parametrize(
    ("epoch", "epochs", "expected"),
    [
        (1, 30, 0.0),
        (3, 30, 0.0),
        (4, 30, 1.0 / 7.0),
        (9, 30, 6.0 / 7.0),
        (10, 30, 1.0),
        (30, 30, 1.0),
        (1, 100, 0.0),
        (10, 100, 0.0),
        (11, 100, 1.0 / 21.0),
        (30, 100, 20.0 / 21.0),
        (31, 100, 1.0),
        (100, 100, 1.0),
    ],
)
def test_relative_open_ratio_uses_exact_integer_milestones(
    epoch: int, epochs: int, expected: float
) -> None:
    assert relative_open_ratio(epoch, epochs) == pytest.approx(expected)


def test_relative_open_ratio_does_not_round_large_integer_milestones() -> None:
    epochs = 9_999_999_999_999_981
    exact_identity_end = (epochs + 9) // 10

    assert relative_open_ratio(exact_identity_end, epochs) == 0.0


@pytest.mark.parametrize(
    ("epoch", "epochs", "error"),
    [
        (True, 30, TypeError),
        (1.0, 30, TypeError),
        (1, False, TypeError),
        (1, 30.0, TypeError),
        (0, 30, ValueError),
        (-1, 30, ValueError),
        (31, 30, ValueError),
        (1, 0, ValueError),
        (1, -1, ValueError),
    ],
)
def test_relative_open_ratio_rejects_invalid_progress(
    epoch: object, epochs: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="epoch"):
        relative_open_ratio(epoch, epochs)  # type: ignore[arg-type]


@pytest.mark.parametrize("channels", [0, -1, 1.5, True, "8"])
def test_pr_ira_rejects_invalid_channels(channels: object) -> None:
    with pytest.raises((TypeError, ValueError), match="channels"):
        PRIRA(channels)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("alpha_max", 0.0),
        ("alpha_max", -0.1),
        ("alpha_max", float("inf")),
        ("alpha_max", float("nan")),
        ("alpha_max", True),
        ("epsilon", 0.0),
        ("epsilon", -1e-6),
        ("epsilon", float("inf")),
        ("epsilon", float("nan")),
        ("epsilon", False),
    ],
)
def test_pr_ira_rejects_invalid_scalar_configuration(
    keyword: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError), match=keyword):
        PRIRA(8, **{keyword: value})  # type: ignore[arg-type]


def test_pr_ira_keeps_epsilon_keyword_only() -> None:
    module = PRIRA(8, 0.15, epsilon=1e-5)

    assert module.alpha_max == 0.15
    assert module.epsilon == 1e-5
    with pytest.raises(TypeError):
        PRIRA(8, 0.15, 1e-5)  # type: ignore[misc]


def test_pr_ira_defaults_and_local_gate_architecture_are_frozen() -> None:
    module = PRIRA(8)

    assert module.channels == 8
    assert module.alpha_max == 0.20
    assert module.epsilon == 1e-6
    assert module.amplitude.ndim == 0
    assert module.amplitude.item() == 0.0
    assert len(module.local_blocks) == 2
    for block in module.local_blocks:
        assert block.depthwise.in_channels == 8
        assert block.depthwise.out_channels == 8
        assert block.depthwise.groups == 8

    channel_final = module.channel_gate[-1]
    assert isinstance(channel_final, nn.Conv2d)
    assert torch.count_nonzero(channel_final.weight) == 0
    assert channel_final.bias is not None
    assert torch.count_nonzero(channel_final.bias) == 0
    assert isinstance(module.spatial_gate, nn.Conv2d)
    assert torch.count_nonzero(module.spatial_gate.weight) == 0
    assert module.spatial_gate.bias is not None
    assert torch.count_nonzero(module.spatial_gate.bias) == 0


def test_pr_ira_starts_as_bit_exact_bchw_identity_with_half_gates() -> None:
    module = PRIRA(8)
    module.set_training_progress(30, 30)
    x = torch.randn(2, 8, 9, 7)

    output = module(x)

    assert output.shape == x.shape
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    diagnostics = module.diagnostics
    assert set(diagnostics) == {
        "effective_amplitude",
        "gate_mean",
        "gate_max",
        "residual_rms_ratio",
    }
    assert diagnostics["effective_amplitude"].item() == 0.0
    assert diagnostics["gate_mean"].item() == pytest.approx(0.25)
    assert diagnostics["gate_max"].item() == pytest.approx(0.25)
    assert diagnostics["residual_rms_ratio"].item() == 0.0
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in diagnostics.values()
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_zero_amplitude_is_bit_exact_for_signed_zero_and_practical_values(
    dtype: torch.dtype,
) -> None:
    module = PRIRA(2).to(dtype=dtype)
    module.set_training_progress(30, 30)
    finfo = torch.finfo(dtype)
    x = torch.tensor(
        [
            -0.0,
            0.0,
            300.0,
            -300.0,
            finfo.tiny,
            -finfo.tiny,
            1.0,
            -1.0,
        ],
        dtype=dtype,
    ).reshape(1, 2, 2, 2)

    output = module(x)

    _assert_floating_bits_equal(output, x)
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in module.diagnostics.values()
    )
    assert module.diagnostics["effective_amplitude"].item() == 0.0
    assert module.diagnostics["residual_rms_ratio"].item() == 0.0


def test_open_zero_amplitude_identity_preserves_analytical_amplitude_gradient() -> None:
    module = PRIRA(4).double()
    module.set_training_progress(30, 30)
    x = torch.randn(2, 4, 5, 3, dtype=torch.float64, requires_grad=True)
    probe = torch.randn_like(x)

    with torch.no_grad():
        d_raw = module.local_blocks(x) - x
        d_raw_rms = _stable_spatial_rms(d_raw)
        stock_rms = _stable_spatial_rms(x)
        residual = d_raw / (d_raw_rms + module.epsilon) * stock_rms
        magnitude = d_raw.abs()
        channel_gate = torch.sigmoid(module.channel_gate(magnitude))
        spatial_gate = torch.sigmoid(
            module.spatial_gate(
                torch.cat(
                    (
                        magnitude.mean(dim=1, keepdim=True),
                        magnitude.amax(dim=1, keepdim=True),
                    ),
                    dim=1,
                )
            )
        )
        expected_amplitude_gradient = (
            module.alpha_max * (probe * channel_gate * spatial_gate * residual).sum()
        )

    output = module(x)
    (output * probe).sum().backward()

    _assert_floating_bits_equal(output, x)
    assert module.amplitude.grad is not None
    assert torch.isfinite(module.amplitude.grad)
    assert module.amplitude.grad.item() != 0.0
    torch.testing.assert_close(
        module.amplitude.grad,
        expected_amplitude_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


def test_active_nonfinite_gate_fails_closed() -> None:
    module = PRIRA(2)
    module.set_training_progress(30, 30)
    with torch.no_grad():
        assert module.spatial_gate.bias is not None
        module.spatial_gate.bias.fill_(float("nan"))
    x = torch.randn(1, 2, 3, 3)

    with pytest.raises(RuntimeError):
        module(x)


def test_active_nonfinite_local_transform_fails_closed() -> None:
    module = PRIRA(2)
    module.set_training_progress(30, 30)
    with torch.no_grad():
        bias = module.local_blocks[0].depthwise.bias
        assert bias is not None
        bias.fill_(float("nan"))

    with pytest.raises(RuntimeError):
        module(torch.randn(1, 2, 3, 3))


def test_active_nonfinite_normalized_residual_fails_closed() -> None:
    module = PRIRA(2)
    module.set_training_progress(30, 30)
    module.epsilon = 0.0
    with torch.no_grad():
        module.amplitude.fill_(0.7)
        for block in module.local_blocks:
            block.depthwise.weight.zero_()
            assert block.depthwise.bias is not None
            block.depthwise.bias.zero_()
            block.pointwise.weight.zero_()
            assert block.pointwise.bias is not None
            block.pointwise.bias.zero_()

    with pytest.raises(RuntimeError):
        module(torch.ones(1, 2, 3, 3))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_closed_schedule_fast_path_is_bit_exact_for_extreme_values(
    dtype: torch.dtype,
) -> None:
    module = PRIRA(2).to(dtype=dtype)
    module.set_training_progress(3, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.75)
        transform_bias = module.local_blocks[0].depthwise.bias
        assert transform_bias is not None
        transform_bias.fill_(float("nan"))
        assert module.spatial_gate.bias is not None
        module.spatial_gate.bias.fill_(float("nan"))
    finfo = torch.finfo(dtype)
    x = torch.tensor(
        [-0.0, 0.0, finfo.max, -finfo.max, 1.0, -1.0, finfo.tiny, -finfo.tiny],
        dtype=dtype,
    ).reshape(1, 2, 2, 2)

    output = module(x)

    _assert_floating_bits_equal(output, x)
    assert all(
        torch.isfinite(value).item() and not value.requires_grad
        for value in module.diagnostics.values()
    )
    assert module.diagnostics["effective_amplitude"].item() == 0.0
    assert module.diagnostics["residual_rms_ratio"].item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_full_open_nonzero_amplitude_uses_finite_promoted_rms(
    dtype: torch.dtype,
) -> None:
    module = PRIRA(4).to(dtype=dtype, device="cpu")
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.7)
    x = torch.full((2, 4, 6, 5), 300.0, dtype=dtype, device="cpu")

    output = module(x)

    assert output.dtype == dtype
    assert output.device == x.device
    assert torch.isfinite(output).all()
    assert not torch.equal(output, x)
    assert all(
        torch.isfinite(value).item()
        and not value.requires_grad
        and value.grad_fn is None
        for value in module.diagnostics.values()
    )
    assert 0.0 < module.diagnostics["residual_rms_ratio"].item()
    assert module.diagnostics["residual_rms_ratio"].item() <= (
        module.alpha_max + 1e-5
    )


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (torch.bfloat16, 1e19),
        (torch.float32, 1e19),
        (torch.float64, 1e200),
    ],
)
def test_large_finite_stock_rms_stays_finite_for_zero_residual(
    dtype: torch.dtype,
    value: float,
) -> None:
    module = PRIRA(2).to(dtype=dtype)
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(10.0)
        for block in module.local_blocks:
            block.depthwise.weight.zero_()
            assert block.depthwise.bias is not None
            block.depthwise.bias.zero_()
            block.pointwise.weight.zero_()
            assert block.pointwise.bias is not None
            block.pointwise.bias.zero_()
    x = torch.full((1, 2, 2, 2), value, dtype=dtype, requires_grad=True)

    output = module(x)
    (output / value).sum().backward()

    _assert_floating_bits_equal(output, x)
    assert all(torch.isfinite(item).item() for item in module.diagnostics.values())
    assert module.diagnostics["residual_rms_ratio"].item() == 0.0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_saturated_low_precision_increment_respects_strict_rms_cap(
    dtype: torch.dtype,
) -> None:
    module = PRIRA(4).to(dtype=dtype)
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(20.0)
        channel_final = module.channel_gate[-1]
        channel_final.weight.zero_()
        assert channel_final.bias is not None
        channel_final.bias.fill_(20.0)
        module.spatial_gate.weight.zero_()
        assert module.spatial_gate.bias is not None
        module.spatial_gate.bias.fill_(20.0)
    x = torch.linspace(-2.0, 2.0, 4 * 6 * 5, dtype=dtype).reshape(1, 4, 6, 5)

    output = module(x)

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    ratio = module.diagnostics["residual_rms_ratio"].item()
    assert 0.0 < ratio <= module.alpha_max + 1e-5


def test_zero_amplitude_supports_torch_func_jvp() -> None:
    module = PRIRA(4).double()
    module.set_training_progress(30, 30)
    x = torch.randn(1, 4, 4, 3, dtype=torch.float64)
    tangent = torch.randn_like(x)

    output, output_tangent = torch.func.jvp(module, (x,), (tangent,))

    _assert_floating_bits_equal(output, x)
    torch.testing.assert_close(output_tangent, tangent, rtol=0, atol=0)


def test_zero_amplitude_supports_torch_func_vmap() -> None:
    module = PRIRA(4).double()
    module.set_training_progress(30, 30)
    x = torch.randn(3, 1, 4, 4, 3, dtype=torch.float64)

    output = torch.func.vmap(module)(x)

    _assert_floating_bits_equal(output, x)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_active_forward_supports_fullgraph_compile() -> None:
    module = PRIRA(4).eval()
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.7)
    x = torch.randn(1, 4, 4, 3)
    expected = module(x)
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    actual = compiled(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pr_ira_matches_the_protected_residual_equation_and_rms_bound() -> None:
    module = PRIRA(8).double()
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.7)
    x = torch.randn(2, 8, 6, 5, dtype=torch.float64)

    output = module(x)

    d_raw = module.local_blocks(x) - x
    d_raw_rms = _stable_spatial_rms(d_raw)
    x_rms = _stable_spatial_rms(x).detach()
    residual = d_raw / (d_raw_rms + module.epsilon) * x_rms
    magnitude = d_raw.abs()
    channel_gate = torch.sigmoid(module.channel_gate(magnitude))
    spatial_summary = torch.cat(
        (
            magnitude.mean(dim=1, keepdim=True),
            magnitude.amax(dim=1, keepdim=True),
        ),
        dim=1,
    )
    spatial_gate = torch.sigmoid(module.spatial_gate(spatial_summary))
    effective_amplitude = module.alpha_max * torch.tanh(module.amplitude)
    expected = x + effective_amplitude * channel_gate * spatial_gate * residual

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    diagnostics = module.diagnostics
    assert diagnostics["effective_amplitude"].item() == pytest.approx(
        module.alpha_max * math.tanh(0.7)
    )
    assert diagnostics["gate_mean"].item() == pytest.approx(0.25)
    assert diagnostics["gate_max"].item() == pytest.approx(0.25)
    assert 0.0 < diagnostics["residual_rms_ratio"].item()
    assert diagnostics["residual_rms_ratio"].item() <= module.alpha_max + 1e-5


def test_pr_ira_detaches_stock_rms_rescaling_from_input_gradient() -> None:
    module = PRIRA(4).double()
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.9)

    actual_input = torch.randn(2, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    actual_loss = module(actual_input).square().mean()
    (actual_gradient,) = torch.autograd.grad(actual_loss, actual_input)

    expected_input = actual_input.detach().clone().requires_grad_(True)
    d_raw = module.local_blocks(expected_input) - expected_input
    d_raw_rms = _stable_spatial_rms(d_raw)
    stock_rms = _stable_spatial_rms(expected_input).detach()
    residual = d_raw / (d_raw_rms + module.epsilon) * stock_rms
    magnitude = d_raw.abs()
    channel_gate = torch.sigmoid(module.channel_gate(magnitude))
    spatial_gate = torch.sigmoid(
        module.spatial_gate(
            torch.cat(
                (
                    magnitude.mean(dim=1, keepdim=True),
                    magnitude.amax(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )
    )
    expected_output = expected_input + (
        module.alpha_max
        * torch.tanh(module.amplitude)
        * channel_gate
        * spatial_gate
        * residual
    )
    expected_loss = expected_output.square().mean()
    (expected_gradient,) = torch.autograd.grad(expected_loss, expected_input)

    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_pr_ira_zero_raw_residual_stays_finite() -> None:
    module = PRIRA(8)
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(10.0)
        for block in module.local_blocks:
            block.depthwise.weight.zero_()
            if block.depthwise.bias is not None:
                block.depthwise.bias.zero_()
            block.pointwise.weight.zero_()
            if block.pointwise.bias is not None:
                block.pointwise.bias.zero_()
    x = torch.zeros(2, 8, 4, 3, requires_grad=True)

    output = module(x)
    output.square().sum().backward()

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    assert module.diagnostics["residual_rms_ratio"].item() == 0.0
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
        assert torch.count_nonzero(parameter.grad) == 0, f"nonzero gradient for {name}"


def test_pr_ira_rejects_invalid_inputs() -> None:
    module = PRIRA(8)

    with pytest.raises(TypeError, match="Tensor"):
        module([[1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BCHW"):
        module(torch.randn(8, 5, 5))
    with pytest.raises(ValueError, match="8 channels"):
        module(torch.randn(1, 4, 5, 5))
    with pytest.raises(TypeError, match="floating"):
        module(torch.ones(1, 8, 5, 5, dtype=torch.int64))
    with pytest.raises(ValueError, match="non-empty"):
        module(torch.empty(0, 8, 5, 5))
    with pytest.raises(ValueError, match="non-empty"):
        module(torch.empty(1, 8, 0, 5))


def test_pr_ira_rejects_device_and_non_autocast_dtype_mismatches() -> None:
    module = PRIRA(4).float()

    with pytest.raises(ValueError, match="dtype"):
        module(torch.randn(1, 4, 3, 3, dtype=torch.float64))

    meta_module = PRIRA(4).to(device="meta")
    with pytest.raises(ValueError, match="device"):
        meta_module(torch.randn(1, 4, 3, 3))


def test_pr_ira_allows_real_cpu_autocast_activation_dtype() -> None:
    module = PRIRA(4).float()
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.7)
    x = torch.full((1, 4, 4, 3), 300.0, dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(x)

    assert output.dtype == x.dtype
    assert output.device == x.device
    assert torch.isfinite(output).all()


def test_progress_setter_controls_a_non_persistent_runtime_schedule() -> None:
    module = PRIRA(8)
    with torch.no_grad():
        module.amplitude.fill_(0.5)
    x = torch.randn(1, 8, 5, 5)

    module.set_training_progress(3, 30)
    identity = module(x)
    module.set_training_progress(4, 30)
    ramp = module(x)
    module.set_training_progress(10, 30)
    opened = module(x)

    torch.testing.assert_close(identity, x, rtol=0, atol=0)
    assert not torch.equal(ramp, x)
    assert not torch.equal(opened, ramp)
    assert module.open_ratio == 1.0
    assert "_open_ratio" not in module.state_dict()
    assert "_open_ratio" in dict(module.named_buffers())

    with pytest.raises(ValueError, match="epoch"):
        module.set_training_progress(31, 30)


def test_pr_ira_construction_is_deterministic_and_preserves_public_rng() -> None:
    torch.manual_seed(71)
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )

    first = _state_dict(PRIRA(8))

    torch.testing.assert_close(torch.random.get_rng_state(), cpu_state, rtol=0, atol=0)
    if torch.cuda.is_available():
        for actual, expected in zip(
            torch.cuda.get_rng_state_all(), cuda_states, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    second = _state_dict(PRIRA(8))
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_pr_ira_state_dict_round_trip_preserves_output_after_progress_setter() -> None:
    source = PRIRA(8)
    source.set_training_progress(9, 30)
    with torch.no_grad():
        source.amplitude.fill_(0.6)
    x = torch.randn(1, 8, 7, 5)
    expected = source(x)

    restored = PRIRA(8)
    restored.load_state_dict(source.state_dict(), strict=True)
    assert restored.open_ratio == 0.0
    restored.set_training_progress(9, 30)

    assert source.state_dict().keys() == restored.state_dict().keys()
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pr_ira_preserves_floating_dtype_and_device(dtype: torch.dtype) -> None:
    module = PRIRA(8).to(dtype=dtype, device="cpu")
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.4)
    x = torch.randn(1, 8, 5, 7, dtype=dtype, device="cpu")

    output = module(x)

    assert output.dtype == dtype
    assert output.device == x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pr_ira_preserves_cuda_rng_dtype_and_device() -> None:
    cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    module = PRIRA(8).to(dtype=torch.float16, device="cuda")
    for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_states, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    module.set_training_progress(30, 30)
    with torch.no_grad():
        module.amplitude.fill_(0.4)
    x = torch.randn(1, 8, 8, 8, dtype=torch.float16, device="cuda")

    output = module(x)

    assert output.dtype == torch.float16
    assert output.device == x.device
    assert torch.isfinite(output).all()

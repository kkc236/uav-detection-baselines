from __future__ import annotations

import math

import pytest
import torch

from src.vsf_rmr import VSFRMR, ordered_scale_weights


def feature_pyramid(*, batch: int = 2, channels: int = 256):
    return [
        torch.randn(batch, channels, 16, 20),
        torch.randn(batch, channels, 8, 10),
        torch.randn(batch, channels, 4, 5),
    ]


def test_ordered_scale_weights_mix_only_adjacent_levels():
    scale = torch.tensor([0.25, 0.95, 1.00, 1.40, 1.95]).reshape(1, 1, 1, 5)

    alpha3, alpha4, alpha5 = ordered_scale_weights(scale)

    torch.testing.assert_close(alpha3 + alpha4 + alpha5, torch.ones_like(scale))
    assert torch.count_nonzero(alpha5[..., :2]).item() == 0
    assert torch.count_nonzero(alpha3[..., 2:]).item() == 0
    assert torch.all(alpha3 >= 0)
    assert torch.all(alpha4 >= 0)
    assert torch.all(alpha5 >= 0)


def test_forward_preserves_shapes_and_scale_field_range():
    module = VSFRMR(channels=256, route_channels=32)
    features = feature_pyramid()

    outputs = module(features)
    state = module.pop_auxiliary_state()

    assert [tuple(value.shape) for value in outputs] == [tuple(value.shape) for value in features]
    assert state is not None
    assert state.scale_field.shape == (2, 1, 16, 20)
    assert state.global_scale.shape == (2, 1, 1, 1)
    assert torch.all((state.scale_field > 0) & (state.scale_field < 2))
    assert torch.isfinite(state.scale_field).all()


def test_zero_initialized_channel_scales_make_exact_identity():
    module = VSFRMR(channels=256, route_channels=32)
    features = feature_pyramid()

    outputs = module(features)

    for actual, expected in zip(outputs, features):
        assert torch.equal(actual, expected)
    for gamma in module.gamma:
        assert torch.count_nonzero(gamma).item() == 0


def test_initial_global_scale_is_about_point_nine_five():
    module = VSFRMR(channels=8, route_channels=4, norm_groups=4)
    features = feature_pyramid(batch=3, channels=8)

    module(features)
    state = module.pop_auxiliary_state()

    assert state is not None
    expected = 2.0 / (1.0 + math.exp(0.1))
    torch.testing.assert_close(state.global_scale, torch.full_like(state.global_scale, expected))


def test_training_cache_is_single_use_and_is_cleared_before_next_forward():
    module = VSFRMR(channels=8, route_channels=4, norm_groups=4)
    first = feature_pyramid(batch=1, channels=8)
    second = [value + 4.0 for value in feature_pyramid(batch=1, channels=8)]

    module(first)
    first_state = module.peek_auxiliary_state()
    module(second)
    second_state = module.pop_auxiliary_state()

    assert first_state is not None
    assert second_state is not None
    assert first_state.scale_field is not second_state.scale_field
    assert module.pop_auxiliary_state() is None


def test_eval_forward_never_retains_auxiliary_state():
    module = VSFRMR(channels=8, route_channels=4, norm_groups=4).eval()

    with torch.no_grad():
        module(feature_pyramid(batch=1, channels=8))

    assert module.peek_auxiliary_state() is None


@pytest.mark.parametrize(
    "features, message",
    [
        (feature_pyramid(channels=8)[:2], "three feature levels"),
        (
            [torch.randn(1, 8, 16, 20), torch.randn(1, 8, 7, 10), torch.randn(1, 8, 4, 5)],
            "2x and 4x",
        ),
        (
            [torch.randn(1, 8, 16, 20), torch.randn(1, 7, 8, 10), torch.randn(1, 8, 4, 5)],
            "8 channels",
        ),
    ],
)
def test_invalid_feature_contract_raises(features, message: str):
    module = VSFRMR(channels=8, route_channels=4, norm_groups=4)

    with pytest.raises(ValueError, match=message):
        module(features)


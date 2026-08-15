import torch
import torch.nn.functional as F

from src.btd_se import BTDSE, ring_sum


def explicit_ring_sum(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    kernel = torch.ones((channels, 1, 9, 9), dtype=x.dtype, device=x.device)
    kernel[:, :, 2:7, 2:7] = 0
    return F.conv2d(x, kernel, padding=4, groups=channels)


def test_ring_sum_matches_explicit_ring_kernel_at_borders_and_center():
    x = torch.arange(2 * 3 * 7 * 8, dtype=torch.float32).reshape(2, 3, 7, 8)

    actual = ring_sum(x)
    expected = explicit_ring_sum(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


def test_zero_initialized_module_preserves_original_pair():
    module = BTDSE(channels=256, embedding_channels=32, tau=1.0)
    pair = torch.randn(2, 512, 11, 13)

    output = module(pair)

    torch.testing.assert_close(output, pair)
    assert torch.count_nonzero(module.gamma).item() == 0
    assert module.last_background_reliability.shape == (2, 1, 11, 13)
    assert module.last_saliency.shape == (2, 1, 11, 13)
    assert module.last_residual.shape == (2, 256, 11, 13)
    assert module.last_normalizer.shape == (2, 1, 11, 13)
    assert torch.isfinite(module.last_normalizer).all()
    assert (module.last_normalizer >= 0).all()


def test_module_backward_is_finite_after_residual_path_is_enabled():
    module = BTDSE(channels=8, embedding_channels=4, tau=1.0)
    with torch.no_grad():
        module.gamma.fill_(0.1)
    pair = torch.randn(2, 16, 9, 9, requires_grad=True)

    output = module(pair)
    loss = output.square().mean()
    loss.backward()

    assert pair.grad is not None
    assert torch.isfinite(pair.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

from __future__ import annotations

import math

import torch
from torch import nn

from src.ioqc_sa_probe import P3SamplingProbe


class FakeCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_heads = 1
        self.n_levels = 3
        self.n_points = 2
        self.sampling_offsets = nn.Linear(2, 12)
        self.attention_weights = nn.Linear(2, 6)

    def forward(self, query, reference_boxes, value, value_shapes, value_mask=None):
        return query


class FakeDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_attn = FakeCrossAttention()


class FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer(), FakeDecoderLayer()])


def inputs(dtype: torch.dtype = torch.float32):
    query = torch.tensor([[[0.25, -0.5]]], dtype=dtype)
    reference = torch.tensor([[[[0.5, 0.5, 0.4, 0.2]]]], dtype=dtype)
    value = torch.zeros((1, 84, 2), dtype=dtype)
    shapes = [(8, 8), (4, 4), (2, 2)]
    return query, reference, value, shapes


def test_probe_computes_exact_p3_moments_in_float32_from_half_inputs():
    attention = FakeCrossAttention().train()
    with torch.no_grad():
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.copy_(
            torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        attention.attention_weights.weight.zero_()
        attention.attention_weights.bias.copy_(
            torch.tensor([0.0, math.log(3.0), -10.0, -10.0, -10.0, -10.0])
        )
    probe = P3SamplingProbe(attention)
    query, reference, value, shapes = inputs(torch.float16)

    probe.capture(query, reference, value, shapes)
    stats = probe.last_statistics

    assert stats is not None
    assert stats.center.dtype == torch.float32
    assert stats.extent.dtype == torch.float32
    assert stats.p3_mass.dtype == torch.float32
    assert stats.p3_shape == (8, 8)
    # FP32 prevents further loss, but cannot recover the quantization already present in the FP16 input box.
    torch.testing.assert_close(stats.center[0, 0], torch.tensor([0.55, 0.5]), atol=3e-5, rtol=0)
    torch.testing.assert_close(
        stats.extent[0, 0],
        torch.tensor([math.sqrt(0.0075 + 1e-6), math.sqrt(1e-6)]),
        atol=3e-5,
        rtol=0,
    )
    assert stats.valid[0, 0]
    assert torch.isfinite(stats.p3_mass).all()


def test_probe_invalidates_zero_p3_mass_without_nan():
    attention = FakeCrossAttention().train()
    with torch.no_grad():
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.zero_()
        attention.attention_weights.weight.zero_()
        attention.attention_weights.bias.copy_(torch.tensor([-100.0, -100.0, 0.0, 0.0, 0.0, 0.0]))
    probe = P3SamplingProbe(attention)

    probe.capture(*inputs())
    stats = probe.last_statistics

    assert stats is not None
    assert not stats.valid[0, 0]
    assert torch.isfinite(stats.center).all()
    assert torch.isfinite(stats.extent).all()


def test_probe_statistics_keep_gradients_for_query_and_projection_weights():
    torch.manual_seed(3)
    attention = FakeCrossAttention().train()
    probe = P3SamplingProbe(attention)
    query, reference, value, shapes = inputs()
    query.requires_grad_()

    probe.capture(query, reference, value, shapes)
    stats = probe.last_statistics
    assert stats is not None
    (stats.center.sum() + stats.extent.sum() + stats.p3_mass.sum()).backward()

    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert attention.sampling_offsets.weight.grad is not None
    assert torch.isfinite(attention.sampling_offsets.weight.grad).all()
    assert attention.attention_weights.weight.grad is not None
    assert torch.isfinite(attention.attention_weights.weight.grad).all()


def test_attach_observes_only_final_decoder_layer_and_can_be_removed():
    decoder = FakeDecoder().train()
    probe = P3SamplingProbe()
    probe.attach(decoder)
    query, reference, value, shapes = inputs()

    decoder.layers[0].cross_attn(query, reference, value, shapes)
    assert probe.last_statistics is None

    decoder.layers[-1].cross_attn(query, reference, value, shapes)
    assert probe.last_statistics is not None

    probe.clear()
    probe.remove()
    decoder.layers[-1].cross_attn(query, reference, value, shapes)
    assert probe.last_statistics is None


def test_probe_does_not_capture_in_evaluation_mode():
    decoder = FakeDecoder().eval()
    probe = P3SamplingProbe()
    probe.attach(decoder)

    decoder.layers[-1].cross_attn(*inputs())

    assert probe.last_statistics is None

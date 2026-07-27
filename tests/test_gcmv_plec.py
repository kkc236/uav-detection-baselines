from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from src.gcmv_geometry import build_plec_geometry
from src.gcmv_plec import (
    ChannelLayerNorm,
    PLECOutput,
    PhasePreservingLocalEvidenceCanonicalizer,
    sample_local_phases,
    uniform_bilinear_canonicalize,
)
from src.sbr_geometry import LetterboxTransform, Tile


def test_public_plec_sampling_contract_is_importable():
    assert PLECOutput is not None
    assert callable(sample_local_phases)
    assert callable(uniform_bilinear_canonicalize)


def identity_geometry(*, height: int = 6, width: int = 8):
    transform = LetterboxTransform(
        source_width=width,
        source_height=height,
        network_shape=(height, width),
        gain=1.0,
        pad=0.0,
    )
    tiles = tuple(Tile(0, 0, width, height, index) for index in range(4))
    return build_plec_geometry(
        source_shapes=[(height, width)],
        tiles=[tiles],
        global_transforms=[transform],
        local_transforms=[[transform] * 4],
        global_feature_shape=(height, width),
        local_feature_shape=(height, width),
    )


def ramp_features(*, height: int = 6, width: int = 8, channels: int = 1):
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return [
        (x + 10.0 * y + 100.0 * view)
        .reshape(1, 1, height, width)
        .expand(1, channels, height, width)
        .clone()
        for view in range(4)
    ]


def test_phase_sampler_matches_analytic_ramp_values():
    geometry = identity_geometry()
    features = ramp_features()
    phase = torch.tensor(
        [
            (-1 / 3, -1 / 3),
            (0, -1 / 3),
            (1 / 3, -1 / 3),
            (-1 / 3, 0),
            (0, 0),
            (1 / 3, 0),
            (-1 / 3, 1 / 3),
            (0, 1 / 3),
            (1 / 3, 1 / 3),
        ]
    )

    sampled = sample_local_phases(features, geometry)

    assert sampled.shape == (1, 4, 9, 1, 6, 8)
    for view in range(4):
        expected = 3.0 + phase[:, 0] + 10.0 * (2.0 + phase[:, 1]) + 100.0 * view
        torch.testing.assert_close(sampled[0, view, :, 0, 2, 3], expected)


def test_invalid_phase_samples_are_exact_zero_and_alignment_is_explicit():
    geometry = identity_geometry()
    sample_valid = geometry.sample_valid.clone()
    sample_valid[:, 2, 0, 2, 3] = False
    geometry = replace(geometry, sample_valid=sample_valid)

    with patch("src.gcmv_plec.F.grid_sample", wraps=F.grid_sample) as grid_sample:
        sampled = sample_local_phases(ramp_features(), geometry)

    assert sampled[0, 2, 0, 0, 2, 3].item() == 0.0
    assert grid_sample.call_args.kwargs["align_corners"] is False
    assert grid_sample.call_args.kwargs["mode"] == "bilinear"
    assert grid_sample.call_args.kwargs["padding_mode"] == "zeros"


def test_uniform_reference_uses_center_phase_and_uniform_valid_views():
    height, width = 2, 3
    geometry = identity_geometry(height=height, width=width)
    center_valid = torch.zeros_like(geometry.center_valid)
    center_valid[:, 0, :, 0, 0] = True
    center_valid[:, :2, :, 0, 1] = True
    center_valid[:, :, :, 1, :] = True
    geometry = replace(geometry, center_valid=center_valid)
    features = [
        torch.full((1, 1, height, width), float(view + 1))
        for view in range(4)
    ]

    output = uniform_bilinear_canonicalize(features, geometry)

    assert output.canonical[0, 0, 0, 0].item() == 1.0
    assert output.canonical[0, 0, 0, 1].item() == 1.5
    assert output.canonical[0, 0, 0, 2].item() == 0.0
    torch.testing.assert_close(
        output.canonical[0, 0, 1],
        torch.full((width,), 2.5),
    )
    assert output.valid_count[0, 0, 0, 0].item() == 1.0
    assert output.valid_count[0, 0, 0, 1].item() == 2.0
    assert output.valid_count[0, 0, 0, 2].item() == 0.0
    assert output.overlap_weights[0, :, 0, 0, 2].count_nonzero().item() == 0
    assert output.edge_prior[0, 0, 0, 2].item() == 0.0


def test_uniform_reference_backpropagates_only_through_features():
    geometry = identity_geometry()
    features = [feature.requires_grad_() for feature in ramp_features()]

    output = uniform_bilinear_canonicalize(features, geometry)
    output.canonical.square().mean().backward()

    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad).item() > 0
    for value in (
        geometry.sample_grid,
        geometry.subcell_offset,
        geometry.magnification,
        geometry.edge_distance,
    ):
        assert not value.requires_grad
        assert value.grad is None


def test_full_plec_constructs_the_frozen_trainable_layers():
    module = PhasePreservingLocalEvidenceCanonicalizer(
        channels=256,
        embedding_hidden=64,
        overlap_hidden=64,
        use_phase_embedding=True,
        use_view_embedding=True,
        use_metadata_embedding=True,
        learned_overlap=True,
    )

    assert isinstance(module.view_embedding, nn.Embedding)
    assert module.view_embedding.num_embeddings == 4
    assert module.view_embedding.embedding_dim == 256
    assert module.phase_mlp[0].in_features == 2
    assert module.phase_mlp[0].out_features == 64
    assert module.phase_mlp[2].out_features == 256
    assert module.metadata_mlp[0].in_features == 3
    assert module.metadata_mlp[0].out_features == 64
    assert module.metadata_mlp[2].out_features == 256
    assert module.phase_reducer.in_channels == 9 * 256
    assert module.phase_reducer.out_channels == 256
    assert module.phase_reducer.groups == 256
    assert module.spatial_mixer.kernel_size == (3, 3)
    assert module.spatial_mixer.groups == 256
    assert module.pointwise.kernel_size == (1, 1)
    assert isinstance(module.overlap_head, nn.Sequential)
    assert isinstance(module.output_norm, ChannelLayerNorm)

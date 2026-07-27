from __future__ import annotations

import torch

from src.gcmv_fusion import (
    GCMVEvidenceInjectionModule,
    GeometryConstrainedGlobalLocalFusion,
    ProtectedEvidenceGate,
)
from src.gcmv_plec import PLECOutput


def inputs(
    *,
    batch: int = 2,
    channels: int = 16,
    height: int = 7,
    width: int = 9,
):
    global_p3 = torch.randn(batch, channels, height, width)
    local = torch.randn_like(global_p3)
    valid_count = torch.full((batch, 1, height, width), 2.0)
    edge_prior = torch.full((batch, 1, height, width), 0.75)
    return global_p3, local, valid_count, edge_prior


def test_gglf_outputs_fixed_window_attention_and_diagnostics():
    module = GeometryConstrainedGlobalLocalFusion(
        channels=16,
        interaction_channels=8,
        window_size=3,
        num_heads=4,
    )
    global_p3, local, valid_count, edge_prior = inputs()

    output = module(global_p3, local, valid_count, edge_prior)

    assert output.correction.shape == global_p3.shape
    assert output.confidence.shape == valid_count.shape
    assert output.attention.shape == (2, 4, 9, 7, 9)
    assert output.attention_entropy.shape == valid_count.shape
    assert output.tiny_map.shape == valid_count.shape
    torch.testing.assert_close(
        output.attention.sum(dim=2),
        torch.ones(2, 4, 7, 9),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.isfinite(output.correction).all()
    assert torch.all((output.confidence >= 0) & (output.confidence <= 1))


def test_gglf_masks_empty_local_evidence_to_exact_zero():
    module = GeometryConstrainedGlobalLocalFusion(
        channels=16,
        interaction_channels=8,
    )
    global_p3, local, valid_count, edge_prior = inputs()
    valid_count.zero_()
    edge_prior.zero_()

    output = module(global_p3, local, valid_count, edge_prior)

    assert torch.count_nonzero(output.correction).item() == 0
    assert torch.count_nonzero(output.confidence).item() == 0
    assert torch.count_nonzero(output.attention).item() == 0
    assert torch.count_nonzero(output.attention_entropy).item() == 0
    assert torch.count_nonzero(output.tiny_map).item() == 0


def test_gglf_local_impulse_has_bounded_five_by_five_support():
    torch.manual_seed(3)
    module = GeometryConstrainedGlobalLocalFusion(
        channels=4,
        interaction_channels=4,
        window_size=3,
        num_heads=2,
    )
    global_p3 = torch.zeros(1, 4, 9, 9)
    local_zero = torch.zeros_like(global_p3)
    local_impulse = local_zero.clone()
    local_impulse[:, 0, 4, 4] = 1.0
    valid_count = torch.ones(1, 1, 9, 9)
    edge_prior = torch.ones_like(valid_count)

    zero = module(
        global_p3, local_zero, valid_count, edge_prior
    ).correction
    impulse = module(
        global_p3, local_impulse, valid_count, edge_prior
    ).correction
    difference = (impulse - zero).abs().sum(dim=1)[0]
    outside = torch.ones(9, 9, dtype=torch.bool)
    outside[2:7, 2:7] = False

    assert torch.count_nonzero(difference[outside]).item() == 0
    assert torch.count_nonzero(difference[~outside]).item() > 0


def test_gglf_all_trainable_families_receive_gradients():
    module = GeometryConstrainedGlobalLocalFusion(
        channels=16,
        interaction_channels=8,
        num_heads=4,
    )
    values = inputs(batch=1)
    output = module(*values)

    (
        output.correction.square().mean()
        + output.tiny_map.mean()
        + output.attention.square().mean()
    ).backward()

    for prefix in (
        "global_norm",
        "local_norm",
        "query",
        "key",
        "value",
        "relative_position_bias",
        "global_projection",
        "residual_reduce",
        "detail_mixer",
        "evidence_project",
        "tiny_head",
    ):
        gradients = [
            parameter.grad
            for name, parameter in module.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_peg_is_exact_global_identity_at_zero_residual_scalar():
    module = ProtectedEvidenceGate(channels=16)
    global_p3, correction, valid_count, edge_prior = inputs()
    confidence = torch.rand_like(valid_count)
    tiny_map = torch.rand_like(valid_count)
    plec_confidence = (valid_count / 4.0) * edge_prior

    output = module(
        global_p3,
        correction,
        tiny_map,
        confidence,
        plec_confidence,
        edge_prior,
        valid_count,
    )

    assert torch.equal(output.enhanced, global_p3)
    assert module.rho.item() == 0.0
    assert output.gamma.item() == 0.0
    assert torch.all((output.gate >= 0) & (output.gate <= 1))
    assert torch.all((output.gate_hat >= 0) & (output.gate_hat <= 1))
    torch.testing.assert_close(
        output.gate_hat,
        torch.full_like(output.gate_hat, 0.5),
    )


def test_peg_zero_scalar_receives_first_step_gradient():
    module = ProtectedEvidenceGate(channels=16)
    global_p3, correction, valid_count, edge_prior = inputs(batch=1)
    global_p3.requires_grad_()
    confidence = torch.rand_like(valid_count)
    tiny_map = torch.rand_like(valid_count)
    plec_confidence = (valid_count / 4.0) * edge_prior

    output = module(
        global_p3,
        correction,
        tiny_map,
        confidence,
        plec_confidence,
        edge_prior,
        valid_count,
    )
    output.enhanced.square().mean().backward()

    assert module.rho.grad is not None
    assert module.rho.grad.abs().sum() > 0
    assert torch.equal(output.enhanced, global_p3)


def test_integrated_module_has_full_family_gradients_in_open_guard_audit():
    module = GCMVEvidenceInjectionModule(
        channels=16,
        interaction_channels=8,
        num_heads=4,
    )
    module.peg.rho.data.fill_(1.0)
    global_p3, local, valid_count, edge_prior = inputs(batch=1)
    plec_output = PLECOutput(
        canonical=local,
        valid_count=valid_count,
        edge_prior=edge_prior,
        overlap_weights=torch.full(
            (1, 4, 1, 7, 9),
            0.25,
        ),
    )

    output = module(global_p3, plec_output)
    output.enhanced.square().mean().backward()

    for prefix in ("gglf.", "peg."):
        gradients = [
            parameter.grad
            for name, parameter in module.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0

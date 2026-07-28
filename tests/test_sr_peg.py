import torch

from src.sr_peg import ScaleRiskProtectedEvidenceGate


def _inputs() -> dict[str, torch.Tensor | bool]:
    return {
        "canonical_queries": torch.randn(2, 12, 32),
        "global_context": torch.randn(2, 12, 32),
        "geometry_embedding": torch.randn(2, 12, 32),
        "local_scores": torch.full((2, 12, 1), 0.5),
        "anchor_mask": torch.tensor(
            [
                [[True]] * 12,
                [[False]] * 12,
            ],
            dtype=torch.bool,
        ),
        "global_queries": torch.randn(2, 3, 32),
        "global_boxes": torch.full((2, 3, 4), 0.2),
        "global_scores": torch.full((2, 3, 1), 0.5),
        "local_valid_mask": torch.ones(2, 12, dtype=torch.bool),
        "residual_enabled": True,
    }


def test_sr_peg_emits_four_trainable_query_outputs():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    output = module(**_inputs())

    assert output.tiny_utility_logits.shape == (2, 12, 1)
    assert output.non_tiny_risk_logits.shape == (2, 12, 1)
    assert output.anchor_admission_logits.shape == (2, 12, 1)
    assert output.global_retain_logits.shape == (2, 3, 1)
    assert output.score_residual.shape == (2, 12, 1)
    assert output.score_residual.abs().max() <= 1
    (
        output.adjusted_local_scores.sum()
        + output.tiny_utility_logits.sum()
        + output.non_tiny_risk_logits.sum()
        + output.anchor_admission_logits.sum()
        + output.global_retain_logits.sum()
    ).backward()
    assert module.local_trunk[0].weight.grad is not None
    assert module.global_attention.in_proj_weight.grad is not None


def test_sr_peg_zero_initialization_is_score_noop():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    inputs = _inputs()
    output = module(**inputs)

    torch.testing.assert_close(
        output.adjusted_local_scores,
        inputs["local_scores"],
    )
    torch.testing.assert_close(
        output.score_residual,
        torch.zeros_like(inputs["local_scores"]),
    )


def test_sr_peg_bypass_returns_original_score_object_but_keeps_gate_logits():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    inputs = _inputs()
    inputs["residual_enabled"] = False
    output = module(**inputs)

    assert output.adjusted_local_scores is inputs["local_scores"]
    assert output.tiny_utility_logits.shape == (2, 12, 1)
    assert output.non_tiny_risk_logits.shape == (2, 12, 1)
    assert output.anchor_admission_logits.shape == (2, 12, 1)
    assert output.global_retain_logits.shape == (2, 3, 1)
    torch.testing.assert_close(
        output.score_residual,
        torch.zeros_like(inputs["local_scores"]),
    )


def test_sr_peg_rejects_invalid_masks_and_score_bounds():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    inputs = _inputs()
    inputs["local_valid_mask"] = torch.ones(2, 12)
    try:
        module(**inputs)
    except ValueError as error:
        assert "local_valid_mask" in str(error)
    else:
        raise AssertionError("non-bool mask must fail closed")

    inputs = _inputs()
    inputs["local_scores"] = torch.full((2, 12, 1), 1.1)
    try:
        module(**inputs)
    except ValueError as error:
        assert "local_scores" in str(error)
    else:
        raise AssertionError("out-of-range local scores must fail closed")


def test_anchor_prior_prefers_fixed_membership_before_learning():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    inputs = _inputs()
    inputs["canonical_queries"] = inputs["canonical_queries"][:1].repeat(2, 1, 1)
    inputs["global_context"] = inputs["global_context"][:1].repeat(2, 1, 1)
    inputs["geometry_embedding"] = inputs["geometry_embedding"][:1].repeat(2, 1, 1)
    inputs["local_scores"] = inputs["local_scores"][:1].repeat(2, 1, 1)
    inputs["anchor_mask"] = torch.tensor(
        [[[True]] * 12, [[False]] * 12],
        dtype=torch.bool,
    )
    output = module(**inputs)

    assert torch.all(
        output.anchor_admission_logits[0]
        > output.anchor_admission_logits[1]
    )


def test_anchor_admission_head_receives_gradient():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    output = module(**_inputs())
    output.anchor_admission_logits.square().mean().backward()

    assert module.anchor_delta_head.weight.grad is not None


def test_global_retain_path_receives_anchor_admission_evidence():
    module = ScaleRiskProtectedEvidenceGate(query_dim=32, num_heads=4)
    inputs = _inputs()
    with torch.no_grad():
        module.global_retain_head[-1].weight.fill_(0.1)
        module.tiny_utility_head.bias.fill_(2.0)
    high_utility = module(**inputs).global_retain_logits

    with torch.no_grad():
        module.tiny_utility_head.bias.fill_(-2.0)
    low_utility = module(**inputs).global_retain_logits

    assert not torch.allclose(high_utility, low_utility)

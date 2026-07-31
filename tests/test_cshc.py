import torch

from src.cshc import C2CandidateFusion, DySample, SparseC2CandidateGenerator


def test_dysample_preserves_shape_and_has_finite_gradients():
    module = DySample(channels=8, scale=2, groups=4)
    feature = torch.randn(2, 8, 5, 7, requires_grad=True)

    output = module(feature)

    assert output.shape == (2, 8, 10, 14)
    output.square().mean().backward()
    assert torch.isfinite(feature.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_sparse_generator_emits_topk_tokens_valid_logit_anchors_and_map():
    module = SparseC2CandidateGenerator(channels=8, hidden_dim=16, candidates=6, anchor_size=0.025)
    feature = torch.zeros(1, 8, 3, 4)
    with torch.no_grad():
        module.objectness.weight.zero_()
        module.objectness.bias.fill_(-2.0)
        module.objectness.weight[0, 0, 1, 1] = 5.0
    feature[0, 0, 1, 2] = 1.0

    output = module(feature)

    assert output.tokens.shape == (1, 6, 16)
    assert output.class_logits.shape == (1, 6, 10)
    assert output.anchor_logits.shape == (1, 6, 4)
    assert output.objectness_logits.shape == (1, 1, 3, 4)
    assert torch.isfinite(output.anchor_logits).all()
    assert torch.all((output.anchor_logits.sigmoid() > 0) & (output.anchor_logits.sigmoid() < 1))


def test_c2_fusion_outputs_requested_channels_at_c2_resolution():
    fusion = C2CandidateFusion(in_channels=12, out_channels=8)

    output = fusion(torch.randn(2, 12, 10, 14))

    assert output.shape == (2, 8, 10, 14)

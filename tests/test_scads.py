from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules.transformer import MLP

from src.fdr_head import FDRDeformableTransformerDecoder, build_distribution_heads
from src.fdr_math import Integral, bbox2distance, cxcywh_to_xyxy, weighting_function
from src.scads import (
    AdaptiveIntegral,
    ScaleConditionedSupportRouter,
    build_support_projects,
    continuous_edge_offsets,
    smallest_covering_support,
    translate_with_project,
)
from src.scads_head import SCADSFDRDeformableTransformerDecoder


def test_support_projects_are_ordered_and_include_exact_base_project() -> None:
    projects = build_support_projects()
    assert projects.shape == (3, 33)
    assert torch.all(projects[:, 1:] >= projects[:, :-1])
    torch.testing.assert_close(projects[1], weighting_function(), rtol=0, atol=0)
    assert projects[0, -1] < projects[1, -1] < projects[2, -1]


def test_adaptive_integral_base_one_hot_matches_fixed_integral() -> None:
    generator = torch.Generator().manual_seed(3407)
    logits = torch.randn((2, 5, 132), generator=generator)
    route = torch.zeros(2, 5, 3)
    route[..., 1] = 1.0
    actual = AdaptiveIntegral()(logits, route)
    expected = Integral()(logits)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_query_specific_project_is_monotonic_convex_combination() -> None:
    integral = AdaptiveIntegral()
    route = torch.tensor(
        [[[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]], dtype=torch.float32
    )
    project = integral.effective_project(route)
    assert project.shape == (1, 2, 33)
    assert torch.all(project[..., 1:] >= project[..., :-1])
    torch.testing.assert_close(project[..., 16], torch.zeros_like(project[..., 16]))


def test_base_project_target_encoding_matches_fixed_fdr() -> None:
    references = torch.tensor(
        [[0.50, 0.50, 0.20, 0.15], [0.30, 0.40, 0.08, 0.10]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[0.39, 0.41, 0.62, 0.58], [0.25, 0.34, 0.36, 0.47]],
        dtype=torch.float32,
    )
    offsets = continuous_edge_offsets(references, targets)
    projects = build_support_projects()[1].expand(2, -1)
    actual = translate_with_project(offsets, projects)
    expected = bbox2distance(references, targets)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


def test_smallest_covering_support_uses_narrow_then_base_then_wide() -> None:
    projects = build_support_projects()
    offsets = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4], [2.2, -2.1, 1.0, 0.5], [5.0, 0.0, -4.5, 1.0], [9.0, 0.0, 0.0, 0.0]]
    )
    targets, overflow = smallest_covering_support(offsets, projects, margin_ratio=0.0)
    assert targets.tolist() == [0, 1, 2, 2]
    assert overflow.tolist() == [False, False, False, True]


def test_router_is_private_rng_isolated_and_input_detached() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    first = ScaleConditionedSupportRouter(16, private_seed=20_000)
    after = torch.random.get_rng_state()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    second = ScaleConditionedSupportRouter(16, private_seed=20_000)
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    hidden = torch.randn(2, 4, 16, requires_grad=True)
    raw_boxes = torch.rand(2, 4, 4, requires_grad=True)
    boxes = torch.cat(
        [raw_boxes[..., :2], raw_boxes[..., 2:] * 0.2 + 0.05], dim=-1
    )
    logits, weights = first(hidden, boxes)
    assert logits.shape == weights.shape == (2, 4, 3)
    assert torch.all(weights[..., 1] > 0.999)
    (logits.square().mean() + weights[..., 0].mean()).backward()
    assert hidden.grad is None
    assert raw_boxes.grad is None
    assert first.output_layer.weight.grad is not None


class _FakeLayer(nn.Module):
    def forward(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        feats: torch.Tensor,
        shapes: list[list[int]],
        padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        del reference, feats, shapes, padding_mask, attn_mask
        return output + 0.01 * query_pos


class _FakeStockDecoder(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(6)])
        self.hidden_dim = hidden
        self.num_layers = 6
        self.eval_idx = 5


class _QueryPos(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(4, hidden, bias=False)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.proj(boxes)


def _decoder_inputs(hidden: int = 16):
    torch.manual_seed(7)
    embed = torch.randn(2, 5, hidden, requires_grad=True)
    refs = torch.randn(2, 5, 4)
    feats = torch.randn(2, 3, hidden)
    shapes = [[1, 3]]
    pre = MLP(hidden, hidden, 4, 3)
    dist = build_distribution_heads(hidden, 6, private_seed=10_000)
    scores = nn.ModuleList([nn.Linear(hidden, 10) for _ in range(6)])
    return embed, refs, feats, shapes, pre, dist, scores


def test_scads_decoder_reuses_one_route_across_all_six_layers() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _decoder_inputs()
    fdr = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=pre
    )
    decoder = SCADSFDRDeformableTransformerDecoder.from_fdr(
        fdr,
        support_ups=(0.25, 0.5, 1.0),
        router_hidden=8,
        temperature=1.0,
        private_seed=20_000,
        base_support_index=1,
    )
    decoder.train()
    boxes, classes = decoder(
        embed, refs, feats, shapes, dist, scores, _QueryPos()
    )
    assert boxes.shape == (6, 2, 5, 4)
    assert classes.shape == (6, 2, 5, 10)
    assert decoder.last_support_logits is not None
    assert decoder.last_support_weights is not None
    assert decoder.last_support_logits.shape == (2, 5, 3)
    assert decoder.last_support_weights.shape == (2, 5, 3)
    assert torch.isfinite(boxes).all()


def test_decoded_box_loss_reaches_router_without_reaching_router_inputs() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _decoder_inputs()
    for head in dist:
        with torch.no_grad():
            head.layers[-1].bias.copy_(torch.linspace(-1.0, 1.0, 132))
    fdr = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=pre
    )
    decoder = SCADSFDRDeformableTransformerDecoder.from_fdr(
        fdr,
        support_ups=(0.25, 0.5, 1.0),
        router_hidden=8,
        temperature=1.0,
        private_seed=20_000,
        base_support_index=1,
    )
    decoder.train()
    boxes, _ = decoder(embed, refs, feats, shapes, dist, scores, _QueryPos())
    boxes[-1].square().mean().backward()
    assert decoder.support_router.output_layer.bias.grad is not None
    assert torch.isfinite(decoder.support_router.output_layer.bias.grad).all()
    assert decoder.support_router.output_layer.bias.grad.abs().sum() > 0


def test_continuous_offsets_accept_cxcywh_targets_after_conversion() -> None:
    references = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    target = torch.tensor([[0.5, 0.5, 0.3, 0.25]])
    offsets = continuous_edge_offsets(references, cxcywh_to_xyxy(target))
    assert offsets.shape == (1, 4)
    assert torch.isfinite(offsets).all()

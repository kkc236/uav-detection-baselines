from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules.transformer import DeformableTransformerDecoder

from src.lpr_head import LPRDeformableTransformerDecoder, LocalizationPriorRefiner, box_geometry_prior


def test_geometry_prior_is_finite_for_tiny_boxes() -> None:
    boxes = torch.tensor([[[0.5, 0.25, 1e-12, 2e-12]]])

    prior = box_geometry_prior(boxes)

    assert prior.shape == (1, 1, 6)
    assert torch.isfinite(prior).all()
    torch.testing.assert_close(prior[..., :2], torch.tensor([[[0.0, -0.5]]]))


def test_zero_gate_is_bitwise_identity_and_alpha_gets_gradient() -> None:
    module = LocalizationPriorRefiner(hidden_dim=256, seed=3407)
    hidden = torch.randn(2, 5, 256, requires_grad=True)
    boxes = torch.rand(2, 5, 4).mul(0.8).add(0.1).requires_grad_()

    refined = module(hidden, boxes)

    assert torch.equal(refined, boxes)
    weights = torch.linspace(0.5, 1.5, refined.numel(), device=refined.device).reshape_as(refined)
    (refined * weights).sum().backward()
    assert module.alpha.grad is not None
    assert module.alpha.grad.abs().item() > 0


def test_positive_gate_keeps_refined_boxes_bounded_and_changes_output() -> None:
    module = LocalizationPriorRefiner(hidden_dim=32, seed=3408)
    module.alpha.data.fill_(0.4)
    hidden = torch.randn(2, 3, 32)
    boxes = torch.rand(2, 3, 4).mul(0.8).add(0.1)

    refined = module(hidden, boxes)

    assert torch.all(refined > 0)
    assert torch.all(refined < 1)
    assert not torch.equal(refined, boxes)


def test_refiner_construction_does_not_advance_global_rng() -> None:
    torch.manual_seed(17)
    expected = torch.rand(4)
    torch.manual_seed(17)

    LocalizationPriorRefiner(hidden_dim=256, seed=3407)
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _RecordingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[torch.Tensor] = []

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        del feats, shapes, padding_mask, attn_mask
        self.references.append(refer_bbox.detach().clone())
        return embed + query_pos


def _decoder_fixture() -> tuple[
    DeformableTransformerDecoder,
    torch.Tensor,
    torch.Tensor,
    list[int],
    nn.ModuleList,
    nn.ModuleList,
    nn.Module,
]:
    torch.manual_seed(23)
    stock = DeformableTransformerDecoder(hidden_dim=4, decoder_layer=_RecordingLayer(), num_layers=3)
    embed = torch.randn(2, 5, 4)
    refer_bbox = torch.randn(2, 5, 4)
    feats = torch.empty(0)
    bbox_head = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
    score_head = nn.ModuleList(nn.Linear(4, 2) for _ in range(3))
    return stock, embed, refer_bbox, [], bbox_head, score_head, nn.Identity()


def _recorded_references(decoder: nn.Module) -> list[torch.Tensor]:
    return [layer.references[-1] for layer in decoder.layers]


def _clear_references(decoder: nn.Module) -> None:
    for layer in decoder.layers:
        layer.references.clear()


def test_lpr_decoder_zero_gate_matches_stock_and_reference_trajectory() -> None:
    stock, embed, refer_bbox, shapes, bbox_head, score_head, pos_mlp = _decoder_fixture()
    stock.train()
    stock_boxes, stock_scores = stock(embed, refer_bbox, torch.empty(0), shapes, bbox_head, score_head, pos_mlp)
    stock_references = _recorded_references(stock)
    _clear_references(stock)

    lpr = LPRDeformableTransformerDecoder.from_stock(stock)
    lpr.train()
    lpr_boxes, lpr_scores = lpr(embed, refer_bbox, torch.empty(0), shapes, bbox_head, score_head, pos_mlp)

    assert torch.equal(lpr_boxes, stock_boxes)
    assert torch.equal(lpr_scores, stock_scores)
    assert len(lpr.lpr_refiners) == stock.num_layers
    for expected, actual in zip(stock_references, _recorded_references(lpr)):
        assert torch.equal(actual, expected)


def test_lpr_decoder_changes_output_without_changing_reference_trajectory() -> None:
    stock, embed, refer_bbox, shapes, bbox_head, score_head, pos_mlp = _decoder_fixture()
    stock.train()
    stock_boxes, _ = stock(embed, refer_bbox, torch.empty(0), shapes, bbox_head, score_head, pos_mlp)
    stock_references = _recorded_references(stock)
    _clear_references(stock)

    lpr = LPRDeformableTransformerDecoder.from_stock(stock)
    lpr.lpr_refiners[-1].alpha.data.fill_(0.2)
    lpr.train()
    lpr_boxes, _ = lpr(embed, refer_bbox, torch.empty(0), shapes, bbox_head, score_head, pos_mlp)

    assert torch.equal(lpr_boxes[:-1], stock_boxes[:-1])
    assert not torch.equal(lpr_boxes[-1], stock_boxes[-1])
    for expected, actual in zip(stock_references, _recorded_references(lpr)):
        assert torch.equal(actual, expected)


def test_lpr_decoder_zero_gate_matches_stock_in_evaluation() -> None:
    stock, embed, refer_bbox, shapes, bbox_head, score_head, pos_mlp = _decoder_fixture()
    stock.eval()
    stock_boxes, stock_scores = stock(embed, refer_bbox, torch.empty(0), shapes, bbox_head, score_head, pos_mlp)
    stock_references = _recorded_references(stock)
    _clear_references(stock)

    lpr = LPRDeformableTransformerDecoder.from_stock(stock).eval()
    lpr_boxes, lpr_scores = lpr(embed, refer_bbox, torch.empty(0), shapes, bbox_head, score_head, pos_mlp)

    assert torch.equal(lpr_boxes, stock_boxes)
    assert torch.equal(lpr_scores, stock_scores)
    for expected, actual in zip(stock_references, _recorded_references(lpr)):
        assert torch.equal(actual, expected)

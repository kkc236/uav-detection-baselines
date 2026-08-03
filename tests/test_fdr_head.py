from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn
from ultralytics.nn.modules.transformer import MLP

from src.fdr_head import (
    FDRDeformableTransformerDecoder,
    build_distribution_heads,
    cumulative_distribution_logits,
)


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
    def __init__(self, layers: int = 6, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(layers)])
        self.hidden_dim = hidden
        self.num_layers = layers
        self.eval_idx = layers - 1


class _QueryPos(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(4, hidden, bias=False)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.proj(boxes)


def _inputs(batch: int = 2, queries: int = 7, hidden: int = 16):
    torch.manual_seed(7)
    embed = torch.randn(batch, queries, hidden, requires_grad=True)
    reference_logits = torch.randn(batch, queries, 4)
    feats = torch.randn(batch, 3, hidden)
    shapes = [[1, 3]]
    pre_head = MLP(hidden, hidden, 4, 3)
    dist_heads = build_distribution_heads(hidden, 6, private_seed=10_000)
    score_heads = nn.ModuleList([nn.Linear(hidden, 10) for _ in range(6)])
    return embed, reference_logits, feats, shapes, pre_head, dist_heads, score_heads


def test_cumulative_distribution_logits_is_exact_cumsum() -> None:
    deltas = torch.stack(
        [torch.full((1, 2, 132), float(index + 1)) for index in range(6)]
    )
    torch.testing.assert_close(
        cumulative_distribution_logits(deltas),
        deltas.cumsum(dim=0),
        rtol=0,
        atol=0,
    )


def test_private_distribution_initialization_does_not_advance_public_rng() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    heads = build_distribution_heads(16, 6, private_seed=10_000)
    after = torch.random.get_rng_state()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert len(heads) == 6
    for head in heads:
        assert head.layers[-1].out_features == 132
        torch.testing.assert_close(
            head.layers[-1].weight,
            torch.zeros_like(head.layers[-1].weight),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            head.layers[-1].bias,
            torch.zeros_like(head.layers[-1].bias),
            rtol=0,
            atol=0,
        )


def test_private_distribution_seed_is_reproducible_and_distinct() -> None:
    first = build_distribution_heads(16, 6, private_seed=10_000)
    second = build_distribution_heads(16, 6, private_seed=10_000)
    third = build_distribution_heads(16, 6, private_seed=10_001)
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), third.state_dict().values())
    )


def test_training_decoder_exposes_six_layer_fdr_evidence() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs()
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=deepcopy(pre),
    )
    decoder.train()
    boxes, classes = decoder(
        embed,
        refs,
        feats,
        shapes,
        dist,
        scores,
        _QueryPos(16),
    )
    assert boxes.shape == (6, 2, 7, 4)
    assert classes.shape == (6, 2, 7, 10)
    assert decoder.last_corner_logits is not None
    assert decoder.last_corner_logits.shape == (6, 2, 7, 132)
    assert decoder.last_references is not None
    assert decoder.last_references.shape == (6, 2, 7, 4)
    assert decoder.last_pre_bboxes is not None
    assert decoder.last_pre_bboxes.shape == (2, 7, 4)
    assert not decoder.last_references.requires_grad
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(classes).all()
    assert torch.isfinite(decoder.last_corner_logits).all()


def test_cumulative_logits_keep_prior_distribution_gradients() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=2)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    decoder.train()
    boxes, _ = decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    boxes[-1].sum().backward()
    for head in dist:
        grad = head.layers[-1].weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_eval_decoder_returns_only_eval_layer_but_keeps_stock_signature() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=3)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    decoder.eval()
    with torch.no_grad():
        boxes, classes = decoder(
            embed,
            refs,
            feats,
            shapes,
            dist,
            scores,
            _QueryPos(16),
        )
    assert boxes.shape == (1, 1, 3, 4)
    assert classes.shape == (1, 1, 3, 10)
    assert decoder.last_corner_logits is not None
    assert decoder.last_corner_logits.shape == (1, 1, 3, 132)


def test_from_stock_rejects_non_six_layer_decoder() -> None:
    _, _, _, _, pre, _, _ = _inputs()
    with pytest.raises(ValueError, match="six decoder layers"):
        FDRDeformableTransformerDecoder.from_stock(
            _FakeStockDecoder(layers=5),
            pre_bbox_head=pre,
        )


def test_fdr_head_contains_no_excluded_components() -> None:
    _, _, _, _, pre, _, _ = _inputs()
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    forbidden = ("ddf", "teacher", "lqe", "go_lsd", "target_gate")
    assert not any(
        token in name.lower()
        for name, _ in decoder.named_modules()
        for token in forbidden
    )

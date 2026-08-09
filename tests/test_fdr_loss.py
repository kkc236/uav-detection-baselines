from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from ultralytics.models.utils.loss import RTDETRDetectionLoss

from src.fdr_math import bbox2distance, fine_grained_localization_loss
from src.fdr_loss import (
    FDRDetectionLoss,
    adjacent_bin_fgl,
    stock_loss_subtotal,
)


def _mixed_batch() -> dict:
    return {
        "cls": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor([[0.52, 0.48, 0.18, 0.22]], dtype=torch.float32),
        "gt_groups": [1, 0],
    }


def _empty_batch(batch_size: int = 2) -> dict:
    return {
        "cls": torch.empty(0, dtype=torch.long),
        "bboxes": torch.empty(0, 4, dtype=torch.float32),
        "gt_groups": [0] * batch_size,
    }


def _matches_with_one_positive() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (torch.tensor([1], dtype=torch.long), torch.tensor([0], dtype=torch.long)),
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    ]


def _empty_matches(batch_size: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        for _ in range(batch_size)
    ]


def _predictions(
    *, layers: int = 3, batch: int = 2, queries: int = 4, classes: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3407)
    boxes = torch.rand((layers, batch, queries, 4), generator=generator)
    boxes[..., 2:] = boxes[..., 2:] * 0.25 + 0.05
    scores = torch.randn((layers, batch, queries, classes), generator=generator)
    return boxes, scores


def _corner_logits(
    *, layers: int = 3, batch: int = 2, queries: int = 4
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(901)
    return torch.randn(
        (layers, batch, queries, 4 * 33), generator=generator, requires_grad=True
    )


class CountingMatcher(nn.Module):
    def __init__(self, matches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        self.matches = matches
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return [(source.clone(), target.clone()) for source, target in self.matches]


def test_fgl_zero_preserves_every_stock_key_value_and_stock_subtotal_exact() -> None:
    boxes, scores = _predictions(layers=7)
    batch = _mixed_batch()
    matches = _matches_with_one_positive()

    stock = RTDETRDetectionLoss(nc=3, use_vfl=True)
    stock.matcher = CountingMatcher(matches)
    stock_losses = stock((boxes, scores), batch)

    pre_boxes = boxes[1].detach().clone().requires_grad_(True)
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=True
    )
    actual = criterion(
        (boxes, scores),
        batch,
        normal_match_indices=[matches] * 7,
        pre_boxes=pre_boxes,
    )

    assert set(stock_losses).issubset(actual)
    for key, expected in stock_losses.items():
        torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        stock_loss_subtotal(actual), stock_loss_subtotal(stock_losses), rtol=0, atol=0
    )
    assert {"loss_bbox_pre", "loss_giou_pre"}.issubset(actual)
    assert not any("class_pre" in key or "pre_class" in key for key in actual)
    assert not any("fgl" in key for key in actual)


def test_fgl_zero_with_distribution_inputs_has_isolated_zero_keys() -> None:
    boxes, scores = _predictions(layers=2)
    matches = _matches_with_one_positive()
    corners = _corner_logits(layers=1)
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=False
    )

    losses = criterion(
        (boxes, scores),
        _mixed_batch(),
        normal_match_indices=[matches, matches],
        corner_logits=corners,
        pre_boxes=boxes[0].detach().clone(),
    )

    assert {"loss_fgl", "loss_fgl_aux"}.issubset(losses)
    torch.testing.assert_close(losses["loss_fgl"], torch.zeros_like(losses["loss_fgl"]))
    torch.testing.assert_close(
        losses["loss_fgl_aux"], torch.zeros_like(losses["loss_fgl_aux"])
    )


def test_normal_fgl_skips_encoder_and_aligns_six_decoder_layers() -> None:
    stock_boxes, stock_scores = _predictions(layers=7)
    corners = _corner_logits(layers=6)
    matches = _matches_with_one_positive()
    criterion = FDRDetectionLoss(nc=3, use_vfl=True, fgl_weight=0.15)

    losses = criterion(
        (stock_boxes, stock_scores),
        _mixed_batch(),
        normal_match_indices=[matches] * 7,
        corner_logits=corners,
        pre_boxes=stock_boxes[1].detach().clone(),
    )
    (losses["loss_fgl"] + losses["loss_fgl_aux"]).backward()

    assert corners.grad is not None and torch.isfinite(corners.grad).all()
    assert criterion.stock_match_calls == 0
    assert criterion.fgl_extra_match_calls == 0


def test_adjacent_bin_fgl_matches_pinned_primitive_exactly() -> None:
    logits = torch.tensor(
        [
            [math.log(2.0), math.log(3.0), math.log(5.0)],
            [math.log(7.0), math.log(11.0), math.log(13.0)],
        ],
        requires_grad=True,
    )
    left = torch.tensor([0.0, 1.0])
    weight_right = torch.tensor([0.25, 0.60])
    weight_left = 1.0 - weight_right
    matched_iou = torch.tensor([0.5, 0.8], requires_grad=True)

    actual = adjacent_bin_fgl(
        logits,
        left,
        weight_left,
        weight_right,
        matched_iou,
        avg_factor=2.5,
    )
    expected = fine_grained_localization_loss(
        logits,
        left,
        weight_right,
        weight_left,
        weight=matched_iou.detach(),
        avg_factor=2.5,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert matched_iou.grad is None


def test_fgl_target_reference_is_detached_from_preliminary_box() -> None:
    boxes, scores = _predictions(layers=3)
    batch = _mixed_batch()
    layer_zero = _matches_with_one_positive()
    main_layer = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    ]
    pre_boxes = boxes[0].detach().clone().requires_grad_(True)
    corners = _corner_logits(layers=2)
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.15, supervise_pre_boxes=False
    )

    losses = criterion(
        (boxes, scores),
        batch,
        normal_match_indices=[layer_zero, layer_zero, main_layer],
        corner_logits=corners,
        pre_boxes=pre_boxes,
    )
    (losses["loss_fgl"] + losses["loss_fgl_aux"]).backward()

    assert pre_boxes.grad is None
    assert corners.grad is not None and torch.isfinite(corners.grad).all()


def test_pre_box_supervision_uses_layer_zero_assignment_and_has_finite_gradient() -> None:
    boxes, scores = _predictions(layers=3)
    batch = _mixed_batch()
    encoder = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    ]
    decoder_layer_zero = _matches_with_one_positive()
    pre_boxes = boxes[0].detach().clone().requires_grad_(True)
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=True
    )

    losses = criterion(
        (boxes, scores),
        batch,
        normal_match_indices=[encoder, decoder_layer_zero, encoder],
        pre_boxes=pre_boxes,
    )
    (losses["loss_bbox_pre"] + losses["loss_giou_pre"]).backward()

    assert pre_boxes.grad is not None
    assert torch.isfinite(pre_boxes.grad).all()
    assert pre_boxes.grad[0, 1].abs().sum() > 0
    assert torch.equal(pre_boxes.grad[0, 0], torch.zeros(4))
    assert not any("class" in key and "pre" in key for key in losses)


def test_fgl_and_pre_losses_reuse_stock_normal_and_fixed_dn_assignments() -> None:
    layers, batch_size, queries = 3, 2, 4
    boxes, scores = _predictions(layers=layers, batch=batch_size, queries=queries)
    matches = _matches_with_one_positive()
    matcher = CountingMatcher(matches)
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.15, supervise_pre_boxes=True
    )
    criterion.matcher = matcher

    dn_queries = 2
    generator = torch.Generator().manual_seed(77)
    dn_boxes = torch.rand((layers, batch_size, dn_queries, 4), generator=generator)
    dn_boxes[..., 2:] = dn_boxes[..., 2:] * 0.2 + 0.05
    dn_scores = torch.randn((layers, batch_size, dn_queries, 3), generator=generator)
    dn_meta = {
        "dn_pos_idx": [torch.tensor([0]), torch.empty(0, dtype=torch.long)],
        "dn_num_group": 1,
    }
    pre_boxes = boxes[0].detach().clone().requires_grad_(True)
    dn_pre_boxes = dn_boxes[0].detach().clone().requires_grad_(True)

    losses = criterion(
        (boxes, scores),
        _mixed_batch(),
        dn_bboxes=dn_boxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
        corner_logits=_corner_logits(layers=layers - 1),
        pre_boxes=pre_boxes,
        dn_corner_logits=_corner_logits(
            layers=layers, batch=batch_size, queries=dn_queries
        ),
        dn_pre_boxes=dn_pre_boxes,
    )

    assert matcher.calls == layers
    assert criterion.stock_match_calls == layers
    assert criterion.fgl_extra_match_calls == 0
    assert criterion.last_normal_decoder_assignment is not None
    for actual, expected in zip(
        criterion.last_normal_decoder_assignment,
        matches,
    ):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
    assert {
        "loss_fgl",
        "loss_fgl_aux",
        "loss_fgl_dn",
        "loss_fgl_aux_dn",
        "loss_bbox_pre",
        "loss_giou_pre",
        "loss_bbox_pre_dn",
        "loss_giou_pre_dn",
    }.issubset(losses)
    assert not any(
        forbidden in key.lower()
        for key in losses
        for forbidden in ("ddf", "teacher", "lqe", "go_lsd", "gate")
    )
    (
        losses["loss_bbox_pre"]
        + losses["loss_giou_pre"]
        + losses["loss_bbox_pre_dn"]
        + losses["loss_giou_pre_dn"]
    ).backward()
    assert pre_boxes.grad is not None and torch.isfinite(pre_boxes.grad).all()
    assert dn_pre_boxes.grad is not None and torch.isfinite(dn_pre_boxes.grad).all()
    assert matcher.calls == layers


def test_empty_gt_losses_are_zero_finite_and_do_not_call_an_extra_matcher() -> None:
    layers = 2
    boxes, scores = _predictions(layers=layers)
    empty = _empty_matches()
    matcher = CountingMatcher(empty)
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.15, supervise_pre_boxes=True
    )
    criterion.matcher = matcher
    corners = _corner_logits(layers=layers - 1)
    pre_boxes = boxes[0].detach().clone().requires_grad_(True)

    losses = criterion(
        (boxes, scores),
        _empty_batch(),
        corner_logits=corners,
        pre_boxes=pre_boxes,
    )
    total = sum(losses.values())
    total.backward()

    assert matcher.calls == layers
    assert criterion.fgl_extra_match_calls == 0
    assert all(torch.isfinite(value) for value in losses.values())
    for key in ("loss_fgl", "loss_fgl_aux", "loss_bbox_pre", "loss_giou_pre"):
        torch.testing.assert_close(losses[key], torch.zeros_like(losses[key]))
    assert corners.grad is not None and torch.isfinite(corners.grad).all()
    assert pre_boxes.grad is not None and torch.isfinite(pre_boxes.grad).all()


def test_boundary_targets_produce_finite_fgl_backward() -> None:
    reference = torch.tensor(
        [[0.01, 0.01, 0.01, 0.01], [0.99, 0.99, 0.01, 0.01]],
        dtype=torch.float32,
    )
    targets_xyxy = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    left, wr, wl = bbox2distance(reference, targets_xyxy)
    logits = torch.randn((8, 33), requires_grad=True)
    quality = torch.ones(8)

    loss = adjacent_bin_fgl(logits, left, wl, wr, quality, avg_factor=2.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.all((left >= 0) & (left < 32))


def test_fgl_requires_preliminary_boxes_as_detached_target_reference() -> None:
    boxes, scores = _predictions(layers=2)
    matches = _matches_with_one_positive()
    criterion = FDRDetectionLoss(nc=3, use_vfl=True, fgl_weight=0.15)

    with pytest.raises(ValueError, match="pre_boxes are required"):
        criterion(
            (boxes, scores),
            _mixed_batch(),
            normal_match_indices=[matches, matches],
            corner_logits=_corner_logits(layers=1),
        )


def test_pre_box_helper_is_optional() -> None:
    boxes, scores = _predictions(layers=1)
    matches = _matches_with_one_positive()
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=False
    )

    losses = criterion(
        (boxes, scores),
        _mixed_batch(),
        normal_match_indices=[matches],
        pre_boxes=boxes[0].detach().clone().requires_grad_(True),
    )

    assert not any("pre" in key for key in losses)


def test_stock_plus_fgl_is_the_explicit_extension_entrypoint() -> None:
    boxes, scores = _predictions(layers=1)
    matches = _matches_with_one_positive()
    criterion = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=False
    )

    losses = criterion.stock_plus_fgl(
        (boxes, scores),
        _mixed_batch(),
        normal_match_indices=[matches],
    )

    assert {"loss_class", "loss_bbox", "loss_giou"}.issubset(losses)


@pytest.mark.parametrize("bad_layers", [0, 2])
def test_supplied_assignment_count_must_equal_prediction_layers(bad_layers: int) -> None:
    boxes, scores = _predictions(layers=3)
    matches = [_matches_with_one_positive()] * bad_layers
    criterion = FDRDetectionLoss(nc=3, use_vfl=True, fgl_weight=0.0)

    with pytest.raises(ValueError, match="one normal assignment per prediction layer"):
        criterion(
            (boxes, scores),
            _mixed_batch(),
            normal_match_indices=matches,
        )

from __future__ import annotations

import torch

from src.bpdd_loss import BPDDDetectionLoss, BPDDOptions
from src.fdr_loss import FDRDetectionLoss


def _batch() -> dict:
    return {
        "cls": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor([[0.52, 0.48, 0.18, 0.22]], dtype=torch.float32),
        "gt_groups": [1],
    }


def _predictions(layers: int = 3, queries: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2901)
    boxes = torch.rand((layers, 1, queries, 4), generator=generator)
    boxes[..., 2:] = boxes[..., 2:] * 0.2 + 0.05
    scores = torch.randn((layers, 1, queries, 3), generator=generator)
    return boxes, scores


def _matches(query: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.tensor([query]), torch.tensor([0]))]


def _corners(layers: int = 2, queries: int = 3) -> torch.Tensor:
    logits = torch.zeros((layers, 1, queries, 4, 33))
    logits[0, 0, 0, :, 0] = -4.0
    logits[0, 0, 0, :, 1] = 4.0
    logits[1, 0, 0, :, 0] = 4.0
    logits[1, 0, 0, :, 1] = -4.0
    return logits.reshape(layers, 1, queries, 132).requires_grad_(True)


def _assert_mapping_exact(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_disabled_bpdd_returns_the_exact_parent_loss_mapping() -> None:
    boxes, scores = _predictions()
    corners = _corners()
    pre_boxes = boxes[1].detach().clone()
    assignments = [_matches(2), _matches(1), _matches(0)]
    parent = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.15, supervise_pre_boxes=True
    )
    candidate = BPDDDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.15,
        supervise_pre_boxes=True,
        bpdd_options=BPDDOptions(enabled=False),
    )

    expected = parent(
        (boxes, scores),
        _batch(),
        normal_match_indices=assignments,
        corner_logits=corners,
        pre_boxes=pre_boxes,
    )
    actual = candidate(
        (boxes, scores),
        _batch(),
        normal_match_indices=assignments,
        corner_logits=corners,
        pre_boxes=pre_boxes,
    )

    _assert_mapping_exact(actual, expected)
    assert "loss_bpdd" not in actual


def test_zero_weight_bpdd_returns_the_exact_parent_loss_mapping() -> None:
    boxes, scores = _predictions()
    corners = _corners()
    pre_boxes = boxes[1].detach().clone()
    assignments = [_matches(2), _matches(1), _matches(0)]
    parent = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=False
    )
    candidate = BPDDDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.0,
        supervise_pre_boxes=False,
        bpdd_options=BPDDOptions(weight=0.0),
    )

    expected = parent(
        (boxes, scores),
        _batch(),
        normal_match_indices=assignments,
        corner_logits=corners,
        pre_boxes=pre_boxes,
    )
    actual = candidate(
        (boxes, scores),
        _batch(),
        normal_match_indices=assignments,
        corner_logits=corners,
        pre_boxes=pre_boxes,
    )

    _assert_mapping_exact(actual, expected)
    assert "loss_bpdd" not in actual


def test_bpdd_reuses_only_final_stock_match_and_never_calls_matcher_again() -> None:
    boxes, scores = _predictions()
    corners = _corners()
    assignments = [_matches(2), _matches(1), _matches(0)]
    criterion = BPDDDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.0,
        supervise_pre_boxes=False,
        bpdd_options=BPDDOptions(margin=0.0),
    )

    losses = criterion(
        (boxes, scores),
        _batch(),
        normal_match_indices=assignments,
        corner_logits=corners,
        pre_boxes=boxes[1].detach().clone(),
    )
    losses["loss_bpdd"].backward()
    gradient = corners.grad.reshape(2, 1, 3, 4, 33)

    assert criterion.stock_match_calls == 0
    assert criterion.fgl_extra_match_calls == 0
    assert losses["loss_bpdd"].item() > 0
    assert gradient[0, 0, 0].abs().sum() > 0
    torch.testing.assert_close(gradient[0, 0, 1], torch.zeros_like(gradient[0, 0, 1]))
    torch.testing.assert_close(gradient[0, 0, 2], torch.zeros_like(gradient[0, 0, 2]))
    torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]))
    assert criterion.last_bpdd_statistics["matched_queries"].item() == 1


def test_bpdd_empty_final_assignment_adds_finite_zero_loss() -> None:
    boxes, scores = _predictions()
    empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    criterion = BPDDDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.0,
        supervise_pre_boxes=False,
        bpdd_options=BPDDOptions(),
    )
    corners = _corners()

    losses = criterion(
        (boxes, scores),
        {"cls": torch.empty(0, dtype=torch.long), "bboxes": torch.empty(0, 4), "gt_groups": [0]},
        normal_match_indices=[empty, empty, empty],
        corner_logits=corners,
        pre_boxes=boxes[1].detach().clone(),
    )
    losses["loss_bpdd"].backward()

    assert torch.isfinite(losses["loss_bpdd"])
    torch.testing.assert_close(losses["loss_bpdd"], torch.zeros_like(losses["loss_bpdd"]))
    assert corners.grad is not None
    assert criterion.last_bpdd_statistics["matched_queries"].item() == 0


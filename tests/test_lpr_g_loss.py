from __future__ import annotations

import torch

from src.lpr_g_loss import MatchRecordingRTDETRDetectionLoss


def _batch() -> dict:
    return {
        "cls": torch.tensor([1]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "gt_groups": [1],
    }


def test_stock_loss_records_main_normal_match_without_reusing_it_for_aux() -> None:
    criterion = MatchRecordingRTDETRDetectionLoss(nc=3, use_vfl=True)
    boxes = torch.rand(3, 1, 5, 4)
    scores = torch.randn(3, 1, 5, 3)

    losses = criterion((boxes, scores), _batch())

    assert {"loss_giou", "loss_class", "loss_bbox"}.issubset(losses)
    assert criterion.last_stock_match_indices is not None
    assert criterion.normal_match_calls == 3


def test_refinement_loss_uses_recorded_match_and_has_no_class_term() -> None:
    criterion = MatchRecordingRTDETRDetectionLoss(nc=3, use_vfl=True)
    boxes = torch.rand(2, 1, 5, 4)
    scores = torch.randn(2, 1, 5, 3)
    criterion((boxes, scores), _batch())
    refined = boxes[-1].detach().clone().requires_grad_(True)

    losses = criterion.refinement_loss(refined, _batch())
    sum(losses.values()).backward()

    assert set(losses) == {"loss_bbox_refine", "loss_giou_refine"}
    assert refined.grad is not None


def test_denoising_match_does_not_replace_recorded_normal_match() -> None:
    criterion = MatchRecordingRTDETRDetectionLoss(nc=3, use_vfl=True)
    boxes = torch.rand(2, 1, 5, 4)
    scores = torch.randn(2, 1, 5, 3)
    dn_boxes = torch.rand(2, 1, 2, 4)
    dn_scores = torch.randn(2, 1, 2, 3)
    dn_meta = {
        "dn_pos_idx": [torch.tensor([0])],
        "dn_num_group": 1,
    }

    criterion(
        (boxes, scores),
        _batch(),
        dn_bboxes=dn_boxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
    )

    assert criterion.last_stock_match_indices is not None
    assert criterion.normal_match_calls == 2

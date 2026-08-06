from __future__ import annotations

import torch

from src.scads import AdaptiveIntegral
from src.scads_loss import SCADSFDRDetectionLoss


def _batch(empty: bool = False) -> dict:
    if empty:
        return {
            "cls": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "gt_groups": [0, 0],
        }
    return {
        "cls": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor([[0.52, 0.48, 0.18, 0.22]]),
        "gt_groups": [1, 0],
    }


def _matches(empty: bool = False):
    if empty:
        return [
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        ]
    return [
        (torch.tensor([1], dtype=torch.long), torch.tensor([0], dtype=torch.long)),
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    ]


def _inputs(empty: bool = False):
    generator = torch.Generator().manual_seed(901)
    boxes = torch.rand((7, 2, 4, 4), generator=generator)
    boxes[..., 2:] = boxes[..., 2:] * 0.2 + 0.05
    scores = torch.randn((7, 2, 4, 3), generator=generator)
    corners = torch.randn(
        (6, 2, 4, 132), generator=generator, requires_grad=True
    )
    pre_boxes = boxes[1].detach().clone().requires_grad_(True)
    support_logits = torch.tensor(
        [[[0.0, 1.0, -1.0]] * 4] * 2,
        requires_grad=True,
    )
    route_weights = support_logits.softmax(-1)
    integral = AdaptiveIntegral()
    projects = integral.effective_project(route_weights)
    assignments = [_matches(empty)] * 7
    return (
        (boxes, scores),
        corners,
        pre_boxes,
        support_logits,
        projects,
        assignments,
        integral.projects,
    )


def test_scads_fgl_and_route_loss_reuse_stock_assignments_and_backpropagate() -> None:
    predictions, corners, pre_boxes, logits, projects, matches, bank = _inputs()
    criterion = SCADSFDRDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.15,
        supervise_pre_boxes=False,
        support_project_bank=bank,
        scads_route_weight=0.05,
        scads_margin_ratio=0.02,
    )
    losses = criterion(
        predictions,
        _batch(),
        corner_logits=corners,
        pre_boxes=pre_boxes,
        normal_match_indices=matches,
        support_logits=logits,
        support_projects=projects,
    )
    assert {"loss_fgl", "loss_fgl_aux", "loss_scads_route"}.issubset(losses)
    assert criterion.stock_match_calls == 0
    assert criterion.fgl_extra_match_calls == 0
    assert criterion.last_route_positive_count == 1
    assert criterion.last_route_target_counts.sum().item() == 1
    total = losses["loss_fgl"] + losses["loss_fgl_aux"] + losses["loss_scads_route"]
    total.backward()
    assert corners.grad is not None and torch.isfinite(corners.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert pre_boxes.grad is None


def test_empty_gt_scads_losses_are_finite_zero_and_have_finite_backward() -> None:
    predictions, corners, pre_boxes, logits, projects, matches, bank = _inputs(
        empty=True
    )
    criterion = SCADSFDRDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.15,
        supervise_pre_boxes=False,
        support_project_bank=bank,
    )
    losses = criterion(
        predictions,
        _batch(empty=True),
        corner_logits=corners,
        pre_boxes=pre_boxes,
        normal_match_indices=matches,
        support_logits=logits,
        support_projects=projects,
    )
    for key in ("loss_fgl", "loss_fgl_aux", "loss_scads_route"):
        assert torch.isfinite(losses[key])
        torch.testing.assert_close(losses[key], torch.zeros_like(losses[key]))
    sum(losses.values()).backward()
    assert corners.grad is not None and torch.isfinite(corners.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert criterion.last_route_positive_count == 0


def test_scads_requires_projects_for_distribution_supervision() -> None:
    predictions, corners, pre_boxes, logits, _projects, matches, bank = _inputs()
    criterion = SCADSFDRDetectionLoss(
        nc=3,
        use_vfl=True,
        support_project_bank=bank,
    )
    try:
        criterion(
            predictions,
            _batch(),
            corner_logits=corners,
            pre_boxes=pre_boxes,
            normal_match_indices=matches,
            support_logits=logits,
        )
    except ValueError as error:
        assert "adaptive support projects" in str(error)
    else:
        raise AssertionError("SCADS accepted missing adaptive projects")

from __future__ import annotations

import torch

from src.bpdd_capacity_loss import (
    CapacityBPDDOptions,
    bounded_improvement,
    capacity_bpdd_warmup,
    quality_gated_capacity_distillation,
    raw_fdr_targets,
)


def _matches(query: int = 0, target: int = 0):
    return (
        torch.tensor([0], dtype=torch.long),
        torch.tensor([query], dtype=torch.long),
        torch.tensor([target], dtype=torch.long),
    )


def _inputs(layers: int = 2):
    student_corners = torch.zeros(layers, 1, 2, 4, 33, requires_grad=True)
    student_classes = torch.zeros(layers, 1, 2, 3, requires_grad=True)
    expert_corners = torch.zeros(1, 2, 4, 33, requires_grad=True)
    expert_classes = torch.zeros(1, 2, 3, requires_grad=True)
    reference = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]])
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    gt_classes = torch.tensor([1], dtype=torch.long)
    return student_corners, student_classes, expert_corners, expert_classes, reference, gt_boxes, gt_classes


def test_raw_targets_are_unclipped_and_report_support() -> None:
    reference = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    targets = torch.tensor([[0.95, 0.5, 0.01, 0.01]])
    raw, representable = raw_fdr_targets(reference, targets)
    assert raw.shape == representable.shape == (1, 4)
    assert (~representable).any()
    assert raw.abs().max() > 4


def test_bounded_improvement_and_schedule_are_exact() -> None:
    student = torch.tensor([0.50, 0.50, 0.10])
    teacher = torch.tensor([0.30, 0.49, 0.20])
    expected = torch.tensor([0.18 / 0.28, 0.0, 0.0])
    torch.testing.assert_close(bounded_improvement(student, teacher, margin=0.02, tau=0.1), expected)
    assert capacity_bpdd_warmup(10, 10, 20) == 0.0
    assert capacity_bpdd_warmup(15, 10, 20) == 0.5
    assert capacity_bpdd_warmup(20, 10, 20) == 1.0


def test_no_matches_and_identity_mismatch_return_connected_zero() -> None:
    values = _inputs()
    empty = tuple(torch.empty(0, dtype=torch.long) for _ in range(3))
    result = quality_gated_capacity_distillation(
        *values, [empty, empty], empty, CapacityBPDDOptions(), epoch=20
    )
    assert result.loss.item() == 0
    result.loss.backward()
    assert values[0].grad is not None

    values = _inputs()
    result = quality_gated_capacity_distillation(
        *values, [_matches(query=1), _matches(query=1)], _matches(query=0),
        CapacityBPDDOptions(), epoch=20,
    )
    assert result.loss.item() == 0
    assert result.statistics["identity_consistent_ratio"].item() == 0


def test_classification_gate_uses_detached_better_teacher() -> None:
    values = list(_inputs())
    with torch.no_grad():
        values[1][..., 1] = -3.0
        values[3][..., 1] = 3.0
    result = quality_gated_capacity_distillation(
        *values, [_matches(), _matches()], _matches(),
        CapacityBPDDOptions(loc_weight=0.0), epoch=20,
    )
    assert result.classification_loss.item() > 0
    result.loss.backward()
    assert values[1].grad is not None and values[1].grad.abs().sum() > 0
    assert values[3].grad is None
    assert result.statistics["active_class_ratio"].item() > 0


def test_out_of_support_edges_cannot_contribute_localization() -> None:
    values = list(_inputs())
    values[5][0] = torch.tensor([0.95, 0.95, 0.01, 0.01])
    with torch.no_grad():
        values[0].normal_(0, 1)
        values[2].normal_(0, 1)
    result = quality_gated_capacity_distillation(
        *values, [_matches(), _matches()], _matches(),
        CapacityBPDDOptions(cls_weight=0.0), epoch=20,
    )
    assert result.localization_loss.item() == 0
    assert result.statistics["representable_edge_ratio"].item() < 1


def test_warmup_disables_both_distillation_terms_through_epoch_ten() -> None:
    values = list(_inputs())
    with torch.no_grad():
        values[1][..., 1] = -3.0
        values[3][..., 1] = 3.0
    result = quality_gated_capacity_distillation(
        *values, [_matches(), _matches()], _matches(), CapacityBPDDOptions(), epoch=10
    )
    assert result.loss.item() == 0
    assert result.statistics["warmup"].item() == 0

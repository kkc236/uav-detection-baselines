from __future__ import annotations

import pytest
import torch
import src.bpdd_capacity_loss as capacity_loss

from src.bpdd_capacity_loss import (
    CapacityBPDDOptions,
    bounded_improvement,
    capacity_bpdd_schedule,
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


def _inputs(layers: int = 2, queries: int = 2):
    student_corners = torch.zeros(layers, 1, queries, 4, 33, requires_grad=True)
    student_classes = torch.zeros(layers, 1, queries, 3, requires_grad=True)
    expert_corners = torch.zeros(1, queries, 4, 33, requires_grad=True)
    expert_classes = torch.zeros(1, queries, 3, requires_grad=True)
    reference = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]).expand(1, queries, 4)
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


def test_capacity_schedule_can_fade_bpdd_after_epoch_eighty() -> None:
    options = CapacityBPDDOptions(decay_start=80, decay_end=100)
    assert capacity_bpdd_schedule(10, options) == 0.0
    assert capacity_bpdd_schedule(15, options) == 0.5
    assert capacity_bpdd_schedule(20, options) == 1.0
    assert capacity_bpdd_schedule(80, options) == 1.0
    assert capacity_bpdd_schedule(90, options) == 0.5
    assert capacity_bpdd_schedule(100, options) == 0.0


@pytest.mark.parametrize("decay_start,decay_end", [(80, None), (None, 100), (100, 80)])
def test_capacity_options_reject_invalid_decay(
    decay_start: int | None, decay_end: int | None
) -> None:
    with pytest.raises(ValueError, match="decay"):
        CapacityBPDDOptions(decay_start=decay_start, decay_end=decay_end)


def test_capacity_result_reports_effective_schedule_scale() -> None:
    values = _inputs()
    result = quality_gated_capacity_distillation(
        *values,
        [_matches(), _matches()],
        _matches(),
        CapacityBPDDOptions(decay_start=80, decay_end=100),
        epoch=90,
    )
    assert result.statistics["warmup"].item() == 1.0
    assert result.statistics["schedule_scale"].item() == 0.5


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


def test_classification_gate_uses_detached_better_teacher(monkeypatch) -> None:
    values = list(_inputs())
    monkeypatch.setattr(
        capacity_loss,
        "decoded_teacher_iou_gate",
        lambda source, teacher, reference, targets, active_edges, margin=0.0:
        (torch.ones(source.shape[0], dtype=torch.bool),
         torch.ones(source.shape[0])),
    )
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


def test_classification_teacher_without_box_quality_gain_is_rejected() -> None:
    values = list(_inputs())
    with torch.no_grad():
        values[1][..., 1] = -3.0
        values[3][..., 1] = 3.0
    result = quality_gated_capacity_distillation(
        *values, [_matches(), _matches()], _matches(),
        CapacityBPDDOptions(loc_weight=0.0), epoch=20,
    )
    assert result.classification_loss.item() == 0.0
    assert result.statistics["active_classes"].item() == 0


def test_only_layers_with_active_class_kd_enter_layer_mean(monkeypatch) -> None:
    values = list(_inputs())
    monkeypatch.setattr(
        capacity_loss,
        "decoded_teacher_iou_gate",
        lambda source, teacher, reference, targets, active_edges, margin=0.0:
        (torch.ones(source.shape[0], dtype=torch.bool),
         torch.ones(source.shape[0])),
    )
    with torch.no_grad():
        values[1][0, 0, 0, 1] = -3.0
        values[3][0, 0, 1] = 3.0
        values[1][1, 0, 0, 1] = 3.0
        values[3][0, 1] = 0.0
    one = quality_gated_capacity_distillation(
        values[0][:1], values[1][:1], values[2], values[3], values[4], values[5], values[6],
        [_matches()], _matches(), CapacityBPDDOptions(loc_weight=0.0), epoch=20,
    )
    two = quality_gated_capacity_distillation(
        *values, [_matches(), _matches()], _matches(),
        CapacityBPDDOptions(loc_weight=0.0), epoch=20,
    )
    torch.testing.assert_close(two.classification_loss, one.classification_loss)
    assert two.statistics["classification_active_layers"].item() == 1


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


@pytest.mark.parametrize("matches", [0, 1, 2, 4, 5])
def test_capacity_loss_accepts_multiple_identity_consistent_matches(matches: int) -> None:
    values = _inputs(queries=max(matches, 1))
    triple = (
        torch.zeros(matches, dtype=torch.long),
        torch.arange(matches, dtype=torch.long),
        torch.zeros(matches, dtype=torch.long),
    )
    result = quality_gated_capacity_distillation(
        *values, [triple] * values[0].shape[0], triple, CapacityBPDDOptions(), epoch=20
    )
    assert torch.isfinite(result.loss)


@pytest.mark.parametrize(
    "options,epoch",
    [
        (CapacityBPDDOptions(enabled=False), 20),
        (CapacityBPDDOptions(), 0),
        (CapacityBPDDOptions(cls_weight=0), 20),
    ],
)
def test_capacity_loss_short_circuits_after_validating_match_shapes(
    options: CapacityBPDDOptions, epoch: int
) -> None:
    values = _inputs(queries=2)
    triple = (
        torch.zeros(2, dtype=torch.long),
        torch.arange(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    result = quality_gated_capacity_distillation(
        *values, [triple] * values[0].shape[0], triple, options, epoch=epoch
    )
    assert torch.isfinite(result.loss)

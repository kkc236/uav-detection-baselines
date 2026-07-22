import pytest
import torch

from src.ebc_qp_diagnostics import MechanismDiagnosticsAccumulator
from src.ebc_qp_decoder import EBCQPForwardState


def test_mechanism_diagnostics_count_unique_gain_loss_and_p2_quality():
    state = EBCQPForwardState(
        stock_topk_indices=torch.tensor([[0, 1]]),
        p2_topk_indices=torch.tensor([[0, 1, 2]]),
        stock_centers=torch.tensor([[[0.2, 0.2], [0.8, 0.8]]]),
        final_centers=torch.tensor([[[0.5, 0.5], [0.8, 0.8]]]),
        final_boxes=torch.tensor([[[0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
        p2_top_centers=torch.tensor([[[0.49, 0.5], [0.51, 0.5], [0.95, 0.95]]]),
        final_ranking_score=torch.tensor([[0.9, 0.7]]),
        final_sources=torch.tensor([[1, 0]]),
        final_source_indices=torch.tensor([[1, 0]]),
        p2_loss=torch.tensor(0.0),
        ebc_loss=torch.tensor(0.0),
        p2_entry_count=1,
        ordinary_query_count=2,
        encoder_aux_source_is_stock=True,
        competition_active=True,
        ebc_active=True,
        stock_boundary=torch.tensor([0.8]),
        assigned_pairs=[],
        uncovered=[],
        p2_all_boxes=torch.tensor(
            [[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]
        ),
        p2_all_logits=torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]]),
        p2_shape=(1, 3),
        p2_valid_mask=torch.ones(1, 3, 1, dtype=torch.bool),
        c2_p3_rms_ratio=torch.tensor([2.0]),
    )
    batch = {
        "batch_idx": torch.tensor([0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "cls": torch.tensor([[0.0]]),
        "img": torch.zeros(1, 3, 160, 160),
    }
    diagnostics = MechanismDiagnosticsAccumulator(tiny_radius=16.0)

    diagnostics.update(state, batch)
    result = diagnostics.compute()

    assert result["n_gain"] == 1
    assert result["n_loss"] == 0
    assert result["v_replace"] == 1
    assert result["stock_top300_coverage"] == 0.0
    assert result["local_assign_rate"] == 1.0
    assert result["effective_p2_entry_rate"] == pytest.approx(0.5)
    assert result["boundary_gap_mean"] == pytest.approx(0.1)
    assert result["boundary_gap_positive_ratio"] == 1.0
    assert result["p2_foreground_at_50"] == 2
    assert result["p2_unique_gt_at_50"] == 1
    assert result["p2_duplicate_rate_at_50"] == pytest.approx(0.5)
    assert result["p2_background_rate_at_50"] == pytest.approx(1 / 3)
    assert result["unassigned_entry_rate"] == 0.0
    assert result["assigned_entry_mean_iou"] == pytest.approx(1.0)
    assert result["assigned_entry_mean_nwd"] == pytest.approx(1.0)
    assert result["low_quality_entry_rate"] == 0.0
    assert result["score_quality_sample_count"] == 1
    assert result["score_iou_spearman"] is None
    assert result["score_nwd_spearman"] is None
    assert result["c2_p3_rms_ratio"] == pytest.approx(2.0)


def test_mechanism_diagnostics_ignore_inactive_competition_batches():
    state = EBCQPForwardState(
        stock_topk_indices=torch.empty((1, 0), dtype=torch.long),
        p2_topk_indices=torch.empty((1, 0), dtype=torch.long),
        stock_centers=torch.empty((1, 0, 2)),
        final_centers=torch.empty((1, 0, 2)),
        final_boxes=torch.empty((1, 0, 4)),
        p2_top_centers=torch.empty((1, 0, 2)),
        final_ranking_score=torch.empty((1, 0)),
        final_sources=torch.empty((1, 0), dtype=torch.long),
        final_source_indices=torch.empty((1, 0), dtype=torch.long),
        p2_loss=torch.tensor(0.0),
        ebc_loss=torch.tensor(0.0),
        p2_entry_count=0,
        ordinary_query_count=0,
        encoder_aux_source_is_stock=True,
        competition_active=False,
        ebc_active=False,
        stock_boundary=torch.tensor([0.0]),
        assigned_pairs=[],
        uncovered=[],
    )
    diagnostics = MechanismDiagnosticsAccumulator(tiny_radius=16.0)

    diagnostics.update(
        state,
        {"batch_idx": torch.empty(0), "bboxes": torch.empty((0, 4)), "img": torch.zeros(1, 3, 320, 320)},
    )

    assert diagnostics.compute()["active_images"] == 0


def test_mechanism_diagnostics_report_quality_head_calibration_separately():
    centers = torch.tensor([[1 / 6, 0.5], [0.5, 0.5], [5 / 6, 0.5]])
    target_boxes = torch.cat((centers, torch.full((3, 2), 0.1)), dim=1)
    predicted_boxes = target_boxes.clone()
    predicted_boxes[:, 2:] = torch.tensor([[0.1, 0.1], [0.08, 0.08], [0.05, 0.05]])
    state = EBCQPForwardState(
        stock_topk_indices=torch.tensor([[0, 1, 2]]),
        p2_topk_indices=torch.tensor([[0, 1, 2]]),
        stock_centers=centers[None],
        final_centers=centers[None],
        final_boxes=predicted_boxes[None],
        p2_top_centers=centers[None],
        final_ranking_score=torch.tensor([[0.9, 0.8, 0.7]]),
        final_sources=torch.zeros((1, 3), dtype=torch.long),
        final_source_indices=torch.tensor([[0, 1, 2]]),
        p2_loss=torch.tensor(0.0),
        ebc_loss=torch.tensor(0.0),
        p2_entry_count=0,
        ordinary_query_count=3,
        encoder_aux_source_is_stock=True,
        competition_active=True,
        ebc_active=False,
        stock_boundary=torch.tensor([0.6]),
        assigned_pairs=[],
        uncovered=[],
        p2_all_boxes=predicted_boxes[None],
        p2_all_logits=torch.tensor([[[1.0], [1.0], [1.0]]]),
        p2_all_quality_logits=torch.tensor([[3.0, 2.0, 1.0]]),
        p2_shape=(1, 3),
        p2_valid_mask=torch.ones(1, 3, 1, dtype=torch.bool),
    )
    batch = {
        "batch_idx": torch.tensor([0, 0, 0]),
        "bboxes": target_boxes,
        "cls": torch.zeros((3, 1)),
        "img": torch.zeros(1, 3, 160, 160),
    }
    diagnostics = MechanismDiagnosticsAccumulator(tiny_radius=16.0)

    diagnostics.update(state, batch)
    result = diagnostics.compute()

    assert result["quality_logit_sample_count"] == 3
    assert result["quality_logit_iou_spearman"] == pytest.approx(1.0)
    assert result["quality_logit_nwd_spearman"] == pytest.approx(1.0)

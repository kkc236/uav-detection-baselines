import torch

from src.ebc_qp_matching import assign_local_p2, match_centers_inside_boxes


def test_matching_maximizes_gt_coverage_before_distance():
    centers = torch.tensor([[0.20, 0.20], [0.28, 0.20]])
    boxes = torch.tensor([[0.24, 0.20, 0.10, 0.10], [0.28, 0.20, 0.04, 0.04]])

    result = match_centers_inside_boxes(centers, boxes)

    assert result.pairs.tolist() == [[0, 0], [1, 1]]


def test_equal_cost_tie_prefers_low_gt_then_low_token():
    centers = torch.tensor([[0.45, 0.50], [0.55, 0.50]])
    boxes = torch.tensor([[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20]])

    for _ in range(5):
        result = match_centers_inside_boxes(centers, boxes)
        assert result.pairs.tolist() == [[0, 0], [1, 1]]


def test_local_assignment_uses_unique_cells_and_reports_unassigned():
    result = assign_local_p2(
        height=4,
        width=4,
        boxes=torch.tensor([[0.375, 0.375, 0.05, 0.05], [0.375, 0.375, 0.05, 0.05]]),
        valid_mask=torch.ones(16, dtype=torch.bool),
        radius=1,
    )

    assert result.pairs.shape == (2, 2)
    assert result.pairs[:, 1].unique().numel() == 2
    assert result.unassigned_gt.numel() == 0


def test_empty_and_unreachable_inputs_return_typed_empty_tensors():
    result = match_centers_inside_boxes(
        torch.empty(0, 2),
        torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
    )

    assert result.pairs.shape == (0, 2)
    assert result.pairs.dtype == torch.long
    assert result.unassigned_gt.tolist() == [0]

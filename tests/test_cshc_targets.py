import torch

from src.cshc_loss import focal_binary_logits
from src.cshc_targets import build_tiny_center_targets


def test_tiny_centers_land_in_the_expected_cells_and_large_boxes_are_excluded():
    target = build_tiny_center_targets(
        bboxes=torch.tensor([[0.25, 0.75, 0.02, 0.02], [0.75, 0.25, 0.20, 0.20]]),
        batch_idx=torch.tensor([0, 0]),
        batch_size=1,
        height=4,
        width=4,
        tiny_area_threshold=0.0025,
    )

    assert target.shape == (1, 1, 4, 4)
    assert target[0, 0, 3, 1] == 1
    assert target.sum() == 1


def test_focal_binary_logits_is_finite_and_penalizes_wrong_high_confidence():
    target = torch.tensor([[[[1.0, 0.0]]]])
    good = focal_binary_logits(torch.tensor([[[[6.0, -6.0]]]]), target)
    bad = focal_binary_logits(torch.tensor([[[[-6.0, 6.0]]]]), target)

    assert torch.isfinite(good) and torch.isfinite(bad)
    assert good < bad

import torch

from src.btd_se_loss import binary_focal_loss
from src.btd_se_targets import build_auxiliary_targets


def test_targets_mark_objects_and_ignored_boxes_as_unreliable_background():
    bboxes = torch.tensor(
        [
            [0.50, 0.50, 0.25, 0.25],
            [0.15, 0.15, 0.20, 0.20],
        ],
        dtype=torch.float32,
    )
    classes = torch.tensor([[2.0], [-1.0]])
    batch_idx = torch.tensor([0.0, 0.0])

    targets = build_auxiliary_targets(
        bboxes=bboxes,
        classes=classes,
        batch_idx=batch_idx,
        batch_size=1,
        height=16,
        width=16,
    )

    assert targets.background[0, 0, 8, 8].item() == 0
    assert targets.background[0, 0, 2, 2].item() == 0
    assert targets.saliency[0, 0, 8, 8].item() == 1
    assert targets.saliency[0, 0, 2, 2].item() == 0
    assert not targets.saliency_valid[0, 0, 2, 2]
    assert targets.saliency_valid[0, 0, 8, 8]


def test_tiny_object_uses_minimum_one_cell_gaussian_standard_deviation():
    targets = build_auxiliary_targets(
        bboxes=torch.tensor([[0.50, 0.50, 0.01, 0.01]]),
        classes=torch.tensor([[0.0]]),
        batch_idx=torch.tensor([0.0]),
        batch_size=1,
        height=16,
        width=16,
    )

    center = targets.saliency[0, 0, 8, 8]
    neighbor = targets.saliency[0, 0, 8, 9]
    assert center.item() == 1
    assert 0.60 < neighbor.item() < 0.61


def test_overlapping_gaussians_are_combined_by_pixelwise_maximum():
    one = build_auxiliary_targets(
        bboxes=torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
        classes=torch.tensor([[0.0]]),
        batch_idx=torch.tensor([0.0]),
        batch_size=1,
        height=16,
        width=16,
    )
    two = build_auxiliary_targets(
        bboxes=torch.tensor([[0.50, 0.50, 0.20, 0.20], [0.625, 0.50, 0.20, 0.20]]),
        classes=torch.tensor([[0.0], [1.0]]),
        batch_idx=torch.tensor([0.0, 0.0]),
        batch_size=1,
        height=16,
        width=16,
    )

    assert torch.all(two.saliency >= one.saliency)
    assert two.saliency[0, 0, 8, 10].item() == 1


def test_binary_focal_loss_is_finite_and_respects_valid_mask():
    probabilities = torch.tensor([[[[0.8, 0.2], [0.6, 0.4]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    valid = torch.tensor([[[[True, True], [False, False]]]])

    loss = binary_focal_loss(probabilities, target, alpha=0.25, exponent=2.0, valid_mask=valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad[0, 0, 1]).item() == 0


def test_binary_focal_loss_returns_differentiable_zero_when_no_pixels_are_valid():
    probabilities = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    target = torch.zeros_like(probabilities)
    valid = torch.zeros_like(probabilities, dtype=torch.bool)

    loss = binary_focal_loss(probabilities, target, valid_mask=valid)
    loss.backward()

    assert loss.item() == 0
    assert probabilities.grad is not None
    assert torch.count_nonzero(probabilities.grad).item() == 0

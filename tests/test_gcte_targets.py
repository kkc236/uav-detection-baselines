import torch

from src.gcte_targets import (
    build_equivariance_pairs,
    build_quality_targets,
    build_tiny_anchor_mask,
)


def test_quality_target_is_best_same_class_global_iou():
    local_boxes = torch.tensor(
        [
            [
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
                [0.25, 0.25, 0.20, 0.20],
            ]
        ]
    )
    logits = torch.tensor(
        [[[10.0, -10.0], [-10.0, 10.0], [-10.0, 10.0]]]
    )
    gt_boxes = torch.tensor(
        [[0.25, 0.25, 0.20, 0.20], [0.75, 0.75, 0.10, 0.10]]
    )
    gt_classes = torch.tensor([0, 1])

    targets = build_quality_targets(
        local_boxes,
        logits,
        gt_boxes,
        gt_classes,
    )

    torch.testing.assert_close(
        targets,
        torch.tensor([[[1.0], [0.25], [0.0]]]),
        atol=1e-6,
        rtol=0,
    )


def test_quality_targets_are_zero_when_image_has_no_gt():
    targets = build_quality_targets(
        torch.full((1, 4, 4), 0.25),
        torch.zeros(1, 4, 3),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
    )

    assert torch.equal(targets, torch.zeros(1, 4, 1))


def test_equivariance_pairs_join_same_gt_across_distinct_views_only():
    matched_gt = torch.tensor([7, 7, 7, 9, 9, -1])
    views = torch.tensor([0, 1, 1, 2, 3, 0])

    pairs = build_equivariance_pairs(matched_gt, views)

    assert pairs.tolist() == [[0, 1], [0, 2], [3, 4]]


def test_tiny_anchor_mask_uses_frozen_640_effective_size():
    boxes = torch.tensor(
        [
            [
                [0.5, 0.5, 16.0 / 640.0, 16.0 / 640.0],
                [0.5, 0.5, 17.0 / 640.0, 17.0 / 640.0],
                [0.5, 0.5, 8.0 / 640.0, 32.0 / 640.0],
            ]
        ]
    )

    mask = build_tiny_anchor_mask(boxes)

    assert mask.tolist() == [[[True], [False], [True]]]

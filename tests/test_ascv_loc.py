from __future__ import annotations

import torch
from torch import nn

from src.ascv_loc import (
    ASCV_LAMBDA,
    ASCV_TILE_RATIO,
    ASCV_TINY_BOUNDARY_PX,
    ASCV_WARMUP_EPOCHS,
    ascv_warmup,
    build_local_targets,
    compute_ascv_loc_loss,
    join_matches_by_target_id,
    local_to_full_xywh,
    preserve_batchnorm_buffers,
    select_target_anchored_crops,
)


def test_frozen_protocol_constants() -> None:
    assert ASCV_TILE_RATIO == 0.60
    assert ASCV_TINY_BOUNDARY_PX == 16.0
    assert ASCV_LAMBDA == 0.1
    assert ASCV_WARMUP_EPOCHS == 3


def test_warmup_is_frozen_three_epoch_linear_ramp() -> None:
    assert ascv_warmup(0) == 1 / 3
    assert ascv_warmup(1) == 2 / 3
    assert ascv_warmup(2) == 1.0
    assert ascv_warmup(8) == 1.0


def test_target_anchored_crop_is_deterministic_and_contains_anchor() -> None:
    boxes = torch.tensor(
        [
            [0.20, 0.25, 0.10, 0.10],
            [0.80, 0.75, 0.08, 0.06],
        ],
        dtype=torch.float32,
    )
    batch_indices = torch.tensor([0, 0])

    first = select_target_anchored_crops(
        boxes=boxes,
        batch_indices=batch_indices,
        batch_size=1,
        image_hw=(640, 640),
        image_keys=["sample.jpg"],
    )
    second = select_target_anchored_crops(
        boxes=boxes,
        batch_indices=batch_indices,
        batch_size=1,
        image_hw=(640, 640),
        image_keys=["sample.jpg"],
    )

    assert torch.equal(first, second)
    assert first.shape == (1, 4)
    x1, y1, x2, y2 = first[0].tolist()
    assert x2 - x1 == 384
    assert y2 - y1 == 384
    contained = []
    for cx, cy, width, height in boxes.tolist():
        bx1 = (cx - width / 2) * 640
        by1 = (cy - height / 2) * 640
        bx2 = (cx + width / 2) * 640
        by2 = (cy + height / 2) * 640
        contained.append(bx1 >= x1 and by1 >= y1 and bx2 <= x2 and by2 <= y2)
    assert any(contained)


def test_local_targets_keep_complete_boxes_and_ignore_boundary_intersections() -> None:
    boxes = torch.tensor(
        [
            [0.25, 0.25, 0.10, 0.10],  # complete in [0, 0, 384, 384]
            [0.60, 0.25, 0.10, 0.10],  # intersects right crop boundary
            [0.85, 0.85, 0.10, 0.10],  # outside
        ],
        dtype=torch.float32,
    )
    classes = torch.tensor([1, 2, 3])
    batch_indices = torch.tensor([0, 0, 0])
    crops = torch.tensor([[0, 0, 384, 384]], dtype=torch.long)

    local = build_local_targets(
        full_boxes=boxes,
        classes=classes,
        batch_indices=batch_indices,
        crops=crops,
        image_hw=(640, 640),
    )

    assert local.gt_ids.tolist() == [0]
    assert local.classes.tolist() == [1]
    assert local.batch_indices.tolist() == [0]
    assert local.groups == [1]
    assert torch.allclose(local.boxes[0], torch.tensor([0.4166667, 0.4166667, 0.1666667, 0.1666667]))


def test_local_to_full_round_trip_geometry() -> None:
    local = torch.tensor([[0.5, 0.5, 0.25, 0.50]], dtype=torch.float32)
    crops = torch.tensor([[128, 64, 512, 448]], dtype=torch.long)
    mapped = local_to_full_xywh(local, crops, image_hw=(640, 640))
    expected = torch.tensor([[0.5, 0.4, 0.15, 0.30]], dtype=torch.float32)
    assert torch.allclose(mapped, expected)


def test_match_join_uses_shared_original_target_identity() -> None:
    full_matches = [
        (torch.tensor([4, 7]), torch.tensor([0, 1])),
        (torch.tensor([2]), torch.tensor([2])),
    ]
    local_matches = [
        (torch.tensor([8]), torch.tensor([0])),
        (torch.tensor([5]), torch.tensor([1])),
    ]
    local_gt_ids = torch.tensor([1, 2])

    joined = join_matches_by_target_id(
        full_matches=full_matches,
        local_matches=local_matches,
        local_gt_ids=local_gt_ids,
    )

    assert joined.batch_indices.tolist() == [0, 1]
    assert joined.full_query_indices.tolist() == [7, 2]
    assert joined.local_query_indices.tolist() == [8, 5]
    assert joined.gt_ids.tolist() == [1, 2]
    assert joined.local_target_indices.tolist() == [0, 1]


def test_tiny_pair_detaches_local_teacher_and_updates_full_student() -> None:
    full = torch.tensor([[0.50, 0.50, 0.02, 0.02]], requires_grad=True)
    local = torch.tensor([[0.50, 0.50, 0.025, 0.025]], requires_grad=True)
    gt = torch.tensor([[0.50, 0.50, 0.02, 0.02]])
    crops = torch.tensor([[128, 128, 512, 512]], dtype=torch.long)

    result = compute_ascv_loc_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=gt,
        pair_crops=crops,
        image_hw=(640, 640),
    )
    result.loss.backward()

    assert result.pair_count == 1
    assert result.tiny_pair_count == 1
    assert result.non_tiny_pair_count == 0
    assert full.grad is not None and full.grad.abs().sum() > 0
    assert local.grad is None or torch.equal(local.grad, torch.zeros_like(local.grad))


def test_non_tiny_pair_detaches_full_teacher_and_updates_local_student() -> None:
    full = torch.tensor([[0.50, 0.50, 0.10, 0.10]], requires_grad=True)
    local = torch.tensor([[0.50, 0.50, 0.20, 0.20]], requires_grad=True)
    gt = torch.tensor([[0.50, 0.50, 0.10, 0.10]])
    crops = torch.tensor([[128, 128, 512, 512]], dtype=torch.long)

    result = compute_ascv_loc_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=gt,
        pair_crops=crops,
        image_hw=(640, 640),
    )
    result.loss.backward()

    assert result.pair_count == 1
    assert result.tiny_pair_count == 0
    assert result.non_tiny_pair_count == 1
    assert local.grad is not None and local.grad.abs().sum() > 0
    assert full.grad is None or torch.equal(full.grad, torch.zeros_like(full.grad))


def test_empty_pairs_return_differentiable_finite_zero() -> None:
    full = torch.empty((0, 4), requires_grad=True)
    local = torch.empty((0, 4), requires_grad=True)
    result = compute_ascv_loc_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=torch.empty((0, 4)),
        pair_crops=torch.empty((0, 4), dtype=torch.long),
        image_hw=(640, 640),
    )
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert result.pair_count == 0
    assert torch.equal(full.grad, torch.empty((0, 4)))
    assert torch.equal(local.grad, torch.empty((0, 4)))


def test_exact_16px_effective_size_uses_tiny_direction() -> None:
    full = torch.tensor([[0.50, 0.50, 0.025, 0.025]], requires_grad=True)
    local = torch.tensor([[0.50, 0.50, 0.03, 0.03]], requires_grad=True)
    gt = torch.tensor([[0.50, 0.50, 0.025, 0.025]])
    crops = torch.tensor([[128, 128, 512, 512]], dtype=torch.long)

    result = compute_ascv_loc_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=gt,
        pair_crops=crops,
        image_hw=(640, 640),
    )
    result.loss.backward()

    assert result.tiny_pair_count == 1
    assert result.non_tiny_pair_count == 0
    assert full.grad is not None and full.grad.abs().sum() > 0
    assert local.grad is None


def test_crop_is_invariant_to_other_images_inserted_before_batch_item() -> None:
    target = torch.tensor([[0.20, 0.25, 0.10, 0.10]], dtype=torch.float32)
    alone = select_target_anchored_crops(
        boxes=target,
        batch_indices=torch.tensor([0]),
        batch_size=1,
        image_hw=(640, 640),
        image_keys=["same.jpg"],
    )
    with_prefix = select_target_anchored_crops(
        boxes=torch.cat((torch.tensor([[0.8, 0.8, 0.1, 0.1]]), target)),
        batch_indices=torch.tensor([0, 1]),
        batch_size=2,
        image_hw=(640, 640),
        image_keys=["other.jpg", "same.jpg"],
    )

    torch.testing.assert_close(alone[0], with_prefix[1])


def test_nonfinite_predictions_are_rejected_without_mutating_inputs() -> None:
    full = torch.tensor([[0.50, 0.50, float("nan"), 0.05]], requires_grad=True)
    local = torch.tensor([[0.50, 0.50, 0.06, 0.06]], requires_grad=True)
    gt = torch.tensor([[0.50, 0.50, 0.05, 0.05]])
    crops = torch.tensor([[128, 128, 512, 512]], dtype=torch.long)
    local_before = local.detach().clone()
    gt_before = gt.clone()

    try:
        compute_ascv_loc_loss(
            full_pred_boxes=full,
            local_pred_boxes=local,
            full_gt_boxes=gt,
            pair_crops=crops,
            image_hw=(640, 640),
        )
    except RuntimeError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite predictions must fail closed")

    torch.testing.assert_close(local.detach(), local_before)
    torch.testing.assert_close(gt, gt_before)


def test_local_forward_preserves_batchnorm_buffers_and_keeps_gradients() -> None:
    module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)).train()
    before = {name: value.clone() for name, value in module.named_buffers()}
    image = torch.rand(2, 3, 8, 8, requires_grad=True)

    with preserve_batchnorm_buffers(module):
        output = module(image)
    output.sum().backward()

    for name, value in module.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert module[0].weight.grad is not None and module[0].weight.grad.abs().sum() > 0


def test_teacher_advantage_diagnostics_follow_frozen_directions() -> None:
    gt = torch.tensor(
        [
            [0.5, 0.5, 0.02, 0.02],
            [0.5, 0.5, 0.10, 0.10],
        ]
    )
    full = torch.tensor(
        [
            [0.5, 0.5, 0.04, 0.04],  # worse tiny student
            [0.5, 0.5, 0.10, 0.10],  # better non-tiny teacher
        ],
        requires_grad=True,
    )
    crops = torch.tensor([[128, 128, 512, 512], [128, 128, 512, 512]])
    local = torch.tensor(
        [
            [0.5, 0.5, 0.0333333, 0.0333333],  # maps to 0.02 full width
            [0.5, 0.5, 0.25, 0.25],  # maps to 0.15 full width
        ],
        requires_grad=True,
    )

    result = compute_ascv_loc_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=gt,
        pair_crops=crops,
        image_hw=(640, 640),
    )

    assert result.tiny_teacher_advantage_sum.item() > 0
    assert result.tiny_teacher_win_count == 1
    assert result.non_tiny_teacher_advantage_sum.item() > 0
    assert result.non_tiny_teacher_win_count == 1

from __future__ import annotations

import pytest
import torch

from src.tascv import (
    TASCV_TINY_BOUNDARY_PX,
    compute_tascv_loss,
)


def _crop(count: int) -> torch.Tensor:
    return torch.tensor(
        [[128, 128, 512, 512] for _ in range(count)],
        dtype=torch.long,
    )


def test_exact_16px_pair_uses_detached_local_teacher() -> None:
    full = torch.tensor(
        [[0.50, 0.50, 0.025, 0.025]],
        requires_grad=True,
    )
    local = torch.tensor(
        [[0.50, 0.50, 0.050, 0.050]],
        requires_grad=True,
    )
    target = torch.tensor([[0.50, 0.50, 0.025, 0.025]])

    result = compute_tascv_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=target,
        pair_crops=_crop(1),
        image_hw=(640, 640),
    )
    result.loss.backward()

    assert TASCV_TINY_BOUNDARY_PX == 16.0
    assert result.matched_pair_count == 1
    assert result.auxiliary_tiny_pair_count == 1
    assert result.excluded_non_tiny_pair_count == 0
    assert full.grad is not None and full.grad.abs().sum() > 0
    assert local.grad is None or torch.equal(
        local.grad,
        torch.zeros_like(local.grad),
    )


def test_non_tiny_pair_has_zero_auxiliary_contribution_and_gradient() -> None:
    full = torch.tensor(
        [[0.50, 0.50, 0.10, 0.10]],
        requires_grad=True,
    )
    local = torch.tensor(
        [[0.50, 0.50, 0.20, 0.20]],
        requires_grad=True,
    )
    target = torch.tensor([[0.50, 0.50, 0.10, 0.10]])

    result = compute_tascv_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=target,
        pair_crops=_crop(1),
        image_hw=(640, 640),
    )
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert result.matched_pair_count == 1
    assert result.auxiliary_tiny_pair_count == 0
    assert result.excluded_non_tiny_pair_count == 1
    assert full.grad is not None
    assert torch.equal(full.grad, torch.zeros_like(full.grad))
    assert local.grad is not None
    assert torch.equal(local.grad, torch.zeros_like(local.grad))


def test_mixed_batch_loss_equals_the_tiny_pair_alone() -> None:
    full = torch.tensor(
        [
            [0.50, 0.50, 0.030, 0.030],
            [0.50, 0.50, 0.100, 0.100],
        ],
        requires_grad=True,
    )
    local = torch.tensor(
        [
            [0.50, 0.50, 0.0416667, 0.0416667],
            [0.50, 0.50, 0.300, 0.300],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [0.50, 0.50, 0.020, 0.020],
            [0.50, 0.50, 0.100, 0.100],
        ]
    )

    mixed = compute_tascv_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=targets,
        pair_crops=_crop(2),
        image_hw=(640, 640),
    )
    tiny_only = compute_tascv_loss(
        full_pred_boxes=full[:1],
        local_pred_boxes=local[:1],
        full_gt_boxes=targets[:1],
        pair_crops=_crop(1),
        image_hw=(640, 640),
    )

    torch.testing.assert_close(mixed.loss, tiny_only.loss)
    assert mixed.auxiliary_tiny_pair_count == 1
    assert mixed.excluded_non_tiny_pair_count == 1
    mixed.loss.backward()
    assert full.grad is not None
    assert full.grad[0, 2].item() > 0
    assert torch.equal(full.grad[1], torch.zeros_like(full.grad[1]))
    assert local.grad is None or torch.equal(
        local.grad,
        torch.zeros_like(local.grad),
    )


def test_loss_matches_independent_centered_box_reference() -> None:
    result = compute_tascv_loss(
        full_pred_boxes=torch.tensor(
            [[0.5, 0.5, 0.04, 0.04]],
            requires_grad=True,
        ),
        local_pred_boxes=torch.tensor(
            [[0.5, 0.5, 1 / 30, 1 / 30]],
            requires_grad=True,
        ),
        full_gt_boxes=torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
        pair_crops=_crop(1),
        image_hw=(640, 640),
    )

    # The mapped teacher is a centered 0.02 square. L1 is 0.04 and the
    # nested-box GIoU equals IoU=0.25, so 0.04 + 1 - 0.25 = 0.79.
    torch.testing.assert_close(
        result.loss,
        torch.tensor(0.79),
        rtol=1e-5,
        atol=1e-6,
    )


def test_empty_and_tiny_empty_cases_return_differentiable_fp32_zero() -> None:
    for full, local, targets, crops in (
        (
            torch.empty((0, 4), requires_grad=True),
            torch.empty((0, 4), requires_grad=True),
            torch.empty((0, 4)),
            torch.empty((0, 4), dtype=torch.long),
        ),
        (
            torch.tensor(
                [[0.5, 0.5, 0.1, 0.1]],
                dtype=torch.float16,
                requires_grad=True,
            ),
            torch.tensor(
                [[0.5, 0.5, 0.2, 0.2]],
                dtype=torch.float16,
                requires_grad=True,
            ),
            torch.tensor(
                [[0.5, 0.5, 0.1, 0.1]],
                dtype=torch.float16,
            ),
            _crop(1),
        ),
    ):
        result = compute_tascv_loss(
            full_pred_boxes=full,
            local_pred_boxes=local,
            full_gt_boxes=targets,
            pair_crops=crops,
            image_hw=(640, 640),
        )
        result.loss.backward()

        assert result.loss.dtype == torch.float32
        assert torch.isfinite(result.loss)
        assert result.loss.item() == 0.0


def test_loss_requires_the_frozen_640_frame() -> None:
    with pytest.raises(ValueError, match="640"):
        compute_tascv_loss(
            full_pred_boxes=torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
            local_pred_boxes=torch.tensor([[0.5, 0.5, 0.04, 0.04]]),
            full_gt_boxes=torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
            pair_crops=_crop(1),
            image_hw=(1280, 1280),
        )


@pytest.mark.parametrize(
    ("target", "crop", "message"),
    (
        (
            [0.99, 0.50, 0.04, 0.04],
            [128, 128, 512, 512],
            "outside the full image",
        ),
        (
            [0.50, 0.50, 0.02, 0.02],
            [-1, 128, 383, 512],
            "outside the full image",
        ),
        (
            [0.20, 0.20, 0.10, 0.10],
            [128, 128, 512, 512],
            "not fully contained",
        ),
    ),
)
def test_paired_target_and_crop_geometry_fails_closed(
    target,
    crop,
    message,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        compute_tascv_loss(
            full_pred_boxes=torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
            local_pred_boxes=torch.tensor([[0.5, 0.5, 0.04, 0.04]]),
            full_gt_boxes=torch.tensor([target]),
            pair_crops=torch.tensor([crop]),
            image_hw=(640, 640),
        )


@pytest.mark.parametrize(
    "crop",
    (
        [128, 128, 511, 512],
        [128.5, 128, 512.5, 512],
    ),
)
def test_crop_v2_geometry_requires_integer_384_square(crop) -> None:
    with pytest.raises(RuntimeError, match="crop-v2"):
        compute_tascv_loss(
            full_pred_boxes=torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
            local_pred_boxes=torch.tensor([[0.5, 0.5, 0.04, 0.04]]),
            full_gt_boxes=torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
            pair_crops=torch.tensor([crop]),
            image_hw=(640, 640),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_crops_are_normalized_to_the_cuda_prediction_device() -> None:
    device = torch.device("cuda")
    result = compute_tascv_loss(
        full_pred_boxes=torch.tensor(
            [[0.5, 0.5, 0.03, 0.03]],
            device=device,
            requires_grad=True,
        ),
        local_pred_boxes=torch.tensor(
            [[0.5, 0.5, 0.04, 0.04]],
            device=device,
            requires_grad=True,
        ),
        full_gt_boxes=torch.tensor(
            [[0.5, 0.5, 0.02, 0.02]],
            device=device,
        ),
        pair_crops=_crop(1),
        image_hw=(640, 640),
    )

    assert result.loss.device == device
    assert result.loss.dtype == torch.float32
    assert result.tiny_teacher_advantage_sum.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prediction_and_target_devices_must_match() -> None:
    with pytest.raises(ValueError, match="same device"):
        compute_tascv_loss(
            full_pred_boxes=torch.tensor(
                [[0.5, 0.5, 0.03, 0.03]],
                device="cuda",
            ),
            local_pred_boxes=torch.tensor(
                [[0.5, 0.5, 0.04, 0.04]],
            ),
            full_gt_boxes=torch.tensor(
                [[0.5, 0.5, 0.02, 0.02]],
                device="cuda",
            ),
            pair_crops=_crop(1),
            image_hw=(640, 640),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("full", float("nan"), "non-finite full predictions"),
        ("local", float("inf"), "non-finite local predictions"),
        ("target", float("nan"), "non-finite full targets"),
        ("target_width", 0.0, "degenerate full targets"),
    ),
)
def test_invalid_geometry_fails_closed(field, value, message) -> None:
    full = torch.tensor([[0.5, 0.5, 0.02, 0.02]])
    local = torch.tensor([[0.5, 0.5, 0.04, 0.04]])
    target = torch.tensor([[0.5, 0.5, 0.02, 0.02]])
    if field == "full":
        full[0, 0] = value
    elif field == "local":
        local[0, 0] = value
    elif field == "target":
        target[0, 0] = value
    else:
        target[0, 2] = value

    with pytest.raises(RuntimeError, match=message):
        compute_tascv_loss(
            full_pred_boxes=full,
            local_pred_boxes=local,
            full_gt_boxes=target,
            pair_crops=_crop(1),
            image_hw=(640, 640),
        )


def test_tiny_teacher_advantage_diagnostics_are_detached() -> None:
    full = torch.tensor(
        [[0.5, 0.5, 0.04, 0.04]],
        requires_grad=True,
    )
    local = torch.tensor(
        [[0.5, 0.5, 0.0333333, 0.0333333]],
        requires_grad=True,
    )
    target = torch.tensor([[0.5, 0.5, 0.02, 0.02]])

    result = compute_tascv_loss(
        full_pred_boxes=full,
        local_pred_boxes=local,
        full_gt_boxes=target,
        pair_crops=_crop(1),
        image_hw=(640, 640),
    )

    assert result.tiny_teacher_advantage_sum.item() > 0
    assert result.tiny_teacher_win_count == 1
    assert result.tiny_teacher_advantage_sum.requires_grad is False

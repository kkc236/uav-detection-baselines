from __future__ import annotations

import torch

from src.vsf_rmr_stress import centered_scale_boxes, vertical_perspective_boxes


def test_centered_scale_keeps_center_object_and_scales_its_size():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.1]])

    transformed, keep = centered_scale_boxes(boxes, image_size=(640, 640), factor=1.25)

    assert keep.tolist() == [True]
    torch.testing.assert_close(transformed, torch.tensor([[0.5, 0.5, 0.25, 0.125]]))


def test_centered_scale_drops_box_when_less_than_half_area_remains():
    boxes = torch.tensor([[0.98, 0.5, 0.10, 0.10]])

    _, keep = centered_scale_boxes(boxes, image_size=(640, 640), factor=1.25)

    assert keep.tolist() == [False]


def test_vertical_perspective_is_deterministic_and_clipped():
    boxes = torch.tensor([[0.25, 0.2, 0.1, 0.1], [0.75, 0.8, 0.1, 0.1]])

    first, first_keep = vertical_perspective_boxes(boxes, image_size=(640, 640), coefficient=5e-4)
    second, second_keep = vertical_perspective_boxes(boxes, image_size=(640, 640), coefficient=5e-4)

    torch.testing.assert_close(first, second)
    assert torch.equal(first_keep, second_keep)
    assert torch.all((first >= 0) & (first <= 1))
    assert not torch.equal(first, boxes)


def test_zero_perspective_is_identity():
    boxes = torch.tensor([[0.25, 0.2, 0.1, 0.1], [0.75, 0.8, 0.1, 0.1]])

    transformed, keep = vertical_perspective_boxes(boxes, image_size=(640, 640), coefficient=0.0)

    torch.testing.assert_close(transformed, boxes)
    assert keep.tolist() == [True, True]


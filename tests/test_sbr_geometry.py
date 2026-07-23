import math

import numpy as np
import pytest

from src.sbr_geometry import (
    LetterboxTransform,
    Tile,
    inverse_letterbox_xyxy,
    non_overlapping_tiles,
    overlapping_tiles,
    tile_to_global_xyxy,
)


def test_overlapping_tiles_are_ordered_half_open_and_have_frozen_overlap():
    tiles = overlapping_tiles(101, 80)
    assert [(t.left, t.top, t.right, t.bottom) for t in tiles] == [
        (0, 0, 61, 48),
        (40, 0, 101, 48),
        (0, 32, 61, 80),
        (40, 32, 101, 80),
    ]
    assert [t.index for t in tiles] == [0, 1, 2, 3]
    assert all(t.right > t.left and t.bottom > t.top for t in tiles)


def test_overlapping_tiles_work_for_portrait_even_dimensions():
    tiles = overlapping_tiles(80, 100)
    assert [(t.left, t.top, t.right, t.bottom) for t in tiles] == [
        (0, 0, 48, 60),
        (32, 0, 80, 60),
        (0, 40, 48, 100),
        (32, 40, 80, 100),
    ]


def test_arm_f_partitions_axes_with_odd_remainder_and_covers_image_once():
    tiles = non_overlapping_tiles(101, 81)
    assert [(t.left, t.top, t.right, t.bottom) for t in tiles] == [
        (0, 0, 50, 40),
        (50, 0, 101, 40),
        (0, 40, 50, 81),
        (50, 40, 101, 81),
    ]
    area = sum((t.right - t.left) * (t.bottom - t.top) for t in tiles)
    assert area == 101 * 81


def test_letterbox_inverse_roundtrip_padding_and_gain():
    transform = LetterboxTransform.from_view(width=1000, height=500, imgsz=640)
    assert transform.new_shape == (640, 640)
    assert transform.imgsz == 640
    view_box = np.array([0.0, 0.0, 1000.0, 500.0])
    network_box = np.array(
        [
            view_box[0] * transform.gain_x + transform.pad_x,
            view_box[1] * transform.gain_y + transform.pad_y,
            view_box[2] * transform.gain_x + transform.pad_x,
            view_box[3] * transform.gain_y + transform.pad_y,
        ]
    )
    restored = inverse_letterbox_xyxy(network_box, transform)
    np.testing.assert_allclose(restored, view_box, atol=0.5)


def test_inverse_letterbox_can_expand_normalized_network_boxes():
    transform = LetterboxTransform.from_view(width=1000, height=500, imgsz=640)
    normalized = np.array([transform.pad_x / 640, transform.pad_y / 640, (640 - transform.pad_x) / 640, (640 - transform.pad_y) / 640])
    restored = inverse_letterbox_xyxy(normalized, transform, normalized=True)
    np.testing.assert_allclose(restored, [0.0, 0.0, 1000.0, 500.0], atol=0.5)


def test_tile_to_global_offsets_once_and_clips():
    tile = Tile(left=40, top=32, right=101, bottom=80, index=3)
    boxes = np.array([[0.0, 0.0, 61.0, 48.0], [55.0, -2.0, 70.0, 60.0]])
    mapped = tile_to_global_xyxy(boxes, tile, width=101, height=80)
    np.testing.assert_allclose(mapped[0], [40, 32, 101, 80])
    np.testing.assert_allclose(mapped[1], [95, 30, 101, 80])


@pytest.mark.parametrize(
    "fn,args",
    [
        (overlapping_tiles, (0, 20)),
        (overlapping_tiles, (20, 0)),
        (non_overlapping_tiles, (1, 20)),
        (non_overlapping_tiles, (20, 1)),
    ],
)
def test_invalid_dimensions_fail_closed(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_invalid_boxes_and_zero_gain_fail_closed():
    transform = LetterboxTransform(gain=(1.0, 1.0), pad=(0.0, 0.0))
    with pytest.raises(ValueError):
        inverse_letterbox_xyxy([[1.0, 2.0, 0.0, 3.0]], transform)
    with pytest.raises(ValueError):
        inverse_letterbox_xyxy([[0.0, 0.0, 1.0, 1.0]], LetterboxTransform(gain=0.0))

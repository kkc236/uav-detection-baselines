import numpy as np
import torch

from src.gcte_views import (
    build_frozen_view_geometry,
    build_local_to_global_homography,
)
from src.sbr_geometry import LetterboxTransform, overlapping_tiles


def _map(matrix: torch.Tensor, point):
    value = matrix @ torch.tensor([point[0], point[1], 1.0])
    return value[:2] / value[2]


def test_local_homography_maps_tile_corners_into_global_letterbox():
    width, height = 1000, 500
    tile = overlapping_tiles(width, height)[3]
    global_transform = LetterboxTransform.from_view(
        width=width,
        height=height,
        imgsz=640,
    )
    local_transform = LetterboxTransform.from_view(
        width=tile.width,
        height=tile.height,
        imgsz=640,
    )

    matrix = build_local_to_global_homography(
        tile=tile,
        global_transform=global_transform,
        local_transform=local_transform,
    )

    local_top_left = (
        local_transform.pad_x / 640.0,
        local_transform.pad_y / 640.0,
    )
    expected_top_left = torch.tensor(
        [
            (tile.left * global_transform.gain_x + global_transform.pad_x)
            / 640.0,
            (tile.top * global_transform.gain_y + global_transform.pad_y)
            / 640.0,
        ]
    )
    torch.testing.assert_close(
        _map(matrix, local_top_left),
        expected_top_left,
        atol=1e-6,
        rtol=0,
    )


def test_frozen_view_geometry_has_four_ordered_300_query_views():
    geometry = build_frozen_view_geometry(
        source_shapes=[(500, 1000), (720, 1280)],
        queries_per_view=300,
    )

    assert geometry.homography.shape == (2, 1200, 3, 3)
    assert geometry.crop_metadata.shape == (2, 1200, 6)
    assert geometry.view_index[0].tolist() == np.repeat(
        np.arange(4),
        300,
    ).tolist()
    assert geometry.valid_mask.all()


def test_geometry_round_trip_is_sub_micro_for_in_bounds_points():
    geometry = build_frozen_view_geometry(
        source_shapes=[(500, 1000)],
        queries_per_view=300,
    )
    point = torch.tensor([0.5, 0.5, 1.0])

    for index in (0, 300, 600, 900):
        matrix = geometry.homography[0, index]
        mapped = matrix @ point
        restored = torch.linalg.inv(matrix) @ mapped
        restored = restored[:2] / restored[2]
        torch.testing.assert_close(
            restored,
            point[:2],
            atol=1e-6,
            rtol=0,
        )


def test_crop_metadata_freezes_overlap_ratio_and_resize_factors():
    geometry = build_frozen_view_geometry(
        source_shapes=[(500, 1000)],
        queries_per_view=300,
    )

    first = geometry.crop_metadata[0, 0]
    last = geometry.crop_metadata[0, 900]
    torch.testing.assert_close(first[:4], torch.tensor([0.0, 0.0, 0.6, 0.6]))
    torch.testing.assert_close(last[:4], torch.tensor([0.4, 0.4, 0.6, 0.6]))
    assert bool((first[4:] > 0).all())
    assert bool((last[4:] > 0).all())

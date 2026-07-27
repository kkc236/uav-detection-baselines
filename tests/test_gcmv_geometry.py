import math

import pytest
import torch

from src.gcmv_geometry import PLECGeometry, build_plec_geometry
from src.sbr_geometry import LetterboxTransform, Tile, overlapping_tiles


def test_public_geometry_contract_is_importable():
    assert PLECGeometry is not None
    assert callable(build_plec_geometry)
    assert LetterboxTransform is not None
    assert Tile is not None


def identity_geometry(*, height: int = 4, width: int = 6) -> PLECGeometry:
    transform = LetterboxTransform(
        source_width=width,
        source_height=height,
        network_shape=(height, width),
        gain=(1.0, 1.0),
        pad=(0.0, 0.0),
        resized_width=width,
        resized_height=height,
    )
    views = tuple(Tile(0, 0, width, height, index) for index in range(4))
    return build_plec_geometry(
        source_shapes=[(height, width)],
        tiles=[views],
        global_transforms=[transform],
        local_transforms=[[transform] * 4],
        global_feature_shape=(height, width),
        local_feature_shape=(height, width),
    )


def test_identity_transform_places_center_phase_on_same_lattice():
    height, width = 4, 6
    geometry = identity_geometry(height=height, width=width)

    center = geometry.sample_grid[0, 0, 4]
    expected_x = 2.0 * (torch.arange(width, dtype=center.dtype) + 0.5) / width - 1.0
    expected_y = 2.0 * (torch.arange(height, dtype=center.dtype) + 0.5) / height - 1.0
    torch.testing.assert_close(center[..., 0], expected_x.expand(height, width))
    torch.testing.assert_close(center[..., 1], expected_y[:, None].expand(height, width))
    torch.testing.assert_close(
        geometry.subcell_offset[0, 0, 4],
        torch.zeros_like(geometry.subcell_offset[0, 0, 4]),
    )


def test_phase_order_is_row_major_and_spans_one_global_cell():
    height, width = 4, 6
    geometry = identity_geometry(height=height, width=width)
    center = geometry.sample_grid[0, 0, 4]
    displacement = geometry.sample_grid[0, 0] - center.unsqueeze(0)
    phase = torch.tensor(
        [
            (-1 / 3, -1 / 3),
            (0, -1 / 3),
            (1 / 3, -1 / 3),
            (-1 / 3, 0),
            (0, 0),
            (1 / 3, 0),
            (-1 / 3, 1 / 3),
            (0, 1 / 3),
            (1 / 3, 1 / 3),
        ],
        dtype=displacement.dtype,
    )
    expected = phase.clone()
    expected[:, 0] *= 2.0 / width
    expected[:, 1] *= 2.0 / height
    expected = expected[:, None, None, :].expand_as(displacement)

    torch.testing.assert_close(displacement, expected)


def test_non_integer_magnification_is_preserved():
    height, width = 120, 180
    tile_views = tuple(Tile(0, 0, width, height, index) for index in range(4))
    global_transform = LetterboxTransform(
        source_width=width,
        source_height=height,
        network_shape=(80, 100),
        gain=(0.5, 0.5),
        pad=(5.0, 10.0),
    )
    local_transform = LetterboxTransform(
        source_width=width,
        source_height=height,
        network_shape=(90, 150),
        gain=(0.75, 0.75),
        pad=(2.0, 3.0),
    )

    geometry = build_plec_geometry(
        source_shapes=[(height, width)],
        tiles=[tile_views],
        global_transforms=[global_transform],
        local_transforms=[[local_transform] * 4],
        global_feature_shape=(7, 11),
        local_feature_shape=(13, 17),
    )

    expected_x = 100 / 11 / 0.5 * 0.75 * 17 / 150
    expected_y = 80 / 7 / 0.5 * 0.75 * 13 / 90
    torch.testing.assert_close(
        geometry.magnification[0, 0, :, 0, 0],
        torch.tensor([expected_x, expected_y]),
        rtol=0.0,
        atol=1e-6,
    )
    assert not math.isclose(expected_x, round(expected_x))
    assert not math.isclose(expected_y, round(expected_y))


def test_60_percent_tiles_produce_expected_coverage_and_edge_distance():
    height = width = 100
    image_tiles = overlapping_tiles(width, height)
    global_transform = LetterboxTransform(
        source_width=width,
        source_height=height,
        network_shape=(height, width),
        gain=1.0,
        pad=0.0,
    )
    local_transforms = [
        LetterboxTransform(
            source_width=tile.width,
            source_height=tile.height,
            network_shape=(tile.height, tile.width),
            gain=1.0,
            pad=0.0,
        )
        for tile in image_tiles
    ]

    geometry = build_plec_geometry(
        source_shapes=[(height, width)],
        tiles=[image_tiles],
        global_transforms=[global_transform],
        local_transforms=[local_transforms],
        global_feature_shape=(10, 10),
        local_feature_shape=(6, 6),
    )
    valid_count = geometry.center_valid.sum(dim=1)

    assert valid_count[0, 0, 0, 0].item() == 1
    assert valid_count[0, 0, 0, 4].item() == 2
    assert valid_count[0, 0, 4, 0].item() == 2
    assert valid_count[0, 0, 4, 4].item() == 4
    assert geometry.edge_distance[0, 0, 0, 2, 2] > geometry.edge_distance[0, 0, 0, 0, 0]
    assert torch.all(geometry.edge_distance >= 0)
    assert torch.all(geometry.edge_distance <= 1)
    assert torch.count_nonzero(
        geometry.edge_distance.masked_select(~geometry.center_valid)
    ).item() == 0


def test_geometry_to_moves_only_floating_tensors_and_keeps_metadata():
    geometry = identity_geometry()

    moved = geometry.to(device="cpu", dtype=torch.float64)

    for value in (
        moved.sample_grid,
        moved.subcell_offset,
        moved.magnification,
        moved.edge_distance,
    ):
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
        assert not value.requires_grad
    assert moved.sample_valid.dtype == torch.bool
    assert moved.center_valid.dtype == torch.bool
    assert moved.local_feature_shape == geometry.local_feature_shape
    assert moved.global_feature_shape == geometry.global_feature_shape


def valid_geometry_arguments():
    height, width = 12, 20
    image_tiles = tuple(Tile(0, 0, width, height, index) for index in range(4))
    transform = LetterboxTransform(
        source_width=width,
        source_height=height,
        network_shape=(height, width),
        gain=1.0,
        pad=0.0,
    )
    return {
        "source_shapes": [(height, width)],
        "tiles": [image_tiles],
        "global_transforms": [transform],
        "local_transforms": [[transform] * 4],
        "global_feature_shape": (3, 5),
        "local_feature_shape": (3, 5),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda args: args.update(source_shapes=[]), "batch"),
        (lambda args: args.update(tiles=[args["tiles"][0][:3]]), "four"),
        (
            lambda args: args.update(
                tiles=[
                    tuple(
                        Tile(tile.left, tile.top, tile.right, tile.bottom, 0)
                        for tile in args["tiles"][0]
                    )
                ]
            ),
            "order",
        ),
        (lambda args: args.update(global_feature_shape=(0, 5)), "feature"),
        (lambda args: args.update(source_shapes=[(0, 20)]), "source"),
        (lambda args: args.update(dtype=torch.int64), "floating"),
    ],
)
def test_invalid_geometry_contract_fails_closed(mutation, message):
    arguments = valid_geometry_arguments()
    mutation(arguments)

    with pytest.raises((TypeError, ValueError), match=message):
        build_plec_geometry(**arguments)


def test_tile_and_transform_metadata_must_match_source_frame():
    arguments = valid_geometry_arguments()
    arguments["tiles"] = [
        (
            Tile(0, 0, 21, 12, 0),
            *arguments["tiles"][0][1:],
        )
    ]
    with pytest.raises(ValueError, match="bounds"):
        build_plec_geometry(**arguments)

    arguments = valid_geometry_arguments()
    wrong = LetterboxTransform(
        source_width=19,
        source_height=12,
        network_shape=(12, 20),
        gain=1.0,
        pad=0.0,
    )
    arguments["global_transforms"] = [wrong]
    with pytest.raises(ValueError, match="source"):
        build_plec_geometry(**arguments)


def test_geometry_tensors_are_finite_and_non_trainable():
    geometry = identity_geometry()

    for value in (
        geometry.sample_grid,
        geometry.subcell_offset,
        geometry.magnification,
        geometry.edge_distance,
    ):
        assert torch.isfinite(value).all()
        assert not value.requires_grad


def matrix_geometry(matrix: torch.Tensor) -> PLECGeometry:
    arguments = valid_geometry_arguments()
    arguments["global_to_source"] = [matrix]
    return build_plec_geometry(**arguments)


def test_recorded_identity_affine_matches_letterbox_identity():
    baseline = build_plec_geometry(**valid_geometry_arguments())
    recorded = matrix_geometry(torch.eye(3))

    torch.testing.assert_close(recorded.sample_grid, baseline.sample_grid)
    torch.testing.assert_close(
        recorded.magnification,
        baseline.magnification,
    )


def test_recorded_scale_translation_maps_global_centers_to_source():
    matrix = torch.tensor(
        [[0.5, 0.0, 1.0], [0.0, 0.5, 2.0], [0.0, 0.0, 1.0]]
    )
    geometry = matrix_geometry(matrix)
    center = geometry.sample_grid[0, 0, 4]
    # Global feature (3x5) is sampled on a 12x20 network frame.
    global_x = (0.5 * 20 / 5)
    global_y = (0.5 * 12 / 3)
    source_x = 0.5 * global_x + 1.0
    source_y = 0.5 * global_y + 2.0
    expected_x = 2.0 * source_x / 20 - 1.0
    expected_y = 2.0 * source_y / 12 - 1.0

    torch.testing.assert_close(center[0, 0, 0], torch.tensor(expected_x))
    torch.testing.assert_close(center[0, 0, 1], torch.tensor(expected_y))


def test_recorded_horizontal_flip_reverses_source_lattice():
    matrix = torch.tensor(
        [[-1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    geometry = matrix_geometry(matrix)
    center = geometry.sample_grid[0, 0, 4]

    assert center[0, 0, 0] > center[0, -1, 0]
    torch.testing.assert_close(
        geometry.magnification[0, 0, :, 0, 0],
        torch.ones(2),
    )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (
            torch.tensor(
                [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "axis-aligned",
        ),
        (
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "singular",
        ),
        (
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, 0.0, 1.0]]
            ),
            "affine",
        ),
    ],
)
def test_recorded_global_transform_fails_closed(matrix, message):
    with pytest.raises(ValueError, match=message):
        matrix_geometry(matrix)

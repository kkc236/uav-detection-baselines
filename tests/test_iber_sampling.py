from __future__ import annotations

import ast
import inspect
from collections.abc import Callable

import pytest
import torch

import src.iber_sampling as iber_sampling
from src.iber_sampling import (
    rgb_normal_radii,
    sample_f3_boundary_evidence,
    sample_rgb_boundary_evidence,
)


def _coordinate_field(axis: str, *, channels: int, size: int = 64) -> torch.Tensor:
    coordinates = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    if axis == "x":
        field = coordinates.view(1, 1, 1, size).expand(1, channels, size, size)
    else:
        field = coordinates.view(1, 1, size, 1).expand(1, channels, size, size)
    return field.clone()


def _repeat_per_channel(
    values: tuple[float, float, float, float], channels: int
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(4, 1).expand(4, channels)


def test_boundary_grid_uses_exact_tangential_positions_in_ltrb_order() -> None:
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.25]]])
    normal_positions = torch.tensor([[[-1 / 64, 0.0, 1 / 64]]])

    grid = iber_sampling._boundary_grid_fp32(boxes, normal_positions)
    normalized = grid.add(1).div(2)

    assert normalized.shape == (1, 1, 4, 3, 3, 2)
    horizontal = torch.tensor((0.375, 0.500, 0.625)).view(3, 1).expand(3, 3)
    vertical = torch.tensor((0.4375, 0.5000, 0.5625)).view(3, 1).expand(3, 3)
    torch.testing.assert_close(normalized[0, 0, 0, :, :, 1], vertical, rtol=0, atol=0)
    torch.testing.assert_close(
        normalized[0, 0, 1, :, :, 0], horizontal, rtol=0, atol=0
    )
    torch.testing.assert_close(normalized[0, 0, 2, :, :, 1], vertical, rtol=0, atol=0)
    torch.testing.assert_close(
        normalized[0, 0, 3, :, :, 0], horizontal, rtol=0, atol=0
    )

    ordered_edge_coordinates = torch.stack(
        (
            normalized[0, 0, 0, :, 1, 0],
            normalized[0, 0, 1, :, 1, 1],
            normalized[0, 0, 2, :, 1, 0],
            normalized[0, 0, 3, :, 1, 1],
        )
    )
    expected_ltrb = torch.tensor((0.25, 0.375, 0.75, 0.625)).view(4, 1)
    torch.testing.assert_close(
        ordered_edge_coordinates, expected_ltrb.expand(4, 3), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("minimum", "expected_near", "expected_far"),
    [
        pytest.param(0.01, 1 / 256, 2 / 256, id="lower-clamp"),
        pytest.param(0.10, 0.08 * 0.10, 0.20 * 0.10, id="interior"),
        pytest.param(0.50, 4 / 256, 8 / 256, id="upper-clamp"),
    ],
)
def test_rgb_normal_radii_use_exact_clipped_formulas_at_non_640_size(
    minimum: float, expected_near: float, expected_far: float
) -> None:
    boxes = torch.tensor(
        [[[0.5, 0.5, minimum, minimum]]], dtype=torch.float64
    )

    near, far = rgb_normal_radii(boxes, image_size=256)

    assert near.dtype is torch.float32
    assert far.dtype is torch.float32
    torch.testing.assert_close(
        near, torch.tensor([[expected_near]], dtype=torch.float32), rtol=0, atol=1e-8
    )
    torch.testing.assert_close(
        far, torch.tensor([[expected_far]], dtype=torch.float32), rtol=0, atol=1e-8
    )


def test_rgb_shape_and_left_edge_signed_contrast() -> None:
    images = torch.zeros(1, 3, 640, 640)
    images[..., 320:] = 1
    boxes = torch.tensor([[[0.625, 0.5, 0.25, 0.25]]])

    evidence = sample_rgb_boundary_evidence(images, boxes, image_size=640)

    assert evidence.shape == (1, 1, 4, 15)
    assert torch.isfinite(evidence).all()
    left_near_signed = evidence[0, 0, 0, 3:6]
    assert torch.all(left_near_signed > 0.9)


@pytest.mark.parametrize(
    ("axis", "positive_edge", "negative_edge", "orthogonal_edges"),
    [
        ("x", 0, 2, (1, 3)),
        ("y", 1, 3, (0, 2)),
    ],
)
@pytest.mark.parametrize("modality", ["rgb", "f3"])
def test_coordinate_fields_verify_every_edge_normal_orientation(
    axis: str,
    positive_edge: int,
    negative_edge: int,
    orthogonal_edges: tuple[int, int],
    modality: str,
) -> None:
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    if modality == "rgb":
        values = _coordinate_field(axis, channels=3)
        evidence = sample_rgb_boundary_evidence(values, boxes, image_size=640)
        signed_contrasts = (
            evidence[0, 0, :, 3:6].mean(dim=-1),
            evidence[0, 0, :, 9:12].mean(dim=-1),
        )
    else:
        values = _coordinate_field(axis, channels=32)
        evidence = sample_f3_boundary_evidence(values, boxes, image_size=640)
        signed_contrasts = (evidence[0, 0, :, 32:64].mean(dim=-1),)

    for signed in signed_contrasts:
        assert signed[positive_edge] > 0
        assert signed[negative_edge] < 0
        assert signed[orthogonal_edges[0]].abs() < 1e-6
        assert signed[orthogonal_edges[1]].abs() < 1e-6


@pytest.mark.parametrize(
    ("axis", "expected_edge", "expected_near", "expected_far", "negative_edge"),
    [
        pytest.param(
            "x",
            (0.25, 0.50, 0.75, 0.50),
            (1 / 32, 0.0, -1 / 32, 0.0),
            (1 / 16, 0.0, -1 / 16, 0.0),
            2,
            id="left-right",
        ),
        pytest.param(
            "y",
            (0.50, 0.375, 0.50, 0.625),
            (0.0, 1 / 32, 0.0, -1 / 32),
            (0.0, 1 / 16, 0.0, -1 / 16),
            3,
            id="top-bottom",
        ),
    ],
)
def test_rgb_evidence_has_exact_edge_signed_and_absolute_slice_layout(
    axis: str,
    expected_edge: tuple[float, float, float, float],
    expected_near: tuple[float, float, float, float],
    expected_far: tuple[float, float, float, float],
    negative_edge: int,
) -> None:
    images = _coordinate_field(axis, channels=3)
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.25]]])

    evidence = sample_rgb_boundary_evidence(images, boxes, image_size=256)[0, 0]

    edge = evidence[:, 0:3]
    near_signed = evidence[:, 3:6]
    near_absolute = evidence[:, 6:9]
    far_signed = evidence[:, 9:12]
    far_absolute = evidence[:, 12:15]
    torch.testing.assert_close(
        edge, _repeat_per_channel(expected_edge, 3), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        near_signed, _repeat_per_channel(expected_near, 3), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        far_signed, _repeat_per_channel(expected_far, 3), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(near_absolute, near_signed.abs(), rtol=0, atol=0)
    torch.testing.assert_close(far_absolute, far_signed.abs(), rtol=0, atol=0)
    assert torch.all(near_signed[negative_edge] < 0)
    assert torch.all(far_signed[negative_edge] < 0)
    assert torch.all(near_absolute[negative_edge] > 0)
    assert torch.all(far_absolute[negative_edge] > 0)


@pytest.mark.parametrize(
    ("axis", "expected_edge", "expected_signed", "negative_edge"),
    [
        pytest.param(
            "x",
            (0.25, 0.50, 0.75, 0.50),
            (1 / 32, 0.0, -1 / 32, 0.0),
            2,
            id="left-right",
        ),
        pytest.param(
            "y",
            (0.50, 0.375, 0.50, 0.625),
            (0.0, 1 / 32, 0.0, -1 / 32),
            3,
            id="top-bottom",
        ),
    ],
)
def test_f3_evidence_has_exact_edge_signed_and_absolute_slice_layout(
    axis: str,
    expected_edge: tuple[float, float, float, float],
    expected_signed: tuple[float, float, float, float],
    negative_edge: int,
) -> None:
    features = _coordinate_field(axis, channels=32)
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.25]]])

    evidence = sample_f3_boundary_evidence(features, boxes, image_size=256)[0, 0]

    edge = evidence[:, 0:32]
    signed = evidence[:, 32:64]
    absolute = evidence[:, 64:96]
    torch.testing.assert_close(
        edge, _repeat_per_channel(expected_edge, 32), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        signed, _repeat_per_channel(expected_signed, 32), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(absolute, signed.abs(), rtol=0, atol=0)
    assert torch.all(signed[negative_edge] < 0)
    assert torch.all(absolute[negative_edge] > 0)


@pytest.mark.parametrize(
    ("minimum", "expected_radius"),
    [
        pytest.param(0.01, 1 / 256, id="lower-clamp"),
        pytest.param(0.10, 0.08 * 0.10, id="interior"),
        pytest.param(0.50, 4 / 256, id="upper-clamp"),
    ],
)
def test_f3_public_sampler_uses_image_size_for_exact_radius_regimes(
    minimum: float, expected_radius: float
) -> None:
    features = _coordinate_field("x", channels=32)
    boxes = torch.tensor([[[0.5, 0.5, minimum, minimum]]])

    evidence = sample_f3_boundary_evidence(features, boxes, image_size=256)

    left_signed_contrast = evidence[0, 0, 0, 32:64]
    torch.testing.assert_close(
        left_signed_contrast,
        torch.full_like(left_signed_contrast, 2 * expected_radius),
        rtol=0,
        atol=1e-6,
    )


def test_f3_shape_and_constant_features_have_zero_contrasts() -> None:
    features = torch.full((2, 32, 20, 24), 3.25)
    boxes = torch.tensor(
        [
            [[0.3, 0.4, 0.2, 0.1], [0.7, 0.6, 0.3, 0.4]],
            [[0.5, 0.5, 0.4, 0.2], [0.2, 0.8, 0.1, 0.1]],
        ]
    )

    evidence = sample_f3_boundary_evidence(features, boxes, image_size=640)

    assert evidence.shape == (2, 2, 4, 96)
    torch.testing.assert_close(
        evidence[..., 32:], torch.zeros_like(evidence[..., 32:]), rtol=0, atol=0
    )


def test_f3_gradients_reach_features_but_not_boxes() -> None:
    features = torch.randn(1, 32, 16, 16, requires_grad=True)
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]], requires_grad=True)

    sample_f3_boundary_evidence(features, boxes, image_size=640).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert boxes.grad is None


def test_rgb_gradients_reach_images_but_not_boxes() -> None:
    images = torch.randn(1, 3, 16, 16, requires_grad=True)
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]], requires_grad=True)

    sample_rgb_boundary_evidence(images, boxes, image_size=640).sum().backward()

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert boxes.grad is None


def test_out_of_image_grid_reaches_real_border_sampling_without_preclamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = torch.zeros(1, 3, 16, 16)
    images[..., 0] = 7
    boxes = torch.tensor([[[-0.05, 0.5, 0.02, 0.5]]], dtype=torch.float64)
    captured_grids: list[torch.Tensor] = []
    real_grid_sample = iber_sampling.F.grid_sample

    def recording_grid_sample(
        input_tensor: torch.Tensor, grid: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        captured_grids.append(grid.detach().clone())
        return real_grid_sample(input_tensor, grid, **kwargs)

    monkeypatch.setattr(iber_sampling.F, "grid_sample", recording_grid_sample)

    evidence = sample_rgb_boundary_evidence(images, boxes, image_size=640)

    assert len(captured_grids) == 1
    assert captured_grids[0].dtype is torch.float32
    assert captured_grids[0][..., 0].max() < -1
    torch.testing.assert_close(
        evidence[..., :3], torch.full_like(evidence[..., :3], 7), rtol=0, atol=0
    )
    torch.testing.assert_close(
        evidence[..., 3:], torch.zeros_like(evidence[..., 3:]), rtol=0, atol=0
    )


def test_tiny_border_and_mildly_out_of_image_boxes_are_finite_and_reproducible() -> None:
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(1, 3, 16, 16, generator=generator)
    features = torch.randn(1, 32, 8, 8, generator=generator)
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1e-12, 1e-12],
                [1.0, 1.0, 1e-12, 1e-12],
                [1.05, -0.05, 0.1, 0.1],
            ]
        ]
    )

    rgb_first = sample_rgb_boundary_evidence(images, boxes, image_size=640)
    rgb_second = sample_rgb_boundary_evidence(images, boxes, image_size=640)
    f3_first = sample_f3_boundary_evidence(features, boxes, image_size=640)
    f3_second = sample_f3_boundary_evidence(features, boxes, image_size=640)

    assert torch.isfinite(rgb_first).all()
    assert torch.isfinite(f3_first).all()
    torch.testing.assert_close(rgb_first, rgb_second, rtol=0, atol=0)
    torch.testing.assert_close(f3_first, f3_second, rtol=0, atol=0)


def test_real_sampling_uses_align_corners_false_pixel_geometry() -> None:
    x_indices = torch.arange(4, dtype=torch.float32)
    images = x_indices.view(1, 1, 1, 4).expand(1, 3, 4, 4).clone()
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])

    evidence = sample_rgb_boundary_evidence(images, boxes, image_size=640)

    torch.testing.assert_close(
        evidence[0, 0, 0, :3], torch.full((3,), 0.5), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: rgb_normal_radii(torch.ones(1, 4), image_size=640),
            ValueError,
            "boxes must have shape",
        ),
        (
            lambda: rgb_normal_radii(
                torch.ones(1, 1, 4, dtype=torch.int64), image_size=640
            ),
            TypeError,
            "boxes must be a floating-point tensor",
        ),
        (
            lambda: sample_rgb_boundary_evidence(
                torch.ones(1, 3, 8), torch.ones(1, 1, 4), image_size=640
            ),
            ValueError,
            "images must have shape",
        ),
        (
            lambda: sample_rgb_boundary_evidence(
                torch.ones(1, 4, 8, 8), torch.ones(1, 1, 4), image_size=640
            ),
            ValueError,
            "images must have shape",
        ),
        (
            lambda: sample_rgb_boundary_evidence(
                torch.ones(1, 3, 8, 8, dtype=torch.int64),
                torch.ones(1, 1, 4),
                image_size=640,
            ),
            TypeError,
            "images must be a floating-point tensor",
        ),
        (
            lambda: sample_f3_boundary_evidence(
                torch.ones(1, 31, 8, 8), torch.ones(1, 1, 4), image_size=640
            ),
            ValueError,
            "features must have shape",
        ),
        (
            lambda: sample_f3_boundary_evidence(
                torch.ones(1, 32, 8, 8, dtype=torch.int64),
                torch.ones(1, 1, 4),
                image_size=640,
            ),
            TypeError,
            "features must be a floating-point tensor",
        ),
        (
            lambda: sample_f3_boundary_evidence(
                torch.ones(2, 32, 8, 8), torch.ones(1, 1, 4), image_size=640
            ),
            ValueError,
            "same batch size",
        ),
        (
            lambda: sample_rgb_boundary_evidence(
                torch.ones(1, 3, 8, 8), torch.ones(1, 1, 4), image_size=0
            ),
            ValueError,
            "image_size must be positive",
        ),
        (
            lambda: sample_f3_boundary_evidence(
                torch.ones(1, 32, 8, 8), torch.ones(1, 1, 4), image_size=-1
            ),
            ValueError,
            "image_size must be positive",
        ),
    ],
)
def test_invalid_inputs_are_rejected_clearly(
    call: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        call()


def test_source_contains_only_sparse_functional_boundary_sampling() -> None:
    source = inspect.getsource(iber_sampling)
    lowered = source.lower()

    for forbidden in ("conv2d", "attention", "deform", "pyramid", "sobel"):
        assert forbidden not in lowered

    tree = ast.parse(source)
    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    grid_sample_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "grid_sample"
    ]
    assert grid_sample_calls
    for call in grid_sample_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords["mode"], ast.Constant)
        assert keywords["mode"].value == "bilinear"
        assert isinstance(keywords["padding_mode"], ast.Constant)
        assert keywords["padding_mode"].value == "border"
        assert isinstance(keywords["align_corners"], ast.Constant)
        assert keywords["align_corners"].value is False

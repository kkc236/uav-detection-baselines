from __future__ import annotations

import importlib
import importlib.util
import math

import pytest
import torch


PINNED_COMMIT = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"


@pytest.fixture(scope="module")
def fdr():
    spec = importlib.util.find_spec("src.fdr_head")
    assert spec is not None, "src.fdr_head must implement the pinned D-FINE FDR math"
    return importlib.import_module("src.fdr_head")


def test_official_source_is_bound_to_exact_revision(fdr) -> None:
    assert fdr.OFFICIAL_DFINE_COMMIT == PINNED_COMMIT
    assert fdr.OFFICIAL_DFINE_SOURCE_URL == (
        "https://github.com/Peterande/D-FINE/tree/"
        f"{PINNED_COMMIT}/src/zoo/dfine"
    )
    assert PINNED_COMMIT in fdr.OFFICIAL_DFINE_UTILS_URL
    assert PINNED_COMMIT in fdr.OFFICIAL_DFINE_DECODER_URL
    assert PINNED_COMMIT in fdr.OFFICIAL_DFINE_CRITERION_URL


def test_weighting_function_matches_reg32_scale4_golden(fdr) -> None:
    expected = torch.tensor(
        [
            -4.000000000000,
            -2.000000000000,
            -1.788130973629,
            -1.591224775369,
            -1.408224685281,
            -1.238148612163,
            -1.080083823052,
            -0.933182044932,
            -0.796654912379,
            -0.669769736709,
            -0.551845573915,
            -0.442249570307,
            -0.340393566226,
            -0.245730939616,
            -0.157753672517,
            -0.075989624725,
            0.000000000000,
            0.075989624725,
            0.157753672517,
            0.245730939616,
            0.340393566226,
            0.442249570307,
            0.551845573915,
            0.669769736709,
            0.796654912379,
            0.933182044932,
            1.080083823052,
            1.238148612163,
            1.408224685281,
            1.591224775369,
            1.788130973629,
            2.000000000000,
            4.000000000000,
        ],
        dtype=torch.float32,
    )

    actual = fdr.weighting_function(reg_max=32, up=0.5, reg_scale=4.0)

    assert actual.dtype == torch.float32
    assert actual.shape == (33,)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, -actual.flip(0), rtol=0, atol=0)
    assert torch.all(actual[1:] > actual[:-1])


def test_weighting_function_preserves_requested_dtype_and_rejects_invalid_bins(fdr) -> None:
    project = fdr.weighting_function(
        reg_max=32,
        up=torch.tensor([0.5], dtype=torch.float64),
        reg_scale=torch.tensor([4.0], dtype=torch.float64),
    )
    assert project.dtype == torch.float64
    with pytest.raises(ValueError, match="even integer"):
        fdr.weighting_function(reg_max=31, up=0.5, reg_scale=4.0)


def test_normalized_box_conversions_match_hand_calculated_values(fdr) -> None:
    boxes = torch.tensor(
        [[0.50, 0.50, 0.20, 0.40], [0.25, 0.75, 0.50, 0.50]],
        dtype=torch.float32,
    )
    expected_xyxy = torch.tensor(
        [[0.40, 0.30, 0.60, 0.70], [0.00, 0.50, 0.50, 1.00]],
        dtype=torch.float32,
    )

    actual_xyxy = fdr.cxcywh_to_xyxy(boxes)

    torch.testing.assert_close(actual_xyxy, expected_xyxy, rtol=0, atol=1e-7)
    torch.testing.assert_close(fdr.xyxy_to_cxcywh(actual_xyxy), boxes, rtol=0, atol=1e-7)


def test_distance2bbox_matches_normalized_ltrb_geometry(fdr) -> None:
    reference = torch.tensor([[0.50, 0.50, 0.20, 0.40]], dtype=torch.float32)
    distances = torch.tensor([[1.00, 0.50, 1.00, 0.50]], dtype=torch.float32)
    expected = torch.tensor([[0.50, 0.50, 0.30, 0.50]], dtype=torch.float32)

    actual = fdr.distance2bbox(reference, distances, reg_scale=4.0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_adjacent_bin_soft_labels_interpolate_and_saturate_boundaries(fdr) -> None:
    project = fdr.weighting_function(32, 0.5, 4.0)
    midpoint = (project[16] + project[17]) / 2
    values = torch.tensor([-5.0, -4.0, midpoint.item(), 0.0, 4.0, 5.0])

    left, weight_right, weight_left = fdr.adjacent_bin_soft_labels(
        values, reg_max=32, reg_scale=4.0, up=0.5
    )

    torch.testing.assert_close(left, torch.tensor([0.0, 0.0, 16.0, 16.0, 31.9, 31.9]))
    torch.testing.assert_close(weight_right, torch.tensor([0.0, 0.0, 0.5, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(weight_left, torch.tensor([1.0, 1.0, 0.5, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(weight_left + weight_right, torch.ones_like(weight_left))


def test_bbox2distance_encodes_values_that_decode_to_target_box(fdr) -> None:
    reference = torch.tensor([[0.50, 0.50, 0.20, 0.40]], dtype=torch.float32)
    target_xyxy = torch.tensor([[0.35, 0.25, 0.65, 0.75]], dtype=torch.float32)
    expected_distances = torch.tensor([1.0, 0.5, 1.0, 0.5], dtype=torch.float32)
    project = fdr.weighting_function(32, 0.5, 4.0)

    left, weight_right, weight_left = fdr.bbox2distance(
        reference, target_xyxy, reg_max=32, reg_scale=4.0, up=0.5
    )
    left_index = left.long()
    interpolated = (
        project[left_index] * weight_left + project[left_index + 1] * weight_right
    )

    torch.testing.assert_close(interpolated, expected_distances, rtol=0, atol=2e-6)
    decoded = fdr.distance2bbox(reference, interpolated.reshape(1, 4), reg_scale=4.0)
    torch.testing.assert_close(
        fdr.cxcywh_to_xyxy(decoded), target_xyxy, rtol=0, atol=2e-6
    )
    assert not left.requires_grad
    assert not weight_right.requires_grad
    assert not weight_left.requires_grad


def test_integral_matches_selected_nonuniform_bins(fdr) -> None:
    bins = 33
    chosen = (0, 16, 17, 32)
    logits = torch.full((1, 4 * bins), -80.0, dtype=torch.float32)
    for edge, index in enumerate(chosen):
        logits[0, edge * bins + index] = 80.0
    expected = torch.tensor(
        [[-4.0, 0.0, 0.075989624725, 4.0]], dtype=torch.float32
    )

    actual = fdr.Integral(reg_max=32, up=0.5, reg_scale=4.0)(logits)

    assert actual.shape == (1, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_zero_distribution_decodes_to_unchanged_reference_box(fdr) -> None:
    reference = torch.tensor(
        [[[0.25, 0.30, 0.10, 0.20], [0.70, 0.80, 0.30, 0.10]]],
        dtype=torch.float32,
    )
    logits = torch.zeros((1, 2, 4 * 33), dtype=torch.float32)

    distances = fdr.Integral()(logits)
    decoded = fdr.distance2bbox(reference, distances, reg_scale=4.0)

    torch.testing.assert_close(distances, torch.zeros_like(distances), rtol=0, atol=1e-7)
    torch.testing.assert_close(decoded, reference, rtol=0, atol=1e-7)


def test_integral_rejects_non_four_edge_shape(fdr) -> None:
    with pytest.raises(ValueError, match=r"4 \* \(reg_max \+ 1\)"):
        fdr.Integral()(torch.zeros(2, 131))


def test_fine_grained_localization_loss_matches_manual_cross_entropy(fdr) -> None:
    logits = torch.tensor(
        [[math.log(2.0), math.log(3.0), math.log(5.0)],
         [math.log(7.0), math.log(11.0), math.log(13.0)]],
        dtype=torch.float32,
        requires_grad=True,
    )
    left = torch.tensor([0.0, 1.0])
    weight_right = torch.tensor([0.25, 0.60])
    weight_left = 1.0 - weight_right
    quality = torch.tensor([0.5, 0.8])
    normalizer = 2.5
    row0 = 0.75 * math.log(10.0 / 2.0) + 0.25 * math.log(10.0 / 3.0)
    row1 = 0.40 * math.log(31.0 / 11.0) + 0.60 * math.log(31.0 / 13.0)
    expected = (0.5 * row0 + 0.8 * row1) / normalizer

    loss = fdr.fine_grained_localization_loss(
        logits,
        left,
        weight_right,
        weight_left,
        weight=quality,
        avg_factor=normalizer,
    )

    torch.testing.assert_close(loss, torch.tensor(expected), rtol=1e-6, atol=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_fgl_reduction_none_returns_one_loss_per_edge(fdr) -> None:
    logits = torch.zeros((4, 33), dtype=torch.float32)
    left = torch.tensor([0.0, 4.0, 16.0, 31.0])
    weight_right = torch.tensor([0.0, 0.25, 0.5, 1.0])
    weight_left = 1.0 - weight_right

    loss = fdr.unimodal_distribution_focal_loss(
        logits,
        left,
        weight_right,
        weight_left,
        reduction="none",
    )

    assert loss.shape == (4,)
    torch.testing.assert_close(loss, torch.full((4,), math.log(33.0)))


def test_float32_and_cpu_amp_paths_are_finite(fdr) -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4 * 33, dtype=torch.float32, requires_grad=True)
    integral = fdr.Integral()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        distances = integral(logits)
        loss = distances.square().mean()

    assert distances.dtype in (torch.float32, torch.bfloat16)
    assert torch.isfinite(distances).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

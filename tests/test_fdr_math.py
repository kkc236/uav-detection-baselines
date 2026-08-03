from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


PINNED_COMMIT = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
OFFICIAL_CLONE = Path(
    os.environ.get(
        "DFINE_OFFICIAL_CLONE",
        str(Path(os.environ.get("TEMP", "/tmp")) / "D-FINE-7fe2f888"),
    )
)
OFFICIAL_DFINE_DIR = OFFICIAL_CLONE / "src" / "zoo" / "dfine"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fdr_math():
    spec = importlib.util.find_spec("src.fdr_math")
    assert spec is not None, "src.fdr_math must own the pure FDR primitives"
    return importlib.import_module("src.fdr_math")


@pytest.fixture(scope="module")
def official_dfine():
    """Load pinned official modules without importing D-FINE's src package."""

    package_name = "_official_dfine_7fe2f888"
    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(OFFICIAL_DFINE_DIR)]
    sys.modules[package_name] = package
    loaded_names = [package_name]
    try:
        loaded = {}
        for module_name in ("box_ops", "dfine_utils"):
            qualified_name = f"{package_name}.{module_name}"
            spec = importlib.util.spec_from_file_location(
                qualified_name, OFFICIAL_DFINE_DIR / f"{module_name}.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified_name] = module
            loaded_names.append(qualified_name)
            spec.loader.exec_module(module)
            loaded[module_name] = module
        yield loaded
    finally:
        for name in reversed(loaded_names):
            sys.modules.pop(name, None)


def _mechanical_integral(logits: torch.Tensor, project: torch.Tensor, reg_max: int) -> torch.Tensor:
    shape = logits.shape
    probabilities = F.softmax(logits.reshape(-1, reg_max + 1), dim=1)
    values = F.linear(probabilities, project.to(logits.device)).reshape(-1, 4)
    return values.reshape(list(shape[:-1]) + [-1])


def _mechanical_fgl(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight_right: torch.Tensor,
    weight_left: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    reduction: str = "sum",
    avg_factor: float | None = None,
) -> torch.Tensor:
    dis_left = label.long()
    dis_right = dis_left + 1
    loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
        -1
    ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
    if weight is not None:
        loss = loss * weight.float()
    if avg_factor is not None:
        return loss.sum() / avg_factor
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def test_fixed_official_clone_is_exact_clean_revision() -> None:
    assert (OFFICIAL_DFINE_DIR / "dfine_utils.py").is_file()
    actual = subprocess.run(
        ["git", "-C", str(OFFICIAL_CLONE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(OFFICIAL_CLONE), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert actual == PINNED_COMMIT
    assert status == ""


def test_math_module_pins_exact_official_sources(fdr_math) -> None:
    assert fdr_math.OFFICIAL_DFINE_COMMIT == PINNED_COMMIT
    assert fdr_math.OFFICIAL_DFINE_SOURCE_URL == (
        "https://github.com/Peterande/D-FINE/tree/"
        f"{PINNED_COMMIT}/src/zoo/dfine"
    )
    assert PINNED_COMMIT in fdr_math.OFFICIAL_DFINE_UTILS_URL
    assert PINNED_COMMIT in fdr_math.OFFICIAL_DFINE_DECODER_URL
    assert PINNED_COMMIT in fdr_math.OFFICIAL_DFINE_CRITERION_URL


@pytest.mark.parametrize("deploy", [False, True], ids=["train", "deploy"])
def test_weighting_function_is_float32_exact_to_official(
    fdr_math, official_dfine, deploy: bool
) -> None:
    up = torch.tensor([0.5], dtype=torch.float32)
    reg_scale = torch.tensor([4.0], dtype=torch.float32)

    expected = official_dfine["dfine_utils"].weighting_function(
        32, up.clone(), reg_scale.clone(), deploy=deploy
    )
    actual = fdr_math.weighting_function(
        32, up.clone(), reg_scale.clone(), deploy=deploy
    )

    assert expected.dtype == actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_translate_gt_is_float32_exact_to_official(fdr_math, official_dfine) -> None:
    values = torch.tensor(
        [-5.0, -4.0, -2.0, -1.3, -0.01, 0.0, 0.01, 0.7, 2.0, 4.0, 5.0],
        dtype=torch.float32,
    )
    up = torch.tensor([0.5], dtype=torch.float32)
    reg_scale = torch.tensor([4.0], dtype=torch.float32)

    expected = official_dfine["dfine_utils"].translate_gt(
        values.clone(), 32, reg_scale.clone(), up.clone()
    )
    actual = fdr_math.translate_gt(
        values.clone(), 32, reg_scale.clone(), up.clone()
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)
    alias = fdr_math.adjacent_bin_soft_labels(
        values.clone(), 32, reg_scale.clone(), up.clone()
    )
    for alias_tensor, expected_tensor in zip(alias, expected):
        assert torch.equal(alias_tensor, expected_tensor)


@pytest.mark.parametrize("shape", [(7, 4), (2, 7, 4)], ids=["unbatched", "batched"])
def test_distance2bbox_is_float32_exact_to_official(
    fdr_math, official_dfine, shape: tuple[int, ...]
) -> None:
    generator = torch.Generator().manual_seed(41)
    points = torch.rand(shape, generator=generator, dtype=torch.float32)
    points[..., 2:] = points[..., 2:] * 0.4 + 0.05
    distances = torch.randn(shape, generator=generator, dtype=torch.float32)
    reg_scale = torch.tensor([4.0], dtype=torch.float32)

    expected = official_dfine["dfine_utils"].distance2bbox(
        points.clone(), distances.clone(), reg_scale.clone()
    )
    actual = fdr_math.distance2bbox(
        points.clone(), distances.clone(), reg_scale.clone()
    )

    assert torch.equal(actual, expected)


def test_bbox2distance_is_float32_exact_to_official(fdr_math, official_dfine) -> None:
    points = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.40],
            [0.20, 0.80, 0.10, 0.15],
            [0.75, 0.25, 0.30, 0.12],
        ],
        dtype=torch.float32,
    )
    bbox = torch.tensor(
        [
            [0.35, 0.25, 0.65, 0.75],
            [0.00, 0.60, 0.35, 1.00],
            [0.55, 0.05, 1.00, 0.45],
        ],
        dtype=torch.float32,
    )
    up = torch.tensor([0.5], dtype=torch.float32)
    reg_scale = torch.tensor([4.0], dtype=torch.float32)

    expected = official_dfine["dfine_utils"].bbox2distance(
        points.clone(), bbox.clone(), 32, reg_scale.clone(), up.clone()
    )
    actual = fdr_math.bbox2distance(
        points.clone(), bbox.clone(), 32, reg_scale.clone(), up.clone()
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)
        assert not actual_tensor.requires_grad


def test_normalized_box_conversions_are_exact_to_official_box_ops(
    fdr_math, official_dfine
) -> None:
    cxcywh = torch.tensor(
        [[0.50, 0.50, 0.20, 0.40], [0.25, 0.75, 0.50, 0.50]],
        dtype=torch.float32,
    )
    xyxy = official_dfine["box_ops"].box_cxcywh_to_xyxy(cxcywh)

    assert torch.equal(fdr_math.cxcywh_to_xyxy(cxcywh), xyxy)
    assert torch.equal(
        fdr_math.xyxy_to_cxcywh(xyxy),
        official_dfine["box_ops"].box_xyxy_to_cxcywh(xyxy),
    )


def test_integral_is_exact_to_mechanical_official_reference(fdr_math) -> None:
    generator = torch.Generator().manual_seed(73)
    logits = torch.randn((2, 5, 4 * 33), generator=generator, dtype=torch.float32)
    project = fdr_math.weighting_function(
        32, torch.tensor([0.5]), torch.tensor([4.0])
    )

    expected = _mechanical_integral(logits, project, 32)
    actual_default = fdr_math.Integral()(logits)
    actual_explicit = fdr_math.Integral()(logits, project)

    assert torch.equal(actual_default, expected)
    assert torch.equal(actual_explicit, expected)


def test_zero_distribution_preserves_reference_with_official_float32_rounding(
    fdr_math, official_dfine
) -> None:
    reference = torch.tensor(
        [[[0.25, 0.30, 0.10, 0.20], [0.70, 0.80, 0.30, 0.10]]],
        dtype=torch.float32,
    )
    logits = torch.zeros((1, 2, 4 * 33), dtype=torch.float32)

    distances = fdr_math.Integral()(logits)
    decoded = fdr_math.distance2bbox(reference, distances, torch.tensor([4.0]))
    expected_decoded = official_dfine["dfine_utils"].distance2bbox(
        reference, distances, torch.tensor([4.0])
    )

    assert torch.equal(distances, torch.zeros_like(distances))
    assert torch.equal(decoded, expected_decoded)
    torch.testing.assert_close(decoded, reference, rtol=0, atol=6e-8)


@pytest.mark.parametrize(
    ("weight", "reduction", "avg_factor"),
    [
        (None, "none", None),
        (None, "mean", None),
        (None, "sum", None),
        (torch.tensor([0.2, 0.4, 0.6, 0.8]), "sum", 2.5),
    ],
    ids=["none", "mean", "sum", "weighted-average"],
)
def test_fgl_is_exact_to_mechanical_official_reference(
    fdr_math,
    weight: torch.Tensor | None,
    reduction: str,
    avg_factor: float | None,
) -> None:
    generator = torch.Generator().manual_seed(97)
    pred = torch.randn((4, 33), generator=generator, dtype=torch.float32)
    label = torch.tensor([0.0, 5.0, 16.0, 31.0], dtype=torch.float32)
    weight_right = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float32)
    weight_left = 1.0 - weight_right

    expected = _mechanical_fgl(
        pred,
        label,
        weight_right,
        weight_left,
        weight=weight,
        reduction=reduction,
        avg_factor=avg_factor,
    )
    actual = fdr_math.fine_grained_localization_loss(
        pred,
        label,
        weight_right,
        weight_left,
        weight=weight,
        reduction=reduction,
        avg_factor=avg_factor,
    )
    official_name = fdr_math.unimodal_distribution_focal_loss(
        pred,
        label,
        weight_right,
        weight_left,
        weight=weight,
        reduction=reduction,
        avg_factor=avg_factor,
    )

    assert torch.equal(actual, expected)
    assert torch.equal(official_name, expected)


def test_float32_and_cpu_amp_paths_remain_finite_and_differentiable(fdr_math) -> None:
    generator = torch.Generator().manual_seed(101)
    logits = torch.randn(
        (2, 3, 4 * 33), generator=generator, dtype=torch.float32, requires_grad=True
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        distances = fdr_math.Integral()(logits)
        loss = distances.square().mean()

    assert torch.isfinite(distances).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_decoder_box_path_imports_math_without_duplicating_it() -> None:
    head_source = (REPOSITORY_ROOT / "src" / "fdr_head.py").read_text("utf-8")
    assert "from src.fdr_math import" in head_source
    assert "def weighting_function(" not in head_source
    assert "def bbox2distance(" not in head_source
    assert "def fine_grained_localization_loss(" not in head_source

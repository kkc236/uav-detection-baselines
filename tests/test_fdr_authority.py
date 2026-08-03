"""Authority and golden-reference contract for the pinned D-FINE FDR source."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch


PIN = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
REPOSITORY = "https://github.com/Peterande/D-FINE"
ROOT = Path(__file__).resolve().parents[1] / "third_party" / "dfine_7fe2f888"
AUTHORITY = ROOT / "AUTHORITY.json"
REFERENCE = ROOT / "reference_fdr.py"

EXPECTED_SOURCES = {
    "dfine_decoder.py": {
        "path": "src/zoo/dfine/dfine_decoder.py",
        "sha256": "7fe0e798493d18ea71a3c337f3116a48a0fd28b20b3dbba5d27e75d8bc6bac14",
    },
    "dfine_utils.py": {
        "path": "src/zoo/dfine/dfine_utils.py",
        "sha256": "d05209405e6c680620fcd847fa62c37a445b1a5838a55ba6c8b6f738aee576d8",
    },
    "dfine_criterion.py": {
        "path": "src/zoo/dfine/dfine_criterion.py",
        "sha256": "ca9938808c8ba59fa2e7aeb3754ddbc415efd51127f7ffdd83e0f68be3423908",
    },
    "dfine_hgnetv2.yml": {
        "path": "configs/dfine/include/dfine_hgnetv2.yml",
        "sha256": "29630acb451a8d34b0a5a04501c7619e226a8b5ca4b10cfd82c5ff11b7d996ce",
    },
}


def _load_authority() -> dict:
    assert AUTHORITY.is_file(), f"pinned authority is missing: {AUTHORITY}"
    return json.loads(AUTHORITY.read_text(encoding="utf-8"))


def _load_reference():
    assert REFERENCE.is_file(), f"test-only FDR reference is missing: {REFERENCE}"
    spec = importlib.util.spec_from_file_location("dfine_7fe2f888_reference", REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dfine_authority_is_commit_pinned_and_self_hashing():
    authority = _load_authority()
    assert authority["schema_version"] == 1
    assert authority["repository"] == REPOSITORY
    assert authority["commit"] == PIN
    assert authority["usage"] == "test-only golden reference"
    assert set(authority["sources"]) == set(EXPECTED_SOURCES)
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == authority[
        "vendored_reference_sha256"
    ]


def test_every_source_is_bound_to_the_fixed_commit_and_sha256():
    authority = _load_authority()
    for name, expected in EXPECTED_SOURCES.items():
        source = authority["sources"][name]
        assert source["repository_path"] == expected["path"]
        assert source["sha256"] == expected["sha256"]
        assert source["source_url"] == (
            f"{REPOSITORY}/blob/{PIN}/{expected['path']}"
        )


def test_license_and_authorship_are_recorded_truthfully():
    authority = _load_authority()
    license_record = authority["license"]
    assert license_record == {
        "spdx": "Apache-2.0",
        "repository_path": "LICENSE",
        "sha256": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
        "source_url": f"{REPOSITORY}/blob/{PIN}/LICENSE",
        "copyright": [
            "Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.",
            "Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR); "
            "Copyright (c) 2023 lyuwenyu. All Rights Reserved.",
        ],
    }
    text = REFERENCE.read_text(encoding="utf-8")
    for phrase in (
        "TEST-ONLY GOLDEN REFERENCE",
        "Apache-2.0",
        "The D-FINE Authors",
        "Modified from RT-DETR",
        PIN,
    ):
        assert phrase in text


def test_reference_exports_only_the_required_fdr_formulas():
    reference = _load_reference()
    assert reference.__all__ == (
        "Integral",
        "weighting_function",
        "translate_gt",
        "distance2bbox",
        "bbox2distance",
        "unimodal_distribution_focal_loss",
    )


def test_reference_weighting_integral_and_box_transform_are_operational():
    reference = _load_reference()
    up = torch.tensor([0.5], dtype=torch.float64)
    scale = torch.tensor([4.0], dtype=torch.float64)
    project = reference.weighting_function(32, up, scale)
    assert project.shape == (33,)
    torch.testing.assert_close(project, -project.flip(0), rtol=0, atol=0)
    assert project[0].item() == -4.0
    assert project[16].item() == 0.0
    assert project[-1].item() == 4.0

    logits = torch.full((1, 1, 4, 33), -100.0, dtype=torch.float64)
    logits[..., 16] = 100.0
    offsets = reference.Integral(32)(logits.reshape(1, 1, 132), project)
    torch.testing.assert_close(offsets, torch.zeros_like(offsets), rtol=0, atol=1e-80)

    points = torch.tensor([[[0.5, 0.5, 0.2, 0.1]]], dtype=torch.float64)
    boxes = reference.distance2bbox(points, offsets, scale)
    torch.testing.assert_close(boxes, points, rtol=0, atol=1e-15)


def test_reference_target_interpolation_and_fgl_are_operational():
    reference = _load_reference()
    up = torch.tensor([0.5], dtype=torch.float32)
    scale = torch.tensor([4.0], dtype=torch.float32)
    values = torch.tensor(
        [-9.0, -8.0, -1e-6, 0.0, 1e-6, 8.0, 9.0], dtype=torch.float32
    )
    indices, weight_right, weight_left = reference.translate_gt(values, 32, scale, up)
    assert torch.all((indices >= 0) & (indices < 32))
    torch.testing.assert_close(
        weight_left + weight_right,
        torch.ones_like(weight_left),
        rtol=0,
        atol=0,
    )

    points = torch.tensor([[0.5, 0.5, 0.2, 0.1]], dtype=torch.float32)
    target_xyxy = torch.tensor([[0.4, 0.45, 0.6, 0.55]], dtype=torch.float32)
    labels, wr, wl = reference.bbox2distance(points, target_xyxy, 32, scale, up)
    assert labels.shape == wr.shape == wl.shape == (4,)
    assert torch.isfinite(labels).all()
    torch.testing.assert_close(wl + wr, torch.ones_like(wl), rtol=0, atol=0)

    pred = torch.zeros((2, 33), dtype=torch.float64)
    label = torch.tensor([0.0, 1.0], dtype=torch.float64)
    wr = torch.tensor([0.25, 0.75], dtype=torch.float64)
    wl = 1.0 - wr
    sample_weight = torch.tensor([0.5, 1.0], dtype=torch.float64)
    loss = reference.unimodal_distribution_focal_loss(
        pred, label, wr, wl, weight=sample_weight, avg_factor=2.0
    )
    expected = torch.tensor(math.log(33.0) * 0.75, dtype=torch.float64)
    torch.testing.assert_close(loss, expected, rtol=1e-15, atol=0)

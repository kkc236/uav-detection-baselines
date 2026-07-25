from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.saded_stock_evaluation_protocol import (
    EXPECTED_DATASET_SIGNATURE,
    EXPECTED_IMAGE_LIST_SHA256,
    build_image_authority,
    frozen_route_contract,
    reject_forbidden,
)


def test_frozen_route_contract_matches_final_method() -> None:
    contract = frozen_route_contract()
    assert contract["tiny_effective_size"] == 16.0
    assert contract["large_effective_size"] == 96.0
    assert contract["match_iou_strictly_greater_than"] == 0.5
    assert contract["fragment_ios"] == 0.5
    assert contract["max_det"] == 300
    assert contract["views"] == ["full", "TL", "TR", "BL", "BR"]


def test_image_authority_binds_order_size_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"a")
    (root / "b.jpg").write_bytes(b"bb")
    authority = build_image_authority(root, ["b.jpg", "a.jpg"])
    assert authority["image_count"] == 2
    assert [row["image_id"] for row in authority["images"]] == [
        "b.jpg",
        "a.jpg",
    ]
    assert [row["size"] for row in authority["images"]] == [2, 1]
    assert all(len(row["sha256"]) == 64 for row in authority["images"])


def test_image_authority_rejects_escape_duplicate_and_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"a")
    with pytest.raises(ValueError, match="unique"):
        build_image_authority(root, ["a.jpg", "a.jpg"])
    with pytest.raises(ValueError, match="canonical"):
        build_image_authority(root, ["../a.jpg"])
    with pytest.raises(ValueError, match="missing"):
        build_image_authority(root, ["missing.jpg"])


def test_authority_constants_are_the_sealed_r0_dev_val_values() -> None:
    assert (
        EXPECTED_IMAGE_LIST_SHA256
        == "87C1B9FE8CD39CAF7F46494E7FE55DC4315573B64EE83A5B71778DBF55933B3A"
    )
    assert (
        EXPECTED_DATASET_SIGNATURE
        == "A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AE"
        "EF15EB1F0AE2A571228A"
    )


def test_forbidden_paths_are_rejected_recursively(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test-dev"):
        reject_forbidden({"nested": [str(tmp_path / "test-dev")]})
    reject_forbidden({"ordinary": json.dumps(["dev-val"])})

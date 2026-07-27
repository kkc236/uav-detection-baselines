from pathlib import Path

import pytest

from scripts.cache_gcqf_evidence import (
    EXPECTED_BASELINE_SHA256,
    build_parser,
    canonical_image_id,
)


def test_cache_cli_freezes_authorities_and_300_query_protocol():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "baseline.pt",
            "--data",
            "visdrone.yaml",
            "--split",
            "train",
            "--dataset-signature",
            "B" * 64,
            "--output",
            "cache",
        ]
    )

    assert args.expected_baseline_sha256 == EXPECTED_BASELINE_SHA256
    assert args.queries_per_view == 300
    assert args.views == 5
    assert args.amp is True
    assert args.batch == 1


def test_canonical_image_id_uses_dataset_relative_identity(tmp_path):
    root = tmp_path / "images"
    image = root / "val" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    assert canonical_image_id(image, root) == "val/a.jpg"


def test_canonical_image_id_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        canonical_image_id(
            tmp_path / "other" / "a.jpg",
            tmp_path / "images",
        )

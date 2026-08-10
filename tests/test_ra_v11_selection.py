from __future__ import annotations

from pathlib import Path

import pytest

from src.lpr_protocol import select_hashed_subset, subset_signature
import src.ra_v11_selection as selection


def _fixture_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, tuple[Path, ...]]:
    root = tmp_path / "VisDrone"
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (root / kind / split).mkdir(parents=True)

    train_count, val_count, screen_count, selection_count = 32, 5, 3, 10
    train = []
    for index in range(train_count):
        image = root / "images" / "train" / f"train-{index:03d}.jpg"
        image.write_bytes(f"train-image-{index}".encode("ascii"))
        (root / "labels" / "train" / f"train-{index:03d}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="ascii"
        )
        train.append(image)
    for index in range(val_count):
        (root / "images" / "val" / f"val-{index:03d}.jpg").write_bytes(
            f"val-image-{index}".encode("ascii")
        )
        (root / "labels" / "val" / f"val-{index:03d}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="ascii"
        )

    monkeypatch.setattr(selection, "TRAIN_IMAGE_COUNT", train_count)
    monkeypatch.setattr(selection, "VAL_IMAGE_COUNT", val_count)
    monkeypatch.setattr(selection, "SCREEN_IMAGE_COUNT", screen_count)
    monkeypatch.setattr(selection, "SELECTION_IMAGE_COUNT", selection_count)
    monkeypatch.setattr(
        selection, "SCREEN30_SELECTION_IMAGE_COUNT", selection_count
    )
    monkeypatch.setattr(selection, "EXPECTED_DATASET_SHA256", "D" * 64)
    monkeypatch.setattr(
        selection,
        "dataset_signature",
        lambda _root: {"file_count": 50, "sha256": "D" * 64},
    )

    screen = select_hashed_subset(train, root=root, fraction=0.10)
    monkeypatch.setattr(selection, "EXPECTED_SUBSET_SHA256", subset_signature(screen, root=root))
    remaining = [path for path in train if path not in set(screen)]
    selected = tuple(
        sorted(remaining, key=lambda path: selection._selection_rank(path, root))[:selection_count]
    )
    selected_set = set(selected)
    screen30_selected = tuple(
        sorted(
            [path for path in remaining if path not in selected_set],
            key=lambda path: selection._selection_rank(
                path, root, selection.SCREEN30_SELECTION_SALT
            ),
        )[:selection_count]
    )
    sizes = ((0.01, 0.01), (0.03, 0.03), (0.10, 0.10))
    for class_id, image in enumerate((*selected, *screen30_selected)):
        width, height = sizes[class_id % len(sizes)]
        (root / "labels" / "train" / f"{image.stem}.txt").write_text(
            f"{class_id % 10} 0.5 0.5 {width} {height}\n", encoding="ascii"
        )
    return root, selected, screen30_selected


def test_selection_is_deterministic_disjoint_complete_and_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, expected, expected_screen30 = _fixture_dataset(tmp_path, monkeypatch)
    output = tmp_path / "authority" / "selection.txt"
    screen30_output = tmp_path / "authority" / "screen30.txt"

    report = selection.build_ra_v11_selection_authority(
        root, output, screen30_output
    )

    assert selection.select_ra_v11_paths(root) == expected
    assert report["counts"] == {
        "train": 32,
        "screen647": 3,
        "remaining": 29,
        "selection": 10,
        "screen30_selection": 10,
        "official_val": 5,
        "duplicate_paths": 0,
        "duplicate_image_content": 0,
    }
    assert report["overlap"] == {
        "screen647_paths": 0,
        "screen10_screen30_paths": 0,
        "screen10_screen30_image_content": 0,
        "official_val_stems": 0,
        "official_val_image_content": 0,
    }
    assert report["selection"]["objects"] == 10
    assert all(report["selection"]["class_counts"].values())
    assert sum(report["selection"]["scale_counts"].values()) == 10
    assert set(report["selection"]["scale_counts"]) == {"tiny", "small", "regular"}
    assert len(report["selection"]["relative_path_sha256"]) == 64
    assert len(report["selection"]["image_manifest_sha256"]) == 64
    assert len(report["selection"]["label_manifest_sha256"]) == 64
    assert report["screen30_selection"]["objects"] == 10
    assert all(report["screen30_selection"]["class_counts"].values())
    assert output.read_text(encoding="utf-8").splitlines() == [
        str(path.resolve()) for path in expected
    ]
    assert screen30_output.read_text(encoding="utf-8").splitlines() == [
        str(path.resolve()) for path in expected_screen30
    ]
    with pytest.raises(FileExistsError, match="refusing to replace"):
        selection.build_ra_v11_selection_authority(root, output)


def test_selection_rejects_official_val_content_overlap_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, expected, _ = _fixture_dataset(tmp_path, monkeypatch)
    (root / "images" / "val" / "val-000.jpg").write_bytes(expected[0].read_bytes())
    output = tmp_path / "selection.txt"

    with pytest.raises(ValueError, match="official val by image content"):
        selection.build_ra_v11_selection_authority(root, output)
    assert not output.exists()


def test_selection_rejects_invalid_labels_and_missing_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, expected, _ = _fixture_dataset(tmp_path, monkeypatch)
    label = root / "labels" / "train" / f"{expected[-1].stem}.txt"
    label.write_text("10 0.5 0.5 0.1 0.1\n", encoding="ascii")

    with pytest.raises(ValueError, match="invalid class id"):
        selection.build_ra_v11_selection_authority(root, tmp_path / "invalid.txt")

    label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="ascii")
    with pytest.raises(ValueError, match="does not contain all classes"):
        selection.build_ra_v11_selection_authority(root, tmp_path / "missing.txt")


def test_selection_salt_and_scale_boundaries_are_frozen() -> None:
    assert selection.SELECTION_SALT == b"ra-glgm-v1.1-selection-v1\0"
    assert (
        selection.SCREEN30_SELECTION_SALT
        == b"ra-glgm-v1.1-screen30-selection-v1\0"
    )
    assert selection._scale_name(0.025, 0.025) == "small"
    assert selection._scale_name(0.05, 0.05) == "regular"
    just_below_tiny = (16.0 - 1e-6) / 640.0
    assert selection._scale_name(just_below_tiny, just_below_tiny) == "tiny"

"""Deterministic train-derived selection authority for RA-GLGM v1.1."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.lpr_protocol import (
    CATEGORY_NAMES,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    dataset_signature,
    file_sha256,
    select_hashed_subset,
    subset_signature,
)


TRAIN_IMAGE_COUNT = 6_471
SCREEN_IMAGE_COUNT = 647
SELECTION_IMAGE_COUNT = 548
SCREEN30_SELECTION_IMAGE_COUNT = 548
VAL_IMAGE_COUNT = 548
SELECTION_SALT = b"ra-glgm-v1.1-selection-v1\0"
SCREEN30_SELECTION_SALT = b"ra-glgm-v1.1-screen30-selection-v1\0"

_SCALE_NAMES = ("tiny", "small", "regular")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path escapes VisDrone root: {path}") from error


def _regular_files(directory: Path, suffix: str) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(f"authoritative directory is missing or unsafe: {directory}")
    paths = sorted(directory.glob(f"*{suffix}"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError(f"authoritative files must be regular non-symlinks: {directory}")
    return paths


def _validated_layout(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    root = dataset_root.resolve()
    if dataset_root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"authoritative VisDrone root is missing or unsafe: {root}")
    signature = dataset_signature(root)
    if signature.get("sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "VisDrone dataset SHA256 mismatch: "
            f"expected={EXPECTED_DATASET_SHA256}, actual={signature.get('sha256')}"
        )

    train_images = _regular_files(root / "images" / "train", ".jpg")
    val_images = _regular_files(root / "images" / "val", ".jpg")
    train_labels = _regular_files(root / "labels" / "train", ".txt")
    val_labels = _regular_files(root / "labels" / "val", ".txt")
    if len(train_images) != TRAIN_IMAGE_COUNT or len(val_images) != VAL_IMAGE_COUNT:
        raise ValueError(
            "VisDrone image count mismatch: "
            f"train={len(train_images)}, val={len(val_images)}"
        )
    if len(train_labels) != TRAIN_IMAGE_COUNT or len(val_labels) != VAL_IMAGE_COUNT:
        raise ValueError(
            "VisDrone label count mismatch: "
            f"train={len(train_labels)}, val={len(val_labels)}"
        )
    if {path.stem for path in train_images} != {path.stem for path in train_labels}:
        raise ValueError("VisDrone train image/label stems differ")
    if {path.stem for path in val_images} != {path.stem for path in val_labels}:
        raise ValueError("VisDrone val image/label stems differ")
    return train_images, val_images


def _selection_rank(
    path: Path, root: Path, salt: bytes = SELECTION_SALT
) -> tuple[bytes, str]:
    relative = _relative(path, root)
    return hashlib.sha256(salt + relative.encode("utf-8")).digest(), relative


def _select_from_train(
    root: Path, train_images: Sequence[Path]
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    screen = select_hashed_subset(train_images, root=root, fraction=0.10)
    if len(screen) != SCREEN_IMAGE_COUNT:
        raise ValueError(f"Screen subset count mismatch: {len(screen)}")
    screen_sha = subset_signature(screen, root=root)
    if screen_sha != EXPECTED_SUBSET_SHA256:
        raise ValueError(
            "Screen647 SHA256 mismatch: "
            f"expected={EXPECTED_SUBSET_SHA256}, actual={screen_sha}"
        )
    screen_set = {path.resolve() for path in screen}
    remaining = [path for path in train_images if path.resolve() not in screen_set]
    expected_remaining = TRAIN_IMAGE_COUNT - SCREEN_IMAGE_COUNT
    if len(remaining) != expected_remaining:
        raise ValueError(
            f"post-Screen candidate count mismatch: {len(remaining)} != {expected_remaining}"
        )
    selected = tuple(
        sorted(remaining, key=lambda path: _selection_rank(path, root))[
            :SELECTION_IMAGE_COUNT
        ]
    )
    if len(selected) != SELECTION_IMAGE_COUNT or len(
        {path.resolve() for path in selected}
    ) != len(selected):
        raise ValueError("selection set count or path uniqueness is invalid")
    if screen_set.intersection(path.resolve() for path in selected):
        raise ValueError("selection set overlaps frozen Screen647")
    selected_set = {path.resolve() for path in selected}
    screen30_candidates = [
        path for path in remaining if path.resolve() not in selected_set
    ]
    screen30_selected = tuple(
        sorted(
            screen30_candidates,
            key=lambda path: _selection_rank(path, root, SCREEN30_SELECTION_SALT),
        )[:SCREEN30_SELECTION_IMAGE_COUNT]
    )
    if len(screen30_selected) != SCREEN30_SELECTION_IMAGE_COUNT or len(
        {path.resolve() for path in screen30_selected}
    ) != len(screen30_selected):
        raise ValueError("Screen30 selection set count or path uniqueness is invalid")
    screen30_set = {path.resolve() for path in screen30_selected}
    if screen_set & screen30_set or selected_set & screen30_set:
        raise ValueError("Screen30 selection set overlaps an earlier experiment split")
    return selected, screen30_selected


def select_ra_v11_paths(dataset_root: str | Path) -> tuple[Path, ...]:
    """Select 548 train images after excluding the frozen Screen647 authority."""

    root = Path(dataset_root).resolve()
    train_images, _ = _validated_layout(root)
    selected, _ = _select_from_train(root, train_images)
    return selected


def select_ra_v11_screen30_paths(dataset_root: str | Path) -> tuple[Path, ...]:
    """Select the disjoint train-derived Screen30 development set."""

    root = Path(dataset_root).resolve()
    train_images, _ = _validated_layout(root)
    _, selected = _select_from_train(root, train_images)
    return selected


def _manifest_sha256(paths: Iterable[Path], root: Path) -> tuple[str, dict[Path, str]]:
    digest = hashlib.sha256()
    file_hashes: dict[Path, str] = {}
    for path in paths:
        resolved = path.resolve()
        relative = _relative(resolved, root)
        checksum = file_sha256(resolved)
        file_hashes[resolved] = checksum
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper(), file_hashes


def _scale_name(width: float, height: float) -> str:
    area_640 = width * 640.0 * height * 640.0
    if area_640 < 16.0**2:
        return "tiny"
    if area_640 < 32.0**2:
        return "small"
    return "regular"


def _parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label width at {path}:{line_number}")
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ValueError(f"non-numeric YOLO label at {path}:{line_number}") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite YOLO label at {path}:{line_number}")
        class_value, center_x, center_y, width, height = values
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < len(CATEGORY_NAMES):
            raise ValueError(f"invalid class id at {path}:{line_number}")
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            raise ValueError(f"invalid normalized center at {path}:{line_number}")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"invalid normalized size at {path}:{line_number}")
        rows.append((class_id, center_x, center_y, width, height))
    return rows


def _object_statistics(label_paths: Sequence[Path]) -> dict[str, Any]:
    class_counts = {name: 0 for name in CATEGORY_NAMES}
    scale_counts = {name: 0 for name in _SCALE_NAMES}
    scale_class_counts = {
        scale: {name: 0 for name in CATEGORY_NAMES} for scale in _SCALE_NAMES
    }
    object_count = 0
    for label in label_paths:
        for class_id, _, _, width, height in _parse_label(label):
            category = CATEGORY_NAMES[class_id]
            scale = _scale_name(width, height)
            class_counts[category] += 1
            scale_counts[scale] += 1
            scale_class_counts[scale][category] += 1
            object_count += 1
    missing = [name for name, count in class_counts.items() if count == 0]
    if missing:
        raise ValueError(f"selection set does not contain all classes: {missing}")
    return {
        "objects": object_count,
        "class_counts": class_counts,
        "scale_counts": scale_counts,
        "scale_class_counts": scale_class_counts,
    }


def _write_create_only_list(destination: Path, paths: Sequence[Path]) -> Path:
    if destination.is_symlink():
        raise FileExistsError(f"refusing to replace selection list symlink: {destination}")
    resolved = destination.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{path.resolve()}\n" for path in paths).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags, 0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace selection list: {resolved}") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return resolved


def build_ra_v11_selection_authority(
    dataset_root: str | Path,
    output_list: str | Path,
    screen30_output_list: str | Path | None = None,
) -> dict[str, Any]:
    """Create disjoint train-derived Screen10 and Screen30 validation authorities."""

    root = Path(dataset_root).resolve()
    train_images, val_images = _validated_layout(root)
    selected, screen30_selected = _select_from_train(root, train_images)
    screen = select_hashed_subset(train_images, root=root, fraction=0.10)
    screen_resolved = {path.resolve() for path in screen}
    selected_resolved = {path.resolve() for path in selected}
    screen30_resolved = {path.resolve() for path in screen30_selected}
    if (
        screen_resolved & selected_resolved
        or screen_resolved & screen30_resolved
        or selected_resolved & screen30_resolved
    ):
        raise ValueError("train-derived experiment splits overlap")

    selected_stems = {path.stem for path in (*selected, *screen30_selected)}
    val_stems = {path.stem for path in val_images}
    if selected_stems & val_stems:
        raise ValueError("selection set overlaps official val by image stem")

    label_paths = tuple(root / "labels" / "train" / f"{path.stem}.txt" for path in selected)
    screen30_label_paths = tuple(
        root / "labels" / "train" / f"{path.stem}.txt"
        for path in screen30_selected
    )
    if any(
        path.is_symlink() or not path.is_file()
        for path in (*label_paths, *screen30_label_paths)
    ):
        raise ValueError("selection image/label path mapping is invalid")
    image_manifest_sha, selected_image_hashes = _manifest_sha256(selected, root)
    label_manifest_sha, _ = _manifest_sha256(label_paths, root)
    screen30_image_manifest_sha, screen30_image_hashes = _manifest_sha256(
        screen30_selected, root
    )
    screen30_label_manifest_sha, _ = _manifest_sha256(screen30_label_paths, root)
    if len(set(selected_image_hashes.values())) != len(selected):
        raise ValueError("selection set contains duplicate image content")
    if len(set(screen30_image_hashes.values())) != len(screen30_selected):
        raise ValueError("Screen30 selection set contains duplicate image content")
    if set(selected_image_hashes.values()) & set(screen30_image_hashes.values()):
        raise ValueError("Screen10 and Screen30 selections overlap by image content")
    _, val_image_hashes = _manifest_sha256(val_images, root)
    if (
        set(selected_image_hashes.values()) & set(val_image_hashes.values())
        or set(screen30_image_hashes.values()) & set(val_image_hashes.values())
    ):
        raise ValueError("selection set overlaps official val by image content")

    statistics = _object_statistics(label_paths)
    relative_paths = [_relative(path, root) for path in selected]
    relative_signature = subset_signature(selected, root=root)
    screen30_statistics = _object_statistics(screen30_label_paths)
    screen30_relative_paths = [_relative(path, root) for path in screen30_selected]
    screen30_relative_signature = subset_signature(screen30_selected, root=root)
    list_path = _write_create_only_list(Path(output_list), selected)
    screen30_destination = (
        Path(screen30_output_list)
        if screen30_output_list is not None
        else Path(output_list).with_name("screen30-dev.txt")
    )
    screen30_list_path = _write_create_only_list(
        screen30_destination, screen30_selected
    )
    return {
        "format_version": 1,
        "design": "ra-glgm-v1.1-selection-set",
        "algorithm": "exclude frozen Screen647; create two disjoint salted SHA256-ranked development sets",
        "salt_sha256": hashlib.sha256(SELECTION_SALT).hexdigest().upper(),
        "screen30_salt_sha256": hashlib.sha256(
            SCREEN30_SELECTION_SALT
        ).hexdigest().upper(),
        "dataset_root": str(root),
        "counts": {
            "train": len(train_images),
            "screen647": len(screen),
            "remaining": len(train_images) - len(screen),
            "selection": len(selected),
            "screen30_selection": len(screen30_selected),
            "official_val": len(val_images),
            "duplicate_paths": (
                len(selected) - len(selected_resolved)
                + len(screen30_selected) - len(screen30_resolved)
            ),
            "duplicate_image_content": (
                len(selected) - len(set(selected_image_hashes.values()))
                + len(screen30_selected) - len(set(screen30_image_hashes.values()))
            ),
        },
        "screen647": {
            "relative_path_sha256": subset_signature(screen, root=root),
        },
        "selection": {
            "relative_paths": relative_paths,
            "relative_path_sha256": relative_signature,
            "image_manifest_sha256": image_manifest_sha,
            "label_manifest_sha256": label_manifest_sha,
            "absolute_list": {
                "path": str(list_path),
                "sha256": file_sha256(list_path),
                "count": len(selected),
            },
            **statistics,
        },
        "screen30_selection": {
            "relative_paths": screen30_relative_paths,
            "relative_path_sha256": screen30_relative_signature,
            "image_manifest_sha256": screen30_image_manifest_sha,
            "label_manifest_sha256": screen30_label_manifest_sha,
            "absolute_list": {
                "path": str(screen30_list_path),
                "sha256": file_sha256(screen30_list_path),
                "count": len(screen30_selected),
            },
            **screen30_statistics,
        },
        "overlap": {
            "screen647_paths": 0,
            "screen10_screen30_paths": 0,
            "screen10_screen30_image_content": 0,
            "official_val_stems": 0,
            "official_val_image_content": 0,
        },
    }


__all__ = [
    "SCREEN_IMAGE_COUNT",
    "SCREEN30_SELECTION_IMAGE_COUNT",
    "SCREEN30_SELECTION_SALT",
    "SELECTION_IMAGE_COUNT",
    "SELECTION_SALT",
    "TRAIN_IMAGE_COUNT",
    "VAL_IMAGE_COUNT",
    "build_ra_v11_selection_authority",
    "select_ra_v11_paths",
    "select_ra_v11_screen30_paths",
]

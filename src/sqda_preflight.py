from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import yaml


EXPECTED_FULL_DATASET_SHA256 = "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
EXPECTED_VAL_CONTENT_SHA256 = "A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A"
EXPECTED_TRAIN10_SHA256 = "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
EXPECTED_SPLIT_COUNTS = {
    "train": 6471,
    "val": 548,
}
VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _files_with_suffixes(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def parse_yolo_label(path: Path, class_count: int = 10) -> int:
    rows = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        try:
            class_value, cx, cy, width, height = map(float, fields)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric YOLO field") from error
        if not class_value.is_integer() or not 0 <= int(class_value) < class_count:
            raise ValueError(f"{path}:{line_number}: class is outside [0,{class_count - 1}]")
        values = (cx, cy, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite box value")
        if not 0 <= cx <= 1 or not 0 <= cy <= 1:
            raise ValueError(f"{path}:{line_number}: box center is outside [0,1]")
        if not 0 < width <= 1 or not 0 < height <= 1:
            raise ValueError(f"{path}:{line_number}: box size is outside (0,1]")
        rows += 1
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dataset_signature(dataset_root: str | Path, splits: Iterable[str] = ("train", "val")) -> str:
    """Reproduce the frozen baseline semantic signature over images and labels."""
    root = Path(dataset_root).expanduser().resolve()
    files = sorted(
        path
        for directory in ("images", "labels")
        for split in splits
        for path in (root / directory / split).glob("**/*")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def split_content_signature(dataset_root: str | Path, split: str) -> str:
    """Hash images, labels, and ignore labels using the frozen val-content protocol."""
    root = Path(dataset_root).expanduser().resolve()
    paths = []
    for folder in (
        root / "images" / split,
        root / "labels" / split,
        root / "labels_ignore" / split,
    ):
        if folder.exists():
            paths.extend(path for path in folder.rglob("*") if path.is_file())
    lines = [
        f"{_file_sha256(path).lower()}  {path.relative_to(root).as_posix()}"
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_visdrone_dataset(
    dataset_root: str | Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    expected = expected_counts or EXPECTED_SPLIT_COUNTS
    split_report = {}
    total_boxes = 0
    for split, expected_count in expected.items():
        images = _files_with_suffixes(root / "images" / split, IMAGE_SUFFIXES)
        labels = _files_with_suffixes(root / "labels" / split, {".txt"})
        if len(images) != expected_count or len(labels) != expected_count:
            raise RuntimeError(
                f"{split} count mismatch: expected {expected_count}, "
                f"images={len(images)}, labels={len(labels)}"
            )
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise RuntimeError(
                f"{split} image/label stem mismatch: "
                f"missing_labels={sorted(image_stems - label_stems)[:5]}, "
                f"missing_images={sorted(label_stems - image_stems)[:5]}"
            )
        boxes = sum(parse_yolo_label(label) for label in labels)
        total_boxes += boxes
        split_report[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": boxes,
        }
    semantic_signature = dataset_signature(root, expected)
    validation_signature = (
        split_content_signature(root, "val") if "val" in expected else None
    )
    if expected_counts is None and semantic_signature != EXPECTED_FULL_DATASET_SHA256:
        raise RuntimeError(
            f"full dataset SHA256 mismatch: expected {EXPECTED_FULL_DATASET_SHA256}, "
            f"got {semantic_signature}"
        )
    if (
        expected_counts is None
        and validation_signature != EXPECTED_VAL_CONTENT_SHA256
    ):
        raise RuntimeError(
            f"val content SHA256 mismatch: expected {EXPECTED_VAL_CONTENT_SHA256}, "
            f"got {validation_signature}"
        )
    return {
        "root": str(root),
        "splits": split_report,
        "total_files": sum(
            report["images"] + report["labels"] for report in split_report.values()
        ),
        "total_boxes": total_boxes,
        "signature": semantic_signature,
        "expected_signature": (
            EXPECTED_FULL_DATASET_SHA256 if expected_counts is None else None
        ),
        "val_content_signature": validation_signature,
        "expected_val_content_signature": (
            EXPECTED_VAL_CONTENT_SHA256 if expected_counts is None else None
        ),
    }


def write_dataset_yaml(dataset_root: str | Path, destination: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": VISDRONE_NAMES,
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    if Path(loaded["path"]).resolve() != root:
        raise RuntimeError("generated dataset YAML does not resolve to the requested root")
    return output

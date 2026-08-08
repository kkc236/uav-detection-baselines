#!/usr/bin/env python3
"""Audit a YOLO-format VisDrone dataset and write a reproducible inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=6471)
    parser.add_argument("--expected-val", type=int, default=548)
    parser.add_argument("--hash-content", action="store_true")
    parser.add_argument("--reference", type=Path, help="Optional prior audit whose immutable fields must match.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_root(data_path: Path, configured: str) -> Path:
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise ValueError(
            f"formal experiments require an absolute data.yaml path value, got {configured!r} in {data_path}"
        )
    return root.resolve()


def resolve_split(root: Path, value: str | list[str]) -> list[Path]:
    entries = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for entry in entries:
        path = Path(entry).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path.is_dir():
            images.extend(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = Path(line).expanduser()
                item = item.resolve() if item.is_absolute() else (path.parent / item).resolve()
                if item.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(item)
        else:
            raise FileNotFoundError(f"split path does not exist: {path}")
    return sorted(set(images))


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part == "images"]
    if not indexes:
        raise ValueError(f"image path has no 'images' component: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def update_inventory_hash(digest, root: Path, path: Path, include_content: bool) -> str | None:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(path.stat().st_size).encode("ascii"))
    digest.update(b"\0")
    content_hash = sha256_file(path) if include_content else None
    if content_hash:
        digest.update(content_hash.encode("ascii"))
    digest.update(b"\n")
    return content_hash


def audit_split(root: Path, images: list[Path], class_count: int, hash_content: bool) -> tuple[dict, set[str]]:
    classes: Counter[int] = Counter()
    boxes = 0
    empty_labels = 0
    digest = hashlib.sha256()
    image_hashes: set[str] = set()
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(image)
        try:
            with Image.open(image) as opened:
                width, height = opened.size
                opened.verify()
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image dimensions: {image} -> {(width, height)}")
        except Exception as error:
            raise ValueError(f"cannot decode image {image}: {error}") from error
        label = label_path(image)
        if not label.is_file():
            raise FileNotFoundError(f"missing label for {image}: {label}")
        image_hash = update_inventory_hash(digest, root, image, hash_content)
        if image_hash:
            image_hashes.add(image_hash)
        update_inventory_hash(digest, root, label, hash_content)
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid YOLO row at {label}:{line_number}: {line}")
            values = [float(field) for field in fields]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite label at {label}:{line_number}")
            class_id = int(values[0])
            if values[0] != class_id or not 0 <= class_id < class_count:
                raise ValueError(f"invalid class at {label}:{line_number}: {values[0]}")
            x, y, width, height = values[1:]
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"invalid normalized box at {label}:{line_number}: {values[1:]}")
            boxes += 1
            classes[class_id] += 1
    report = {
        "images": len(images),
        "labels": len(images),
        "empty_labels": empty_labels,
        "boxes": boxes,
        "class_box_counts": {str(index): classes[index] for index in range(class_count)},
        "inventory_sha256": digest.hexdigest().upper(),
        "inventory_hash_includes_content": hash_content,
    }
    return report, image_hashes


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = data.get("names")
    class_count = len(names) if isinstance(names, (dict, list)) else 0
    if class_count != 10:
        raise ValueError(f"VisDrone experiment requires exactly 10 classes, got {class_count}")
    root = resolve_root(data_path, data["path"])
    canonical_names = {str(key): value for key, value in names.items()} if isinstance(names, dict) else names
    ordered_names = (
        [canonical_names[str(index)] for index in range(class_count)]
        if isinstance(canonical_names, dict) and set(canonical_names) == {str(index) for index in range(class_count)}
        else canonical_names
    )
    if ordered_names != VISDRONE_NAMES:
        raise ValueError(f"unexpected VisDrone class mapping: {ordered_names}")
    split_images = {split: resolve_split(root, data[split]) for split in ("train", "val")}
    path_overlap = set(split_images["train"]) & set(split_images["val"])
    if path_overlap:
        raise ValueError(f"train/val path leakage detected: {sorted(path_overlap)[:5]}")
    splits = {}
    content_hashes = {}
    for split in ("train", "val"):
        splits[split], content_hashes[split] = audit_split(
            root, split_images[split], class_count, args.hash_content
        )
    if args.hash_content and (content_overlap := content_hashes["train"] & content_hashes["val"]):
        raise ValueError(f"train/val image-content leakage detected: {sorted(content_overlap)[:5]}")
    expected = {"train": args.expected_train, "val": args.expected_val}
    for split, expected_count in expected.items():
        if expected_count >= 0 and splits[split]["images"] != expected_count:
            raise ValueError(f"{split} image count mismatch: expected {expected_count}, got {splits[split]['images']}")
    payload = {
        "schema": "visdrone-yolo-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(data_path),
        "data_yaml_sha256": sha256_file(data_path),
        "dataset_root": str(root),
        "class_count": class_count,
        "names": canonical_names,
        "splits": splits,
    }
    if args.reference:
        reference = json.loads(args.reference.resolve().read_text(encoding="utf-8"))
        immutable_fields = ("data_yaml_sha256", "dataset_root", "class_count", "names", "splits")
        differences = [field for field in immutable_fields if reference.get(field) != payload.get(field)]
        if differences:
            raise RuntimeError(f"dataset differs from reference audit in fields: {differences}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

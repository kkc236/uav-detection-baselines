#!/usr/bin/env python3
"""Download and convert official VisDrone train/val data for the GLGM experiment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image
from ultralytics.utils import ASSETS_URL, TQDM
from ultralytics.utils.downloads import download


NAMES = [
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    return parser.parse_args()


def convert_split(root: Path, source_name: str, split: str) -> dict[str, int]:
    source = root / source_name
    source_images = source / "images"
    source_annotations = source / "annotations"
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for image in source_images.glob("*.jpg"):
        target = images / image.name
        if not target.exists():
            image.replace(target)
    stats = {
        "source_rows": 0,
        "ignored_rows": 0,
        "invalid_nonpositive_size_rows": 0,
        "converted_rows": 0,
    }
    for annotation in TQDM(sorted(source_annotations.glob("*.txt")), desc=f"Converting {split}"):
        image_path = images / annotation.with_suffix(".jpg").name
        with Image.open(image_path) as opened:
            width, height = opened.size
        rows = []
        for raw_line in annotation.read_text(encoding="utf-8").strip().splitlines():
            row = raw_line.split(",")
            stats["source_rows"] += 1
            if len(row) < 6:
                raise ValueError(f"invalid VisDrone row in {annotation}: {raw_line}")
            if row[4] == "0":
                stats["ignored_rows"] += 1
                continue
            x, y, box_width, box_height = map(int, row[:4])
            if box_width <= 0 or box_height <= 0:
                stats["invalid_nonpositive_size_rows"] += 1
                continue
            class_id = int(row[5]) - 1
            if not 0 <= class_id < len(NAMES):
                raise ValueError(f"invalid VisDrone class in {annotation}: {raw_line}")
            rows.append(
                f"{class_id} {(x + box_width / 2) / width:.6f} {(y + box_height / 2) / height:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}\n"
            )
            stats["converted_rows"] += 1
        (labels / annotation.name).write_text("".join(rows), encoding="utf-8")
    shutil.rmtree(source)
    return stats


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    weights_dir = args.weights_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    expected = {"train": 6471, "val": 548}
    complete = all(len(list((root / "images" / split).glob("*.jpg"))) == count for split, count in expected.items())
    if not complete:
        sources = {
            "VisDrone2019-DET-train": "train",
            "VisDrone2019-DET-val": "val",
        }
        urls = [f"{ASSETS_URL}/{source}.zip" for source in sources]
        download(urls, dir=root, threads=2, delete=True, exist_ok=True)
        for source, split in sources.items():
            convert_split(root, source, split)
    weight_path = weights_dir / "rtdetr-x.pt"
    if not weight_path.is_file():
        download(f"{ASSETS_URL}/rtdetr-x.pt", dir=weights_dir, unzip=False)
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(NAMES)},
    }
    data_path = root / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(data_path)
    print(weight_path)


if __name__ == "__main__":
    main()

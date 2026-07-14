from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import requests
from PIL import Image


ASSET_BASE_URL = "https://ultralytics.com/assets"
ARCHIVES = {
    "train": "VisDrone2019-DET-train.zip",
    "val": "VisDrone2019-DET-val.zip",
    "test": "VisDrone2019-DET-test-dev.zip",
}
SOURCE_FOLDERS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def convert_visdrone_row(row: str, *, image_width: int, image_height: int) -> str | None:
    parts = row.split(",")
    if len(parts) < 6 or parts[4] == "0":
        return None

    box = _convert_box(parts, image_width=image_width, image_height=image_height)
    if box is None:
        return None
    cls = int(parts[5]) - 1
    x_center, y_center, w_norm, h_norm = box
    return f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def convert_visdrone_ignore_row(row: str, *, image_width: int, image_height: int) -> str | None:
    parts = row.split(",")
    if len(parts) < 6 or parts[4] != "0":
        return None

    box = _convert_box(parts, image_width=image_width, image_height=image_height)
    if box is None:
        return None
    return " ".join(f"{value:.6f}" for value in box)


def _convert_box(parts: list[str], *, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = map(int, parts[:4])
    if image_width <= 0 or image_height <= 0 or w <= 0 or h <= 0:
        return None
    return (
        (x + w / 2) / image_width,
        (y + h / 2) / image_height,
        w / image_width,
        h / image_height,
    )


def download_with_resume(url: str, destination: Path, *, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resume_at = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if response.status_code == 200 and resume_at:
            resume_at = 0
        response.raise_for_status()
        mode = "ab" if resume_at else "wb"
        with destination.open(mode + "") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)


def extract_archive(archive_path: Path, dataset_dir: Path) -> None:
    with ZipFile(archive_path) as archive:
        archive.extractall(dataset_dir)


def convert_split(dataset_dir: Path, split: str) -> None:
    source_dir = dataset_dir / SOURCE_FOLDERS[split]
    source_images = source_dir / "images"
    source_annotations = source_dir / "annotations"
    target_images = dataset_dir / "images" / split
    target_labels = dataset_dir / "labels" / split
    target_ignore_labels = dataset_dir / "labels_ignore" / split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    target_ignore_labels.mkdir(parents=True, exist_ok=True)

    for image_path in source_images.glob("*.jpg"):
        target = target_images / image_path.name
        if not target.exists():
            target.write_bytes(image_path.read_bytes())

    for annotation_path in source_annotations.glob("*.txt"):
        image_path = target_images / annotation_path.with_suffix(".jpg").name
        width, height = Image.open(image_path).size
        rows = []
        ignore_rows = []
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            converted = convert_visdrone_row(line, image_width=width, image_height=height)
            if converted is not None:
                rows.append(converted)
            converted_ignore = convert_visdrone_ignore_row(line, image_width=width, image_height=height)
            if converted_ignore is not None:
                ignore_rows.append(converted_ignore)
        (target_labels / annotation_path.name).write_text("\n".join(rows), encoding="utf-8")
        (target_ignore_labels / annotation_path.name).write_text("\n".join(ignore_rows), encoding="utf-8")


def prepare_visdrone(dataset_dir: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        archive = dataset_dir / ARCHIVES[split]
        if not archive.exists():
            download_with_resume(f"{ASSET_BASE_URL}/{ARCHIVES[split]}", archive)
        if not (dataset_dir / SOURCE_FOLDERS[split]).exists():
            extract_archive(archive, dataset_dir)
        convert_split(dataset_dir, split)

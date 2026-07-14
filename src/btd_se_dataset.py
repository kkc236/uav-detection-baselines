from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics.models.rtdetr.val import RTDETRDataset


def ignore_label_path(image_file: str | Path) -> Path:
    image_path = Path(image_file)
    if image_path.parent.parent.name != "images":
        raise ValueError(f"expected image path under images/<split>, received {image_path}")
    dataset_root = image_path.parent.parent.parent
    split = image_path.parent.name
    return dataset_root / "labels_ignore" / split / image_path.with_suffix(".txt").name


def load_ignore_boxes(path: str | Path) -> np.ndarray:
    label_path = Path(path)
    if not label_path.exists() or not label_path.read_text(encoding="ascii").strip():
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    if boxes.shape[1] != 4:
        raise ValueError(f"ignore label {label_path} must contain four normalized box values per row")
    return boxes


def append_ignored_boxes(label: dict) -> dict:
    boxes = load_ignore_boxes(ignore_label_path(label["im_file"]))
    if not len(boxes):
        return label

    ignored_classes = np.full((len(boxes), 1), -1.0, dtype=np.float32)
    label["cls"] = np.concatenate((label["cls"], ignored_classes), axis=0)
    label["bboxes"] = np.concatenate((label["bboxes"], boxes), axis=0)
    return label


class BTDSEVisDroneDataset(RTDETRDataset):
    """RT-DETR dataset that carries VisDrone ignored boxes through training transforms as class -1."""

    def get_labels(self) -> list[dict]:
        labels = super().get_labels()
        if self.augment:
            for label in labels:
                append_ignored_boxes(label)
        return labels

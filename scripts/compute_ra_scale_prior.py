"""Recompute the canonical RA-GLGM v1.2 natural-image scale prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


QUANTILES = np.linspace(0.0, 1.0, 21, dtype=np.float64)


def _image_path(images: Path, stem: str) -> Path:
    matches = [
        path
        for suffix in (".jpg", ".jpeg", ".png", ".bmp")
        if (path := images / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one image for {stem}, found {matches}")
    return matches[0]


def compute_prior(dataset_root: str | Path) -> dict:
    root = Path(dataset_root).resolve()
    labels = root / "labels" / "train"
    images = root / "images" / "train"
    areas: list[float] = []
    class_counts = [0] * 10
    files = sorted(labels.glob("*.txt"))
    for label_path in files:
        with Image.open(_image_path(images, label_path.stem)) as image:
            width, height = image.size
        ratio = 640.0 / max(width, height)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            class_id = int(float(fields[0]))
            if not 0 <= class_id < 10:
                continue
            area = (
                float(fields[3])
                * width
                * ratio
                * float(fields[4])
                * height
                * ratio
            )
            if not math.isfinite(area) or area <= 0:
                raise RuntimeError(f"invalid area in {label_path}: {line}")
            areas.append(area)
            class_counts[class_id] += 1
    values = np.asarray(areas, dtype=np.float64)
    knots = np.quantile(values, QUANTILES, method="linear")
    payload = {
        "format_version": 1,
        "source": str(root).replace("\\", "/"),
        "images": len(files),
        "instances": int(values.size),
        "class_counts": class_counts,
        "letterbox_imgsz": 640,
        "area_definition": (
            "original normalized box transformed by "
            "640/max(image_width,image_height)"
        ),
        "quantile_probabilities": QUANTILES.tolist(),
        "area_knots": knots.tolist(),
        "log_area_knots": np.log(knots).tolist(),
        "minimum_area": float(values.min()),
        "maximum_area": float(values.max()),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = compute_prior(args.dataset_root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.write_text(rendered, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

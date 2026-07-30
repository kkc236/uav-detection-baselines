from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


VISDRONE_NAMES = (
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
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_coco_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    class_count: int = 10,
) -> dict:
    images_root = Path(images_dir).expanduser().resolve()
    labels_root = Path(labels_dir).expanduser().resolve()
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(f"missing image or label directory: {images_root}, {labels_root}")
    if not 1 <= class_count <= len(VISDRONE_NAMES):
        raise ValueError(f"class_count must be in [1,{len(VISDRONE_NAMES)}]")

    images = []
    annotations = []
    annotation_id = 1
    image_paths = sorted(
        path for path in images_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        image_id = image_path.stem
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        label_path = labels_root / f"{image_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number} must contain five fields")
            class_id, center_x, center_y, box_width, box_height = map(float, fields)
            integer_class = int(class_id)
            if class_id != integer_class or not 0 <= integer_class < class_count:
                raise ValueError(f"{label_path}:{line_number} has invalid class {class_id}")
            pixel_width = box_width * width
            pixel_height = box_height * height
            left = (center_x - box_width / 2.0) * width
            top = (center_y - box_height / 2.0) * height
            if pixel_width <= 0 or pixel_height <= 0:
                raise ValueError(f"{label_path}:{line_number} has a non-positive box")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": integer_class + 1,
                    "bbox": [left, top, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "info": {"description": "VisDrone YOLO labels converted in memory for COCO area evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": VISDRONE_NAMES[index]}
            for index in range(class_count)
        ],
    }


def evaluate_predictions(dataset: dict, predictions_path: str | Path) -> dict:
    predictions = Path(predictions_path).expanduser().resolve()
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    coco_ground_truth = COCO()
    coco_ground_truth.dataset = dataset
    coco_ground_truth.createIndex()
    coco_predictions = coco_ground_truth.loadRes(str(predictions))
    evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
    evaluator.params.imgIds = [image["id"] for image in dataset["images"]]
    evaluator.params.catIds = [category["id"] for category in dataset["categories"]]
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ap_small": float(evaluator.stats[3]),
        "ap_medium": float(evaluator.stats[4]),
        "ap_large": float(evaluator.stats[5]),
        "ar_small": float(evaluator.stats[9]),
        "ar_medium": float(evaluator.stats[10]),
        "ar_large": float(evaluator.stats[11]),
        "max_dets": int(evaluator.params.maxDets[-1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and SQDA-SGC with COCO small-object area metrics."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=548)
    parser.add_argument("--expected-annotations", type=int, default=38759)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = build_coco_dataset(args.images, args.labels)
    if len(dataset["images"]) != args.expected_images:
        raise RuntimeError(
            f"expected {args.expected_images} images, found {len(dataset['images'])}"
        )
    if len(dataset["annotations"]) != args.expected_annotations:
        raise RuntimeError(
            f"expected {args.expected_annotations} annotations, "
            f"found {len(dataset['annotations'])}"
        )
    baseline = evaluate_predictions(dataset, args.baseline_predictions)
    candidate = evaluate_predictions(dataset, args.candidate_predictions)
    report = {
        "protocol": {
            "area_small": [0, 32**2],
            "coordinate_space": "original image pixels",
            "iou_thresholds": [0.5, 0.95, 0.05],
            "max_dets": 300,
            "images": len(dataset["images"]),
            "annotations": len(dataset["annotations"]),
        },
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            key: candidate[key] - baseline[key]
            for key in baseline
            if key != "max_dets"
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

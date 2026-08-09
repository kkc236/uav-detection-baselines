"""Locked independent evaluator for RA-GLGM Screen30 and Formal100 tails."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ra_experiment_protocol import (  # noqa: E402
    RA_EXPERIMENT_PROTOCOL_SHA256,
    file_sha256,
    read_json,
    read_jsonl,
    validate_runtime_identity,
)
from src.rtdetr_ra_glgm import register_ra_glgm_decoder  # noqa: E402


AREA_RANGES = (
    ("all", 0.0, 1.0e10),
    ("tiny", 0.0, 16.0**2),
    ("small", 16.0**2, 32.0**2),
    ("medium", 32.0**2, 96.0**2),
    ("large", 96.0**2, 1.0e10),
)
COCO_CATEGORY_IDS = tuple(range(1, 11))
MAX_DETECTIONS_PER_IMAGE = 300


def _coco_category_id(class_id: int) -> int:
    """Match Ultralytics' non-COCO JSON export, which is one-indexed."""

    if not 0 <= class_id < len(COCO_CATEGORY_IDS):
        raise ValueError(f"invalid zero-indexed VisDrone class: {class_id}")
    return class_id + 1


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite evaluator metric: {name}={value!r}")
    return result


def _parse_epochs(value: str) -> list[int]:
    epochs = [int(item) for item in value.split(",") if item.strip()]
    if not epochs or epochs != sorted(set(epochs)) or any(epoch <= 0 for epoch in epochs):
        raise ValueError("epochs must be unique positive integers in ascending order")
    return epochs


def _load_authority(path: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("protocol_sha256") != RA_EXPERIMENT_PROTOCOL_SHA256:
        raise ValueError("foreign RA evaluator protocol manifest")
    evaluator = manifest.get("locked_evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("locked evaluator authority is missing")
    self_path = Path(__file__).resolve()
    if Path(str(evaluator.get("path", ""))).resolve() != self_path:
        raise ValueError("running evaluator path differs from protocol authority")
    digest = file_sha256(self_path)
    if digest != str(evaluator.get("sha256", "")).upper():
        raise ValueError("running evaluator bytes differ from protocol authority")
    if runtime.get("locked_evaluator_sha256") != digest:
        raise ValueError("run was not bound to this locked evaluator")
    return manifest


def _dataset(data_yaml: Path) -> tuple[Path, list[str], list[Path]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(data["path"])).resolve()
    val = Path(str(data["val"]))
    val = val.resolve() if val.is_absolute() else (root / val).resolve()
    if not val.is_dir():
        raise FileNotFoundError(f"validation image directory is missing: {val}")
    names_value = data["names"]
    if isinstance(names_value, Mapping):
        names = [str(names_value.get(index, names_value.get(str(index)))) for index in range(10)]
    else:
        names = [str(name) for name in names_value]
    if len(names) != 10 or any(name in {"None", ""} for name in names):
        raise ValueError("locked evaluator requires ten ordered VisDrone classes")
    images = sorted(val.glob("*.jpg"))
    if len(images) != 548:
        raise ValueError(f"locked evaluator requires 548 validation images, got {len(images)}")
    return root, names, images


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, value in enumerate(parts) if value == "images"]
    if not indexes:
        raise ValueError(f"validation image path has no images component: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _coco_ground_truth(images: Sequence[Path], names: Sequence[str]) -> tuple[dict[str, Any], dict[str, int]]:
    dataset: dict[str, Any] = {
        "info": {"description": "VisDrone locked RA-GLGM validation"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": _coco_category_id(index), "name": name}
            for index, name in enumerate(names)
        ],
    }
    image_ids: dict[str, int] = {}
    annotation_id = 1
    for image_id, image in enumerate(images, 1):
        stem = image.stem
        if stem in image_ids:
            raise ValueError(f"duplicate validation image stem: {stem}")
        image_ids[stem] = image_id
        with Image.open(image) as opened:
            width, height = opened.size
        dataset["images"].append(
            {"id": image_id, "file_name": image.name, "width": width, "height": height}
        )
        label = _label_path(image)
        if not label.is_file():
            raise FileNotFoundError(f"validation label is missing: {label}")
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid validation label at {label}:{line_number}")
            class_value, cx, cy, box_width, box_height = map(float, fields)
            class_id = int(class_value)
            if class_value != class_id or not 0 <= class_id < 10:
                raise ValueError(f"invalid validation class at {label}:{line_number}")
            width_px, height_px = box_width * width, box_height * height
            x_px = (cx - box_width / 2) * width
            y_px = (cy - box_height / 2) * height
            if width_px <= 0 or height_px <= 0:
                raise ValueError(f"non-positive validation box at {label}:{line_number}")
            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": _coco_category_id(class_id),
                    "bbox": [x_px, y_px, width_px, height_px],
                    "area": width_px * height_px,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    if annotation_id - 1 != 38_759:
        raise ValueError(
            f"locked evaluator requires 38,759 validation objects, got {annotation_id - 1}"
        )
    return dataset, image_ids


def _validated_predictions(
    predictions: Sequence[Mapping[str, Any]],
    image_ids: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Convert Ultralytics JSON rows while enforcing conf/max_det authority."""
    converted = []
    prediction_counts = {stem: 0 for stem in image_ids}
    for prediction in predictions:
        stem = str(prediction["image_id"])
        if stem not in image_ids:
            raise ValueError(f"prediction refers to foreign validation image: {stem}")
        prediction_counts[stem] += 1
        if prediction_counts[stem] > MAX_DETECTIONS_PER_IMAGE:
            raise ValueError(
                f"validation image {stem} exceeds max_det={MAX_DETECTIONS_PER_IMAGE}"
            )
        category_id = int(prediction["category_id"])
        if category_id not in COCO_CATEGORY_IDS:
            raise ValueError(
                f"prediction category_id does not match Ultralytics one-indexed export: {category_id}"
            )
        converted.append(
            {
                "image_id": image_ids[stem],
                "category_id": category_id,
                "bbox": [
                    _finite(value, f"prediction_{stem}_bbox")
                    for value in prediction["bbox"]
                ],
                "score": _finite(prediction["score"], f"prediction_{stem}_score"),
            }
        )
    if not converted:
        raise ValueError("locked RT-DETR evaluation produced no predictions")
    return converted


def _coco_area_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
    image_ids: Mapping[str, int],
) -> dict[str, float]:
    converted = _validated_predictions(predictions, image_ids)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = dict(ground_truth)
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(converted)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(image_ids.values())
    evaluator.params.catIds = list(COCO_CATEGORY_IDS)
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.params.areaRng = [[lower, upper] for _, lower, upper in AREA_RANGES]
    evaluator.params.areaRngLbl = [name for name, _, _ in AREA_RANGES]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # [IoU, recall, class, area, maxDet]

    def mean_valid(values: np.ndarray, name: str) -> float:
        valid = values[values > -1]
        if not valid.size:
            raise ValueError(f"COCOeval produced no valid values for {name}")
        return _finite(valid.mean(), name)

    return {
        "ap_tiny": mean_valid(precision[:, :, :, 1, 2], "ap_tiny"),
        "ap_small": mean_valid(precision[:, :, :, 2, 2], "ap_small"),
    }


def _checkpoint_queue(run: Path) -> dict[int, Mapping[str, Any]]:
    rows = read_jsonl(run / "publication-queue.jsonl")
    result = {}
    for row in rows:
        epoch = int(row["completed_epoch"])
        if epoch in result:
            raise ValueError(f"duplicate checkpoint queue epoch: {epoch}")
        result[epoch] = row
    return result


def evaluate(
    *,
    run_dir: str | Path,
    protocol_manifest: str | Path,
    epochs: Sequence[int],
    output: str | Path,
) -> list[dict[str, Any]]:
    from ultralytics import RTDETR

    register_ra_glgm_decoder()
    run = Path(run_dir).resolve()
    runtime = read_json(run / "ra-run.json")
    identity = runtime.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("RA runtime identity is missing")
    variant, stage = str(identity.get("variant")), str(identity.get("stage"))
    validate_runtime_identity(runtime, variant=variant, stage=stage)
    manifest = _load_authority(Path(protocol_manifest).resolve(), runtime)
    if identity != manifest["run_identities"].get(f"{variant}_{stage}"):
        raise ValueError("RA run identity differs from protocol authority")
    data_yaml = Path(str(runtime["data"])).resolve()
    _, names, images = _dataset(data_yaml)
    ground_truth, image_ids = _coco_ground_truth(images, names)
    queue = _checkpoint_queue(run)
    rows: list[dict[str, Any]] = []
    evaluator_sha = file_sha256(__file__)
    for epoch in epochs:
        checkpoint = run / "weights" / f"epoch{epoch - 1}.pt"
        queued = queue.get(epoch)
        if queued is None or Path(str(queued.get("checkpoint", ""))).resolve() != checkpoint:
            raise ValueError(f"checkpoint queue authority is missing for epoch {epoch}")
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != str(queued.get("checkpoint_sha256", "")).upper():
            raise ValueError(f"checkpoint SHA256 mismatch at epoch {epoch}")
        model = RTDETR(str(checkpoint))
        has_ra = any(module.__class__.__name__ == "RAGLGM" for module in model.model.modules())
        if has_ra != (variant == "ra_glgm"):
            raise ValueError(f"checkpoint architecture differs from arm at epoch {epoch}")
        result = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=640,
            batch=8,
            workers=8,
            device="0",
            max_det=300,
            conf=0.001,
            half=False,
            plots=False,
            save_json=True,
            project=str((run / "locked-evaluator").resolve()),
            name=f"epoch{epoch:04d}",
            exist_ok=False,
            verbose=False,
        )
        box = result.box
        if len(box.ap_class_index) != 10:
            raise ValueError(f"epoch {epoch} did not evaluate all ten classes")
        class_ap_by_id = {
            int(class_id): _finite(box.ap[index], f"class_{class_id}_ap")
            for index, class_id in enumerate(box.ap_class_index)
        }
        class_ap = [class_ap_by_id[index] for index in range(10)]
        predictions = list(model.validator.jdict)
        area = _coco_area_metrics(predictions, ground_truth, image_ids)
        row = {
            "completed_epoch": epoch,
            "variant": variant,
            "stage": stage,
            "run_id": identity["run_id"],
            "evaluator_sha256": evaluator_sha,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "precision": _finite(result.results_dict["metrics/precision(B)"], "precision"),
            "recall": _finite(result.results_dict["metrics/recall(B)"], "recall"),
            "map50": _finite(result.results_dict["metrics/mAP50(B)"], "map50"),
            "map75": _finite(np.nanmean(box.all_ap[:, 5]), "map75"),
            "map": _finite(result.results_dict["metrics/mAP50-95(B)"], "map"),
            **area,
            "class_ap": class_ap,
            "model_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            "speed_ms_per_image": {
                name: _finite(value, f"speed_{name}") for name, value in result.speed.items()
            },
        }
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace locked evaluation: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate(
        run_dir=args.run_dir,
        protocol_manifest=args.protocol_manifest,
        epochs=_parse_epochs(args.epochs),
        output=args.output,
    )
    print(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

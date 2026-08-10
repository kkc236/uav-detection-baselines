"""Locked independent evaluator for RA-GLGM Screen30 and Formal100 tails."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
    BASELINE_PARAMETERS,
    RA_EXPERIMENT_PROTOCOL,
    file_sha256,
    ignore_sidecar_signature,
    load_ra_authority,
    read_json,
    read_jsonl,
    validate_runtime_identity,
)
from src.fdr_protocol import canonical_json_bytes  # noqa: E402
from src.lpr_protocol import CATEGORY_NAMES, dataset_signature  # noqa: E402
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
LETTERBOX_SIZE = 640
IGNORE_DETECTION_IOF_THRESHOLD = 0.5
EVALUATION_CONFIDENCE = 0.001


@dataclass(frozen=True)
class _LetterboxGeometry:
    source_width: int
    source_height: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float

    def xywh(self, box: Sequence[float]) -> list[float]:
        if len(box) != 4:
            raise ValueError("bbox must contain x, y, width, height")
        x, y, width, height = map(float, box)
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError("bbox contains a non-finite coordinate")
        if width < 0 or height < 0:
            raise ValueError("bbox width and height must be non-negative")
        return [
            x * self.scale_x + self.pad_x,
            y * self.scale_y + self.pad_y,
            width * self.scale_x,
            height * self.scale_y,
        ]


def _letterbox_geometry(source_width: int, source_height: int) -> _LetterboxGeometry:
    """Match Ultralytics' centered 640 letterbox, including resize rounding."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    ratio = min(LETTERBOX_SIZE / source_width, LETTERBOX_SIZE / source_height)
    resized_width = round(source_width * ratio)
    resized_height = round(source_height * ratio)
    return _LetterboxGeometry(
        source_width=source_width,
        source_height=source_height,
        scale_x=resized_width / source_width,
        scale_y=resized_height / source_height,
        pad_x=(LETTERBOX_SIZE - resized_width) / 2.0,
        pad_y=(LETTERBOX_SIZE - resized_height) / 2.0,
    )


def _intersection_over_detection(first: Sequence[float], second: Sequence[float]) -> float:
    """Return intersection over the first (detection) xywh box area."""

    x1, y1, width1, height1 = map(float, first)
    x2, y2, width2, height2 = map(float, second)
    area = max(width1, 0.0) * max(height1, 0.0)
    if area <= 0:
        return 0.0
    intersection_width = max(0.0, min(x1 + width1, x2 + width2) - max(x1, x2))
    intersection_height = max(0.0, min(y1 + height1, y2 + height2) - max(y1, y2))
    return intersection_width * intersection_height / area


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
    manifest = load_ra_authority(path, repository_root=ROOT)
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


def _dataset(
    data_yaml: Path, *, expected_images: int
) -> tuple[Path, list[str], list[Path], Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(data["path"])).resolve()
    val = Path(str(data["val"]))
    val = val.resolve() if val.is_absolute() else (root / val).resolve()
    if val.is_dir():
        images = sorted(val.glob("*.jpg"))
    elif val.is_file():
        images = []
        for line in val.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image = Path(line.strip())
            image = image.resolve() if image.is_absolute() else (root / image).resolve()
            if not image.is_file() or image.suffix.lower() != ".jpg":
                raise FileNotFoundError(f"validation list contains a missing image: {image}")
            images.append(image)
        if len(set(images)) != len(images):
            raise ValueError("validation list must contain unique image paths")
    else:
        raise FileNotFoundError(f"validation image source is missing: {val}")
    names_value = data["names"]
    if isinstance(names_value, Mapping):
        names = [str(names_value.get(index, names_value.get(str(index)))) for index in range(10)]
    else:
        names = [str(name) for name in names_value]
    if names != list(CATEGORY_NAMES):
        raise ValueError("locked evaluator VisDrone category order differs from authority")
    if len(images) != expected_images:
        raise ValueError(
            f"locked evaluator requires {expected_images} validation images, got {len(images)}"
        )
    return root, names, images, val


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, value in enumerate(parts) if value == "images"]
    if not indexes:
        raise ValueError(f"validation image path has no images component: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _ignore_label_path(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, value in enumerate(parts) if value == "images"]
    if not indexes:
        raise ValueError(f"validation image path has no images component: {image}")
    parts[indexes[-1]] = "labels_ignore"
    return Path(*parts).with_suffix(".txt")


def _ignore_boxes(image: Path, geometry: _LetterboxGeometry) -> list[list[float]]:
    path = _ignore_label_path(image)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"validation ignore sidecar is missing: {path}")
    result: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid ignore sidecar at {path}:{line_number}")
        cx, cy, width, height = map(float, fields)
        if not all(math.isfinite(value) for value in (cx, cy, width, height)) or not (
            0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1
        ):
            raise ValueError(f"invalid normalized ignore box at {path}:{line_number}")
        source_box = [
            (cx - width / 2) * geometry.source_width,
            (cy - height / 2) * geometry.source_height,
            width * geometry.source_width,
            height * geometry.source_height,
        ]
        result.append(geometry.xywh(source_box))
    return result


def _coco_ground_truth(
    images: Sequence[Path],
    names: Sequence[str],
    *,
    expected_objects: int | None,
) -> tuple[
    dict[str, Any],
    dict[str, int],
    dict[str, _LetterboxGeometry],
    dict[str, list[list[float]]],
]:
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
    geometries: dict[str, _LetterboxGeometry] = {}
    ignored: dict[str, list[list[float]]] = {}
    annotation_id = 1
    for image_id, image in enumerate(images, 1):
        stem = image.stem
        if stem in image_ids:
            raise ValueError(f"duplicate validation image stem: {stem}")
        image_ids[stem] = image_id
        with Image.open(image) as opened:
            width, height = opened.size
        geometry = _letterbox_geometry(width, height)
        geometries[stem] = geometry
        ignored[stem] = _ignore_boxes(image, geometry)
        dataset["images"].append(
            {
                "id": image_id,
                "file_name": image.name,
                "width": LETTERBOX_SIZE,
                "height": LETTERBOX_SIZE,
            }
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
            letterboxed = geometry.xywh([x_px, y_px, width_px, height_px])
            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": _coco_category_id(class_id),
                    "bbox": letterboxed,
                    "area": letterboxed[2] * letterboxed[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    if expected_objects is not None and annotation_id - 1 != expected_objects:
        raise ValueError(
            f"locked evaluator requires {expected_objects:,} validation objects, "
            f"got {annotation_id - 1}"
        )
    return dataset, image_ids, geometries, ignored


def _validated_predictions(
    predictions: Sequence[Mapping[str, Any]],
    image_ids: Mapping[str, int],
    geometries: Mapping[str, _LetterboxGeometry],
    ignored: Mapping[str, Sequence[Sequence[float]]],
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
        geometry = geometries.get(stem)
        if geometry is None:
            raise ValueError(f"prediction image geometry is missing: {stem}")
        source_bbox = [
            _finite(value, f"prediction_{stem}_bbox") for value in prediction["bbox"]
        ]
        bbox = geometry.xywh(source_bbox)
        if any(
            _intersection_over_detection(bbox, ignore_box)
            >= IGNORE_DETECTION_IOF_THRESHOLD
            for ignore_box in ignored.get(stem, ())
        ):
            continue
        converted.append(
            {
                "image_id": image_ids[stem],
                "category_id": category_id,
                "bbox": bbox,
                "score": _finite(prediction["score"], f"prediction_{stem}_score"),
            }
        )
    if not converted:
        raise ValueError("locked RT-DETR evaluation produced no predictions")
    return converted


def _micro_precision_recall(eval_images: Sequence[Mapping[str, Any] | None]) -> dict[str, float]:
    """Compute fixed-threshold micro P/R from COCO matches, excluding ignored entries."""

    true_positive = false_positive = false_negative = 0
    all_area = [AREA_RANGES[0][1], AREA_RANGES[0][2]]
    for item in eval_images:
        if (
            item is None
            or int(item.get("maxDet", -1)) != MAX_DETECTIONS_PER_IMAGE
            or list(item.get("aRng", ())) != all_area
        ):
            continue
        scores = np.asarray(item["dtScores"], dtype=np.float64)
        matched = np.asarray(item["dtMatches"])[0] > 0
        detection_ignored = np.asarray(item["dtIgnore"])[0].astype(bool)
        selected = scores >= EVALUATION_CONFIDENCE
        true_positive += int(np.count_nonzero(selected & matched & ~detection_ignored))
        false_positive += int(np.count_nonzero(selected & ~matched & ~detection_ignored))

        ground_truth_ids = np.asarray(item["gtIds"])
        ground_truth_ignored = np.asarray(item["gtIgnore"]).astype(bool)
        matched_ground_truth_ids = np.asarray(item["dtMatches"])[0][
            selected & matched & ~detection_ignored
        ]
        valid_ground_truth_ids = ground_truth_ids[~ground_truth_ignored]
        false_negative += int(
            np.count_nonzero(~np.isin(valid_ground_truth_ids, matched_ground_truth_ids))
        )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    if precision_denominator <= 0 or recall_denominator <= 0:
        raise ValueError("COCOeval produced no valid detections or ground truths for micro P/R")
    return {
        "precision": true_positive / precision_denominator,
        "recall": true_positive / recall_denominator,
    }


def _coco_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
    image_ids: Mapping[str, int],
    geometries: Mapping[str, _LetterboxGeometry],
    ignored: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, float]:
    converted = _validated_predictions(predictions, image_ids, geometries, ignored)
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

    metrics: dict[str, Any] = {
        "map": mean_valid(precision[:, :, :, 0, 2], "map"),
        "map50": mean_valid(precision[0, :, :, 0, 2], "map50"),
        "map75": mean_valid(precision[5, :, :, 0, 2], "map75"),
        "ap_tiny": mean_valid(precision[:, :, :, 1, 2], "ap_tiny"),
        "ap_small": mean_valid(precision[:, :, :, 2, 2], "ap_small"),
        "class_ap": [
            mean_valid(precision[:, :, index, 0, 2], f"class_{index}_ap")
            for index in range(len(COCO_CATEGORY_IDS))
        ],
    }
    metrics.update(_micro_precision_recall(evaluator.evalImgs))
    return metrics


def _checkpoint_queue(run: Path) -> dict[int, Mapping[str, Any]]:
    rows = read_jsonl(run / "publication-queue.jsonl")
    result = {}
    for row in rows:
        epoch = int(row["completed_epoch"])
        if epoch in result:
            raise ValueError(f"duplicate checkpoint queue epoch: {epoch}")
        result[epoch] = row
    return result


def _ordered_names(value: Mapping[int, str] | Sequence[str]) -> list[str]:
    if isinstance(value, Mapping):
        return [str(value.get(index, value.get(str(index)))) for index in range(10)]
    return [str(name) for name in value]


def _validate_checkpoint_model(model: Any, *, variant: str) -> int:
    """Require the frozen FDR baseline or the one exact P3 RA insertion."""

    names = _ordered_names(model.names)
    if names != list(CATEGORY_NAMES):
        raise ValueError("checkpoint VisDrone category order differs from authority")
    named_ra = [
        (name, module)
        for name, module in model.model.named_modules()
        if module.__class__.__name__ == "RAGLGM"
    ]
    if variant == "baseline":
        if named_ra:
            raise ValueError("baseline checkpoint contains an RA-GLGM module")
    else:
        if len(named_ra) != 1 or named_ra[0][0] != "model.28.ra_glgm":
            raise ValueError("method checkpoint does not contain the unique frozen P3 RA-GLGM")
        if getattr(named_ra[0][1], "private_parameter_count", None) != int(
            RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]
        ):
            raise ValueError("method checkpoint RA-GLGM private parameter count differs")
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    expected = BASELINE_PARAMETERS + (
        int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"])
        if variant == "ra_glgm"
        else 0
    )
    if parameters != expected:
        raise ValueError(
            f"checkpoint parameter count differs: expected={expected}, actual={parameters}"
        )
    return parameters


def _load_saved_predictions(result: Any) -> list[dict[str, Any]]:
    """Read the validator's structured JSON output across Ultralytics versions."""

    prediction_path = Path(result.save_dir).resolve() / "predictions.json"
    if prediction_path.is_symlink() or not prediction_path.is_file():
        raise FileNotFoundError(
            f"locked evaluator prediction JSON is missing: {prediction_path}"
        )
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise ValueError("locked evaluator prediction JSON must contain a list of objects")
    return [dict(row) for row in payload]


def _bind_evaluation_row(
    row: Mapping[str, Any], previous_sha256: str
) -> dict[str, Any]:
    """Bind one locked metric row to its predecessor and canonical bytes."""

    normalized_previous = str(previous_sha256).upper()
    if len(normalized_previous) != 64 or any(
        character not in "0123456789ABCDEF" for character in normalized_previous
    ):
        raise ValueError("previous evaluation row SHA256 is invalid")
    payload = {
        **dict(row),
        "previous_evaluation_row_sha256": normalized_previous,
    }
    payload["evaluation_row_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest().upper()
    return payload


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
    expected_epochs = {
        "screen": RA_EXPERIMENT_PROTOCOL["evaluation"]["screen_evaluated_epochs"],
        "formal": RA_EXPERIMENT_PROTOCOL["evaluation"]["formal_evaluated_epochs"],
        "explore50": RA_EXPERIMENT_PROTOCOL["evaluation"]["explore50_evaluated_epochs"],
    }.get(stage)
    if expected_epochs is None or list(epochs) != list(expected_epochs):
        raise ValueError(f"locked evaluator epochs differ from frozen {stage} authority")
    data_yaml = Path(str(runtime["data"])).resolve()
    dataset_authority = manifest.get("dataset_authority")
    if not isinstance(dataset_authority, Mapping):
        raise ValueError("locked evaluator dataset authority is missing")
    selection_name = {
        "explore50": "selection_set",
    }.get(stage)
    selection = dataset_authority.get(selection_name) if selection_name else None
    if stage == "explore50":
        if not isinstance(selection, Mapping):
            raise ValueError(f"{stage} selection-set authority is missing")
        expected_images = int(selection.get("images", -1))
        expected_objects = int(selection.get("objects", -1))
    else:
        expected_images = int(RA_EXPERIMENT_PROTOCOL["dataset"]["val_images"])
        expected_objects = 38_759
    dataset_root, names, images, validation_source = _dataset(
        data_yaml, expected_images=expected_images
    )
    if dataset_root != Path(str(dataset_authority.get("root", ""))).resolve():
        raise ValueError("locked evaluator dataset root differs from authority")
    if dataset_signature(dataset_root) != dataset_authority.get("positive"):
        raise ValueError("locked evaluator positive dataset differs from authority")
    if ignore_sidecar_signature(dataset_root) != dataset_authority.get("ignore"):
        raise ValueError("locked evaluator ignore sidecars differ from authority")
    if stage == "explore50":
        if validation_source != Path(str(selection.get("path", ""))).resolve():
            raise ValueError(f"{stage} validation list path differs from authority")
        if file_sha256(validation_source) != str(selection.get("sha256", "")).upper():
            raise ValueError(f"{stage} validation list SHA256 differs from authority")
    elif validation_source != (dataset_root / "images" / "val").resolve():
        raise ValueError(f"{stage} must use the authoritative official val split")
    ground_truth, image_ids, geometries, ignored = _coco_ground_truth(
        images, names, expected_objects=expected_objects
    )
    queue = _checkpoint_queue(run)
    rows: list[dict[str, Any]] = []
    previous_evaluation_sha256 = "0" * 64
    evaluator_sha = file_sha256(__file__)
    for epoch in epochs:
        checkpoint = run / "weights" / f"epoch{epoch - 1}.pt"
        queued = queue.get(epoch)
        if (
            queued is None
            or queued.get("run_id") != identity["run_id"]
            or queued.get("variant") != variant
            or queued.get("stage") != stage
            or queued.get("status") != "pending"
            or Path(str(queued.get("checkpoint", ""))).resolve() != checkpoint
        ):
            raise ValueError(f"checkpoint queue authority is missing for epoch {epoch}")
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != str(queued.get("checkpoint_sha256", "")).upper():
            raise ValueError(f"checkpoint SHA256 mismatch at epoch {epoch}")
        model = RTDETR(str(checkpoint))
        model_parameters = _validate_checkpoint_model(model, variant=variant)
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
        predictions = _load_saved_predictions(result)
        prediction_path = Path(result.save_dir).resolve() / "predictions.json"
        independent = _coco_metrics(
            predictions,
            ground_truth,
            image_ids,
            geometries,
            ignored,
        )
        row = _bind_evaluation_row({
            "completed_epoch": epoch,
            "variant": variant,
            "stage": stage,
            "run_id": identity["run_id"],
            "evaluator_sha256": evaluator_sha,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "predictions_artifact": {
                "path": str(prediction_path),
                "sha256": file_sha256(prediction_path),
            },
            **independent,
            "model_parameters": model_parameters,
            "ultralytics_diagnostics": {
                "precision": _finite(
                    result.results_dict["metrics/precision(B)"], "ultralytics_precision"
                ),
                "recall": _finite(
                    result.results_dict["metrics/recall(B)"], "ultralytics_recall"
                ),
                "map50": _finite(
                    result.results_dict["metrics/mAP50(B)"], "ultralytics_map50"
                ),
                "map": _finite(
                    result.results_dict["metrics/mAP50-95(B)"], "ultralytics_map"
                ),
            },
            "speed_ms_per_image": {
                name: _finite(value, f"speed_{name}") for name, value in result.speed.items()
            },
        }, previous_evaluation_sha256)
        rows.append(row)
        previous_evaluation_sha256 = row["evaluation_row_sha256"]
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

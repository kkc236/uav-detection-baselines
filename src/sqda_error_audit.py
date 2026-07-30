from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


COCO_SMALL_AREA = 32.0**2
COCO_MEDIUM_AREA = 96.0**2


def _area_bin(area: float) -> str:
    if not math.isfinite(area) or area < 0.0:
        raise ValueError(f"box area must be finite and non-negative, got {area}")
    if area < COCO_SMALL_AREA:
        return "small"
    if area < COCO_MEDIUM_AREA:
        return "medium"
    return "large"


def _box_area(box: Sequence[float]) -> float:
    if len(box) != 4:
        raise ValueError(f"bbox must contain [x,y,w,h], got {box}")
    width, height = float(box[2]), float(box[3])
    if not math.isfinite(width) or not math.isfinite(height):
        raise ValueError(f"bbox size must be finite, got {box}")
    return max(0.0, width) * max(0.0, height)


def _iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("IoU boxes must each contain [x,y,w,h]")
    first_x, first_y, first_w, first_h = (float(value) for value in first)
    second_x, second_y, second_w, second_h = (float(value) for value in second)
    values = (first_x, first_y, first_w, first_h, second_x, second_y, second_w, second_h)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("IoU boxes must contain only finite values")
    first_w, first_h = max(0.0, first_w), max(0.0, first_h)
    second_w, second_h = max(0.0, second_w), max(0.0, second_h)
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first_w * first_h + second_w * second_h - intersection
    return intersection / union if union > 0.0 else 0.0


def _fresh_bin() -> dict[str, Any]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "_tp_scores": [],
        "_fp_scores": [],
        "_tp_ious": [],
    }


def _record(
    bins: Mapping[str, dict[str, Any]],
    bin_name: str,
    metric: str,
    value: float | None = None,
) -> None:
    for name in ("all", bin_name):
        bucket = bins[name]
        bucket[metric] += 1
        if metric == "tp" and value is not None:
            bucket["_tp_scores"].append(value)
        if metric == "fp" and value is not None:
            bucket["_fp_scores"].append(value)


def _record_true_positive(
    bins: Mapping[str, dict[str, Any]],
    bin_name: str,
    score: float,
    iou: float,
) -> None:
    for name in ("all", bin_name):
        bucket = bins[name]
        bucket["tp"] += 1
        bucket["_tp_scores"].append(score)
        bucket["_tp_ious"].append(iou)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _validate_prediction(prediction: Mapping[str, Any]) -> tuple[Any, Any, list[float], float]:
    required = ("image_id", "category_id", "bbox", "score")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise ValueError(f"prediction is missing required keys: {missing}")
    box = prediction["bbox"]
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
        raise ValueError("prediction bbox must be a numeric sequence")
    canonical_box = [float(value) for value in box]
    _box_area(canonical_box)
    score = float(prediction["score"])
    if not math.isfinite(score):
        raise ValueError(f"prediction score must be finite, got {score}")
    return prediction["image_id"], prediction["category_id"], canonical_box, score


def summarize_detection_errors(
    dataset: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize fixed-threshold class-aware detection errors by COCO area bin."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0,1]")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0,1]")

    image_ids = {image["id"] for image in dataset.get("images", [])}
    category_ids = {category["id"] for category in dataset.get("categories", [])}
    if not image_ids or not category_ids:
        raise ValueError("dataset must contain images and categories")
    ground_truth: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
        if image_id not in image_ids or category_id not in category_ids:
            raise ValueError("annotation references an unknown image or category")
        box = annotation.get("bbox")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
            raise ValueError("annotation bbox must be a numeric sequence")
        canonical_box = [float(value) for value in box]
        area = float(annotation.get("area", _box_area(canonical_box)))
        ground_truth[(image_id, category_id)].append(
            {"bbox": canonical_box, "bin": _area_bin(area), "matched": False}
        )

    grouped_predictions: dict[tuple[Any, Any], list[tuple[list[float], float]]] = defaultdict(list)
    for prediction in predictions:
        image_id, category_id, box, score = _validate_prediction(prediction)
        if image_id not in image_ids or category_id not in category_ids:
            raise ValueError("prediction references an unknown image or category")
        if score >= confidence_threshold:
            grouped_predictions[(image_id, category_id)].append((box, score))

    bins = {name: _fresh_bin() for name in ("all", "small", "medium", "large")}
    for key in sorted(set(ground_truth) | set(grouped_predictions), key=repr):
        targets = ground_truth[key]
        for box, score in sorted(grouped_predictions[key], key=lambda value: value[1], reverse=True):
            best_index, best_iou = None, -1.0
            for index, target in enumerate(targets):
                if target["matched"]:
                    continue
                iou = _iou_xywh(box, target["bbox"])
                if iou > best_iou:
                    best_index, best_iou = index, iou
            if best_index is not None and best_iou >= iou_threshold:
                target = targets[best_index]
                target["matched"] = True
                _record_true_positive(bins, target["bin"], score, best_iou)
            else:
                _record(bins, _area_bin(_box_area(box)), "fp", score)
        for target in targets:
            if not target["matched"]:
                _record(bins, target["bin"], "fn")

    return {
        name: {
            "tp": bucket["tp"],
            "fp": bucket["fp"],
            "fn": bucket["fn"],
            "mean_tp_score": _mean(bucket["_tp_scores"]),
            "mean_fp_score": _mean(bucket["_fp_scores"]),
            "mean_tp_iou": _mean(bucket["_tp_ious"]),
        }
        for name, bucket in bins.items()
    }


def compare_error_summaries(
    baseline: Mapping[str, Mapping[str, float | int | None]],
    candidate: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Return candidate-minus-baseline error counts and compatible mean deltas."""
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate error bins differ")
    count_keys = ("tp", "fp", "fn")
    mean_keys = ("mean_tp_score", "mean_fp_score", "mean_tp_iou")
    result: dict[str, dict[str, float | int | None]] = {}
    for bin_name in sorted(baseline):
        before, after = baseline[bin_name], candidate[bin_name]
        if set(before) != set(after):
            raise ValueError(f"baseline and candidate metrics differ for {bin_name}")
        result[bin_name] = {
            key: int(after[key]) - int(before[key])
            for key in count_keys
        }
        for key in mean_keys:
            before_value, after_value = before[key], after[key]
            result[bin_name][key] = (
                None
                if before_value is None or after_value is None
                else float(after_value) - float(before_value)
            )
    return result


def precision_recall_f1_curve(
    dataset: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Build the complete class-aware greedy P/R/F1 curve without threshold tuning."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0,1]")
    image_ids = {image["id"] for image in dataset.get("images", [])}
    category_ids = {category["id"] for category in dataset.get("categories", [])}
    if not image_ids or not category_ids:
        raise ValueError("dataset must contain images and categories")
    ground_truth: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
        if image_id not in image_ids or category_id not in category_ids:
            raise ValueError("annotation references an unknown image or category")
        box = annotation.get("bbox")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
            raise ValueError("annotation bbox must be a numeric sequence")
        ground_truth[(image_id, category_id)].append(
            {"bbox": [float(value) for value in box], "matched": False}
        )

    ranked_predictions = []
    for prediction in predictions:
        image_id, category_id, box, score = _validate_prediction(prediction)
        if image_id not in image_ids or category_id not in category_ids:
            raise ValueError("prediction references an unknown image or category")
        ranked_predictions.append((score, image_id, category_id, box))
    ranked_predictions.sort(key=lambda value: value[0], reverse=True)

    true_positives = 0
    false_positives = 0
    total_ground_truth = sum(len(targets) for targets in ground_truth.values())
    points: list[dict[str, float | int]] = []
    for score, image_id, category_id, box in ranked_predictions:
        targets = ground_truth[(image_id, category_id)]
        best_index, best_iou = None, -1.0
        for index, target in enumerate(targets):
            if target["matched"]:
                continue
            iou = _iou_xywh(box, target["bbox"])
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_index is not None and best_iou >= iou_threshold:
            targets[best_index]["matched"] = True
            true_positives += 1
        else:
            false_positives += 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / total_ground_truth if total_ground_truth else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        points.append(
            {
                "confidence_threshold": score,
                "tp": true_positives,
                "fp": false_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    best = max(
        points,
        key=lambda point: (float(point["f1"]), float(point["confidence_threshold"])),
        default={
            "confidence_threshold": None,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    )
    return {
        "iou_threshold": iou_threshold,
        "matching": "class-aware_score-descending_greedy",
        "ground_truth": total_ground_truth,
        "points": points,
        "best_f1": best,
    }

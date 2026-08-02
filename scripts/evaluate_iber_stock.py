"""Create the immutable IBER-BE stock baseline authority."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    execution_environment,
    file_sha256,
    write_immutable_report,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
)
from src.rtdetr_iber import FrozenIBERAdapter  # noqa: E402


AMENDED_GATE_STATUS = "passed_with_runtime_amendment"
EXPECTED_CATEGORY_SHA256 = (
    "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
)
BASELINE_REFERENCE_ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 4090",
    "driver": "550.142",
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "cuda": "12.1",
    "ultralytics": "8.4.90",
}
EVALUATION_CONSTANTS = {
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "conf": 0.001,
    "max_det": 300,
    "nms": False,
    "half": False,
    "repeats": 3,
}
IOU_THRESHOLDS = torch.linspace(0.50, 0.95, 10)


def _canonical_metric_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _assert_repeated_evaluations(
    repeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(repeats) != EVALUATION_CONSTANTS["repeats"]:
        raise ValueError("stock authority requires exactly 3 repeats")
    if not repeats:
        raise ValueError("stock authority repeats cannot be empty")
    reference = _canonical_metric_bytes(repeats[0])
    for index, report in enumerate(repeats, start=1):
        if not report or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in report.values()
        ):
            raise ValueError(f"stock repeat {index} has invalid metrics")
        if _canonical_metric_bytes(report) != reference:
            raise ValueError(f"stock repeat {index} differs from repeat 1")
    return dict(repeats[0])


def build_stock_authority_report(
    *,
    repeats: Sequence[Mapping[str, Any]],
    baseline_path: Path,
    baseline_bytes: int,
    baseline_sha256: str,
    dataset_sha256: str,
    category_sha256: str,
    execution_environment: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Bind three exact stock evaluations to all amended authorities."""
    stock = _assert_repeated_evaluations(repeats)
    actual = (
        baseline_sha256.upper(),
        dataset_sha256.upper(),
        category_sha256.upper(),
    )
    expected = (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_DATASET_SHA256,
        EXPECTED_CATEGORY_SHA256,
    )
    if actual != expected:
        raise ValueError("IBER-BE stock authority artifact mismatch")
    if dict(execution_environment) != execution_environment_authority():
        raise ValueError("IBER-BE stock authority execution environment mismatch")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in source_commit)
    ):
        raise ValueError("IBER-BE stock authority source commit is invalid")
    if type(baseline_bytes) is not int or baseline_bytes < 1:
        raise ValueError("IBER-BE stock authority baseline byte count is invalid")
    return {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "status": AMENDED_GATE_STATUS,
        "source_commit": source_commit.lower(),
        "baseline_checkpoint": {
            "path": str(Path(baseline_path).resolve()),
            "bytes": baseline_bytes,
            "sha256": baseline_sha256.upper(),
        },
        "dataset_sha256": dataset_sha256.upper(),
        "category_sha256": category_sha256.upper(),
        "baseline_reference_environment": dict(BASELINE_REFERENCE_ENVIRONMENT),
        "execution_environment": dict(execution_environment),
        "runtime_amendment": dict(RUNTIME_AMENDMENT),
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "evaluation_constants": dict(EVALUATION_CONSTANTS),
        "repeat_count": len(repeats),
        "repeat_exact": True,
        "stock": stock,
    }


def execution_environment_authority() -> dict[str, Any]:
    """Return the frozen amended execution environment as a mutable copy."""
    return execution_environment()


def current_execution_environment() -> dict[str, Any]:
    """Measure the server runtime instead of trusting configured labels."""
    import torchvision
    import ultralytics

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().splitlines()
    if len(query) != 1:
        raise RuntimeError("IBER-BE requires exactly one visible evaluation GPU")
    gpu, memory, driver = [field.strip() for field in query[0].split(",")]
    return {
        "gpu": gpu,
        "reported_memory_mib": int(memory),
        "driver": driver,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": str(torch.version.cuda),
        "ultralytics": ultralytics.__version__,
    }


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _seed_evaluation() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes.split(2, dim=-1)
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _area_bucket(boxes: torch.Tensor, *, image_size: int) -> torch.Tensor:
    area_pixels = boxes[:, 2:].clamp_min(0).prod(-1) * float(image_size**2)
    return torch.where(
        area_pixels < float(16**2),
        torch.zeros_like(area_pixels, dtype=torch.long),
        torch.where(
            area_pixels < float(32**2),
            torch.ones_like(area_pixels, dtype=torch.long),
            torch.full_like(area_pixels, 2, dtype=torch.long),
        ),
    )


def _validate_record(record: Mapping[str, torch.Tensor], *, prediction: bool) -> None:
    required = {"boxes", "classes"} | ({"scores"} if prediction else set())
    if set(record) != required:
        raise ValueError("stock evaluation record schema mismatch")
    if record["boxes"].ndim != 2 or record["boxes"].shape[-1] != 4:
        raise ValueError("stock evaluation boxes must have shape [N,4]")
    if record["classes"].shape != (len(record["boxes"]),):
        raise ValueError("stock evaluation classes must have shape [N]")
    if prediction and record["scores"].shape != (len(record["boxes"]),):
        raise ValueError("stock evaluation scores must have shape [N]")


def _match_predictions(
    prediction_boxes: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> np.ndarray:
    correct = np.zeros((prediction_boxes.shape[0], len(IOU_THRESHOLDS)), dtype=bool)
    if prediction_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        return correct
    iou = box_iou(_cxcywh_to_xyxy(target_boxes), _cxcywh_to_xyxy(prediction_boxes))
    iou = (iou * (target_classes[:, None] == prediction_classes[None, :])).cpu().numpy()
    for column, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if not matches.shape[0]:
            continue
        if matches.shape[0] > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), column] = True
    return correct


def _summarize_ap(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    bucket: int | None,
) -> dict[str, float]:
    true_positives: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    predicted_classes: list[np.ndarray] = []
    target_classes: list[np.ndarray] = []
    for prediction, target in zip(predictions, targets):
        _validate_record(prediction, prediction=True)
        _validate_record(target, prediction=False)
        pred_boxes = prediction["boxes"].detach().float().cpu()
        pred_scores = prediction["scores"].detach().float().cpu()
        pred_classes = prediction["classes"].detach().long().cpu()
        gt_boxes = target["boxes"].detach().float().cpu()
        gt_classes = target["classes"].detach().long().cpu()
        if bucket is not None:
            pred_mask = _area_bucket(pred_boxes, image_size=640) == bucket
            gt_mask = _area_bucket(gt_boxes, image_size=640) == bucket
            pred_boxes, pred_scores, pred_classes = (
                pred_boxes[pred_mask],
                pred_scores[pred_mask],
                pred_classes[pred_mask],
            )
            gt_boxes, gt_classes = gt_boxes[gt_mask], gt_classes[gt_mask]
        true_positives.append(
            _match_predictions(pred_boxes, pred_classes, gt_boxes, gt_classes)
        )
        confidences.append(pred_scores.numpy())
        predicted_classes.append(pred_classes.numpy())
        target_classes.append(gt_classes.numpy())
    tp = np.concatenate(true_positives) if true_positives else np.zeros((0, 10), bool)
    conf = np.concatenate(confidences) if confidences else np.zeros(0, np.float32)
    pred_cls = np.concatenate(predicted_classes) if predicted_classes else np.zeros(0, np.int64)
    target_cls = np.concatenate(target_classes) if target_classes else np.zeros(0, np.int64)
    if target_cls.size == 0 or pred_cls.size == 0:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0}
    result = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    precision, recall, ap = result[2], result[3], result[5]
    if ap.size == 0:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "map": float(ap.mean()),
        "ap50": float(ap[:, 0].mean()),
        "ap75": float(ap[:, 5].mean()),
        "precision": float(precision.mean()) if precision.size else 0.0,
        "recall": float(recall.mean()) if recall.size else 0.0,
    }


def compute_detection_metrics(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, float]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("stock predictions and targets must have equal nonzero counts")
    full = _summarize_ap(predictions, targets, bucket=None)
    tiny = _summarize_ap(predictions, targets, bucket=0)
    small = _summarize_ap(predictions, targets, bucket=1)
    metrics = {
        "map": full["map"],
        "ap50": full["ap50"],
        "ap75": full["ap75"],
        "ap_tiny": tiny["map"],
        "ap_small": small["map"],
        "precision": full["precision"],
        "recall": full["recall"],
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite IBER-BE stock metrics")
    return metrics


def _build_validation_loader(
    dataset_root: Path,
    baseline_checkpoint: Path,
    device: torch.device,
    *,
    save_dir: Path,
):
    from ultralytics.models.rtdetr.val import RTDETRValidator

    data = {
        "path": str(dataset_root.resolve()),
        "train": str((dataset_root / "images" / "train").resolve()),
        "val": str((dataset_root / "images" / "val").resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": len(CATEGORY_NAMES),
        "channels": 3,
    }
    validator = RTDETRValidator(
        save_dir=save_dir,
        args={
            "model": str(baseline_checkpoint.resolve()),
            "data": data,
            "task": "detect",
            "mode": "val",
            "split": "val",
            "imgsz": EVALUATION_CONSTANTS["imgsz"],
            "batch": EVALUATION_CONSTANTS["batch"],
            "workers": EVALUATION_CONSTANTS["workers"],
            "device": "0",
            "max_det": EVALUATION_CONSTANTS["max_det"],
            "nms": EVALUATION_CONSTANTS["nms"],
            "cache": False,
            "conf": EVALUATION_CONSTANTS["conf"],
            "half": EVALUATION_CONSTANTS["half"],
            "rect": False,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
        }
    )
    validator.data = data
    validator.device = device
    loader = validator.get_dataloader(data["val"], EVALUATION_CONSTANTS["batch"])
    if len(loader.dataset) != 548:
        raise ValueError(f"IBER-BE validation image count mismatch: {len(loader.dataset)}")
    return loader, validator


def _batch_targets(batch: dict[str, Any], image_index: int) -> dict[str, torch.Tensor]:
    mask = batch["batch_idx"].view(-1).long() == image_index
    return {
        "boxes": batch["bboxes"][mask].detach().float().cpu(),
        "classes": batch["cls"][mask].view(-1).detach().long().cpu(),
    }


def _batch_predictions(
    postprocessed: torch.Tensor, image_index: int
) -> dict[str, torch.Tensor]:
    prediction = postprocessed[image_index].detach().float().cpu()
    prediction = prediction[prediction[:, 4] > EVALUATION_CONSTANTS["conf"]]
    return {
        "boxes": prediction[:, :4],
        "scores": prediction[:, 4],
        "classes": prediction[:, 5].long(),
    }


def _evaluate_stock_once(
    adapter: FrozenIBERAdapter,
    loader: Any,
    validator: Any,
    *,
    device: torch.device,
) -> dict[str, float]:
    _seed_evaluation()
    adapter.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = validator.preprocess(raw_batch)
            output = adapter.forward_evidence(batch["img"])
            if not torch.equal(output.stock_boxes, output.refined_boxes):
                raise RuntimeError("zero-initialized IBER-BE changed stock boxes")
            postprocessed = adapter.detector.model[-1].postprocess(
                output.stock_boxes, output.stock_scores.sigmoid()
            )
            for image_index in range(batch["img"].shape[0]):
                predictions.append(_batch_predictions(postprocessed, image_index))
                targets.append(_batch_targets(batch, image_index))
    if len(targets) != 548:
        raise RuntimeError(
            f"IBER-BE stock authority processed {len(targets)} images instead of 548"
        )
    return compute_detection_metrics(predictions, targets)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline = args.baseline_checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    from ultralytics import RTDETR

    device = torch.device("cuda:0")
    detector = RTDETR(str(baseline)).model.to(device).eval()
    detector.requires_grad_(False)
    with FrozenIBERAdapter.from_detector(
        detector,
        private_seed=10_000,
        probe="b3",
        image_size=EVALUATION_CONSTANTS["imgsz"],
        rho=0.05,
    ).to(device).eval() as adapter:
        loader, validator = _build_validation_loader(
            dataset_root,
            baseline,
            device,
            save_dir=args.output.resolve().parent / "stock-validator",
        )
        repeats = [
            _evaluate_stock_once(adapter, loader, validator, device=device)
            for _ in range(EVALUATION_CONSTANTS["repeats"])
        ]
    report = build_stock_authority_report(
        repeats=repeats,
        baseline_path=baseline,
        baseline_bytes=baseline.stat().st_size,
        baseline_sha256=file_sha256(baseline),
        dataset_sha256=str(dataset_signature(dataset_root)["sha256"]),
        category_sha256=category_mapping_sha256(CATEGORY_NAMES),
        execution_environment=current_execution_environment(),
        source_commit=_source_commit(),
    )
    write_immutable_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

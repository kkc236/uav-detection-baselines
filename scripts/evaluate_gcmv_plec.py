#!/usr/bin/env python3
"""Paired source-frame evaluation for a GCMV PLEC method/control screen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

from src.gcmv_data import GCMVRTDETRDataset
from src.rtdetr_gcmv_plec import GCMVPLECDetectionModel
from src.sbr_artifacts import (
    atomic_write_json,
    environment_info,
    load_dataset,
    sha256_file,
)
from src.sbr_geometry import LetterboxTransform, inverse_letterbox_xyxy
from src.sbr_metrics import evaluate_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"
CONFIDENCE_THRESHOLD = 0.001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate matched GCMV PLEC and stock checkpoints."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--method-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    return parser


def decode_source_predictions(
    prediction: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    imgsz: int = 640,
) -> dict[str, list]:
    """Decode normalized RT-DETR xywh predictions into source-frame xyxy."""

    if prediction.ndim != 2 or prediction.shape[1] != 6:
        raise ValueError("prediction must have shape [N,6]")
    if source_height <= 0 or source_width <= 0 or imgsz <= 0:
        raise ValueError("source dimensions and imgsz must be positive")
    values = prediction.detach().float().cpu().numpy()
    finite = np.isfinite(values).all(axis=1)
    keep = finite & (values[:, 4] >= CONFIDENCE_THRESHOLD)
    query_indices = np.flatnonzero(keep)
    values = values[keep]
    if not len(values):
        return {
            "pred_boxes": [],
            "pred_scores": [],
            "pred_classes": [],
            "pred_source": [],
            "pred_query": [],
        }
    center_x, center_y, width, height = values[:, :4].T
    network_boxes = np.column_stack(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
    )
    transform = LetterboxTransform.from_view(
        width=source_width,
        height=source_height,
        imgsz=imgsz,
    )
    source_boxes = inverse_letterbox_xyxy(
        network_boxes,
        transform,
        normalized=True,
    )
    source_boxes[:, [0, 2]] = np.clip(
        source_boxes[:, [0, 2]], 0.0, float(source_width)
    )
    source_boxes[:, [1, 3]] = np.clip(
        source_boxes[:, [1, 3]], 0.0, float(source_height)
    )
    nonempty = (
        (source_boxes[:, 2] > source_boxes[:, 0])
        & (source_boxes[:, 3] > source_boxes[:, 1])
    )
    values = values[nonempty]
    source_boxes = source_boxes[nonempty]
    query_indices = query_indices[nonempty]
    return {
        "pred_boxes": source_boxes.tolist(),
        "pred_scores": values[:, 4].astype(float).tolist(),
        "pred_classes": values[:, 5].astype(int).tolist(),
        "pred_source": [0] * len(values),
        "pred_query": query_indices.astype(int).tolist(),
    }


def metric_deltas(
    control: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, float]:
    """Return common scalar method-minus-control metric deltas."""

    deltas = {}
    for key in sorted(set(control) & set(method)):
        left, right = control[key], method[key]
        if (
            isinstance(left, (int, float, np.number))
            and not isinstance(left, (bool, np.bool_))
            and isinstance(right, (int, float, np.number))
            and not isinstance(right, (bool, np.bool_))
        ):
            delta = float(right) - float(left)
            if math.isfinite(delta):
                deltas[key] = delta
    return deltas


def jsonable(value: Any) -> Any:
    """Convert evaluator output to canonical-JSON-compatible primitives."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_dataset(
    data: dict,
    *,
    batch: int,
) -> GCMVRTDETRDataset:
    hyp = get_cfg(
        overrides={
            "imgsz": 640,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "mask_ratio": 4,
            "overlap_mask": True,
            "bgr": 0.0,
        }
    )
    return GCMVRTDETRDataset(
        img_path=data["val"],
        imgsz=640,
        local_imgsz=1088,
        batch_size=batch,
        augment=False,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        prefix="paired-eval: ",
        classes=None,
        data=data,
        fraction=1.0,
    )


def _load_model(
    *,
    model_path: str,
    checkpoint_path: str,
    data: dict,
    device: torch.device,
    method: bool,
) -> tuple[GCMVPLECDetectionModel, dict[str, Any]]:
    checkpoint = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint must contain a mapping")
    weights = payload.get("ema") or payload.get("model")
    if not isinstance(weights, torch.nn.Module):
        raise RuntimeError("checkpoint has no model or EMA module")
    model = GCMVPLECDetectionModel(
        model_path,
        nc=int(data["nc"]),
        ch=int(data["channels"]),
        verbose=False,
    )
    model.load(weights)
    model.gcmv_enabled = method
    model.eval().to(device)
    summary = {
        "path": checkpoint.as_posix(),
        "sha256": sha256_file(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "gamma_ref": float(
            model.reference_adapter.gamma_ref.detach().float().cpu().item()
        ),
    }
    del payload, weights
    return model, summary


def _prediction_tensor(output: Any) -> torch.Tensor:
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.ndim != 3
        or prediction.shape[-1] != 6
    ):
        raise RuntimeError("unexpected RT-DETR inference output")
    return prediction


def _metric_row(
    image: Mapping[str, Any],
    decoded: Mapping[str, list],
) -> dict[str, Any]:
    return {
        "image_id": image["relative_path"],
        "width": int(image["width"]),
        "height": int(image["height"]),
        **decoded,
        "gt_boxes": [list(box) for box in image["gt_boxes"]],
        "gt_classes": [int(value) for value in image["gt_classes"]],
        "ignore_boxes": [list(box) for box in image["ignore_boxes"]],
        "effective_gain": min(
            640.0 / float(image["width"]),
            640.0 / float(image["height"]),
            1.0,
        ),
    }


def _run_arm(
    *,
    arm: str,
    checkpoint: str,
    model_path: str,
    data: dict,
    dataset: GCMVRTDETRDataset,
    authority: Mapping[str, Any],
    batch: int,
    workers: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    is_method = arm == "method"
    model, checkpoint_summary = _load_model(
        model_path=model_path,
        checkpoint_path=checkpoint,
        data=data,
        device=device,
        method=is_method,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=dataset.collate_fn,
    )
    image_by_path = {
        str(Path(image["path"]).resolve()): image
        for image in authority["images"]
    }
    rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            image = raw_batch["img"].to(
                device, non_blocking=device.type == "cuda"
            ).float() / 255
            local_views = None
            source_shapes = raw_batch["source_shape"]
            if is_method:
                local_views = raw_batch["local_views"].to(
                    device, non_blocking=True
                ).float() / 255
                source_shapes = source_shapes.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model.predict(
                    image,
                    local_views=local_views,
                    source_shapes=source_shapes if is_method else None,
                )
            predictions = _prediction_tensor(output)
            for index, image_path in enumerate(raw_batch["im_file"]):
                source_height, source_width = (
                    int(value)
                    for value in raw_batch["source_shape"][index].tolist()
                )
                decoded = decode_source_predictions(
                    predictions[index],
                    source_height=source_height,
                    source_width=source_width,
                )
                key = str(Path(image_path).resolve())
                if key not in image_by_path:
                    raise RuntimeError(f"dataset authority mismatch: {key}")
                rows.append(_metric_row(image_by_path[key], decoded))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    runtime = {
        "elapsed_seconds": elapsed,
        "milliseconds_per_image": 1000.0 * elapsed / max(len(rows), 1),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "checkpoint": checkpoint_summary,
    }
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, runtime


def evaluate(args: argparse.Namespace) -> Path:
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("batch must be positive and workers non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("paired GCMV PLEC evaluation requires CUDA")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(f"cuda:{args.device}")
    data = check_det_dataset(args.data, autodownload=False)
    authority = load_dataset(args.data, split="val")
    dataset = _build_dataset(data, batch=args.batch)
    if len(dataset) != int(authority["image_count"]):
        raise RuntimeError("validation dataset count does not match authority")

    rows: dict[str, list[dict[str, Any]]] = {}
    runtime: dict[str, Any] = {}
    for arm, checkpoint in (
        ("control", args.control_checkpoint),
        ("method", args.method_checkpoint),
    ):
        rows[arm], runtime[arm] = _run_arm(
            arm=arm,
            checkpoint=checkpoint,
            model_path=args.model,
            data=data,
            dataset=dataset,
            authority=authority,
            batch=args.batch,
            workers=args.workers,
            device=device,
        )
    metrics = {
        arm: evaluate_dataset(arm_rows)
        for arm, arm_rows in rows.items()
    }
    result = {
        "schema_version": "gcmv-plec-paired-screen/v1",
        "data": {
            "yaml": str(Path(args.data).resolve()),
            "yaml_sha256": authority["yaml_hash"],
            "dataset_signature": authority["dataset_signature"],
            "image_count": authority["image_count"],
        },
        "metrics": metrics,
        "deltas_method_minus_control": metric_deltas(
            metrics["control"], metrics["method"]
        ),
        "runtime": runtime,
        "environment": environment_info(),
    }
    atomic_write_json(output, jsonable(result))
    return output


def main() -> None:
    args = build_parser().parse_args()
    print(evaluate(args))


if __name__ == "__main__":
    main()

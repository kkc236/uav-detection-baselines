"""Pure helpers for live ACR-EG checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import gc
from hashlib import sha256
from pathlib import Path
import time
from typing import Any

import torch


LIVE_EVALUATION_SCHEMA = "gcte-acr-eg-live-evaluation/v1"
LIVE_ENDPOINT = "live-global-plus-four-local-views"


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def prediction_tensor_to_sbr_row(
    predictions: torch.Tensor,
    image: Mapping[str, Any],
    *,
    source: int = 0,
    network_to_source: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Convert normalized RT-DETR `[xywh,score,class]` to one SBR row."""

    if predictions.ndim != 2 or predictions.shape[1] != 6:
        raise ValueError("predictions must have shape [N,6]")
    values = predictions.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("predictions must be finite")
    width = int(image["width"])
    height = int(image["height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scores = values[:, 4]
    classes = values[:, 5]
    if bool(((scores < 0.0) | (scores > 1.0)).any()):
        raise ValueError("prediction scores must be in [0,1]")
    if bool((classes < 0.0).any()) or not bool(torch.equal(classes, classes.round())):
        raise ValueError("prediction classes must be non-negative integers")

    xywh = values[:, :4]
    half = xywh[:, 2:] / 2.0
    xyxy = torch.cat((xywh[:, :2] - half, xywh[:, :2] + half), dim=1)
    if network_to_source is None:
        scale = torch.tensor([width, height, width, height], dtype=xyxy.dtype)
        xyxy = xyxy * scale
    else:
        matrix = network_to_source.detach().to(device="cpu", dtype=xyxy.dtype)
        if matrix.shape != (3, 3) or not bool(torch.isfinite(matrix).all()):
            raise ValueError("network_to_source must be a finite 3x3 matrix")
        xyxy = xyxy * 640.0
        x1, y1, x2, y2 = xyxy.unbind(dim=1)
        corners = torch.stack(
            (
                torch.stack((x1, y1), dim=1),
                torch.stack((x2, y1), dim=1),
                torch.stack((x1, y2), dim=1),
                torch.stack((x2, y2), dim=1),
            ),
            dim=1,
        )
        homogeneous = torch.cat(
            (corners, torch.ones((*corners.shape[:2], 1), dtype=xyxy.dtype)),
            dim=2,
        )
        mapped = homogeneous @ matrix.T
        divisor = mapped[..., 2:3]
        if bool((divisor.abs() <= torch.finfo(mapped.dtype).eps).any()):
            raise ValueError("network_to_source maps a corner to infinity")
        mapped = mapped[..., :2] / divisor
        minimum = mapped.amin(dim=1)
        maximum = mapped.amax(dim=1)
        xyxy = torch.cat((minimum, maximum), dim=1)
    xyxy[:, 0] = xyxy[:, 0].clamp(0.0, float(width))
    xyxy[:, 2] = xyxy[:, 2].clamp(0.0, float(width))
    xyxy[:, 1] = xyxy[:, 1].clamp(0.0, float(height))
    xyxy[:, 3] = xyxy[:, 3].clamp(0.0, float(height))

    count = int(values.shape[0])
    return {
        "image_id": str(image["relative_path"]),
        "width": width,
        "height": height,
        "pred_boxes": xyxy.tolist(),
        "pred_scores": scores.tolist(),
        "pred_classes": classes.to(dtype=torch.long).tolist(),
        "pred_source": [int(source)] * count,
        "pred_query": list(range(count)),
        "gt_boxes": [list(value) for value in image["gt_boxes"]],
        "gt_classes": [int(value) for value in image["gt_classes"]],
        "ignore_boxes": [list(value) for value in image["ignore_boxes"]],
        "effective_gain": min(640.0 / width, 640.0 / height, 1.0),
    }


def numeric_deltas(
    baseline: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, float]:
    """Return method-minus-baseline deltas for shared real-valued metrics."""

    result: dict[str, float] = {}
    for name in sorted(set(baseline) & set(method)):
        left = baseline[name]
        right = method[name]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[name] = float(right) - float(left)
    return result


def build_result(
    *,
    baseline_metrics: Mapping[str, Any],
    method_metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    baseline: Mapping[str, Any],
    dataset: Mapping[str, Any],
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical top-level live evaluation result."""

    deltas = numeric_deltas(baseline_metrics, method_metrics)
    return {
        "schema_version": LIVE_EVALUATION_SCHEMA,
        "endpoint": LIVE_ENDPOINT,
        "checkpoint": dict(checkpoint),
        "baseline": dict(baseline),
        "dataset": dict(dataset),
        "runtime": dict(runtime),
        "source": dict(source),
        "metrics": {
            "Baseline": dict(baseline_metrics),
            "ACR-EG": dict(method_metrics),
        },
        "deltas": deltas,
        "decision": {
            "exceeds_baseline_mAP": deltas.get("mAP50-95", float("-inf")) > 0.0,
            "tiny_improves": deltas.get("AP-tiny-SBR", float("-inf")) > 0.0,
        },
    }


def _prediction_tensor(value: object) -> torch.Tensor:
    candidate = value[0] if isinstance(value, tuple) and value else value
    if not isinstance(candidate, torch.Tensor):
        raise RuntimeError("ACR_EG_LIVE_PREDICTION_OUTPUT_DRIFT")
    if candidate.ndim != 3 or candidate.shape[0] != 1 or candidate.shape[2] != 6:
        raise RuntimeError("ACR_EG_LIVE_PREDICTION_LAYOUT_DRIFT")
    if candidate.shape[1] != 300:
        raise RuntimeError("ACR_EG_LIVE_MAX_DET_DRIFT")
    return candidate[0]


def _load_checkpoint_model(
    path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
    integrated: bool,
    expected_epoch: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    from src.rtdetr_acr_eg import (
        ACREGDetectionModel,
        ACR_EG_EXTRA_PREFIX,
        ACR_EG_STATE_KEY_COUNT,
    )

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != expected_sha256.upper():
        raise ValueError(
            f"ACR_EG_LIVE_CHECKPOINT_SHA_MISMATCH expected={expected_sha256.upper()} actual={observed}"
        )
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("ACR_EG_LIVE_CHECKPOINT_NOT_MAPPING")
    model = payload.get("ema") or payload.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("ACR_EG_LIVE_CHECKPOINT_HAS_NO_MODEL")
    epoch = payload.get("epoch")
    if integrated:
        if type(model) is not ACREGDetectionModel:
            raise ValueError("ACR_EG_LIVE_MODEL_IDENTITY_MISMATCH")
        acr_keys = {
            name for name in model.state_dict() if name.startswith(ACR_EG_EXTRA_PREFIX)
        }
        if len(acr_keys) != ACR_EG_STATE_KEY_COUNT:
            raise ValueError("ACR_EG_LIVE_STATE_IDENTITY_MISMATCH")
        if epoch != expected_epoch:
            raise ValueError("ACR_EG_LIVE_EPOCH_MISMATCH")
    elif isinstance(model, ACREGDetectionModel):
        raise ValueError("ACR_EG_LIVE_BASELINE_IS_INTEGRATED")
    model = model.float().to(device)
    model.requires_grad_(False)
    model.eval()
    return model, {
        "path": resolved.as_posix(),
        "sha256": observed,
        "epoch": epoch,
        "model_type": type(model).__name__,
    }


def _run_arm(
    *,
    model: torch.nn.Module,
    dataset: object,
    images: list[Mapping[str, Any]],
    device: torch.device,
    workers: int,
    amp: bool,
    paired: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    if len(dataset) != len(images):
        raise RuntimeError("ACR_EG_LIVE_IMAGE_COUNT_DRIFT")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        image = images[index]
        observed_path = Path(str(batch["im_file"][0])).resolve()
        if observed_path != Path(image["path"]).resolve():
            raise RuntimeError("ACR_EG_LIVE_IMAGE_ORDER_DRIFT")
        global_image = batch["img"].to(device, non_blocking=True).float() / 255.0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp,
        ):
            if paired:
                local_views = (
                    batch["local_views"].to(device, non_blocking=True).float() / 255.0
                )
                source_shapes = batch["source_shape"].to(device, non_blocking=True)
                predictions = model.predict(
                    global_image,
                    local_views=local_views,
                    source_shapes=source_shapes,
                )
                if getattr(model, "last_acr_eg_output", None) is None:
                    raise RuntimeError("ACR_EG_LIVE_SILENT_STOCK_FALLBACK")
            else:
                predictions = model.predict(global_image)
        tensor = _prediction_tensor(predictions)
        rows.append(
            prediction_tensor_to_sbr_row(
                tensor,
                image,
                network_to_source=batch["global_to_source"][0],
            )
        )
        if (index + 1) % 25 == 0 or index + 1 == len(images):
            arm = "ACR-EG" if paired else "Baseline"
            print(f"ACR_EG_LIVE_PROGRESS arm={arm} {index + 1}/{len(images)}", flush=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return rows, {
        "seconds": seconds,
        "milliseconds_per_image": 1000.0 * seconds / len(images),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def run_live_evaluation(args: Any) -> Path:
    """Execute the sealed baseline-versus-live-ACR-EG evaluation."""

    import ultralytics
    from ultralytics.data.utils import check_det_dataset

    from src.gcte_data import build_gcqf_dataset
    from src.sbr_artifacts import (
        atomic_write_json,
        atomic_write_jsonl_gz,
        ensure_empty_output,
        git_provenance,
        load_dataset,
        write_checksums,
    )
    from src.sbr_metrics import evaluate_dataset

    if ultralytics.__version__ != "8.4.90":
        raise RuntimeError("ACR_EG_LIVE_ULTRALYTICS_VERSION_DRIFT")
    if not torch.cuda.is_available() or str(args.device) != "0":
        raise RuntimeError("ACR_EG_LIVE_REQUIRES_CUDA_DEVICE_0")
    device = torch.device("cuda:0")
    data_path = Path(args.data).resolve()
    dataset_manifest = load_dataset(data_path, split="val")
    if dataset_manifest["image_count"] != args.expected_records:
        raise RuntimeError("ACR_EG_LIVE_DATASET_COUNT_DRIFT")
    if dataset_manifest["dataset_signature"].upper() != args.dataset_signature.upper():
        raise RuntimeError("ACR_EG_LIVE_DATASET_SIGNATURE_DRIFT")
    images = list(dataset_manifest["images"])
    if args.limit is not None:
        images = images[: args.limit]
    checked_data = check_det_dataset(str(data_path), autodownload=False)
    dataset = build_gcqf_dataset(checked_data, split="val", batch_size=1)
    if args.limit is not None:
        dataset.im_files = dataset.im_files[: args.limit]
        dataset.labels = dataset.labels[: args.limit]
        dataset.npy_files = dataset.npy_files[: args.limit]
        dataset.ni = args.limit

    output = ensure_empty_output(Path(args.output).resolve())
    baseline_model, baseline_info = _load_checkpoint_model(
        Path(args.baseline_checkpoint),
        expected_sha256=args.expected_baseline_sha256,
        device=device,
        integrated=False,
    )
    baseline_rows, baseline_runtime = _run_arm(
        model=baseline_model,
        dataset=dataset,
        images=images,
        device=device,
        workers=args.workers,
        amp=args.amp,
        paired=False,
    )
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    method_model, checkpoint_info = _load_checkpoint_model(
        Path(args.checkpoint),
        expected_sha256=args.expected_checkpoint_sha256,
        device=device,
        integrated=True,
        expected_epoch=args.expected_epoch,
    )
    method_rows, method_runtime = _run_arm(
        model=method_model,
        dataset=dataset,
        images=images,
        device=device,
        workers=args.workers,
        amp=args.amp,
        paired=True,
    )
    gate = getattr(method_model, "last_acr_eg_output", None)
    if gate is None:
        raise RuntimeError("ACR_EG_LIVE_GATE_OUTPUT_MISSING")

    baseline_metrics = evaluate_dataset(baseline_rows)
    method_metrics = evaluate_dataset(method_rows)
    baseline_predictions = output / "predictions-baseline.jsonl.gz"
    method_predictions = output / "predictions-acr-eg.jsonl.gz"
    atomic_write_jsonl_gz(baseline_predictions, baseline_rows)
    atomic_write_jsonl_gz(method_predictions, method_rows)
    runtime = {
        "device": torch.cuda.get_device_name(device),
        "amp": bool(args.amp),
        "views": {"Baseline": 1, "ACR-EG": 5},
        "Baseline": baseline_runtime,
        "ACR-EG": method_runtime,
    }
    result = build_result(
        baseline_metrics=baseline_metrics,
        method_metrics=method_metrics,
        checkpoint=checkpoint_info,
        baseline=baseline_info,
        dataset={
            "yaml": data_path.as_posix(),
            "yaml_sha256": sha256_file(data_path),
            "image_count": len(images),
            "full_image_count": dataset_manifest["image_count"],
            "signature": dataset_manifest["dataset_signature"].upper(),
        },
        runtime=runtime,
        source=git_provenance(Path(__file__).resolve().parents[1]),
    )
    evaluation = output / "evaluation.json"
    atomic_write_json(evaluation, _stringify_mapping_keys(result))
    write_checksums(
        output / "checksums.sha256",
        (evaluation, baseline_predictions, method_predictions),
        root=output,
    )
    print(
        "ACR_EG_LIVE_EVALUATION_COMPLETE "
        f"map_delta={result['deltas'].get('mAP50-95')} output={evaluation}",
        flush=True,
    )
    return evaluation


__all__ = [
    "LIVE_ENDPOINT",
    "LIVE_EVALUATION_SCHEMA",
    "build_result",
    "numeric_deltas",
    "prediction_tensor_to_sbr_row",
    "run_live_evaluation",
    "sha256_file",
]

"""Run the frozen RT-DETR quality-reordering upper-bound oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_evaluation import compute_detection_metrics  # noqa: E402
from src.iber_protocol import (  # noqa: E402
    EXECUTION_ENVIRONMENT,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    file_sha256,
    select_hashed_subset,
    state_fingerprint,
    subset_signature,
)
from src.rtdetr_quality_oracle import (  # noqa: E402
    ALPHA_GRID,
    DEV_COUNT,
    EXPECTED_DEV_SHA256,
    decide_quality_oracle,
    flattened_topk,
    oracle_topk,
    ordered_path_sha256,
    same_class_iou_quality,
    select_alpha,
    select_internal_dev,
)


IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 8
CONFIDENCE = 0.001
MAX_DET = 300
NMS = False
TRAIN_COUNT = 647
VAL_COUNT = 548
NUM_CLASSES = 10
STOCK_AUTHORITY = {
    "map": 0.24164844987309864,
    "ap50": 0.4143946635382976,
    "ap75": 0.23916375458831637,
    "ap_tiny": 0.10314861659739166,
    "ap_small": 0.24166148504350557,
    "precision": 0.5119369275291381,
    "recall": 0.43525461908044843,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    return file_sha256(Path(path))


def _source_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("source commit must be exactly 40 hexadecimal characters")
    return commit


def _execution_environment() -> dict[str, Any]:
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
        raise RuntimeError("quality oracle requires exactly one visible GPU")
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


def _schema_sha256() -> str:
    payload = {
        "identity": "rtdetr-quality-reranking-oracle-v1",
        "image_size": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "workers": WORKERS,
        "confidence": CONFIDENCE,
        "max_det": MAX_DET,
        "nms": NMS,
        "train_count": TRAIN_COUNT,
        "dev_count": DEV_COUNT,
        "val_count": VAL_COUNT,
        "classes": list(CATEGORY_NAMES),
        "category_sha256": category_mapping_sha256(CATEGORY_NAMES),
        "alpha_grid": list(ALPHA_GRID),
        "stock_authority": STOCK_AUTHORITY,
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def _dev_list_path(cache_root: Path) -> Path:
    root = Path(cache_root).resolve()
    return root.parent / f"{root.name}-internal-dev.txt"


def _write_image_list_create_only(path: Path, paths: Sequence[Path]) -> None:
    payload = "".join(f"{Path(item).resolve()}\n" for item in paths).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"immutable image list differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _prepare_internal_dev(dataset_root: Path, cache_root: Path) -> tuple[Path, ...]:
    root = Path(dataset_root).resolve()
    all_train = sorted((root / "images" / "train").glob("*.jpg"))
    if len(all_train) != 6471:
        raise RuntimeError(f"training image count mismatch: {len(all_train)}")
    authorized = select_hashed_subset(all_train, root=root, fraction=0.10)
    actual_subset_sha = subset_signature(authorized, root=root)
    if len(authorized) != TRAIN_COUNT or actual_subset_sha != EXPECTED_SUBSET_SHA256:
        raise RuntimeError(
            "fixed-subset authority mismatch: "
            f"count={len(authorized)}, sha256={actual_subset_sha}"
        )
    selected = select_internal_dev(authorized, root=root)
    actual_dev_sha = ordered_path_sha256(selected, root=root)
    if actual_dev_sha != EXPECTED_DEV_SHA256:
        raise RuntimeError(f"internal-dev authority mismatch: {actual_dev_sha}")
    _write_image_list_create_only(_dev_list_path(cache_root), selected)
    return selected


def _build_pre_alpha_authority(
    baseline_checkpoint: Path,
    dataset_root: Path,
    dev_paths: Sequence[Path],
) -> dict[str, str]:
    baseline = Path(baseline_checkpoint).resolve()
    root = Path(dataset_root).resolve()
    baseline_sha = _file_sha256(baseline)
    subset_paths = select_hashed_subset(
        sorted((root / "images" / "train").glob("*.jpg")),
        root=root,
        fraction=0.10,
    )
    subset_sha = subset_signature(subset_paths, root=root)
    dev_sha = ordered_path_sha256(dev_paths, root=root)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"baseline authority mismatch: {baseline_sha}")
    if subset_sha != EXPECTED_SUBSET_SHA256:
        raise RuntimeError(f"subset authority mismatch: {subset_sha}")
    if dev_sha != EXPECTED_DEV_SHA256:
        raise RuntimeError(f"internal-dev authority mismatch: {dev_sha}")
    environment = _execution_environment()
    expected_environment = dict(EXECUTION_ENVIRONMENT)
    if environment != expected_environment:
        raise RuntimeError(
            f"execution environment mismatch: expected={expected_environment}, actual={environment}"
        )
    return {
        "baseline_sha256": baseline_sha,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": subset_sha,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "source_commit": _source_commit(),
        "schema_sha256": _schema_sha256(),
        "dev_sha256": dev_sha,
    }


def _assert_full_dataset_authority(dataset_root: Path) -> str:
    dataset_sha = str(dataset_signature(Path(dataset_root).resolve())["sha256"])
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"dataset authority mismatch: {dataset_sha}")
    return dataset_sha


def _device(value: str) -> torch.device:
    if not isinstance(value, str) or value != "0":
        raise ValueError("the frozen quality oracle permits only device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is unavailable")
    return torch.device("cuda:0")


def _assert_cuda0_detector(detector: torch.nn.Module) -> None:
    parameters = tuple(detector.parameters())
    if not parameters or any(parameter.device != torch.device("cuda:0") for parameter in parameters):
        raise RuntimeError("quality oracle detector parameters must be on cuda:0")


def _assert_cuda0_tensor(tensor: torch.Tensor, *, label: str) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.device != torch.device("cuda:0"):
        raise RuntimeError(f"{label} must be on cuda:0")


def _load_detector(checkpoint: Path, device: torch.device):
    from ultralytics import RTDETR

    detector = RTDETR(str(Path(checkpoint).resolve())).model.to(device).eval()
    detector.requires_grad_(False)
    detector.model[-1].export = False
    _assert_cuda0_detector(detector)
    return detector


def _build_validation_loader(
    dataset_root: Path,
    baseline_checkpoint: Path,
    device: torch.device,
    *,
    image_source: Path,
    split_name: str,
    save_dir: Path,
    expected_count: int,
):
    from ultralytics.models.rtdetr.val import RTDETRValidator

    data = {
        "path": str(Path(dataset_root).resolve()),
        "train": str((Path(dataset_root) / "images" / "train").resolve()),
        "val": str(Path(image_source).resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": len(CATEGORY_NAMES),
        "channels": 3,
    }
    validator = RTDETRValidator(
        save_dir=Path(save_dir),
        args={
            "model": str(Path(baseline_checkpoint).resolve()),
            "data": data,
            "task": "detect",
            "mode": "val",
            "split": "val",
            "imgsz": IMAGE_SIZE,
            "batch": BATCH_SIZE,
            "workers": WORKERS,
            "device": "0" if device.type == "cuda" else "cpu",
            "max_det": MAX_DET,
            "nms": NMS,
            "cache": False,
            "conf": CONFIDENCE,
            "half": False,
            "rect": False,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
        },
    )
    validator.data = data
    validator.device = device
    loader = validator.get_dataloader(data["val"], BATCH_SIZE)
    actual = len(loader.dataset)
    if actual != expected_count:
        raise RuntimeError(
            f"{split_name} image count mismatch: expected={expected_count}, actual={actual}"
        )
    return loader, validator


def _state_sha256(detector: torch.nn.Module) -> str:
    return state_fingerprint(detector.state_dict())


def _assert_detector_isolated(detector: torch.nn.Module) -> None:
    if any(parameter.requires_grad for parameter in detector.parameters()):
        raise RuntimeError("quality oracle detector is not frozen")
    if any(parameter.grad is not None for parameter in detector.parameters()):
        raise RuntimeError("quality oracle detector received a gradient")


def _extract_decoder_batch(
    detector: Any,
    images: torch.Tensor,
    *,
    require_cuda_smoke_shape: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    detector.model[-1].export = False
    with torch.inference_mode():
        stock_output, auxiliary = detector.predict(images)
        if not isinstance(auxiliary, tuple) or len(auxiliary) != 5:
            raise RuntimeError("RT-DETR auxiliary decoder tuple is invalid")
        decoder_boxes, decoder_logits, _, _, _ = auxiliary
        boxes = decoder_boxes[-1].detach().float()
        logits = decoder_logits[-1].detach().float()
        reconstructed = detector.model[-1].postprocess(boxes, logits.sigmoid())
    if not torch.equal(reconstructed, stock_output):
        raise RuntimeError("decoder reconstruction differs from stock RT-DETR output")
    expected_batch = BATCH_SIZE if require_cuda_smoke_shape else images.shape[0]
    if boxes.shape != (expected_batch, MAX_DET, 4):
        raise RuntimeError(f"decoder box shape mismatch: {tuple(boxes.shape)}")
    if logits.shape != (expected_batch, MAX_DET, NUM_CLASSES):
        raise RuntimeError(f"decoder logit shape mismatch: {tuple(logits.shape)}")
    if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("decoder evidence contains non-finite values")
    if boxes.requires_grad or logits.requires_grad or stock_output.requires_grad:
        raise RuntimeError("decoder evidence is attached to gradients")
    _assert_detector_isolated(detector)
    return stock_output, boxes, logits


def _batch_target(batch: Mapping[str, Any], image_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["batch_idx"].view(-1).long() == image_index
    boxes = batch["bboxes"][mask].detach().float().cpu().contiguous()
    classes = batch["cls"][mask].view(-1).detach().long().cpu().contiguous()
    return boxes, classes


def _extract_records(
    detector: torch.nn.Module,
    loader: Any,
    validator: Any,
    *,
    device: torch.device,
    expected_count: int,
    run_cuda_smoke: bool,
) -> list[dict[str, Any]]:
    if device != torch.device("cuda:0"):
        raise RuntimeError("quality evidence extraction requires cuda:0")
    _assert_cuda0_detector(detector)
    _assert_detector_isolated(detector)
    state_before = _state_sha256(detector)
    records: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = validator.preprocess(raw_batch)
        images = batch["img"]
        _assert_cuda0_tensor(images, label="preprocessed input")
        _, boxes, logits = _extract_decoder_batch(
            detector,
            images,
            require_cuda_smoke_shape=run_cuda_smoke and batch_index == 0,
        )
        image_ids = batch.get("im_file")
        if not isinstance(image_ids, Sequence) or isinstance(image_ids, (str, bytes)):
            raise RuntimeError("validator batch is missing image identifiers")
        if len(image_ids) != images.shape[0]:
            raise RuntimeError("validator image identifier count mismatch")
        for image_index, image_id in enumerate(image_ids):
            target_boxes, target_classes = _batch_target(batch, image_index)
            records.append(
                {
                    "image_id": Path(str(image_id)).as_posix(),
                    "boxes": boxes[image_index].detach().cpu().contiguous().clone(),
                    "logits": logits[image_index].detach().cpu().contiguous().clone(),
                    "target_boxes": target_boxes.clone(),
                    "target_classes": target_classes.clone(),
                }
            )
    if len(records) != expected_count:
        raise RuntimeError(
            f"quality evidence count mismatch: expected={expected_count}, actual={len(records)}"
        )
    _assert_detector_isolated(detector)
    if _state_sha256(detector) != state_before:
        raise RuntimeError("detector state changed during quality evidence extraction")
    return records


def _prediction_record(postprocessed: torch.Tensor) -> dict[str, torch.Tensor]:
    selected = postprocessed.detach().float().cpu()
    selected = selected[selected[:, 4] > CONFIDENCE]
    return {
        "boxes": selected[:, :4],
        "scores": selected[:, 4],
        "classes": selected[:, 5].long(),
    }


def _evaluate_records(
    records: Sequence[Mapping[str, Any]], *, alphas: Sequence[float]
) -> dict[str, Any]:
    if not records:
        raise ValueError("quality evaluation requires records")
    frozen_alphas = tuple(alphas)
    if not frozen_alphas or any(alpha not in ALPHA_GRID for alpha in frozen_alphas):
        raise ValueError("quality evaluation alpha is outside the frozen grid")
    stock_predictions: list[dict[str, torch.Tensor]] = []
    oracle_predictions = {alpha: [] for alpha in frozen_alphas}
    targets: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for record in records:
            boxes = record["boxes"].detach().float().cpu().unsqueeze(0)
            logits = record["logits"].detach().float().cpu().unsqueeze(0)
            target_boxes = record["target_boxes"].detach().float().cpu()
            target_classes = record["target_classes"].detach().long().cpu()
            stock = flattened_topk(
                boxes, logits.sigmoid(), num_classes=NUM_CLASSES, max_det=MAX_DET
            )[0]
            quality = same_class_iou_quality(
                boxes[0], target_boxes, target_classes, NUM_CLASSES
            ).unsqueeze(0)
            stock_predictions.append(_prediction_record(stock))
            for alpha in frozen_alphas:
                oracle = oracle_topk(
                    boxes,
                    logits,
                    quality,
                    alpha=float(alpha),
                    num_classes=NUM_CLASSES,
                    max_det=MAX_DET,
                )[0]
                oracle_predictions[alpha].append(_prediction_record(oracle))
            targets.append({"boxes": target_boxes, "classes": target_classes})
    return {
        "stock": compute_detection_metrics(
            stock_predictions, targets, image_size=IMAGE_SIZE
        ),
        "oracle": {
            alpha: compute_detection_metrics(
                predictions, targets, image_size=IMAGE_SIZE
            )
            for alpha, predictions in oracle_predictions.items()
        },
    }


def _select_alpha(metrics_by_alpha: Mapping[float, Mapping[str, float]]) -> float:
    return select_alpha(metrics_by_alpha)


def _persist_cache(
    root: Path,
    *,
    dev: list[dict[str, Any]],
    val: list[dict[str, Any]],
    authority: Mapping[str, str],
) -> dict[str, Any]:
    """Keep the evolving Task 2 cache API behind one integration seam."""
    from src.rtdetr_quality_oracle import write_quality_oracle_cache

    return write_quality_oracle_cache(root, dev=dev, val=val, authority=authority)


def _load_cache(
    root: Path, *, authority: Mapping[str, str], manifest_sha256: str
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Keep the verified Task 2 cache loader behind one integration seam."""
    from src.rtdetr_quality_oracle import load_quality_oracle_cache

    return load_quality_oracle_cache(
        root, authority=authority, manifest_sha256=manifest_sha256
    )


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_canonical_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    raw = _canonical_json_bytes(payload)
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _read_canonical_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"immutable report is missing or unsafe: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"immutable report is invalid: {path}") from error
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise RuntimeError(f"immutable report is not canonical: {path}")
    return payload


def _write_or_validate_canonical_json(
    path: Path, payload: Mapping[str, Any]
) -> None:
    path = Path(path)
    raw = _canonical_json_bytes(payload)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise RuntimeError(f"immutable report differs: {path}")
        return
    _write_canonical_json_create_only(path, payload)


_ALPHA_REPORT = "alpha-selection-report.json"
_CACHE_AUTHORITY_REPORT = "cache-manifest-authority.json"
_FINAL_REPORTS = {
    "quality-oracle-report.json",
    "quality-oracle-decision.json",
    "environment-hash-inventory.json",
}


def _validate_report_stage(report_root: Path, cache_root: Path) -> set[str]:
    report_root = Path(report_root)
    cache_root = Path(cache_root)
    if report_root.is_symlink():
        raise RuntimeError("report root must not be a symlink")
    if not report_root.exists():
        report_root.mkdir(parents=True, exist_ok=False)
    if not report_root.is_dir():
        raise RuntimeError("report root must be a directory")
    entries = {path.name for path in report_root.iterdir()}
    allowed = {_ALPHA_REPORT, _CACHE_AUTHORITY_REPORT, *_FINAL_REPORTS}
    if not entries <= allowed:
        raise RuntimeError(f"report stage contains unexpected entries: {sorted(entries - allowed)}")
    if _CACHE_AUTHORITY_REPORT in entries and _ALPHA_REPORT not in entries:
        raise RuntimeError("cache external authority exists without alpha selection")
    if entries & _FINAL_REPORTS and _CACHE_AUTHORITY_REPORT not in entries:
        raise RuntimeError("final reports exist without cache external authority")
    if cache_root.exists() and _CACHE_AUTHORITY_REPORT not in entries:
        raise RuntimeError("cache exists without external authority")
    if _CACHE_AUTHORITY_REPORT in entries and not cache_root.exists():
        raise RuntimeError("cache external authority exists without cache")
    return entries


def _relative_paths(paths: Sequence[Path], *, root: Path) -> list[str]:
    resolved_root = Path(root).resolve()
    return [Path(path).resolve().relative_to(resolved_root).as_posix() for path in paths]


def _bind_record_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_paths: Sequence[Path],
    dataset_root: Path,
    split_name: str,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    actual_paths: list[Path] = []
    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str):
            raise RuntimeError(f"{split_name} record image identity is invalid")
        path = Path(image_id)
        actual_paths.append(path if path.is_absolute() else root / path)
    expected = _relative_paths(tuple(expected_paths), root=root)
    actual = _relative_paths(tuple(actual_paths), root=root)
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise RuntimeError(f"{split_name} identity set mismatch")
    return {
        "count": len(actual),
        "actual_loader_order_sha256": ordered_path_sha256(tuple(actual_paths), root=root),
        "actual_loader_image_paths": actual,
    }


def _expected_official_val_paths(dataset_root: Path) -> tuple[Path, ...]:
    paths = tuple(sorted((Path(dataset_root).resolve() / "images" / "val").glob("*.jpg")))
    if len(paths) != VAL_COUNT:
        raise RuntimeError(
            f"official-val image count mismatch: expected={VAL_COUNT}, actual={len(paths)}"
        )
    return paths


def _alpha_report_payload(
    *,
    authority: Mapping[str, str],
    dataset_root: Path,
    selected_paths: Sequence[Path],
    dev_binding: Mapping[str, Any],
    dev_evaluation: Mapping[str, Any],
    selected_alpha: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "authority": dict(authority),
        "split": {
            "count": DEV_COUNT,
            "selection_order_sha256": EXPECTED_DEV_SHA256,
            "selection_order_image_paths": _relative_paths(selected_paths, root=dataset_root),
            "actual_loader_order_sha256": dev_binding["actual_loader_order_sha256"],
            "actual_loader_image_paths": dev_binding["actual_loader_image_paths"],
        },
        "stock": dev_evaluation["stock"],
        "candidates": {
            format(alpha, "g"): dev_evaluation["oracle"][alpha] for alpha in ALPHA_GRID
        },
        "selected_alpha": selected_alpha,
    }


def _validate_alpha_report(
    path: Path,
    *,
    authority: Mapping[str, str],
    dataset_root: Path,
    selected_paths: Sequence[Path],
) -> tuple[float, dict[str, Any]]:
    payload = _read_canonical_json(path)
    if payload.get("format_version") != 1 or payload.get("authority") != dict(authority):
        raise RuntimeError("alpha-selection report authority mismatch")
    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise RuntimeError("alpha-selection split is invalid")
    expected_selection = _relative_paths(selected_paths, root=dataset_root)
    if (
        split.get("count") != DEV_COUNT
        or split.get("selection_order_sha256") != EXPECTED_DEV_SHA256
        or split.get("selection_order_image_paths") != expected_selection
    ):
        raise RuntimeError("alpha-selection split authority mismatch")
    actual_relative = split.get("actual_loader_image_paths")
    if not isinstance(actual_relative, list) or any(not isinstance(item, str) for item in actual_relative):
        raise RuntimeError("alpha-selection actual loader order is invalid")
    binding = _bind_record_identities(
        _records_from_ids(actual_relative),
        expected_paths=selected_paths,
        dataset_root=dataset_root,
        split_name="internal-dev",
    )
    if binding["actual_loader_order_sha256"] != split.get("actual_loader_order_sha256"):
        raise RuntimeError("alpha-selection actual loader order hash mismatch")
    candidates = payload.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != {format(alpha, "g") for alpha in ALPHA_GRID}:
        raise RuntimeError("alpha-selection candidate grid mismatch")
    metrics = {alpha: candidates[format(alpha, "g")] for alpha in ALPHA_GRID}
    selected_alpha = payload.get("selected_alpha")
    if selected_alpha not in ALPHA_GRID or _select_alpha(metrics) != selected_alpha:
        raise RuntimeError("alpha-selection selected alpha mismatch")
    return float(selected_alpha), payload


def _records_from_ids(image_ids: Sequence[str]) -> list[dict[str, str]]:
    return [{"image_id": image_id} for image_id in image_ids]


def _assert_stock_authority(metrics: Mapping[str, float]) -> dict[str, Any]:
    actual = dict(metrics)
    exact_names = ("map", "ap50", "ap75", "ap_tiny", "ap_small")
    diagnostic_names = ("precision", "recall")
    tolerance = Decimal("1e-8")
    if set(actual) != set(STOCK_AUTHORITY) or any(
        actual.get(name) != STOCK_AUTHORITY[name] for name in exact_names
    ):
        raise RuntimeError(
            f"official-validation stock authority mismatch: expected={STOCK_AUTHORITY}, actual={actual}"
        )
    diagnostic_delta = {
        name: float(
            abs(Decimal(str(actual[name])) - Decimal(str(STOCK_AUTHORITY[name])))
        )
        for name in diagnostic_names
    }
    if any(Decimal(str(delta)) > tolerance for delta in diagnostic_delta.values()):
        raise RuntimeError(
            f"official-validation stock authority mismatch: expected={STOCK_AUTHORITY}, actual={actual}"
        )
    amended = any(delta != 0.0 for delta in diagnostic_delta.values())
    return {
        "status": (
            "passed_with_non_gate_float_amendment" if amended else "passed_exact"
        ),
        "exact_gate_metrics": list(exact_names),
        "diagnostic_delta": diagnostic_delta,
        "tolerance": float(tolerance),
    }


def _cache_authority_payload(
    *,
    authority: Mapping[str, str],
    manifest_sha256: str,
    dev_binding: Mapping[str, Any],
    val_binding: Mapping[str, Any],
    detector: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "authority": dict(authority),
        "manifest_sha256": manifest_sha256,
        "splits": {"dev": dict(dev_binding), "val": dict(val_binding)},
        "detector": dict(detector),
    }


def _validate_cache_authority(
    path: Path, *, authority: Mapping[str, str]
) -> dict[str, Any]:
    payload = _read_canonical_json(path)
    if set(payload) != {
        "format_version",
        "authority",
        "manifest_sha256",
        "splits",
        "detector",
    }:
        raise RuntimeError("cache external authority schema mismatch")
    if payload["format_version"] != 1 or payload["authority"] != dict(authority):
        raise RuntimeError("cache external authority mismatch")
    manifest_sha = payload["manifest_sha256"]
    if (
        not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or manifest_sha != manifest_sha.upper()
        or any(character not in "0123456789ABCDEF" for character in manifest_sha)
    ):
        raise RuntimeError("cache external manifest sha256 is invalid")
    if not isinstance(payload["splits"], Mapping) or set(payload["splits"]) != {"dev", "val"}:
        raise RuntimeError("cache external split authority mismatch")
    if not isinstance(payload["detector"], Mapping):
        raise RuntimeError("cache external detector authority mismatch")
    return payload


def _validate_completed_reports(
    report_root: Path,
    *,
    authority: Mapping[str, str],
    selected_alpha: float,
) -> int | None:
    present = {name for name in _FINAL_REPORTS if (Path(report_root) / name).is_file()}
    if present != _FINAL_REPORTS:
        return None
    official = _read_canonical_json(Path(report_root) / "quality-oracle-report.json")
    decision = _read_canonical_json(Path(report_root) / "quality-oracle-decision.json")
    inventory = _read_canonical_json(Path(report_root) / "environment-hash-inventory.json")
    for payload in (official, decision, inventory):
        if payload.get("format_version") != 1 or payload.get("authority") != dict(authority):
            raise RuntimeError("completed report authority mismatch")
    if official.get("selected_alpha") != selected_alpha or decision.get("selected_alpha") != selected_alpha:
        raise RuntimeError("completed report selected alpha mismatch")
    status = decision.get("status")
    if status not in {"passed", "scientific_failed"}:
        raise RuntimeError("completed decision status is invalid")
    return 0


def _run(args: argparse.Namespace) -> int:
    baseline = Path(args.baseline_checkpoint).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    report_root = Path(args.report_root).resolve()
    _validate_report_stage(report_root, cache_root)
    dev_paths = _prepare_internal_dev(dataset_root, cache_root)
    authority = _build_pre_alpha_authority(baseline, dataset_root, dev_paths)
    device = _device(args.device)
    alpha_report_path = report_root / _ALPHA_REPORT
    cache_authority_path = report_root / _CACHE_AUTHORITY_REPORT
    selected_alpha: float | None = None
    alpha_report: dict[str, Any] | None = None
    if alpha_report_path.exists():
        selected_alpha, alpha_report = _validate_alpha_report(
            alpha_report_path,
            authority=authority,
            dataset_root=dataset_root,
            selected_paths=dev_paths,
        )

    dev_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    cache_authority: dict[str, Any]
    if cache_root.exists():
        if alpha_report is None or not cache_authority_path.is_file():
            raise RuntimeError("cache resume requires alpha and external authority")
        _assert_full_dataset_authority(dataset_root)
        val_paths = _expected_official_val_paths(dataset_root)
        cache_authority = _validate_cache_authority(
            cache_authority_path, authority=authority
        )
        loaded = _load_cache(
            cache_root,
            authority=authority,
            manifest_sha256=cache_authority["manifest_sha256"],
        )
        dev_records = list(loaded["dev"])
        val_records = list(loaded["val"])
        dev_binding = _bind_record_identities(
            dev_records,
            expected_paths=dev_paths,
            dataset_root=dataset_root,
            split_name="internal-dev",
        )
        val_binding = _bind_record_identities(
            val_records,
            expected_paths=val_paths,
            dataset_root=dataset_root,
            split_name="official-val",
        )
        if cache_authority["splits"] != {"dev": dev_binding, "val": val_binding}:
            raise RuntimeError("cache external split authority mismatch")
        if alpha_report["split"]["actual_loader_order_sha256"] != dev_binding[
            "actual_loader_order_sha256"
        ]:
            raise RuntimeError("cache dev order differs from alpha-selection report")
    else:
        if cache_authority_path.exists():
            raise RuntimeError("cache external authority exists without cache")
        detector = _load_detector(baseline, device)
        _assert_cuda0_detector(detector)
        _assert_detector_isolated(detector)
        state_before = _state_sha256(detector)
        dev_loader, dev_validator = _build_validation_loader(
            dataset_root,
            baseline,
            device,
            image_source=_dev_list_path(cache_root),
            split_name="internal-dev",
            save_dir=cache_root.parent / f".{cache_root.name}-validator-dev",
            expected_count=DEV_COUNT,
        )
        dev_records = _extract_records(
            detector,
            dev_loader,
            dev_validator,
            device=device,
            expected_count=DEV_COUNT,
            run_cuda_smoke=True,
        )
        dev_binding = _bind_record_identities(
            dev_records,
            expected_paths=dev_paths,
            dataset_root=dataset_root,
            split_name="internal-dev",
        )
        dev_evaluation = _evaluate_records(dev_records, alphas=ALPHA_GRID)
        computed_alpha = _select_alpha(dev_evaluation["oracle"])
        computed_alpha_report = _alpha_report_payload(
            authority=authority,
            dataset_root=dataset_root,
            selected_paths=dev_paths,
            dev_binding=dev_binding,
            dev_evaluation=dev_evaluation,
            selected_alpha=computed_alpha,
        )
        _write_or_validate_canonical_json(alpha_report_path, computed_alpha_report)
        selected_alpha, alpha_report = _validate_alpha_report(
            alpha_report_path,
            authority=authority,
            dataset_root=dataset_root,
            selected_paths=dev_paths,
        )

        _assert_full_dataset_authority(dataset_root)
        val_paths = _expected_official_val_paths(dataset_root)
        val_loader, val_validator = _build_validation_loader(
            dataset_root,
            baseline,
            device,
            image_source=dataset_root / "images" / "val",
            split_name="official-val",
            save_dir=cache_root.parent / f".{cache_root.name}-validator-val",
            expected_count=VAL_COUNT,
        )
        val_records = _extract_records(
            detector,
            val_loader,
            val_validator,
            device=device,
            expected_count=VAL_COUNT,
            run_cuda_smoke=False,
        )
        val_binding = _bind_record_identities(
            val_records,
            expected_paths=val_paths,
            dataset_root=dataset_root,
            split_name="official-val",
        )
        _assert_detector_isolated(detector)
        state_after = _state_sha256(detector)
        if state_after != state_before:
            raise RuntimeError("detector state changed across the quality oracle")
        _persist_cache(
            cache_root,
            dev=dev_records,
            val=val_records,
            authority=authority,
        )
        detector_authority = {
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
            "gradients": False,
        }
        cache_authority = _cache_authority_payload(
            authority=authority,
            manifest_sha256=_file_sha256(cache_root / "manifest.json"),
            dev_binding=dev_binding,
            val_binding=val_binding,
            detector=detector_authority,
        )
        _write_canonical_json_create_only(cache_authority_path, cache_authority)

    if selected_alpha is None:
        raise RuntimeError("alpha selection was not frozen")
    completed = _validate_completed_reports(
        report_root, authority=authority, selected_alpha=selected_alpha
    )
    if completed is not None:
        return completed

    official = _evaluate_records(val_records, alphas=(selected_alpha,))
    stock_authority_check = _assert_stock_authority(official["stock"])
    oracle_metrics = official["oracle"][selected_alpha]
    decision = decide_quality_oracle(
        stock_map=official["stock"]["map"],
        stock_ap75=official["stock"]["ap75"],
        oracle_map=oracle_metrics["map"],
        oracle_ap75=oracle_metrics["ap75"],
    )

    official_report_path = report_root / "quality-oracle-report.json"
    decision_path = report_root / "quality-oracle-decision.json"
    _write_or_validate_canonical_json(
        official_report_path,
        {
            "format_version": 1,
            "authority": authority,
            "split": {"count": VAL_COUNT, "official_passes": 1},
            "selected_alpha": selected_alpha,
            "stock": official["stock"],
            "stock_authority_check": stock_authority_check,
            "oracle": oracle_metrics,
            "detector": cache_authority["detector"],
            "cache_manifest": {
                "path": str(cache_root / "manifest.json"),
                "sha256": cache_authority["manifest_sha256"],
            },
        },
    )
    _write_or_validate_canonical_json(
        decision_path,
        {
            "format_version": 1,
            "authority": authority,
            "selected_alpha": selected_alpha,
            **decision,
        },
    )
    inventory_path = report_root / "environment-hash-inventory.json"
    _write_or_validate_canonical_json(
        inventory_path,
        {
            "format_version": 1,
            "authority": authority,
            "environment": _execution_environment(),
            "inputs": {
                "baseline_checkpoint": {
                    "path": str(baseline),
                    "bytes": baseline.stat().st_size if baseline.is_file() else None,
                    "sha256": _file_sha256(baseline),
                },
                "dataset_root": str(dataset_root),
                "dev_image_list": {
                    "path": str(_dev_list_path(cache_root)),
                    "sha256": _file_sha256(_dev_list_path(cache_root)),
                },
                "cache_manifest": {
                    "path": str(cache_root / "manifest.json"),
                    "sha256": cache_authority["manifest_sha256"],
                },
                "cache_manifest_authority": {
                    "path": str(cache_authority_path),
                    "sha256": _file_sha256(cache_authority_path),
                },
            },
            "reports": {
                alpha_report_path.name: _file_sha256(alpha_report_path),
                official_report_path.name: _file_sha256(official_report_path),
                decision_path.name: _file_sha256(decision_path),
            },
        },
    )
    return 0 if decision["status"] in {"passed", "scientific_failed"} else 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

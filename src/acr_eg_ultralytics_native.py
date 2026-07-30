"""Ultralytics-native metrics for the sealed ACR-EG checkpoint pair."""

from __future__ import annotations

from collections.abc import Mapping
import gc
from pathlib import Path
import time
from typing import Any

import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.utils.patches import imread

from src.gcte_views import build_local_view_tensor


NATIVE_SCHEMA = "gcte-acr-eg-ultralytics-native/v1"
EXPECTED_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEF"
    "CF3AFEF6C174C6E4F3B1EF810C883099B"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "66E0B8D27706CDA594BE657B20BFD01C"
    "AA536D90B7EA0A05EDC2FEEC11C6E2B4"
)
EXPECTED_DATASET_SIGNATURE = (
    "A9A0C00DC640BCAAEFE9360F5E3B553"
    "82E74E169B5AEEF15EB1F0AE2A571228A"
)


class NativePairedRTDETRDataset(RTDETRDataset):
    """Stock RT-DETR validation samples plus four source-image local views."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        source = imread(self.im_files[index], flags=self.cv2_flag)
        if source is None:
            raise FileNotFoundError(self.im_files[index])
        if source.ndim == 2:
            source = source[..., None]
        if source.shape[2] != 3:
            raise RuntimeError("ACR_EG_NATIVE_SOURCE_CHANNEL_DRIFT")
        source_shape = tuple(int(value) for value in source.shape[:2])
        if tuple(sample["ori_shape"]) != source_shape:
            raise RuntimeError("ACR_EG_NATIVE_SOURCE_SHAPE_DRIFT")
        sample["local_views"] = build_local_view_tensor(source)
        sample["source_shape"] = torch.tensor(source_shape, dtype=torch.long)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        local_views = torch.stack(
            [sample.pop("local_views") for sample in batch]
        )
        source_shapes = torch.stack(
            [sample.pop("source_shape") for sample in batch]
        )
        collated = YOLODataset.collate_fn(batch)
        collated["local_views"] = local_views
        collated["source_shape"] = source_shapes
        return collated


def extract_requested_metrics(det_metrics: object) -> dict[str, float]:
    """Extract the five requested scalars from Ultralytics ``DetMetrics``."""

    box = det_metrics.box
    return {
        "Precision": float(box.mp),
        "Recall": float(box.mr),
        "AP50": float(box.map50),
        "AP75": float(box.map75),
        "mAP50-95": float(box.map),
    }


def validate_native_protocol(args: object) -> None:
    """Fail closed if the paired native-validation protocol drifts."""

    observed = (
        str(args.device),
        int(args.batch),
        int(args.workers),
        int(args.imgsz),
        float(args.conf),
        int(args.max_det),
        int(args.expected_records),
        int(args.expected_epoch),
        str(args.expected_baseline_sha256).upper(),
        str(args.expected_checkpoint_sha256).upper(),
        str(args.dataset_signature).upper(),
        bool(args.amp),
    )
    expected = (
        "0",
        1,
        0,
        640,
        0.001,
        300,
        548,
        99,
        EXPECTED_BASELINE_SHA256,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_DATASET_SIGNATURE,
        True,
    )
    if observed != expected:
        raise ValueError("ACR_EG_NATIVE_PROTOCOL_DRIFT")
    if bool(args.smoke):
        if args.limit != 1:
            raise ValueError("ACR_EG_NATIVE_PROTOCOL_DRIFT")
    elif args.limit is not None:
        raise ValueError("ACR_EG_NATIVE_PROTOCOL_DRIFT")


def require_paired_path(model: object) -> None:
    """Reject a method prediction that silently skipped ACR-EG."""

    if getattr(model, "last_acr_eg_output", None) is None:
        raise RuntimeError("ACR_EG_NATIVE_SILENT_STOCK_FALLBACK")


def predict_native_batch(
    model: object,
    batch: Mapping[str, Any],
    *,
    paired: bool,
) -> object:
    """Run one stock or genuine five-view ACR-EG inference batch."""

    if not paired:
        return model.predict(batch["img"])
    model.last_acr_eg_output = None
    prediction = model.predict(
        batch["img"],
        local_views=batch["local_views"],
        source_shapes=batch["source_shape"],
    )
    require_paired_path(model)
    return prediction


def run_native_arm(
    *,
    model: object,
    dataset: object,
    validator: object,
    device: torch.device,
    workers: int,
    amp: bool,
    paired: bool,
) -> dict[str, Any]:
    """Evaluate one arm through Ultralytics RT-DETR validation metrics."""

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=dataset.collate_fn,
    )
    validator.init_metrics(model)
    seen = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        validator.batch_i = batch_index
        if not paired:
            for key in (
                "local_views",
                "source_shape",
                "source_to_global",
                "global_to_source",
            ):
                batch.pop(key, None)
        batch = validator.preprocess(batch)
        if paired:
            batch["local_views"] = batch["local_views"].float() / 255.0
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            predictions = predict_native_batch(model, batch, paired=paired)
        predictions = validator.postprocess(predictions)
        validator.update_metrics(predictions, batch)
        seen += int(batch["img"].shape[0])
        if seen % 25 == 0 or seen == len(dataset):
            arm = "ACR-EG" if paired else "Baseline"
            print(
                f"ACR_EG_NATIVE_PROGRESS arm={arm} {seen}/{len(dataset)}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    official_results = {
        str(name): float(value)
        for name, value in validator.get_stats().items()
    }
    summary = getattr(validator.metrics, "summary", None)
    per_class = summary() if callable(summary) else []
    return {
        "image_count": seen,
        "metrics": extract_requested_metrics(validator.metrics),
        "official_results": official_results,
        "per_class": per_class,
        "runtime": {
            "seconds": seconds,
            "milliseconds_per_image": 1000.0 * seconds / max(seen, 1),
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


def run_native_evaluation(args: object) -> Path:
    """Run the frozen paired evaluation using Ultralytics RT-DETR metrics."""

    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.rtdetr.val import RTDETRValidator

    from src.acr_eg_live_evaluation import (
        _load_checkpoint_model,
        sha256_file,
    )
    from src.sbr_artifacts import (
        atomic_write_json,
        ensure_empty_output,
        git_provenance,
        load_dataset,
        write_checksums,
    )

    validate_native_protocol(args)
    if ultralytics.__version__ != "8.4.90":
        raise RuntimeError("ACR_EG_NATIVE_ULTRALYTICS_VERSION_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("ACR_EG_NATIVE_REQUIRES_CUDA")

    device = torch.device("cuda:0")
    data_path = Path(args.data).resolve()
    manifest = load_dataset(data_path, split="val")
    if int(manifest["image_count"]) != int(args.expected_records):
        raise RuntimeError("ACR_EG_NATIVE_DATASET_COUNT_DRIFT")
    if (
        str(manifest["dataset_signature"]).upper()
        != str(args.dataset_signature).upper()
    ):
        raise RuntimeError("ACR_EG_NATIVE_DATASET_SIGNATURE_DRIFT")

    checked_data = check_det_dataset(str(data_path), autodownload=False)

    def native_args() -> object:
        return get_cfg(
            overrides={
                "task": "detect",
                "mode": "val",
                "data": str(data_path),
                "model": "",
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "conf": args.conf,
                "iou": 0.7,
                "max_det": args.max_det,
                "rect": False,
                "cache": False,
                "plots": False,
                "save_json": False,
                "save_txt": False,
                "split": "val",
            }
        )

    dataset = NativePairedRTDETRDataset(
        img_path=checked_data["val"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=native_args(),
        rect=False,
        cache=None,
        prefix="native-val: ",
        data=checked_data,
    )
    if args.limit is not None:
        dataset.im_files = dataset.im_files[: args.limit]
        dataset.labels = dataset.labels[: args.limit]
        dataset.npy_files = dataset.npy_files[: args.limit]
        dataset.ni = args.limit
    expected_images = int(args.limit or args.expected_records)
    if len(dataset) != expected_images:
        raise RuntimeError("ACR_EG_NATIVE_DATASET_LENGTH_DRIFT")

    output = ensure_empty_output(Path(args.output).resolve())

    def make_validator(arm: str) -> object:
        validator_args = native_args()
        validator = RTDETRValidator(
            dataloader=None,
            save_dir=output / arm,
            args=validator_args,
        )
        validator.device = device
        validator.data = checked_data
        validator.training = False
        validator.args.quantize = None
        return validator

    baseline_model, baseline_info = _load_checkpoint_model(
        Path(args.baseline_checkpoint),
        expected_sha256=args.expected_baseline_sha256,
        device=device,
        integrated=False,
    )
    baseline_arm = run_native_arm(
        model=baseline_model,
        dataset=dataset,
        validator=make_validator("baseline"),
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
    method_arm = run_native_arm(
        model=method_model,
        dataset=dataset,
        validator=make_validator("acr-eg"),
        device=device,
        workers=args.workers,
        amp=args.amp,
        paired=True,
    )
    del method_model
    gc.collect()
    torch.cuda.empty_cache()

    if (
        baseline_arm["image_count"] != expected_images
        or method_arm["image_count"] != expected_images
    ):
        raise RuntimeError("ACR_EG_NATIVE_EVALUATED_COUNT_DRIFT")

    protocol = {
        "ultralytics": ultralytics.__version__,
        "imgsz": int(args.imgsz),
        "local_views": 4,
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": str(args.device),
        "amp": bool(args.amp),
        "conf": float(args.conf),
        "iou_thresholds": [round(0.5 + 0.05 * index, 2) for index in range(10)],
        "max_det": int(args.max_det),
        "nms": False,
        "metric_backend": "ultralytics.models.rtdetr.val.RTDETRValidator",
    }
    result = build_native_result(
        baseline_metrics=baseline_arm["metrics"],
        method_metrics=method_arm["metrics"],
        baseline=baseline_info,
        checkpoint=checkpoint_info,
        dataset={
            "yaml": data_path.as_posix(),
            "yaml_sha256": sha256_file(data_path),
            "image_count": expected_images,
            "full_image_count": int(manifest["image_count"]),
            "signature": str(manifest["dataset_signature"]).upper(),
        },
        protocol=protocol,
        source=git_provenance(Path(__file__).resolve().parents[1]),
    )
    result["official_results"] = {
        "Baseline": baseline_arm["official_results"],
        "ACR-EG": method_arm["official_results"],
    }
    result["per_class"] = {
        "Baseline": baseline_arm["per_class"],
        "ACR-EG": method_arm["per_class"],
    }
    result["runtime"] = {
        "device": torch.cuda.get_device_name(device),
        "Baseline": baseline_arm["runtime"],
        "ACR-EG": method_arm["runtime"],
    }

    evaluation = output / "evaluation.json"
    atomic_write_json(evaluation, _jsonable(result))
    write_checksums(output / "checksums.sha256", (evaluation,), root=output)
    print(
        "ACR_EG_ULTRALYTICS_NATIVE_COMPLETE "
        f"map_delta={result['deltas']['mAP50-95']} output={evaluation}",
        flush=True,
    )
    return evaluation


def build_native_result(
    *,
    baseline_metrics: Mapping[str, float],
    method_metrics: Mapping[str, float],
    baseline: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one canonical paired native-validation result."""

    metric_names = ("Precision", "Recall", "AP50", "AP75", "mAP50-95")
    deltas = {
        name: float(method_metrics[name]) - float(baseline_metrics[name])
        for name in metric_names
    }
    return {
        "schema_version": NATIVE_SCHEMA,
        "baseline": dict(baseline),
        "checkpoint": dict(checkpoint),
        "dataset": dict(dataset),
        "protocol": dict(protocol),
        "source": dict(source),
        "metrics": {
            "Baseline": dict(baseline_metrics),
            "ACR-EG": dict(method_metrics),
        },
        "deltas": deltas,
    }


__all__ = [
    "EXPECTED_BASELINE_SHA256",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_DATASET_SIGNATURE",
    "NATIVE_SCHEMA",
    "NativePairedRTDETRDataset",
    "build_native_result",
    "extract_requested_metrics",
    "predict_native_batch",
    "require_paired_path",
    "run_native_arm",
    "run_native_evaluation",
    "validate_native_protocol",
]

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRDataset
from ultralytics.utils import ops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_sqda_small_ap import build_coco_dataset, evaluate_predictions
from src.rtdetr_sqda_sgc import (
    BASELINE_SHA256,
    SQDASGCDetectionModel,
    load_inherited_sqda_adapter,
    load_mature_baseline,
    sha256_file,
)
from src.sqda_geometry_diagnosis import (
    DIAGNOSTIC_MODES,
    attach_baseline_threshold_metrics,
    build_branch_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only four-branch diagnosis for a retained SQDA G2 adapter."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-images", type=int, default=548)
    parser.add_argument("--expected-annotations", type=int, default=38759)
    return parser


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(f"cuda:{int(value.split(',')[0])}")


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def _predictions_to_coco(
    decoded: torch.Tensor,
    im_files: list[str],
    ori_shapes: list[tuple[int, int]],
    *,
    image_size: int,
) -> list[dict[str, Any]]:
    if decoded.ndim != 3 or decoded.shape[1:] != (300, 6):
        raise RuntimeError(f"expected decoded Top-300 predictions [B,300,6], got {decoded.shape}")
    records: list[dict[str, Any]] = []
    for prediction, image_file, ori_shape in zip(decoded, im_files, ori_shapes):
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"non-finite prediction for {image_file}")
        box = ops.xywh2xyxy(prediction[:, :4]).float() * float(image_size)
        original_height, original_width = (int(value) for value in ori_shape)
        box[:, [0, 2]] *= original_width / float(image_size)
        box[:, [1, 3]] *= original_height / float(image_size)
        box = ops.xyxy2xywh(box)
        box[:, :2] -= box[:, 2:] / 2.0
        for coordinates, score, category in zip(
            box.cpu().tolist(),
            prediction[:, 4].float().cpu().tolist(),
            prediction[:, 5].long().cpu().tolist(),
        ):
            records.append(
                {
                    "image_id": Path(image_file).stem,
                    "category_id": int(category) + 1,
                    "bbox": [round(float(value), 6) for value in coordinates],
                    "score": round(float(score), 8),
                }
            )
    return records


def _build_loader(data_yaml: Path, workers: int) -> tuple[dict[str, Any], Any]:
    data = check_det_dataset(str(data_yaml), autodownload=False)
    config = get_cfg(
        overrides={
            "task": "detect",
            "mode": "val",
            "data": str(data_yaml),
            "imgsz": 640,
            "batch": 8,
            "workers": workers,
            "rect": False,
            "cache": False,
            "seed": 0,
            "deterministic": True,
            "nms": False,
            "max_det": 300,
        }
    )
    dataset = RTDETRDataset(
        img_path=data["val"],
        imgsz=640,
        batch_size=8,
        augment=False,
        hyp=config,
        rect=False,
        cache=None,
        single_cls=False,
        prefix="SQDA geometry diagnosis: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    return data, build_dataloader(
        dataset,
        batch=8,
        workers=workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
    )


def _tide_status(ground_truth: Path, predictions_by_mode: dict[str, Path]) -> dict[str, Any]:
    if importlib.util.find_spec("tidecv") is None:
        return {"available": False, "reason": "tidecv is not installed"}
    try:
        from tidecv import TIDE, datasets

        reports: dict[str, Any] = {}
        for mode, prediction_path in predictions_by_mode.items():
            tide = TIDE()
            tide.evaluate(
                datasets.COCO(str(ground_truth)),
                datasets.COCOResult(str(prediction_path)),
                mode=TIDE.BOX,
            )
            reports[mode] = {
                "status": "evaluated",
                "main_errors": {
                    str(key): int(value)
                    for key, value in tide.get_main_errors().items()
                },
            }
        return {"available": True, "reports": reports}
    except Exception as error:
        return {"available": True, "status": "failed", "reason": repr(error)}


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    adapter_checkpoint = args.adapter_checkpoint.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    images = args.images.expanduser().resolve()
    labels = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if sha256_file(checkpoint) != BASELINE_SHA256:
        raise ValueError("mature baseline SHA256 mismatch")
    dataset = build_coco_dataset(images, labels)
    if len(dataset["images"]) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(dataset['images'])}")
    if len(dataset["annotations"]) != args.expected_annotations:
        raise RuntimeError(
            f"expected {args.expected_annotations} annotations, found {len(dataset['annotations'])}"
        )
    data, loader = _build_loader(data_yaml, args.workers)
    device = _device(args.device)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    output.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output / "ground-truth-coco.json"
    _write_json(ground_truth_path, dataset)
    predictions_by_mode: dict[str, list[dict[str, Any]]] = {}
    prediction_paths: dict[str, Path] = {}
    summaries: dict[str, dict[str, Any]] = {}
    inheritance_metadata: dict[str, Any] | None = None
    tensor_audits: dict[str, dict[str, str]] = {}
    for mode in DIAGNOSTIC_MODES:
        model = SQDASGCDetectionModel("rtdetr-l.yaml", nc=int(data["nc"]), verbose=False)
        load_mature_baseline(model, checkpoint, expected_sha256=BASELINE_SHA256)
        inherited = load_inherited_sqda_adapter(model, adapter_checkpoint)
        if inheritance_metadata is None:
            inheritance_metadata = inherited
        model.residual_mode = mode
        model = model.to(device).eval()
        adapter_before = _state_sha256(model.sqda_sgc)
        records: list[dict[str, Any]] = []
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            for batch in loader:
                images_tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
                prediction = model(images_tensor)
                decoded = prediction[0] if isinstance(prediction, tuple) else prediction
                records.extend(
                    _predictions_to_coco(
                        decoded,
                        list(batch["im_file"]),
                        list(batch["ori_shape"]),
                        image_size=640,
                    )
                )
        adapter_after = _state_sha256(model.sqda_sgc)
        if adapter_before != adapter_after:
            raise AssertionError(f"read-only diagnostic mutated SQDA adapter in mode={mode}")
        tensor_audits[mode] = {"before": adapter_before, "after": adapter_after}
        prediction_path = output / mode / "predictions.json"
        _write_json(prediction_path, records)
        metrics = evaluate_predictions(dataset, prediction_path)
        predictions_by_mode[mode] = records
        prediction_paths[mode] = prediction_path
        summaries[mode] = build_branch_summary(mode, dataset, records, metrics)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_threshold = attach_baseline_threshold_metrics(
        summaries,
        dataset,
        predictions_by_mode,
    )
    for mode, summary in summaries.items():
        curve = summary.pop("pr_f1_curve")
        curve_path = output / mode / "pr-f1-curve.json"
        _write_json(curve_path, curve)
        summary["pr_f1_curve"] = {
            "path": str(curve_path),
            "sha256": sha256_file(curve_path),
            "best_f1": curve["best_f1"],
            "point_count": len(curve["points"]),
        }
        summary["prediction"] = {
            "path": str(prediction_paths[mode]),
            "sha256": sha256_file(prediction_paths[mode]),
            "entries": len(predictions_by_mode[mode]),
        }
    report = {
        "schema": 1,
        "training_signal": False,
        "git_sha": _git_sha(),
        "protocol": {
            "modes": list(DIAGNOSTIC_MODES),
            "imgsz": 640,
            "batch": 8,
            "max_det": 300,
            "nms": False,
            "amp": True,
            "seed": 0,
            "deterministic": True,
            "fixed_error_threshold": 0.25,
            "fixed_error_iou": 0.50,
            "baseline_f1_threshold": baseline_threshold,
        },
        "baseline": {
            "path": str(checkpoint),
            "sha256": BASELINE_SHA256,
        },
        "inherited_adapter": inheritance_metadata,
        "dataset": {
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": sha256_file(data_yaml),
            "images": len(dataset["images"]),
            "annotations": len(dataset["annotations"]),
        },
        "adapter_tensor_audits": tensor_audits,
        "branches": summaries,
        "tide": _tide_status(ground_truth_path, prediction_paths),
    }
    _write_json(output / "geometry-branch-diagnosis.json", report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_diagnosis(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

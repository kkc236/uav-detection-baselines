"""Evaluate frozen FDR/BPDD/RA checkpoints under one official-val authority."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from ultralytics.models.rtdetr.val import RTDETRValidator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_ra_glgm_checkpoints import (  # noqa: E402
    _coco_ground_truth,
    _coco_metrics,
    _dataset,
)
from src.fdr_protocol import canonical_json_bytes  # noqa: E402
from src.lpr_protocol import CATEGORY_NAMES, dataset_signature  # noqa: E402
from src.ra_experiment_protocol import ignore_sidecar_signature  # noqa: E402
from src.rtdetr_fdr import FDRRTDETRDetectionModel  # noqa: E402
from src.rtdetr_ra_glgm import RAGLGMDetectionModel  # noqa: E402


DESIGN = "fdr-bpdd-ra-glgm-unified-comparison-v1"
EXPECTED_SLOTS = ("A", "B", "C", "D")
EXPECTED_KINDS = {"A": "fdr", "B": "fdr", "C": "ra", "D": "ra"}
EXPECTED_PARAMETERS = {"fdr": 33_156_614, "ra": 33_970_010}
EXPECTED_VAL_IMAGES = 548
EXPECTED_VAL_OBJECTS = 38_759
EVALUATION_PROTOCOL = {
    "split": "val",
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "max_det": 300,
    "conf": 0.001,
    "half": False,
    "nms": False,
    "plots": False,
    "save_json": True,
}


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def manifest_payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest().upper()


def _uppercase_sha256(value: Any, label: str) -> str:
    result = str(value).upper()
    if len(result) != 64 or any(character not in "0123456789ABCDEF" for character in result):
        raise ValueError(f"{label} must be an uppercase-compatible SHA256")
    return result


def _read_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"comparison manifest is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("comparison manifest must contain one JSON object")
    if payload.get("format_version") != 1 or payload.get("design") != DESIGN:
        raise ValueError("comparison manifest format/design mismatch")
    expected = _uppercase_sha256(payload.get("manifest_sha256"), "manifest_sha256")
    actual = manifest_payload_sha256(payload)
    if actual != expected:
        raise ValueError(f"comparison manifest SHA256 mismatch: expected={expected}, actual={actual}")
    return payload


def _validate_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("checkpoints")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_SLOTS):
        raise ValueError("comparison manifest must contain exactly A/B/C/D checkpoints")
    result: list[dict[str, Any]] = []
    for expected_slot, entry in zip(EXPECTED_SLOTS, entries, strict=True):
        if not isinstance(entry, Mapping) or set(entry) != {
            "slot",
            "label",
            "kind",
            "path",
            "sha256",
        }:
            raise ValueError("each checkpoint entry must contain only slot/label/kind/path/sha256")
        slot, kind, label = str(entry["slot"]), str(entry["kind"]), str(entry["label"])
        if slot != expected_slot:
            raise ValueError("comparison checkpoints must be ordered exactly A/B/C/D")
        if kind != EXPECTED_KINDS[slot]:
            raise ValueError(f"checkpoint {slot} kind must be {EXPECTED_KINDS[slot]}")
        if not label.strip():
            raise ValueError(f"checkpoint {slot} label is blank")
        checkpoint = Path(str(entry["path"])).resolve()
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint {slot} is missing: {checkpoint}")
        expected_sha = _uppercase_sha256(entry["sha256"], f"checkpoint {slot} SHA256")
        actual_sha = file_sha256(checkpoint)
        if actual_sha != expected_sha:
            raise ValueError(
                f"checkpoint {slot} SHA256 mismatch: expected={expected_sha}, actual={actual_sha}"
            )
        result.append(
            {
                "slot": slot,
                "label": label,
                "kind": kind,
                "path": checkpoint,
                "sha256": actual_sha,
            }
        )
    return result


def _validate_dataset(payload: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    authority = payload.get("dataset")
    if not isinstance(authority, Mapping):
        raise ValueError("comparison dataset authority is missing")
    required = {
        "data_yaml",
        "data_yaml_sha256",
        "root",
        "positive",
        "ignore",
        "val_images",
        "val_objects",
    }
    if set(authority) != required:
        raise ValueError("comparison dataset authority fields differ from the frozen schema")
    data_yaml = Path(str(authority["data_yaml"])).resolve()
    if data_yaml.is_symlink() or not data_yaml.is_file():
        raise FileNotFoundError(f"comparison data YAML is missing: {data_yaml}")
    expected_yaml_sha = _uppercase_sha256(authority["data_yaml_sha256"], "data YAML SHA256")
    if file_sha256(data_yaml) != expected_yaml_sha:
        raise ValueError("comparison data YAML SHA256 mismatch")
    if int(authority["val_images"]) != EXPECTED_VAL_IMAGES:
        raise ValueError("comparison must use all 548 official validation images")
    if int(authority["val_objects"]) != EXPECTED_VAL_OBJECTS:
        raise ValueError("comparison official-val object count differs from authority")
    root, _, images, validation_source = _dataset(
        data_yaml, expected_images=EXPECTED_VAL_IMAGES
    )
    if root != Path(str(authority["root"])).resolve():
        raise ValueError("comparison dataset root differs from authority")
    if validation_source != (root / "images" / "val").resolve():
        raise ValueError("comparison must use the official validation directory")
    positive = dataset_signature(root)
    ignored = ignore_sidecar_signature(root)
    if positive != authority["positive"]:
        raise ValueError("comparison positive dataset signature mismatch")
    if ignored != authority["ignore"]:
        raise ValueError("comparison ignore-sidecar signature mismatch")
    return data_yaml, {
        "root": root,
        "names": list(CATEGORY_NAMES),
        "images": images,
        "positive": positive,
        "ignore": ignored,
        "data_yaml_sha256": expected_yaml_sha,
    }


def _checkpoint_state(source: Any) -> dict[str, torch.Tensor]:
    if isinstance(source, Mapping):
        state = dict(source)
    elif callable(getattr(source, "state_dict", None)):
        state = dict(source.state_dict())
    else:
        raise TypeError("checkpoint EMA/model does not expose a tensor state_dict")
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("checkpoint EMA/model state must contain only tensors")
    return state


def _install_bpdd_pickle_compatibility() -> None:
    """Let historical training-only BPDD model objects yield their tensor state."""

    module_name = "src.rtdetr_fdr_bpdd"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
    compatibility = types.ModuleType(module_name)
    compatibility.FDRBPDDDetectionModel = FDRRTDETRDetectionModel
    sys.modules[module_name] = compatibility


def _load_deployment_model(checkpoint: Path, *, kind: str) -> tuple[torch.nn.Module, str]:
    _install_bpdd_pickle_compatibility()
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping):
        raise TypeError("checkpoint must contain one mapping artifact")
    source_field = "ema" if artifact.get("ema") is not None else "model"
    source = artifact.get(source_field)
    if source is None:
        raise ValueError("checkpoint contains neither EMA nor model state")
    state = _checkpoint_state(source)
    factory = FDRRTDETRDetectionModel if kind == "fdr" else RAGLGMDetectionModel
    model = factory(nc=len(CATEGORY_NAMES), verbose=False)
    model.load_state_dict(state, strict=True)
    model.names = dict(enumerate(CATEGORY_NAMES))
    model.eval()
    _validate_deployment_model(model, kind=kind)
    return model, source_field


def _validate_deployment_model(model: torch.nn.Module, *, kind: str) -> int:
    if kind not in EXPECTED_PARAMETERS:
        raise ValueError(f"unknown deployment kind: {kind}")
    named_ra = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "RAGLGM"
    ]
    if kind == "fdr" and named_ra:
        raise ValueError("FDR deployment graph contains an RA-GLGM module")
    if kind == "ra" and (
        len(named_ra) != 1 or named_ra[0][0] != "model.28.ra_glgm"
    ):
        raise ValueError("RA deployment graph must contain unique model.28.ra_glgm")
    if any(module.__class__.__name__.startswith("BPDD") for module in model.modules()):
        raise ValueError("BPDD entered the deployment graph")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS[kind]:
        raise ValueError(
            f"{kind} parameter count differs: expected={EXPECTED_PARAMETERS[kind]}, actual={parameters}"
        )
    return parameters


def _evaluate_model(
    model: torch.nn.Module,
    *,
    data_yaml: Path,
    save_dir: Path,
    device: str,
    ground_truth: Mapping[str, Any],
    image_ids: Mapping[str, int],
    geometries: Mapping[str, Any],
    ignored: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, Any]:
    validator = RTDETRValidator(
        save_dir=save_dir,
        args={
            "model": str(data_yaml),
            "data": str(data_yaml),
            "task": "detect",
            "mode": "val",
            **EVALUATION_PROTOCOL,
            "device": device,
            "cache": False,
            "rect": False,
            "save_txt": False,
            "verbose": False,
        },
    )
    validator(model=model)
    prediction_path = Path(validator.save_dir).resolve() / "predictions.json"
    if prediction_path.is_symlink() or not prediction_path.is_file():
        raise FileNotFoundError(f"prediction JSON is missing: {prediction_path}")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list) or not all(isinstance(row, Mapping) for row in predictions):
        raise ValueError("prediction JSON must contain a list of objects")
    metrics = _coco_metrics(predictions, ground_truth, image_ids, geometries, ignored)
    for key in ("precision", "recall", "map", "map50", "map75", "ap_tiny", "ap_small"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise ValueError(f"unified evaluator produced invalid metric: {key}")
    class_ap = metrics.get("class_ap")
    if not isinstance(class_ap, list) or len(class_ap) != len(CATEGORY_NAMES):
        raise ValueError("unified evaluator produced invalid class AP")
    return {
        **metrics,
        "predictions_artifact": {
            "path": str(prediction_path),
            "sha256": file_sha256(prediction_path),
        },
        "processed_images": len(image_ids),
    }


def bind_evaluation_row(row: Mapping[str, Any], previous_sha256: str) -> dict[str, Any]:
    previous = _uppercase_sha256(previous_sha256, "previous evaluation row SHA256")
    payload = {**dict(row), "previous_evaluation_row_sha256": previous}
    payload["evaluation_row_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest().upper()
    return payload


def _write_create_only_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace comparison evaluation: {destination}")
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
    return destination


def evaluate_comparison(
    *,
    manifest: str | Path,
    output: str | Path,
    work_dir: str | Path,
    device: str = "0",
    model_loader: Callable[..., tuple[torch.nn.Module, str]] = _load_deployment_model,
    model_evaluator: Callable[..., dict[str, Any]] = _evaluate_model,
) -> list[dict[str, Any]]:
    payload = _read_manifest(manifest)
    if payload.get("evaluation") != EVALUATION_PROTOCOL:
        raise ValueError("comparison evaluation protocol differs from the frozen protocol")
    entries = _validate_entries(payload)
    data_yaml, dataset = _validate_dataset(payload)
    ground_truth, image_ids, geometries, ignored = _coco_ground_truth(
        dataset["images"], dataset["names"], expected_objects=EXPECTED_VAL_OBJECTS
    )
    work = Path(work_dir).resolve()
    if work.exists() or work.is_symlink():
        raise FileExistsError(f"refusing to reuse comparison work directory: {work}")
    work.mkdir(parents=True)
    manifest_sha = _uppercase_sha256(payload["manifest_sha256"], "manifest_sha256")
    evaluator_sha = file_sha256(__file__)
    previous = "0" * 64
    rows: list[dict[str, Any]] = []
    for entry in entries:
        model, source_field = model_loader(entry["path"], kind=entry["kind"])
        parameters = _validate_deployment_model(model, kind=entry["kind"])
        metrics = model_evaluator(
            model,
            data_yaml=data_yaml,
            save_dir=work / entry["slot"],
            device=device,
            ground_truth=ground_truth,
            image_ids=image_ids,
            geometries=geometries,
            ignored=ignored,
        )
        row = bind_evaluation_row(
            {
                "format_version": 1,
                "design": DESIGN,
                "slot": entry["slot"],
                "label": entry["label"],
                "kind": entry["kind"],
                "manifest_sha256": manifest_sha,
                "evaluator_sha256": evaluator_sha,
                "checkpoint": str(entry["path"]),
                "checkpoint_sha256": entry["sha256"],
                "checkpoint_source_field": source_field,
                "model_parameters": parameters,
                "bpdd_inference_module": False,
                "dataset": {
                    "root": str(dataset["root"]),
                    "data_yaml_sha256": dataset["data_yaml_sha256"],
                    "positive": dataset["positive"],
                    "ignore": dataset["ignore"],
                    "val_images": EXPECTED_VAL_IMAGES,
                    "val_objects": EXPECTED_VAL_OBJECTS,
                },
                "evaluation": dict(EVALUATION_PROTOCOL),
                **metrics,
            },
            previous,
        )
        rows.append(row)
        previous = row["evaluation_row_sha256"]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_create_only_jsonl(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    rows = evaluate_comparison(
        manifest=args.manifest,
        output=args.output,
        work_dir=args.work_dir,
        device=args.device,
    )
    print(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

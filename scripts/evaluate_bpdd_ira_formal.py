"""Evaluate the exact FDR+BPDD+IRA Formal100 EMA on frozen VisDrone val."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml
from torch import nn
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bpdd_formal_evaluation import (  # noqa: E402
    CachedScaleRTDETRValidator,
    SCALE_NAMES,
    state_sha256,
    summarize_native_box_metrics,
    summarize_scale_metrics,
    write_create_only_json,
)
from src.bpdd_protocol import BPDD_PROTOCOL_SHA256  # noqa: E402
from src.bpdd_ira_protocol import (  # noqa: E402
    BPDD_IRA_PROTOCOL,
    BPDD_IRA_PROTOCOL_SHA256,
    FDR_INITIAL_STATE_SHA256,
    build_run_identity,
)
from src.fdr_protocol import FDR_PROTOCOL_SHA256, public_state_sha256  # noqa: E402
from src.lpr_protocol import CATEGORY_NAMES, dataset_signature  # noqa: E402
from src.rtdetr_fdr_bpdd_ira import FDRBPDDIRADetectionModel  # noqa: E402


EVALUATION_PROTOCOL = {
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "conf": 0.001,
    "max_det": 300,
    "nms": False,
}
BENCHMARK_PROTOCOL = {
    "imgsz": 640,
    "batch": 1,
    "half": True,
    "warmup": 50,
    "runs": 200,
}
EXPECTED_VAL_IMAGES = 548
COMBINED_VARIANT = "fdr_bpdd_ira"


@dataclass(frozen=True)
class LoadedCombinedCheckpoint:
    model: nn.Module
    metadata: dict[str, Any]


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON authority is not an object: {path}")
    return payload


def _checkpoint_state(source: Any) -> dict[str, torch.Tensor]:
    if isinstance(source, Mapping):
        state = dict(source)
    elif callable(getattr(source, "state_dict", None)):
        state = dict(source.state_dict())
    else:
        raise TypeError("checkpoint EMA does not expose a tensor state_dict")
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("checkpoint EMA state must contain only tensors")
    return state


def load_exact_combined_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    model_factory: Callable[..., nn.Module] = FDRBPDDIRADetectionModel,
) -> LoadedCombinedCheckpoint:
    """Strictly load exact epoch100 EMA into the combined inference graph."""

    path = Path(checkpoint).resolve()
    if path.name != "epoch99.pt":
        raise ValueError("Formal100 evaluation only accepts exact epoch99.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Formal100 checkpoint not found: {path}")
    actual_sha = file_sha256(path)
    if actual_sha != str(expected_sha256).upper():
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected={expected_sha256}, actual={actual_sha}"
        )
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping) or artifact.get("epoch") != 99:
        raise ValueError("Formal100 checkpoint must contain raw epoch99 for epoch100")
    if artifact.get("optimizer") is None:
        raise ValueError("exact epoch100 training checkpoint optimizer state was stripped")
    source = artifact.get("ema")
    if source is None:
        raise ValueError("Formal100 combined evaluation requires checkpoint EMA")
    state = _checkpoint_state(source)
    model = model_factory(nc=len(CATEGORY_NAMES))
    if model_factory is FDRBPDDIRADetectionModel and type(model) is not FDRBPDDIRADetectionModel:
        raise TypeError("Formal100 must instantiate the exact combined FDR+BPDD+IRA graph")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint does not strictly match the combined graph") from error
    model.eval()
    return LoadedCombinedCheckpoint(
        model=model,
        metadata={
            "kind": "exact-final-ema",
            "completed_epoch": 100,
            "raw_epoch": 99,
            "sha256": actual_sha,
            "sha256_verified": True,
            "source_field": "ema",
            "ema_state_sha256": state_sha256(state),
            "strict_fdr_bpdd_ira_graph": True,
        },
    )


def validate_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    manifest = _read_json(run / "bpdd-ira-run.json")
    identity = manifest.get("run_identity")
    source = manifest.get("source")
    initial = manifest.get("initial_state")
    if manifest.get("format_version") != 1 or not isinstance(identity, Mapping):
        raise ValueError("combined Formal100 run manifest is missing identity")
    if not isinstance(source, Mapping) or not isinstance(initial, Mapping):
        raise ValueError("combined Formal100 authority is incomplete")
    expected_identity = build_run_identity(
        source, stage="formal", variant=COMBINED_VARIANT, seed=0
    )
    if dict(identity) != expected_identity:
        raise ValueError("combined Formal100 run identity or variant is invalid")
    if manifest.get("protocol_sha256") != BPDD_IRA_PROTOCOL_SHA256:
        raise ValueError("combined Formal100 protocol SHA256 mismatch")
    if initial.get("sha256") != FDR_INITIAL_STATE_SHA256:
        raise ValueError("combined Formal100 initial-state authority mismatch")
    data = manifest.get("data")
    if not isinstance(data, str) or "test" in Path(data).name.lower():
        raise ValueError("combined Formal100 evaluation requires val YAML, never test YAML")
    return manifest


def validate_epoch_records(
    path: str | Path,
    *,
    run_identity: Mapping[str, Any],
    checkpoint: str | Path,
) -> dict[str, Any]:
    records_path = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"epoch record line {line_number} is not an object")
        rows.append(payload)
    if len(rows) != 100:
        raise ValueError("epoch records must contain exactly 100 immutable rows")
    epochs = [row.get("completed_epoch") for row in rows]
    if epochs != list(range(1, 101)):
        raise ValueError("epoch records must contain continuous epochs 1-100 exactly once")
    for row in rows:
        for field in ("run_id", "variant", "stage"):
            if row.get(field) != run_identity.get(field):
                raise ValueError(f"epoch record identity mismatch for {field}")
        if row.get("gradients_finite") is not True:
            raise ValueError("epoch record reports non-finite gradients")
    checkpoint_sha = file_sha256(checkpoint)
    if str(rows[-1].get("checkpoint_sha256", "")).upper() != checkpoint_sha:
        raise ValueError("epoch100 record checkpoint SHA256 mismatch")
    ema_sha = str(rows[-1].get("ema_state_sha256", "")).upper()
    if len(ema_sha) != 64:
        raise ValueError("epoch100 record EMA SHA256 is invalid")
    return {
        "path": str(records_path),
        "count": 100,
        "completed_epochs": [1, 100],
        "sha256": file_sha256(records_path),
        "final_checkpoint_sha256": checkpoint_sha,
        "final_ema_state_sha256": ema_sha,
        "all_gradients_finite": True,
    }


def _resolve_yaml_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _default_image_count(path: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(
        1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions
    )


def validate_val_authority(
    data_yaml: str | Path,
    *,
    dataset_root: str | Path,
    signature_fn: Callable[[Path], Mapping[str, Any]] = dataset_signature,
    image_count_fn: Callable[[Path], int] = _default_image_count,
) -> dict[str, Any]:
    yaml_path = Path(data_yaml).resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("val YAML is not a mapping")
    root = Path(dataset_root).resolve()
    yaml_root = _resolve_yaml_path(payload.get("path", root), base=yaml_path.parent)
    if yaml_root != root:
        raise ValueError("val YAML dataset root differs from frozen authority")
    val_value = payload.get("val")
    if not isinstance(val_value, str):
        raise ValueError("val YAML is missing val split")
    normalized = val_value.replace("\\", "/").lower()
    if "test" in normalized or "val" not in normalized:
        raise ValueError("evaluation YAML must bind the val split and must not bind test")
    if int(payload.get("nc", -1)) != 10 or tuple(payload.get("names", ())) != tuple(CATEGORY_NAMES):
        raise ValueError("val YAML category mapping differs from frozen authority")
    val_path = _resolve_yaml_path(val_value, base=yaml_root)
    images = int(image_count_fn(val_path))
    if images != EXPECTED_VAL_IMAGES:
        raise ValueError(f"frozen val authority requires exactly 548 images, got {images}")
    signature = dict(signature_fn(root))
    dataset_sha = str(signature.get("sha256", "")).upper()
    expected_sha = str(BPDD_IRA_PROTOCOL["dataset"]["sha256"]).upper()
    if dataset_sha != expected_sha:
        raise ValueError("frozen VisDrone train/val dataset SHA256 mismatch")
    return {
        "yaml": str(yaml_path),
        "yaml_sha256": file_sha256(yaml_path),
        "split": "val",
        "images": images,
        "dataset_root": str(root),
        "dataset_sha256": dataset_sha,
    }


def run_official_validation(model: Any, *, data: Path, save_dir: Path) -> dict[str, Any]:
    validator = CachedScaleRTDETRValidator(
        save_dir=save_dir,
        args={
            "model": str(data),
            "data": str(data),
            "task": "detect",
            "mode": "val",
            "split": "val",
            **EVALUATION_PROTOCOL,
            "device": "0",
            "cache": False,
            "half": False,
            "rect": False,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
        },
    )
    validator(model=model)
    processed = len(validator.scale_targets)
    if processed != EXPECTED_VAL_IMAGES or len(validator.scale_predictions) != processed:
        raise RuntimeError(f"official val processed {processed} images instead of 548")
    return {
        **summarize_native_box_metrics(validator.metrics.box, CATEGORY_NAMES),
        **summarize_scale_metrics(
            validator.scale_predictions,
            validator.scale_targets,
            class_count=len(CATEGORY_NAMES),
        ),
        "processed_images": processed,
        "prediction_passes": 1,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples or not all(math.isfinite(value) and value > 0 for value in samples):
        raise ValueError("latency samples must be non-empty, finite, and positive")
    median = float(statistics.median(samples))
    return {
        "median_ms": median,
        "p95_ms": _percentile(samples, 0.95),
        "fps": 1000.0 / median,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_efficiency_audit(model: nn.Module, *, device: str) -> dict[str, Any]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal100 FP16 efficiency audit requires CUDA")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gflops = float(get_flops(model, imgsz=BENCHMARK_PROTOCOL["imgsz"]))
    runtime_model = model.to(torch_device).eval().half()
    image = torch.randn(
        1,
        3,
        BENCHMARK_PROTOCOL["imgsz"],
        BENCHMARK_PROTOCOL["imgsz"],
        device=torch_device,
        dtype=torch.float16,
    )
    torch.cuda.reset_peak_memory_stats(torch_device)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(BENCHMARK_PROTOCOL["warmup"]):
            runtime_model.predict(image)
        _synchronize(torch_device)
        for _ in range(BENCHMARK_PROTOCOL["runs"]):
            _synchronize(torch_device)
            start = time.perf_counter()
            runtime_model.predict(image)
            _synchronize(torch_device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "parameters": int(parameters),
        "gflops": gflops,
        "fp16": {
            "device": str(torch_device),
            **latency_summary(samples),
            "peak_memory_mib": torch.cuda.max_memory_allocated(torch_device) / 1024**2,
        },
    }


def _metric_block(payload: Mapping[str, Any]) -> dict[str, float] | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    required = ("precision", "recall", "f1", "map50", "map75", "map")
    try:
        result = {key: float(metrics[key]) for key in required}
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result.values()) else None


def load_reference_authority(
    path: str | Path,
    *,
    expected_variant: str,
    method: str,
) -> dict[str, Any]:
    authority_path = Path(path).resolve()
    payload = _read_json(authority_path)
    metrics = _metric_block(payload)
    if metrics is None:
        raise ValueError(f"{method} reference does not contain complete core metrics")
    failures: list[str] = []
    identity = payload.get("evaluation_identity")
    checkpoint = payload.get("checkpoint")
    if payload.get("format_version") != 1:
        failures.append("format_version")
    if not isinstance(identity, Mapping):
        failures.append("evaluation_identity")
        identity = {}
    if identity.get("variant") != expected_variant:
        failures.append("variant")
    if identity.get("stage") != "formal" or identity.get("seed") != 0:
        failures.append("training_protocol")
    if identity.get("fdr_protocol_sha256") != FDR_PROTOCOL_SHA256:
        failures.append("training_protocol")
    expected_protocol = {
        "fdr": FDR_PROTOCOL_SHA256,
        "fdr_bpdd": BPDD_PROTOCOL_SHA256,
    }.get(expected_variant)
    actual_protocol = identity.get("protocol_sha256")
    if expected_protocol is not None:
        if actual_protocol != expected_protocol:
            failures.append("training_protocol")
    else:
        embedded_protocol = payload.get("protocol")
        if (
            not isinstance(embedded_protocol, Mapping)
            or actual_protocol != public_state_sha256(embedded_protocol)
            or embedded_protocol.get("variant") != expected_variant
        ):
            failures.append("training_protocol")
    if identity.get("split") != "val" or identity.get("images") != EXPECTED_VAL_IMAGES:
        failures.append("split")
    if identity.get("dataset_sha256") != BPDD_IRA_PROTOCOL["dataset"]["sha256"]:
        failures.append("dataset")
    if payload.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        failures.append("evaluator_protocol")
    if payload.get("processed_images") != EXPECTED_VAL_IMAGES or payload.get("prediction_passes") != 1:
        failures.append("evaluator_execution")
    if not isinstance(checkpoint, Mapping) or (
        checkpoint.get("kind") != "exact-final-ema"
        or checkpoint.get("completed_epoch") != 100
        or checkpoint.get("sha256_verified") is not True
    ):
        failures.append("checkpoint")
    strict = not failures
    return {
        "method": method,
        "variant": expected_variant,
        "strict": strict,
        "evidence_level": (
            "strict_reference" if strict else "non_strict_historical_reference"
        ),
        "strict_validation_failures": sorted(set(failures)),
        "authority_path": str(authority_path),
        "authority_sha256": file_sha256(authority_path),
        "metrics": metrics,
        "scales": dict(payload.get("scales", {})),
        "class_details": dict(payload.get("class_details", {})),
    }


def _subtract(current: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(current[key]) - float(reference[key]) for key in current}


def _comparison_delta(
    current: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    class_delta: dict[str, dict[str, float]] = {}
    current_classes = current.get("class_details", {})
    reference_classes = reference.get("class_details", {})
    if isinstance(current_classes, Mapping) and isinstance(reference_classes, Mapping):
        for name in CATEGORY_NAMES:
            left, right = current_classes.get(name), reference_classes.get(name)
            if isinstance(left, Mapping) and isinstance(right, Mapping):
                class_delta[name] = _subtract(
                    {key: left[key] for key in ("map50", "map75", "map")},
                    {key: right[key] for key in ("map50", "map75", "map")},
                )
    current_scales, reference_scales = current.get("scales", {}), reference.get("scales", {})
    scale_delta = {}
    if isinstance(current_scales, Mapping) and isinstance(reference_scales, Mapping):
        scale_delta = {
            name: float(current_scales[name]) - float(reference_scales[name])
            for name in SCALE_NAMES
            if name in current_scales and name in reference_scales
        }
    return {
        "authority_sha256": reference["authority_sha256"],
        "metrics_delta": _subtract(current["metrics"], reference["metrics"]),
        "scale_delta": scale_delta,
        "class_delta": class_delta,
    }


def build_preliminary_comparisons(
    current: Mapping[str, Any],
    *,
    fdr_evaluation: str | Path,
    bpdd_evaluation: str | Path,
    ira_evaluation: str | Path | None,
) -> dict[str, Any]:
    references = [
        load_reference_authority(fdr_evaluation, expected_variant="fdr", method="FDR"),
        load_reference_authority(
            bpdd_evaluation, expected_variant="fdr_bpdd", method="FDR+BPDD"
        ),
    ]
    if ira_evaluation is not None:
        references.append(
            load_reference_authority(
                ira_evaluation, expected_variant="fdr_ira", method="FDR+IRA"
            )
        )
    else:
        references.append(
            {
                "method": "FDR+IRA",
                "variant": "fdr_ira",
                "strict": False,
                "evidence_level": "unavailable",
                "strict_validation_failures": ["authority_unavailable"],
                "authority_path": None,
                "authority_sha256": None,
                "metrics": {
                    key: None
                    for key in ("precision", "recall", "f1", "map50", "map75", "map")
                },
                "scales": {},
                "class_details": {},
            }
        )
    rows = [
        {
            "method": reference["method"],
            "evidence_level": reference["evidence_level"],
            "strict_validation_failures": reference["strict_validation_failures"],
            **reference["metrics"],
        }
        for reference in references
    ]
    rows.append(
        {
            "method": "FDR+BPDD+IRA",
            "evidence_level": "current_exact",
            "strict_validation_failures": [],
            **dict(current["metrics"]),
        }
    )
    strict_references = [reference for reference in references if reference["strict"]]
    result: dict[str, Any] = {
        "comparison_scope": "preliminary_cross_run",
        "strict_paired": False,
        "four_row_summary": rows,
        "strict_delta_table": [
            {
                "method": reference["method"],
                **_comparison_delta(current, reference)["metrics_delta"],
            }
            for reference in strict_references
        ],
        "non_strict_historical_reference": [
            row
            for row, reference in zip(rows, references, strict=False)
            if reference["evidence_level"] == "non_strict_historical_reference"
        ],
        "unavailable_references": [
            row
            for row, reference in zip(rows, references, strict=False)
            if reference["evidence_level"] == "unavailable"
        ],
    }
    by_method = {reference["method"]: reference for reference in references}
    if by_method["FDR"]["strict"]:
        result["against_fdr"] = _comparison_delta(current, by_method["FDR"])
    if by_method["FDR+BPDD"]["strict"]:
        result["against_fdr_bpdd"] = _comparison_delta(current, by_method["FDR+BPDD"])
    if by_method["FDR+IRA"]["strict"]:
        result["against_fdr_ira"] = _comparison_delta(current, by_method["FDR+IRA"])
    return result


def evaluate_formal_checkpoint(
    *,
    run_dir: str | Path,
    checkpoint: str | Path,
    dataset_root: str | Path,
    fdr_evaluation: str | Path,
    bpdd_evaluation: str | Path,
    ira_evaluation: str | Path | None,
    output: str | Path,
    device: str = "cuda:0",
    checkpoint_loader: Callable[..., LoadedCombinedCheckpoint] = load_exact_combined_checkpoint,
    validation_runner: Callable[..., dict[str, Any]] = run_official_validation,
    efficiency_runner: Callable[..., dict[str, Any]] = run_efficiency_audit,
    val_authority_validator: Callable[..., dict[str, Any]] = validate_val_authority,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output_path}")
    manifest = validate_run_manifest(run)
    identity = dict(manifest["run_identity"])
    epoch_records = validate_epoch_records(
        run / "fdr-epochs.jsonl",
        run_identity=identity,
        checkpoint=checkpoint_path,
    )
    data_path = Path(str(manifest["data"])).resolve()
    val_authority = val_authority_validator(
        data_path, dataset_root=Path(dataset_root).resolve()
    )
    loaded = checkpoint_loader(
        checkpoint_path,
        expected_sha256=epoch_records["final_checkpoint_sha256"],
    )
    if loaded.metadata.get("kind") != "exact-final-ema":
        raise ValueError("combined Formal100 evaluation requires exact epoch100 EMA")
    if loaded.metadata.get("ema_state_sha256") != epoch_records["final_ema_state_sha256"]:
        raise ValueError("epoch100 EMA state SHA256 differs from immutable epoch evidence")
    validation = validation_runner(
        loaded.model,
        data=data_path,
        save_dir=output_path.parent / "validator",
    )
    if validation.get("processed_images") != EXPECTED_VAL_IMAGES or validation.get("prediction_passes") != 1:
        raise RuntimeError("combined validation must use one exact 548-image prediction pass")
    efficiency_loaded = checkpoint_loader(
        checkpoint_path,
        expected_sha256=epoch_records["final_checkpoint_sha256"],
    )
    if efficiency_loaded.metadata != loaded.metadata:
        raise ValueError("independent efficiency reload differs from validation EMA")
    efficiency = efficiency_runner(efficiency_loaded.model, device=device)
    comparisons = build_preliminary_comparisons(
        validation,
        fdr_evaluation=fdr_evaluation,
        bpdd_evaluation=bpdd_evaluation,
        ira_evaluation=ira_evaluation,
    )
    report = {
        "format_version": 1,
        "evaluation_identity": {
            **identity,
            "data": str(data_path),
            "dataset_sha256": val_authority["dataset_sha256"],
            "split": "val",
            "images": EXPECTED_VAL_IMAGES,
        },
        "val_authority": val_authority,
        "checkpoint": dict(loaded.metadata),
        "epoch_records": epoch_records,
        "evaluation_protocol": dict(EVALUATION_PROTOCOL),
        "benchmark_protocol": dict(BENCHMARK_PROTOCOL),
        **validation,
        "efficiency": efficiency,
        "comparisons": comparisons,
        "hashes": {
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "ema_state_sha256": loaded.metadata["ema_state_sha256"],
            "epoch_records_sha256": epoch_records["sha256"],
            "data_yaml_sha256": val_authority["yaml_sha256"],
            "fdr_evaluation_sha256": file_sha256(fdr_evaluation),
            "bpdd_evaluation_sha256": file_sha256(bpdd_evaluation),
            "ira_evaluation_sha256": (
                file_sha256(ira_evaluation) if ira_evaluation is not None else None
            ),
            "dataset_sha256": val_authority["dataset_sha256"],
        },
    }
    write_create_only_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fdr-evaluation", type=Path, required=True)
    parser.add_argument("--bpdd-evaluation", type=Path, required=True)
    parser.add_argument("--ira-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_formal_checkpoint(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        fdr_evaluation=args.fdr_evaluation,
        bpdd_evaluation=args.bpdd_evaluation,
        ira_evaluation=args.ira_evaluation,
        output=args.output,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

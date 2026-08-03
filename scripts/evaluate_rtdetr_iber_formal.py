"""Independently compare epoch-100 baseline, stock, and refined IBER-BE outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.github_checkpoint_sync import checkpoint_metadata
from src.iber_formal_protocol import (
    FORMAL_DESIGN_VERSION,
    FORMAL_EPOCHS,
    FORMAL_FROZEN_PROTOCOL,
    validate_formal_manifest,
)
from src.iber_formal_publication import (
    FormalPublicationIdentity,
    FormalPublicationLedger,
)
from src.iber_protocol import EXPECTED_BASELINE_SHA256
from src.lpr_protocol import (
    current_environment,
    dataset_signature,
    environment_violations,
    source_violations,
)
from src.rtdetr_iber_formal import IBERFullRTDETRDetectionModel


FORMAL_VALIDATION_ARGS = {
    "model": "rtdetr-l.yaml",
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "device": "0",
    "max_det": 300,
    "nms": False,
    "cache": False,
    "plots": False,
    "save_json": False,
    "verbose": False,
    "task": "detect",
    "mode": "val",
    "split": "val",
    "rect": False,
}
METRIC_FIELDS = ("map", "map50", "map75", "precision", "recall")


def checkpoint_record(path: str | Path) -> dict[str, Any]:
    metadata = checkpoint_metadata(path)
    return {
        "completed_epoch": metadata.completed_epoch,
        "bytes": metadata.bytes,
        "sha256": metadata.sha256,
    }


def validate_epoch100_checkpoints(
    method_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method = checkpoint_record(method_checkpoint)
    baseline = checkpoint_record(baseline_checkpoint)
    for name, record in (("method", method), ("baseline", baseline)):
        if record["completed_epoch"] != FORMAL_EPOCHS:
            raise ValueError(
                f"formal evaluation requires {name} completed epoch 100, "
                f"got {record['completed_epoch']}"
            )
    return method, baseline


def _model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    source = checkpoint.get("ema")
    if source is None:
        source = checkpoint.get("model")
    if isinstance(source, torch.nn.Module):
        state = source.float().state_dict()
    elif isinstance(source, Mapping):
        state = source.get("state_dict", source)
    else:
        raise ValueError("checkpoint has no loadable EMA/model state")
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint model state is not a tensor state dictionary")
    return dict(state)


def load_baseline(path: str | Path) -> RTDETRDetectionModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _model_state(checkpoint)
    if any("iber_refiner." in name for name in state):
        raise ValueError("baseline checkpoint contains IBER private tensors")
    model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    model.load_state_dict(state, strict=True)
    return model


def load_method(path: str | Path) -> IBERFullRTDETRDetectionModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _model_state(checkpoint)
    if not any("iber_refiner." in name for name in state):
        raise ValueError("method checkpoint is missing IBER private tensors")
    model = IBERFullRTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False, private_seed=10_000
    )
    model.load_state_dict(state, strict=True)
    return model


def _metric_dict(metrics: Any) -> dict[str, float]:
    values = {
        "map": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("formal validation produced non-finite metrics")
    return values


def validate_model(
    model: torch.nn.Module,
    *,
    data: str | Path,
    mode: str | None = None,
) -> dict[str, float]:
    if mode is not None:
        if not isinstance(model, IBERFullRTDETRDetectionModel):
            raise TypeError("only the IBER method model has stock/refined modes")
        model.set_refinement_output(mode)
    validator = RTDETRValidator(args={**FORMAL_VALIDATION_ARGS, "data": str(data)})
    validator(model=model)
    return _metric_dict(validator.metrics)


def _validated_metrics(name: str, values: Mapping[str, Any]) -> dict[str, float]:
    if set(values) != set(METRIC_FIELDS):
        raise ValueError(f"{name} metrics are incomplete")
    result = {field: float(values[field]) for field in METRIC_FIELDS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"{name} metrics are non-finite")
    return result


def build_comparison(
    baseline: Mapping[str, Any],
    method_stock: Mapping[str, Any],
    method_refined: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_values = _validated_metrics("baseline", baseline)
    stock_values = _validated_metrics("method stock", method_stock)
    refined_values = _validated_metrics("method refined", method_refined)

    def delta(first: Mapping[str, float], second: Mapping[str, float]) -> dict[str, float]:
        return {field: first[field] - second[field] for field in METRIC_FIELDS}

    return {
        "baseline": baseline_values,
        "method_stock": stock_values,
        "method_refined": refined_values,
        "delta": {
            "stock_vs_baseline": delta(stock_values, baseline_values),
            "refined_vs_baseline": delta(refined_values, baseline_values),
            "refined_vs_stock": delta(refined_values, stock_values),
        },
    }


def _rms(value: torch.Tensor) -> float:
    data = value.detach().float()
    if data.numel() == 0 or not bool(torch.isfinite(data).all()):
        raise FloatingPointError("formal IBER activity is missing or non-finite")
    return float(data.square().mean().sqrt().cpu())


def validate_evaluation_authority(runtime: Mapping[str, Any]) -> dict[str, Any]:
    if (
        runtime.get("design_version") != FORMAL_DESIGN_VERSION
        or runtime.get("stage") != "formal"
        or runtime.get("seed") != 0
        or runtime.get("epochs") != FORMAL_EPOCHS
        or runtime.get("protocol") != FORMAL_FROZEN_PROTOCOL
    ):
        raise ValueError("method runtime is not the frozen formal100 authority")
    manifest = validate_formal_manifest(runtime.get("manifest", {}))
    violations = environment_violations(current_environment())
    if violations:
        raise ValueError(f"formal evaluation environment mismatch: {violations}")
    drift = source_violations()
    if drift:
        raise ValueError(f"formal evaluation Ultralytics source drift: {drift}")
    dataset = dataset_signature(Path(manifest["dataset_root"]))
    if dataset.get("sha256") != manifest["dataset"]["sha256"]:
        raise ValueError("formal evaluation dataset SHA-256 mismatch")
    return manifest


def validate_method_publication(
    method_record: Mapping[str, Any],
    runtime: Mapping[str, Any],
    ledger_path: str | Path,
) -> dict[str, Any]:
    """Bind the evaluated method checkpoint to verified formal epoch 100."""
    try:
        identity = FormalPublicationIdentity(
            source_commit=str(runtime["source_commit"]),
            protocol_sha256=str(runtime["protocol_sha256"]),
            initial_state_sha256=str(runtime["initial_state_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("formal runtime publication identity is incomplete") from error
    rows = FormalPublicationLedger(ledger_path, identity).records()
    if len(rows) != FORMAL_EPOCHS:
        raise ValueError("formal publication ledger does not reach epoch100")
    record = rows[-1]
    expected = record.get("checkpoint", {})
    if (
        method_record.get("completed_epoch") != FORMAL_EPOCHS
        or expected.get("bytes") != method_record.get("bytes")
        or str(expected.get("sha256", "")).lower()
        != str(method_record.get("sha256", "")).lower()
    ):
        raise ValueError("method checkpoint does not match published epoch100")
    return record


def evaluate_formal(
    method_checkpoint: Path,
    baseline_checkpoint: Path,
    runtime_manifest: Path,
    publication_ledger: Path | None = None,
) -> dict[str, Any]:
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    manifest = validate_evaluation_authority(runtime)
    method_record, baseline_record = validate_epoch100_checkpoints(
        method_checkpoint, baseline_checkpoint
    )
    if baseline_record["sha256"].upper() != EXPECTED_BASELINE_SHA256:
        raise ValueError("formal baseline checkpoint SHA-256 mismatch")
    publication_record = validate_method_publication(
        method_record,
        runtime,
        publication_ledger
        if publication_ledger is not None
        else runtime_manifest.parent / "publication-ledger.jsonl",
    )
    data = manifest["data"]["formal"]["path"]
    baseline_model = load_baseline(baseline_checkpoint)
    baseline_metrics = validate_model(baseline_model, data=data)
    del baseline_model
    stock_model = load_method(method_checkpoint)
    stock_metrics = validate_model(stock_model, data=data, mode="stock")
    del stock_model
    refined_model = load_method(method_checkpoint)
    refined_metrics = validate_model(refined_model, data=data, mode="refined")
    comparison = build_comparison(
        baseline_metrics,
        stock_metrics,
        refined_metrics,
    )
    output = refined_model.last_iber_output
    if output is None:
        raise RuntimeError("formal refined validation produced no IBER activity")
    return {
        "design_version": FORMAL_DESIGN_VERSION,
        "stage": "formal",
        "seed": 0,
        "method_checkpoint": method_record,
        "baseline_checkpoint": baseline_record,
        "publication_record": publication_record,
        "runtime": runtime,
        **comparison,
        "activity": {
            "gate_rms": _rms(output.gates),
            "residual_rms": _rms(output.residuals),
            "f3_boundary_rms": _rms(output.f3_boundary_evidence),
            "rgb_boundary_rms": _rms(output.rgb_boundary_evidence),
        },
    }


def write_immutable_report(path: str | Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed formal report: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--publication-ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    method = args.method_checkpoint.resolve()
    runtime = (
        args.runtime_manifest.resolve()
        if args.runtime_manifest is not None
        else method.parent.parent / "iber_formal_protocol.json"
    )
    report = evaluate_formal(
        method,
        args.baseline_checkpoint.resolve(),
        runtime,
        args.publication_ledger.resolve()
        if args.publication_ledger is not None
        else None,
    )
    write_immutable_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

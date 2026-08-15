"""Run immutable fail-closed I0-I4 gates before BPDD+IRA Formal100."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GATE_ORDER = ("I0", "I1", "I2", "I3", "I4")
FIXED_RUNTIME = {
    "device": "cuda:0",
    "batch": 8,
    "imgsz": 640,
    "amp_scale": 128.0,
    "queries": 300,
}
FORMAL_AUTHORITY = {
    "stage": "formal",
    "epochs": 100,
    "seed": 0,
    "variant": "fdr_bpdd_ira",
    "pretrained": False,
}
GateRunner = Callable[["PreflightContext"], Mapping[str, Any]]


@dataclass(frozen=True)
class PreflightContext:
    protocol_manifest: Path
    initial_state: Path
    dataset_root: Path
    report_root: Path
    repository_root: Path = ROOT

    def __post_init__(self) -> None:
        for field in (
            "protocol_manifest",
            "initial_state",
            "dataset_root",
            "report_root",
            "repository_root",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _record(gate: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {
        "format_version": 1,
        "gate": gate,
        "payload": body,
        "payload_sha256": hashlib.sha256(_canonical(body)).hexdigest().upper(),
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_inputs(context: PreflightContext) -> None:
    if not context.protocol_manifest.is_file() or context.protocol_manifest.is_symlink():
        raise FileNotFoundError(
            f"BPDD IRA protocol manifest not found or unsafe: {context.protocol_manifest}"
        )
    if not context.initial_state.is_file() or context.initial_state.is_symlink():
        raise FileNotFoundError(
            f"FDR initial state not found or unsafe: {context.initial_state}"
        )
    if not context.dataset_root.is_dir() or context.dataset_root.is_symlink():
        raise FileNotFoundError(f"VisDrone root not found or unsafe: {context.dataset_root}")
    if not context.repository_root.is_dir():
        raise FileNotFoundError(f"repository root not found: {context.repository_root}")
    if context.report_root.exists():
        raise FileExistsError(
            f"BPDD IRA preflight report root already exists: {context.report_root}"
        )


def _validate_gate_payload(gate: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(evidence)
    status = payload.get("status")
    if status not in {"passed", "engineering_failed", "scientific_failed", "blocked"}:
        return {
            "status": "engineering_failed",
            "gate": gate,
            "reason": "invalid gate status",
        }
    if payload.get("gate") != gate:
        return {
            "status": "engineering_failed",
            "gate": gate,
            "reason": "gate payload identity mismatch",
        }
    if status == "passed":
        checks = payload.get("checks")
        if isinstance(checks, Mapping) and not all(bool(value) for value in checks.values()):
            return {
                "status": "engineering_failed",
                "gate": gate,
                "reason": "passed gate contains a false check",
                "checks": dict(checks),
            }
    return payload


def run_preflight(
    context: PreflightContext,
    *,
    gate_runners: Mapping[str, GateRunner] | None = None,
) -> dict[str, Any]:
    """Run every gate in order and block all successors after the first failure."""

    _validate_inputs(context)
    supplied = dict(gate_runners or {})
    unknown = set(supplied) - set(GATE_ORDER)
    if unknown:
        raise ValueError(f"unknown BPDD IRA preflight gates: {sorted(unknown)}")
    context.report_root.mkdir(parents=True, exist_ok=False)
    states: dict[str, str] = {}
    hashes: dict[str, str] = {}
    blocked_by: str | None = None
    for gate in GATE_ORDER:
        if blocked_by is not None:
            evidence: dict[str, Any] = {
                "status": "blocked",
                "gate": gate,
                "blocked_by": blocked_by,
            }
        else:
            runner = supplied.get(gate) or globals()[f"run_{gate.lower()}"]
            try:
                evidence = _validate_gate_payload(gate, runner(context))
            except Exception as error:
                evidence = {
                    "status": "engineering_failed",
                    "gate": gate,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
        status = str(evidence["status"])
        if status != "passed" and blocked_by is None:
            blocked_by = gate
        states[gate] = status
        record = _record(gate, evidence)
        _write_create_only(context.report_root / f"{gate}.json", record)
        hashes[gate] = hashlib.sha256(_canonical(record)).hexdigest().upper()

    eligible = all(states.get(gate) == "passed" for gate in GATE_ORDER)
    failed_status = (
        "scientific_failed"
        if any(value == "scientific_failed" for value in states.values())
        else "engineering_failed"
    )
    decision = {
        "status": "passed" if eligible else failed_status,
        "formal_eligible": eligible,
        "gate_states": states,
        "gate_report_sha256": hashes,
        "fixed_runtime": dict(FIXED_RUNTIME),
        "formal_authority": dict(FORMAL_AUTHORITY),
    }
    _write_create_only(
        context.report_root / "decision.json",
        _record("decision", decision),
    )
    return decision


def _manifest_and_artifact(context: PreflightContext) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from src.fdr_protocol import validate_fdr_initial_state

    manifest = json.loads(context.protocol_manifest.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("BPDD IRA protocol manifest format must be 1")
    declared_hash = manifest.get("manifest_sha256")
    if declared_hash is not None:
        unhashed = dict(manifest)
        unhashed.pop("manifest_sha256", None)
        actual_hash = hashlib.sha256(_canonical(unhashed)).hexdigest().upper()
        if str(declared_hash).upper() != actual_hash:
            raise ValueError("BPDD IRA protocol manifest SHA256 mismatch")
    initial = manifest.get("initial_state")
    if not isinstance(initial, Mapping):
        raise ValueError("BPDD IRA protocol has no initial-state authority")
    expected_path = context.initial_state.resolve()
    manifest_path = Path(str(initial.get("path", ""))).resolve()
    if manifest_path != expected_path:
        raise ValueError("BPDD IRA initial-state path differs from authority")
    actual_sha = _file_sha256(expected_path)
    if actual_sha != str(initial.get("sha256", "")).upper():
        raise ValueError("BPDD IRA initial-state SHA256 mismatch")
    artifact = torch.load(expected_path, map_location="cpu", weights_only=False)
    validate_fdr_initial_state(artifact)
    if artifact.get("fingerprints") != initial.get("fingerprints"):
        raise ValueError("BPDD IRA initial-state fingerprints differ")
    return manifest, artifact


def _tree_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    return left == right


def _cpu_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _models(context: PreflightContext, device: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import torch

    from src.fdr_protocol import load_fdr_initial_state
    from src.rtdetr_fdr_bpdd import BPDD_MODEL_CFG, FDRBPDDDetectionModel
    from src.rtdetr_fdr_bpdd_ira import (
        BPDD_IRA_MODEL_CFG,
        FDRBPDDIRADetectionModel,
        load_fdr_bpdd_ira_initial_state,
    )

    _manifest, artifact = _manifest_and_artifact(context)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        bpdd = FDRBPDDDetectionModel(
            BPDD_MODEL_CFG,
            ch=3,
            nc=10,
            verbose=False,
            private_seed=10_000,
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        combined = FDRBPDDIRADetectionModel(
            BPDD_IRA_MODEL_CFG,
            ch=3,
            nc=10,
            verbose=False,
            private_seed=10_000,
            ira_private_seed=20_000,
        )
    load_fdr_initial_state(bpdd, artifact, variant="fdr")
    load_report = load_fdr_bpdd_ira_initial_state(combined, artifact)
    return bpdd.to(device), combined.to(device), artifact, load_report


def _protocol_dataset(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("BPDD IRA protocol payload is missing")
    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("BPDD IRA dataset authority is missing")
    return dataset


def validate_environment_authority(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require every frozen environment field to exist and match exactly."""

    normalized = dict(actual)
    for key, value in expected.items():
        if key not in normalized:
            raise ValueError(f"runtime authority missing {key}")
        if str(normalized[key]) != str(value):
            raise ValueError(f"runtime authority mismatch for {key}")
    return normalized


def run_i0(context: PreflightContext) -> dict[str, Any]:
    """Bind source, environment, YAML, dataset, and initial-state authority."""

    import torch
    import torchvision
    import ultralytics
    import yaml

    from scripts.train_rtdetr_fdr import current_source_identity
    from src.lpr_protocol import dataset_signature

    if not torch.cuda.is_available():
        raise RuntimeError("I0 requires CUDA")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("I0 requires cuda:0 NVIDIA GeForce RTX 4090")
    manifest, artifact = _manifest_and_artifact(context)
    expected_source = manifest.get("source")
    actual_source = current_source_identity(context.repository_root.resolve())
    if not isinstance(expected_source, Mapping) or dict(expected_source) != actual_source:
        raise ValueError("checked-out source differs from BPDD IRA authority")

    expected_dataset = _protocol_dataset(manifest)
    actual_dataset = dataset_signature(context.dataset_root.resolve())
    train_count = len(list((context.dataset_root / "images" / "train").glob("*.jpg")))
    val_count = len(list((context.dataset_root / "images" / "val").glob("*.jpg")))
    if str(actual_dataset.get("sha256")) != str(expected_dataset.get("sha256")):
        raise ValueError("VisDrone dataset SHA256 differs from BPDD IRA authority")
    if train_count != int(expected_dataset.get("train_images", -1)):
        raise ValueError("VisDrone train image count differs from authority")
    if val_count != int(expected_dataset.get("val_images", -1)):
        raise ValueError("VisDrone val image count differs from authority")

    combined_path = context.repository_root / "configs" / "rtdetr-l-fdr-bpdd-ira.yaml"
    bpdd_path = context.repository_root / "configs" / "rtdetr-l-fdr-bpdd.yaml"
    combined = yaml.safe_load(combined_path.read_text(encoding="utf-8"))
    bpdd = yaml.safe_load(bpdd_path.read_text(encoding="utf-8"))
    ira_rows = [row for row in combined.get("head", []) if row[2] == "IRA"]
    decoder = combined.get("head", [])[-1]
    yaml_checks = {
        "one_ira_layer": len(ira_rows) == 1,
        "ira_on_p3": bool(ira_rows and ira_rows[0] == [21, 1, "IRA", [256]]),
        "decoder_sources": decoder[0] == [22, 25, 28],
        "decoder_queries": decoder[3][-1]["num_queries"] == 300,
        "fdr_loss_unchanged": combined.get("fdr_loss") == bpdd.get("fdr_loss"),
        "bpdd_loss_unchanged": combined.get("bpdd_loss") == bpdd.get("bpdd_loss"),
    }
    if not all(yaml_checks.values()):
        raise RuntimeError(f"BPDD IRA YAML contract failed: {yaml_checks}")

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    environment = {
        "model": "Ultralytics RT-DETR-L",
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "ultralytics": str(ultralytics.__version__),
        "cuda": str(torch.version.cuda),
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0),
    }
    expected_environment = manifest["protocol"].get("environment", {})
    validate_environment_authority(environment, expected_environment)
    checks = {
        **yaml_checks,
        "source_exact": True,
        "environment_exact": True,
        "dataset_exact": True,
        "initial_state_exact": bool(artifact.get("fingerprints")),
    }
    return {
        "status": "passed",
        "gate": "I0",
        "checks": checks,
        "source": actual_source,
        "environment": environment,
        "dataset": {
            "sha256": actual_dataset["sha256"],
            "train_images": train_count,
            "val_images": val_count,
        },
        "yaml_sha256": _file_sha256(combined_path),
        "initial_state_sha256": _file_sha256(context.initial_state),
    }


def run_i1(context: PreflightContext) -> dict[str, Any]:
    """Prove exact shared/FDR state and isolate every new tensor to IRA."""

    import torch

    from src.rtdetr_fdr_bpdd_ira import remap_bpdd_ira_shared_key

    bpdd, combined, artifact, load_report = _models(context, torch.device("cpu"))
    bpdd_state = _cpu_state(bpdd)
    combined_state = _cpu_state(combined)
    mismatches = [
        name
        for name, expected in bpdd_state.items()
        if not torch.equal(combined_state[remap_bpdd_ira_shared_key(name)], expected)
    ]
    private = list(load_report["ira_private_keys"])
    checks = {
        "shared_keys_complete": int(load_report["shared_tensor_count"]) == len(bpdd_state),
        "shared_tensors_exact": not mismatches,
        "shared_mismatch_zero": int(load_report["shared_mismatch_count"]) == 0,
        "ira_private_nonempty": bool(private),
        "ira_only_private": all(name.startswith("model.22.") for name in private),
        "rezero_gate_zero": bool(torch.equal(combined.ira.residual_scale, torch.zeros_like(combined.ira.residual_scale))),
    }
    if not all(checks.values()):
        raise RuntimeError(f"BPDD IRA I1 state isolation failed: {checks}")
    return {
        "status": "passed",
        "gate": "I1",
        "checks": checks,
        "shared_tensor_count": len(bpdd_state),
        "ira_private_tensor_count": len(private),
        "initial_fdr_state_sha256": artifact["fingerprints"]["fdr"],
    }


def summarize_gradient_group(parameters: Sequence[Any]) -> dict[str, Any]:
    """Return finite/non-zero gradient evidence for one disjoint parameter group."""

    import torch

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    nonzero = sum(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    squared = sum(
        float(gradient.detach().float().square().sum().cpu())
        for gradient in gradients
        if torch.isfinite(gradient).all()
    )
    return {
        "parameter_tensors": len(parameters),
        "gradient_tensors": len(gradients),
        "finite": finite,
        "nonzero_tensors": nonzero,
        "l2_norm": math.sqrt(squared),
    }


def _gradient_groups(model: Any) -> dict[str, list[Any]]:
    from src.rtdetr_fdr_bpdd_ira import FDRBPDDIRATrainer

    trainer = FDRBPDDIRATrainer.__new__(FDRBPDDIRATrainer)
    trainer.model = model
    return trainer.gradient_parameter_groups()


def _real_batch(context: PreflightContext, device: Any, *, augment: bool) -> tuple[dict[str, Any], str]:
    from src.fdr_runtime_preflight import _build_loader, _move_batch

    loader, signature = _build_loader(context, augment=augment)
    batch = _move_batch(next(iter(loader)), device)
    if int(batch["img"].shape[0]) != 8:
        raise RuntimeError("real VisDrone preflight batch does not contain eight images")
    return batch, signature


def _backward(model: Any, batch: Mapping[str, Any], scaler: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=True):
        loss, _items = model.loss(dict(batch))
    scaler.scale(loss).backward()
    return loss, _gradient_groups(model)


def run_i2(context: PreflightContext) -> dict[str, Any]:
    """Run real batch8 CUDA, ReZero gradients, BPDD, and one AMP128 MuSGD step."""

    import torch

    from src.fdr_protocol import validate_optimizer_coverage
    from src.fdr_runtime_preflight import _musgd

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("I2 requires cuda:0 NVIDIA GeForce RTX 4090")
    device = torch.device("cuda:0")
    _bpdd, model, _artifact, _report = _models(context, device)
    del _bpdd
    batch, data_order_sha256 = _real_batch(context, device, augment=True)
    model.train()
    optimizer = _musgd(model)
    optimizer_coverage = validate_optimizer_coverage(model, optimizer)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=128.0, growth_interval=2**31 - 1
    )

    pristine = _cpu_state(model)
    with torch.no_grad():
        model.ira.residual_scale.fill_(0.1)
    open_loss, open_groups = _backward(model, batch, scaler)
    open_reports = {
        name: summarize_gradient_group(parameters)
        for name, parameters in open_groups.items()
    }
    open_all_ira = open_reports["ira_gradient_norm"]
    if not (
        bool(torch.isfinite(open_loss.detach()))
        and open_all_ira["finite"]
        and open_all_ira["gradient_tensors"] == open_all_ira["parameter_tensors"]
        and open_all_ira["nonzero_tensors"] == open_all_ira["parameter_tensors"]
    ):
        raise RuntimeError(f"I2 open-gate IRA gradients failed: {open_all_ira}")

    model.load_state_dict(pristine, strict=True)
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    loss, groups = _backward(model, batch, scaler)
    scaler.unscale_(optimizer)
    zero_reports = {
        name: summarize_gradient_group(parameters)
        for name, parameters in groups.items()
    }
    ira_parameters = dict(model.ira.named_parameters())
    gate_gradient = ira_parameters["residual_scale"].grad
    gate_live = bool(
        gate_gradient is not None
        and torch.isfinite(gate_gradient).all()
        and torch.count_nonzero(gate_gradient)
    )
    non_gate_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for name, parameter in ira_parameters.items()
        if name != "residual_scale"
    )
    common_live = zero_reports["gradient_norm"]["finite"] and zero_reports["gradient_norm"]["nonzero_tensors"] > 0
    fdr_live = zero_reports["fdr_gradient_norm"]["finite"] and zero_reports["fdr_gradient_norm"]["nonzero_tensors"] > 0
    if not (bool(torch.isfinite(loss.detach())) and common_live and fdr_live and gate_live and non_gate_finite):
        raise RuntimeError(f"I2 zero-gate gradient semantics failed: {zero_reports}")

    for parameters in groups.values():
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)
    stats = {
        name: float(value.detach().float().cpu())
        for name, value in model.last_bpdd_statistics.items()
    }
    losses = {
        name: float(value.detach().float().cpu())
        for name, value in model.last_fdr_losses.items()
    }
    bpdd_finite = bool(stats) and all(math.isfinite(value) for value in stats.values())
    loss_finite = bool(losses) and all(math.isfinite(value) for value in losses.values())
    checks = {
        "real_train_batch8": True,
        "losses_finite": loss_finite,
        "bpdd_statistics_finite": bpdd_finite,
        "bpdd_loss_present": "loss_bpdd" in losses,
        "common_gradient_live": common_live,
        "fdr_gradient_live": fdr_live,
        "rezero_gate_gradient_live": gate_live,
        "rezero_private_gradients_finite": non_gate_finite,
        "open_gate_all_ira_gradients_live": True,
        "optimizer_coverage_exact": optimizer_coverage["tensor_count"] > 0,
        "amp_scale_128": scale_before == scale_after == 128.0,
        "optimizer_step_not_skipped": scale_after >= scale_before,
    }
    if not all(checks.values()):
        raise RuntimeError(f"BPDD IRA I2 runtime failed: {checks}")
    return {
        "status": "passed",
        "gate": "I2",
        "checks": checks,
        "runtime": dict(FIXED_RUNTIME),
        "loss": float(loss.detach().float().cpu()),
        "losses": losses,
        "bpdd": stats,
        "zero_gate_gradients": zero_reports,
        "open_gate_gradients": open_reports,
        "optimizer_coverage": optimizer_coverage,
        "data_order_sha256": data_order_sha256,
    }


def _prediction_tensor(prediction: Any) -> Any:
    import torch

    if isinstance(prediction, torch.Tensor):
        return prediction
    if isinstance(prediction, (tuple, list)):
        for value in prediction:
            try:
                return _prediction_tensor(value)
            except RuntimeError:
                continue
    raise RuntimeError("evaluation output contains no prediction tensor")


def validate_prediction_contract(prediction: Any, *, batch_size: int) -> dict[str, Any]:
    """Require the ordinary RT-DETR [batch, 300, outputs] evaluation contract."""

    import torch

    tensor = _prediction_tensor(prediction)
    if tensor.ndim < 2 or int(tensor.shape[0]) != int(batch_size) or int(tensor.shape[1]) != 300:
        raise RuntimeError("evaluation output violates the 300-query contract")
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("evaluation output contains non-finite values")
    return {
        "batch": int(tensor.shape[0]),
        "queries": int(tensor.shape[1]),
        "finite": True,
    }


def run_i3(context: PreflightContext) -> dict[str, Any]:
    """Prove 300-query evaluation and bit-exact zero-gate BPDD parity."""

    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bpdd, combined, _artifact, _report = _models(context, device)
    batch, data_order_sha256 = _real_batch(context, device, augment=False)
    bpdd.eval()
    combined.eval()
    with torch.inference_mode():
        expected = bpdd.predict(batch["img"])
        actual = combined.predict(batch["img"])
    contract = validate_prediction_contract(actual, batch_size=8)
    parity = _tree_equal(actual, expected)
    checks = {
        "zero_gate": bool(torch.count_nonzero(combined.ira.residual_scale) == 0),
        "zero_gate_parity_exact": parity,
        "batch8": contract["batch"] == 8,
        "queries_300": contract["queries"] == 300,
        "prediction_finite": contract["finite"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"BPDD IRA I3 evaluation parity failed: {checks}")
    return {
        "status": "passed",
        "gate": "I3",
        "checks": checks,
        "prediction": contract,
        "data_order_sha256": data_order_sha256,
    }


def run_i4(context: PreflightContext) -> dict[str, Any]:
    """Prove strict combined checkpoint roundtrip and fail-closed resume."""

    import torch

    from src.fdr_protocol import public_state_sha256
    from src.rtdetr_fdr_bpdd_ira import (
        BPDD_IRA_MODEL_CFG,
        FDRBPDDIRADetectionModel,
        load_exact_fdr_bpdd_ira_resume_state,
    )

    _bpdd, model, _artifact, _report = _models(context, torch.device("cpu"))
    del _bpdd
    with torch.no_grad():
        model.ira.residual_scale.fill_(0.125)
    expected = _cpu_state(model)
    buffer = io.BytesIO()
    torch.save({"state_dict": expected}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    resumed = FDRBPDDIRADetectionModel(
        BPDD_IRA_MODEL_CFG,
        ch=3,
        nc=10,
        verbose=False,
        private_seed=10_000,
        ira_private_seed=20_000,
    )
    load_exact_fdr_bpdd_ira_resume_state(resumed, checkpoint)
    restored = _cpu_state(resumed)

    incomplete_rejected = False
    incomplete = dict(expected)
    incomplete.pop(next(iter(incomplete)))
    try:
        load_exact_fdr_bpdd_ira_resume_state(resumed, {"state_dict": incomplete})
    except ValueError:
        incomplete_rejected = True
    checks = {
        "state_keys_exact": set(expected) == set(restored),
        "state_tensors_exact": _tree_equal(expected, restored),
        "state_hash_exact": public_state_sha256(expected) == public_state_sha256(restored),
        "ira_gate_restored": float(resumed.ira.residual_scale) == 0.125,
        "incomplete_resume_rejected": incomplete_rejected,
    }
    if not all(checks.values()):
        raise RuntimeError(f"BPDD IRA I4 checkpoint/resume failed: {checks}")
    return {
        "status": "passed",
        "gate": "I4",
        "checks": checks,
        "state_sha256": public_state_sha256(restored),
        "tensor_count": len(restored),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision = run_preflight(
        PreflightContext(
            protocol_manifest=args.protocol_manifest,
            initial_state=args.initial_state,
            dataset_root=args.dataset_root,
            report_root=args.report_root,
        )
    )
    print(json.dumps(decision, sort_keys=True, allow_nan=False))
    return 0 if decision["formal_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

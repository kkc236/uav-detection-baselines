#!/usr/bin/env python3
"""Build, train, evaluate, and compare the paired RT-DETR-X GLGM experiment."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIG = PACKAGE_ROOT / "configs" / "rtdetr-x-glgm-control.yaml"
GLGM_CONFIG = PACKAGE_ROOT / "configs" / "rtdetr-x-glgm-only.yaml"
METRIC_KEYS = ("precision", "recall", "f1", "ap50", "ap75", "map50_95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True, help="Ultralytics source checkout containing GLGM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Verify code/config contracts and build paired initial states.")
    preflight.add_argument("--artifact-dir", type=Path, required=True)
    preflight.add_argument("--public-seed", type=int, default=0)
    preflight.add_argument("--private-seed", type=int, default=10000)
    preflight.add_argument("--device", default="0")
    preflight.add_argument("--full-forward", action="store_true")
    preflight.add_argument(
        "--base-weights",
        type=Path,
        help="Optional stock RT-DETR-X checkpoint. Compatible public tensors are remapped into both arms.",
    )

    train = subparsers.add_parser("train", help="Train one arm from its paired initial state or resume checkpoint.")
    train.add_argument("--arm", choices=("control", "glgm"), required=True)
    train.add_argument("--artifact-dir", type=Path, required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--runs-dir", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--device", default="0")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--fraction", type=float, default=1.0)
    train.add_argument("--save-period", type=int, default=5)
    train.add_argument("--name")

    evaluate = subparsers.add_parser("eval", help="Run one independent validation and write canonical JSON.")
    evaluate.add_argument("--arm", choices=("control", "glgm"), required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--train-receipt", type=Path, required=True)
    evaluate.add_argument("--checkpoint-kind", choices=("last", "best"), required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--project", type=Path, required=True)
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=4)
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--device", default="0")

    compare = subparsers.add_parser("compare", help="Compare canonical control and GLGM metric JSON files.")
    compare.add_argument("--control", type=Path, required=True)
    compare.add_argument("--glgm", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--exploratory",
        action="store_true",
        help="Allow the training-device field to differ and mark the comparison non-strict.",
    )

    benchmark = subparsers.add_parser("benchmark", help="Measure synchronized batch-1 model latency on CUDA.")
    benchmark.add_argument("--arm", choices=("control", "glgm"), required=True)
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--train-receipt", type=Path, required=True)
    benchmark.add_argument("--checkpoint-kind", choices=("last", "best"), required=True)
    benchmark.add_argument("--weights", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--device", default="0")
    benchmark.add_argument("--warmup", type=int, default=50)
    benchmark.add_argument("--iterations", type=int, default=200)
    benchmark.add_argument("--half", action="store_true")
    return parser.parse_args()


def import_runtime(repo_dir: Path):
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"not an Ultralytics source checkout: {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    import numpy as np
    import torch
    import ultralytics
    import yaml
    from ultralytics import RTDETR
    from ultralytics.nn.modules.block import GLGM

    imported = Path(ultralytics.__file__).resolve()
    expected = (repo_dir / "ultralytics" / "__init__.py").resolve()
    if imported != expected:
        raise RuntimeError(f"wrong Ultralytics import: expected {expected}, got {imported}")
    return np, torch, ultralytics, yaml, RTDETR, GLGM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tensor_fingerprint(torch, state: dict[str, Any], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(keys):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest().upper()


def canonical_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def mapping_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest().upper()


def ensure_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {name}: {value}")
    return value


def check_config_pair(yaml) -> None:
    control = yaml.safe_load(CONTROL_CONFIG.read_text(encoding="utf-8"))
    method = yaml.safe_load(GLGM_CONFIG.read_text(encoding="utf-8"))
    if control["nc"] != 10 or method["nc"] != 10:
        raise ValueError("paired VisDrone configs must use nc=10")
    if control["head"][2][2] != "nn.Identity" or method["head"][2][2] != "GLGM":
        raise ValueError("the only paired layer must be Identity versus GLGM at head index 2")
    normalized_control = copy.deepcopy(control)
    normalized_method = copy.deepcopy(method)
    normalized_control["head"][2] = ["PAIRED", 1, "PAIRED", [384]]
    normalized_method["head"][2] = ["PAIRED", 1, "PAIRED", [384]]
    if normalized_control != normalized_method:
        raise ValueError("control and GLGM configs differ outside the declared paired layer")


def current_source_hashes(repo_dir: Path) -> dict[str, str]:
    return {
        "block.py": sha256_file(repo_dir / "ultralytics" / "nn" / "modules" / "block.py"),
        "modules_init.py": sha256_file(repo_dir / "ultralytics" / "nn" / "modules" / "__init__.py"),
        "tasks.py": sha256_file(repo_dir / "ultralytics" / "nn" / "tasks.py"),
        "control_yaml": sha256_file(CONTROL_CONFIG),
        "glgm_yaml": sha256_file(GLGM_CONFIG),
        "experiment_script.py": sha256_file(Path(__file__)),
        "audit_script.py": sha256_file(PACKAGE_ROOT / "scripts" / "audit_visdrone.py"),
        "prepare_script.py": sha256_file(PACKAGE_ROOT / "scripts" / "prepare_visdrone.py"),
        "run_pair.sh": sha256_file(PACKAGE_ROOT / "scripts" / "run_glgm_pair.sh"),
        "repo_tree_sha256": tree_fingerprint(repo_dir),
    }


def tree_fingerprint(repo_dir: Path) -> str:
    digest = hashlib.sha256()
    excluded_parts = {".git", "__pycache__", ".pytest_cache", "runs", "weights"}
    excluded_suffixes = {".pyc", ".pyo", ".pt", ".cache"}
    files = [
        path
        for path in repo_dir.rglob("*")
        if path.is_file()
        and not excluded_parts.intersection(path.relative_to(repo_dir).parts)
        and path.suffix.lower() not in excluded_suffixes
    ]
    for path in sorted(files, key=lambda item: item.relative_to(repo_dir).as_posix()):
        relative = path.relative_to(repo_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def pip_freeze_fingerprint() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    normalized = "\n".join(sorted(line.strip() for line in result.stdout.splitlines() if line.strip())) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()


def verify_runtime(manifest: dict[str, Any], torch, ultralytics) -> None:
    actual = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "pip_freeze_sha256": pip_freeze_fingerprint(),
    }
    expected = {key: manifest.get(key) for key in actual}
    if expected != actual:
        raise RuntimeError(f"runtime changed after preflight: expected {expected}, got {actual}")


def load_and_verify_manifest(manifest_path: Path, repo_dir: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run preflight first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest.get("repo_dir", "")).resolve() != repo_dir.resolve():
        raise RuntimeError(f"repository path differs from preflight: {manifest.get('repo_dir')} != {repo_dir}")
    expected = manifest.get("source_sha256")
    actual = current_source_hashes(repo_dir)
    if expected != actual:
        raise RuntimeError(f"source/config hash mismatch after preflight: expected {expected}, got {actual}")
    return manifest


def verify_data_audit(audit_path: Path, data_path: Path) -> dict[str, Any]:
    if not audit_path.is_file():
        raise FileNotFoundError(f"run audit_visdrone.py first: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(data_path)
    if audit.get("data_yaml_sha256") != actual_hash:
        raise RuntimeError("data YAML changed after the VisDrone audit")
    return audit


def verify_train_receipt(
    receipt_path: Path, arm: str, weights_path: Path, checkpoint_kind: str
) -> dict[str, Any]:
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "glgm-train-receipt-v1" or receipt.get("arm") != arm:
        raise RuntimeError(f"invalid train receipt identity: {receipt_path}")
    protocol = receipt.get("protocol")
    if not isinstance(protocol, dict) or receipt.get("protocol_sha256") != mapping_fingerprint(protocol):
        raise RuntimeError(f"training protocol hash mismatch in receipt: {receipt_path}")
    completion = receipt.get("training_completion", {})
    results_path = Path(completion.get("results_csv", ""))
    if (
        not completion.get("all_rows_finite")
        or completion.get("completed_epochs") != completion.get("requested_epochs")
        or completion.get("results_row_count") != completion.get("requested_epochs")
        or not results_path.is_file()
        or sha256_file(results_path) != completion.get("results_csv_sha256")
    ):
        raise RuntimeError(f"training completion evidence is invalid: {receipt_path}")
    item = receipt.get("checkpoints", {}).get(checkpoint_kind, {})
    if item.get("kind") != checkpoint_kind:
        raise RuntimeError(f"checkpoint role is missing from train receipt: {checkpoint_kind}")
    if Path(item.get("path", "")).resolve() != weights_path.resolve():
        raise RuntimeError(f"checkpoint path does not match its receipt role: {weights_path}")
    if sha256_file(weights_path) != item.get("sha256"):
        raise RuntimeError(f"checkpoint hash does not match its receipt role: {weights_path}")
    return receipt


def validate_training_results(results_path: Path, expected_epochs: int) -> dict[str, Any]:
    if not results_path.is_file():
        raise FileNotFoundError(f"missing training results: {results_path}")
    with results_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != expected_epochs:
        raise RuntimeError(f"training produced {len(rows)} result rows, expected {expected_epochs}: {results_path}")
    if not reader.fieldnames:
        raise RuntimeError(f"training results have no columns: {results_path}")
    for row_index, row in enumerate(rows, start=1):
        for field in reader.fieldnames:
            raw = row.get(field)
            if raw is None or not raw.strip():
                raise RuntimeError(f"empty training result at row {row_index}, field {field!r}")
            value = float(raw)
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite training result at row {row_index}, field {field!r}: {raw}")
    return {
        "requested_epochs": expected_epochs,
        "completed_epochs": len(rows),
        "results_csv": str(results_path.resolve()),
        "results_csv_sha256": sha256_file(results_path),
        "results_row_count": len(rows),
        "all_rows_finite": True,
    }


def paired_protocol(protocol: dict[str, Any], exploratory: bool) -> dict[str, Any]:
    normalized = copy.deepcopy(protocol)
    normalized.pop("name", None)
    if exploratory:
        normalized.pop("device", None)
    return normalized


def seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_to_stock_key(name: str) -> str:
    """Map the Identity-aligned target layer index to the stock RT-DETR-X layer index."""
    match = re.match(r"^model\.(\d+)\.(.+)$", name)
    if not match:
        return name
    index = int(match.group(1))
    if index >= 17:
        index -= 1
    return f"model.{index}.{match.group(2)}"


def make_checkpoint(torch, wrapper, model_yaml: Path, arm: str, source_hash: str) -> dict[str, Any]:
    model = copy.deepcopy(wrapper.model).float().cpu()
    model.args = dict(getattr(model, "args", {}) or {})
    model.args.update({"task": "detect", "model": str(model_yaml), "nc": 10})
    return {
        "epoch": -1,
        "best_fitness": None,
        "model": model,
        "ema": None,
        "optimizer": None,
        "scaler": None,
        "train_args": {"task": "detect", "model": str(model_yaml), "data": None, "imgsz": 640},
        "date": datetime.now(timezone.utc).isoformat(),
        "paired_arm": arm,
        "paired_public_sha256": source_hash,
    }


def all_tensors_finite(torch, value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (list, tuple)):
        return all(all_tensors_finite(torch, item) for item in value)
    if isinstance(value, dict):
        return all(all_tensors_finite(torch, item) for item in value.values())
    return True


def run_preflight(args: argparse.Namespace, runtime) -> None:
    np, torch, ultralytics, yaml, RTDETR, GLGM = runtime
    del np
    check_config_pair(yaml)
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(torch, args.public_seed)
    control = RTDETR(str(CONTROL_CONFIG))
    with torch.random.fork_rng(devices=[]):
        seed_everything(torch, args.private_seed)
        method = RTDETR(str(GLGM_CONFIG))

    control_state = control.model.state_dict()
    method_state = method.model.state_dict()
    common_keys = sorted(set(control_state) & set(method_state))
    if set(control_state) != set(common_keys):
        raise RuntimeError("control contains a tensor absent from the GLGM arm")
    if any(control_state[name].shape != method_state[name].shape for name in common_keys):
        raise RuntimeError("a common tensor has different shape across paired arms")
    private_keys = sorted(set(method_state) - set(control_state))
    if not private_keys or not all(name.startswith("model.16.") for name in private_keys):
        raise RuntimeError(f"undeclared GLGM private tensor: {private_keys[:5]}")

    initialization: dict[str, Any] = {"mode": "scratch"}
    if args.base_weights:
        base_weights = args.base_weights.resolve()
        if not base_weights.is_file():
            raise FileNotFoundError(base_weights)
        source = RTDETR(str(base_weights))
        source_state = source.model.state_dict()
        loaded_keys = []
        loaded_parameters = 0
        public_parameters = sum(control_state[name].numel() for name in common_keys)
        for name in common_keys:
            source_name = target_to_stock_key(name)
            if source_name in source_state and source_state[source_name].shape == control_state[name].shape:
                control_state[name] = source_state[source_name].detach().clone()
                loaded_keys.append(name)
                loaded_parameters += control_state[name].numel()
        loaded_fraction = loaded_parameters / public_parameters
        if loaded_fraction < 0.95:
            raise RuntimeError(
                f"pretrained public-parameter coverage is too low: {loaded_fraction:.4%}; "
                "the checkpoint may not be stock RT-DETR-X"
            )
        control.model.load_state_dict(control_state, strict=True)
        initialization = {
            "mode": "stock_rtdetr_x_pretrained",
            "base_weights": str(base_weights),
            "base_weights_sha256": sha256_file(base_weights),
            "loaded_public_tensor_count": len(loaded_keys),
            "loaded_public_parameter_count": loaded_parameters,
            "public_parameter_count": public_parameters,
            "loaded_public_parameter_fraction": loaded_fraction,
        }
        del source, source_state

    paired_state = {name: value.detach().clone() for name, value in method_state.items()}
    for name in common_keys:
        paired_state[name] = control_state[name].detach().clone()
    method.model.load_state_dict(paired_state, strict=True)

    control_state = control.model.state_dict()
    method_state = method.model.state_dict()
    public_hash = tensor_fingerprint(torch, control_state, common_keys)
    if tensor_fingerprint(torch, method_state, common_keys) != public_hash:
        raise RuntimeError("public tensors are not byte-identical after pairing")
    private_hash = tensor_fingerprint(torch, method_state, private_keys)

    control_init = artifact_dir / "control_paired_init.pt"
    glgm_init = artifact_dir / "glgm_paired_init.pt"
    torch.save(make_checkpoint(torch, control, CONTROL_CONFIG, "control", public_hash), control_init)
    torch.save(make_checkpoint(torch, method, GLGM_CONFIG, "glgm", public_hash), glgm_init)

    reloaded_control = RTDETR(str(control_init)).model.state_dict()
    if set(reloaded_control) != set(common_keys):
        raise RuntimeError("reloaded control checkpoint has an unexpected tensor set")
    if tensor_fingerprint(torch, reloaded_control, common_keys) != public_hash:
        raise RuntimeError("reloaded control checkpoint public hash mismatch")
    del reloaded_control
    reloaded_method = RTDETR(str(glgm_init)).model.state_dict()
    if set(reloaded_method) != set(common_keys) | set(private_keys):
        raise RuntimeError("reloaded GLGM checkpoint has an unexpected tensor set")
    if tensor_fingerprint(torch, reloaded_method, common_keys) != public_hash:
        raise RuntimeError("reloaded GLGM checkpoint public hash mismatch")
    if tensor_fingerprint(torch, reloaded_method, private_keys) != private_hash:
        raise RuntimeError("reloaded GLGM checkpoint private hash mismatch")
    del reloaded_method

    module = GLGM(384, 384)
    module_input = torch.randn(2, 384, 20, 20, requires_grad=True)
    module_output = module(module_input)
    module_output.float().mean().backward()
    if module_output.shape != module_input.shape or not all_tensors_finite(torch, module_output):
        raise RuntimeError("GLGM module-only forward contract failed")
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters()):
        raise RuntimeError("GLGM module-only backward contract failed")

    full_forward = None
    if args.full_forward:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        method.model.eval().to(device)
        sample = torch.zeros(1, 3, 640, 640, device=device)
        with torch.inference_mode():
            output = method.model(sample)
        if not all_tensors_finite(torch, output):
            raise RuntimeError("non-finite full-model output")
        full_forward = {
            "device": str(device),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
        method.model.cpu()

    control_params = sum(parameter.numel() for parameter in control.model.parameters())
    method_params = sum(parameter.numel() for parameter in method.model.parameters())
    manifest = {
        "schema": "glgm-paired-preflight-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_dir": str(args.repo_dir.resolve()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "pip_freeze_sha256": pip_freeze_fingerprint(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "public_seed": args.public_seed,
        "private_seed": args.private_seed,
        "initialization": initialization,
        "common_tensor_count": len(common_keys),
        "private_tensor_count": len(private_keys),
        "public_state_sha256": public_hash,
        "private_state_sha256": private_hash,
        "control_parameters": control_params,
        "glgm_parameters": method_params,
        "parameter_delta": method_params - control_params,
        "parameter_delta_percent": 100.0 * (method_params - control_params) / control_params,
        "control_init": {"path": str(control_init), "sha256": sha256_file(control_init)},
        "glgm_init": {"path": str(glgm_init), "sha256": sha256_file(glgm_init)},
        "source_sha256": current_source_hashes(args.repo_dir),
        "checkpoint_reload_verified": True,
        "module_only_forward_backward": True,
        "full_forward": full_forward,
    }
    canonical_json(artifact_dir / "paired_preflight_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "cache": False,
        "seed": args.seed,
        "deterministic": True,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 64,
        "cos_lr": False,
        "patience": 0,
        "amp": True,
        "fraction": args.fraction,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "save": True,
        "save_period": args.save_period,
        "val": True,
        "plots": True,
        "project": str(args.runs_dir.resolve()),
        "name": args.name or f"{args.arm}-seed{args.seed}-e{args.epochs}",
        "exist_ok": False,
    }


def run_train(args: argparse.Namespace, runtime) -> None:
    np, torch, ultralytics, yaml, RTDETR, GLGM = runtime
    del np, yaml, GLGM
    manifest_path = args.artifact_dir / "paired_preflight_manifest.json"
    manifest = load_and_verify_manifest(manifest_path, args.repo_dir)
    verify_runtime(manifest, torch, ultralytics)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    audit = verify_data_audit(args.artifact_dir / "visdrone-audit.json", args.data.resolve())
    if args.resume:
        raise RuntimeError(
            "strict paired mode does not permit single-arm resume; restart both arms from paired initialization"
        )

    init_path = Path(manifest[f"{args.arm}_init"]["path"])
    if sha256_file(init_path) != manifest[f"{args.arm}_init"]["sha256"]:
        raise RuntimeError(f"paired initial checkpoint hash mismatch: {init_path}")
    run_name = args.name or f"{args.arm}-seed{args.seed}-e{args.epochs}"
    expected_save_dir = (args.runs_dir.resolve() / run_name).resolve()
    if expected_save_dir.exists():
        raise FileExistsError(f"refusing to reuse an existing run directory: {expected_save_dir}")
    model = RTDETR(str(init_path))
    protocol = train_kwargs(args)
    model.train(**protocol)
    actual_save_dir = Path(model.trainer.save_dir).resolve()
    if actual_save_dir != expected_save_dir:
        raise RuntimeError(f"unexpected training output directory: {actual_save_dir} != {expected_save_dir}")
    completed_epochs = int(model.trainer.epoch) + 1
    if completed_epochs != args.epochs:
        raise RuntimeError(f"training stopped at epoch {completed_epochs}, expected {args.epochs}")
    training_completion = validate_training_results(actual_save_dir / "results.csv", args.epochs)
    required = {
        "last": actual_save_dir / "weights" / "last.pt",
        "best": actual_save_dir / "weights" / "best.pt",
    }
    for kind, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {kind} checkpoint after training: {path}")
    receipt = {
        "schema": "glgm-train-receipt-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "run_name": run_name,
        "save_dir": str(actual_save_dir),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "public_state_sha256": manifest["public_state_sha256"],
        "initial_checkpoint_sha256": manifest[f"{args.arm}_init"]["sha256"],
        "data_yaml_sha256": audit["data_yaml_sha256"],
        "data_inventory_sha256": {
            split: audit["splits"][split]["inventory_sha256"] for split in ("train", "val")
        },
        "protocol": protocol,
        "protocol_sha256": mapping_fingerprint(protocol),
        "training_completion": training_completion,
        "checkpoints": {
            kind: {"kind": kind, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for kind, path in required.items()
        },
    }
    canonical_json(args.artifact_dir / f"{args.arm}-train-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def run_eval(args: argparse.Namespace, runtime) -> None:
    np, torch, ultralytics, yaml, RTDETR, GLGM = runtime
    del GLGM
    manifest = load_and_verify_manifest(args.manifest.resolve(), args.repo_dir)
    verify_runtime(manifest, torch, ultralytics)
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError(f"missing weights or data YAML: {args.weights}, {args.data}")
    audit = verify_data_audit(args.manifest.resolve().parent / "visdrone-audit.json", args.data.resolve())
    receipt = verify_train_receipt(
        args.train_receipt.resolve(), args.arm, args.weights.resolve(), args.checkpoint_kind
    )
    if receipt["manifest_sha256"] != sha256_file(args.manifest):
        raise RuntimeError("train receipt belongs to a different preflight manifest")
    model = RTDETR(str(args.weights.resolve()))
    has_glgm = any(module.__class__.__name__ == "GLGM" for module in model.model.modules())
    if has_glgm != (args.arm == "glgm"):
        raise RuntimeError(f"evaluation checkpoint architecture does not match requested arm: {args.arm}")
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    configured_names = data_config["names"]

    def class_name(class_id: int) -> str:
        if isinstance(configured_names, dict):
            return str(configured_names.get(class_id, configured_names.get(str(class_id), class_id)))
        return str(configured_names[class_id])

    result = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        max_det=300,
        plots=True,
        project=str(args.project.resolve()),
        name=f"{args.arm}-{args.weights.stem}-{args.split}",
        exist_ok=False,
    )
    box = result.box
    metrics = {
        "precision": ensure_finite(result.results_dict["metrics/precision(B)"], "precision"),
        "recall": ensure_finite(result.results_dict["metrics/recall(B)"], "recall"),
        "f1": ensure_finite(np.nanmean(box.f1), "f1"),
        "ap50": ensure_finite(result.results_dict["metrics/mAP50(B)"], "ap50"),
        "ap75": ensure_finite(np.nanmean(box.all_ap[:, 5]), "ap75"),
        "map50_95": ensure_finite(result.results_dict["metrics/mAP50-95(B)"], "map50_95"),
    }
    per_class = []
    class_indexes = [int(index) for index in box.ap_class_index]
    for row_index, class_id in enumerate(class_indexes):
        per_class.append(
            {
                "id": class_id,
                "name": class_name(class_id),
                "precision": ensure_finite(box.p[row_index], f"class_{class_id}_precision"),
                "recall": ensure_finite(box.r[row_index], f"class_{class_id}_recall"),
                "f1": ensure_finite(box.f1[row_index], f"class_{class_id}_f1"),
                "ap50": ensure_finite(box.ap50[row_index], f"class_{class_id}_ap50"),
                "ap75": ensure_finite(box.all_ap[row_index, 5], f"class_{class_id}_ap75"),
                "map50_95": ensure_finite(box.ap[row_index], f"class_{class_id}_map50_95"),
            }
        )
    payload = {
        "schema": "glgm-independent-evaluation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights),
        "checkpoint_kind": args.checkpoint_kind,
        "train_receipt": str(args.train_receipt.resolve()),
        "train_receipt_sha256": sha256_file(args.train_receipt),
        "manifest_sha256": sha256_file(args.manifest),
        "public_state_sha256": manifest["public_state_sha256"],
        "data": str(args.data.resolve()),
        "data_yaml_sha256": sha256_file(args.data),
        "data_inventory_sha256": {
            split: audit["splits"][split]["inventory_sha256"] for split in ("train", "val")
        },
        "seed": receipt["protocol"]["seed"],
        "epochs": receipt["protocol"]["epochs"],
        "training_protocol": receipt["protocol"],
        "training_protocol_sha256": receipt["protocol_sha256"],
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "max_det": 300,
        "metrics": metrics,
        "per_class": per_class,
        "parameters": parameters,
        "speed_ms_per_image": {key: ensure_finite(value, f"speed_{key}") for key, value in result.speed.items()},
        "validator_save_dir": str(result.save_dir),
    }
    canonical_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_compare(args: argparse.Namespace) -> None:
    control = json.loads(args.control.read_text(encoding="utf-8"))
    method = json.loads(args.glgm.read_text(encoding="utf-8"))
    if control.get("schema") != "glgm-independent-evaluation-v1" or control.get("arm") != "control":
        raise ValueError("invalid control evaluation identity")
    if method.get("schema") != "glgm-independent-evaluation-v1" or method.get("arm") != "glgm":
        raise ValueError("invalid GLGM evaluation identity")
    for field in (
        "manifest_sha256",
        "public_state_sha256",
        "data_yaml_sha256",
        "data_inventory_sha256",
        "seed",
        "epochs",
        "checkpoint_kind",
        "split",
        "imgsz",
        "batch",
        "max_det",
    ):
        if control[field] != method[field]:
            raise ValueError(f"evaluation contract mismatch for {field}: {control[field]} != {method[field]}")
    control_protocol = paired_protocol(control["training_protocol"], args.exploratory)
    method_protocol = paired_protocol(method["training_protocol"], args.exploratory)
    if control_protocol != method_protocol:
        raise ValueError(
            f"training protocol mismatch: control={control_protocol}, glgm={method_protocol}"
        )
    rows = {}
    for key in METRIC_KEYS:
        baseline = ensure_finite(control["metrics"][key], f"control_{key}")
        candidate = ensure_finite(method["metrics"][key], f"glgm_{key}")
        rows[key] = {
            "control": baseline,
            "glgm": candidate,
            "absolute_delta": candidate - baseline,
            "percentage_point_delta": 100.0 * (candidate - baseline),
            "relative_percent": 100.0 * (candidate - baseline) / baseline if baseline else None,
        }
    control_classes = {(row["id"], row["name"]): row for row in control["per_class"]}
    method_classes = {(row["id"], row["name"]): row for row in method["per_class"]}
    if set(control_classes) != set(method_classes):
        raise ValueError("per-class identities differ between evaluations")
    per_class = []
    for identity in sorted(control_classes):
        baseline_row = control_classes[identity]
        candidate_row = method_classes[identity]
        deltas = {key: candidate_row[key] - baseline_row[key] for key in METRIC_KEYS}
        per_class.append({"id": identity[0], "name": identity[1], "delta": deltas})

    speed_keys = sorted(set(control["speed_ms_per_image"]) & set(method["speed_ms_per_image"]))
    speed = {
        key: {
            "control_ms": control["speed_ms_per_image"][key],
            "glgm_ms": method["speed_ms_per_image"][key],
            "delta_ms": method["speed_ms_per_image"][key] - control["speed_ms_per_image"][key],
        }
        for key in speed_keys
    }
    payload = {
        "schema": "glgm-paired-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "control_report": str(args.control.resolve()),
        "glgm_report": str(args.glgm.resolve()),
        "strict_pair": not args.exploratory,
        "allowed_protocol_differences": ["device"] if args.exploratory else [],
        "paired_training_protocol_sha256": mapping_fingerprint(control_protocol),
        "metrics": rows,
        "per_class_delta": per_class,
        "speed_ms_per_image": speed,
        "parameter_delta": method["parameters"] - control["parameters"],
        "parameter_delta_percent": 100.0
        * (method["parameters"] - control["parameters"])
        / control["parameters"],
    }
    canonical_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_benchmark(args: argparse.Namespace, runtime) -> None:
    np, torch, ultralytics, yaml, RTDETR, GLGM = runtime
    del yaml, GLGM
    manifest = load_and_verify_manifest(args.manifest.resolve(), args.repo_dir)
    verify_runtime(manifest, torch, ultralytics)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal latency benchmark")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    receipt = verify_train_receipt(
        args.train_receipt.resolve(), args.arm, args.weights.resolve(), args.checkpoint_kind
    )
    if receipt["manifest_sha256"] != sha256_file(args.manifest):
        raise RuntimeError("train receipt belongs to a different preflight manifest")
    model = RTDETR(str(args.weights.resolve()))
    has_glgm = any(module.__class__.__name__ == "GLGM" for module in model.model.modules())
    if has_glgm != (args.arm == "glgm"):
        raise RuntimeError(f"benchmark checkpoint architecture does not match requested arm: {args.arm}")
    device = torch.device(f"cuda:{args.device}")
    network = model.model.eval().to(device)
    dtype = torch.float16 if args.half else torch.float32
    if args.half:
        network.half()
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            network(sample)
        torch.cuda.synchronize(device)
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = network(sample)
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    if not all_tensors_finite(torch, output):
        raise RuntimeError("non-finite benchmark output")
    mean_ms = ensure_finite(np.mean(timings), "benchmark_mean_ms")
    payload = {
        "schema": "glgm-latency-benchmark-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights),
        "checkpoint_kind": args.checkpoint_kind,
        "train_receipt_sha256": sha256_file(args.train_receipt),
        "manifest_sha256": sha256_file(args.manifest),
        "public_state_sha256": manifest["public_state_sha256"],
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "imgsz": args.imgsz,
        "batch": 1,
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "mean_ms": mean_ms,
        "p50_ms": ensure_finite(np.percentile(timings, 50), "benchmark_p50_ms"),
        "p95_ms": ensure_finite(np.percentile(timings, 95), "benchmark_p95_ms"),
        "fps": 1000.0 / mean_ms,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
    }
    canonical_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    args.repo_dir = args.repo_dir.resolve()
    if args.command == "compare":
        run_compare(args)
        return
    runtime = import_runtime(args.repo_dir)
    if args.command == "preflight":
        run_preflight(args, runtime)
    elif args.command == "train":
        run_train(args, runtime)
    elif args.command == "eval":
        run_eval(args, runtime)
    elif args.command == "benchmark":
        run_benchmark(args, runtime)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

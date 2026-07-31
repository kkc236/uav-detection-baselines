"""Frozen paired-protocol utilities shared by LPR control and method arms."""

from __future__ import annotations

import json
import platform
import subprocess
from hashlib import sha256
from math import floor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


EXPECTED_DATASET_SHA256 = "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
EXPECTED_SUBSET_SHA256 = "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
EXPECTED_ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 4090",
    "driver": "550.142",
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "cuda": "12.1",
    "ultralytics": "8.4.90",
}
CATEGORY_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def _relative_name(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def select_hashed_subset(
    image_paths: Iterable[Path],
    *,
    root: Path,
    fraction: float,
) -> list[Path]:
    """Select a stable subset ranked by the hash of each relative image path."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    candidates = [Path(path) for path in image_paths]
    ranked = sorted(
        candidates,
        key=lambda path: (
            sha256(_relative_name(path, root).encode("utf-8")).digest(),
            _relative_name(path, root),
        ),
    )
    if not ranked:
        return []
    return ranked[: max(1, floor(len(ranked) * fraction))]


def subset_signature(paths: Iterable[Path], *, root: Path) -> str:
    digest = sha256()
    for path in paths:
        digest.update(_relative_name(Path(path), root).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_sha256(path: Path) -> str:
    """Return an uppercase streaming SHA-256 for a protocol artifact."""
    return _file_sha256(Path(path))


def dataset_signature(dataset_root: Path) -> dict[str, int | str]:
    dataset_root = Path(dataset_root).resolve()
    files = sorted(
        path
        for directory in ("images", "labels")
        for split in ("train", "val")
        for path in (dataset_root / directory / split).glob("**/*")
        if path.is_file()
    )
    digest = sha256()
    for path in files:
        digest.update(_relative_name(path, dataset_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return {"file_count": len(files), "sha256": digest.hexdigest().upper()}


def category_mapping_sha256(names: Mapping[int, str] | Sequence[str]) -> str:
    if isinstance(names, Mapping):
        normalized = {int(index): names[index] for index in sorted(names)}
    else:
        normalized = {index: name for index, name in enumerate(names)}
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(payload).hexdigest().upper()


def current_environment() -> dict[str, str | None]:
    import torchvision
    import ultralytics

    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        driver = result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": driver,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
    }


def environment_violations(actual: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in EXPECTED_ENVIRONMENT.items()
        if actual.get(key) != expected
    }


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(state):
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


def build_initial_state(
    control_state: Mapping[str, torch.Tensor],
    lpr_state: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    common_names = set(control_state)
    lpr_names = set(lpr_state)
    missing = common_names - lpr_names
    if missing:
        raise ValueError(f"LPR state is missing common tensors: {sorted(missing)[:5]}")
    for name in common_names:
        if control_state[name].shape != lpr_state[name].shape:
            raise ValueError(f"common tensor shape mismatch: {name}")
    private_names = lpr_names - common_names
    if any("lpr_refiners" not in name for name in private_names):
        raise ValueError("initial state contains an unapproved LPR-private tensor")

    common = {name: value.detach().cpu().clone() for name, value in control_state.items()}
    private = {name: lpr_state[name].detach().cpu().clone() for name in sorted(private_names)}
    return {
        "format_version": 1,
        "common_state": common,
        "lpr_state": private,
        "metadata": dict(metadata),
        "fingerprints": {
            "common": state_fingerprint(common),
            "lpr": state_fingerprint(private),
        },
    }


def load_initial_state(model, artifact: Mapping[str, Any], *, variant: str) -> None:
    if variant not in {"control", "lpr"}:
        raise ValueError(f"unknown paired variant: {variant}")
    common = artifact["common_state"]
    private = artifact["lpr_state"]
    fingerprints = artifact["fingerprints"]
    if state_fingerprint(common) != fingerprints["common"]:
        raise ValueError("common initial-state fingerprint mismatch")
    if state_fingerprint(private) != fingerprints["lpr"]:
        raise ValueError("LPR initial-state fingerprint mismatch")
    expected = dict(common)
    if variant == "lpr":
        expected.update(private)
    model_names = set(model.state_dict())
    if model_names != set(expected):
        missing = sorted(model_names - set(expected))
        unexpected = sorted(set(expected) - model_names)
        raise ValueError(f"initial-state keys do not match model: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.load_state_dict(expected, strict=True)

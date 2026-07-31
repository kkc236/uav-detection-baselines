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
EXPECTED_COMMON_FINGERPRINTS = {
    0: "0B968046FDC89BE5A31581C81F7335A9742BC422503428113637B1CC829F0FA0",
    1: "A73D3A57F5DCF3F62FA4B30329C32204E3A74BC57AA4FFEE577873D14F0A3D65",
    2: "1CCA2D745106F949268B3978722A415439623376D01C1D188B1450C6230AF1B2",
}
EXPECTED_SOURCE_SHA256 = {
    "head.py": "5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F",
    "tasks.py": "B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92",
    "rtdetr-l.yaml": "85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83",
}
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


def ultralytics_source_paths() -> dict[str, Path]:
    import ultralytics

    root = Path(ultralytics.__file__).resolve().parent
    return {
        "head.py": root / "nn" / "modules" / "head.py",
        "tasks.py": root / "nn" / "tasks.py",
        "rtdetr-l.yaml": root / "cfg" / "models" / "rt-detr" / "rtdetr-l.yaml",
    }


def source_violations(paths: Mapping[str, Path] | None = None) -> dict[str, dict[str, Any]]:
    actual_paths = dict(paths or ultralytics_source_paths())
    violations = {}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        path = actual_paths.get(name)
        actual = file_sha256(path) if path is not None and Path(path).is_file() else None
        if actual != expected:
            violations[name] = {
                "expected": expected,
                "actual": actual,
                "path": str(path) if path is not None else None,
            }
    return violations


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


def validate_initial_state_authority(
    path: str | Path,
    *,
    seed: int,
    manifest_record: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = Path(path).resolve()
    if state_path != Path(manifest_record.get("path", "")).resolve():
        raise ValueError("initial-state path does not match protocol manifest")
    if not state_path.is_file():
        raise FileNotFoundError(f"missing paired initial state: {state_path}")
    actual_file_sha = file_sha256(state_path)
    if actual_file_sha != manifest_record.get("sha256"):
        raise ValueError(
            f"initial-state file SHA mismatch: expected={manifest_record.get('sha256')}, actual={actual_file_sha}"
        )
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("initial-state artifact format is invalid")
    if artifact.get("metadata", {}).get("seed") != seed:
        raise ValueError("initial-state seed does not match paired arm")
    fingerprints = artifact.get("fingerprints", {})
    if fingerprints != manifest_record.get("fingerprints"):
        raise ValueError("initial-state fingerprints do not match protocol manifest")
    if state_fingerprint(artifact.get("common_state", {})) != fingerprints.get("common"):
        raise ValueError("common initial-state fingerprint mismatch")
    if state_fingerprint(artifact.get("lpr_state", {})) != fingerprints.get("lpr"):
        raise ValueError("LPR initial-state fingerprint mismatch")
    expected_common = EXPECTED_COMMON_FINGERPRINTS.get(seed)
    if fingerprints.get("common") != expected_common:
        raise ValueError(
            f"common initial-state is not the Linux authority for seed {seed}: "
            f"expected={expected_common}, actual={fingerprints.get('common')}"
        )
    return artifact

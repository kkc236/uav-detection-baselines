from __future__ import annotations

from hashlib import sha256
from math import floor
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn


INNOVATION_MODULES = frozenset({"p2_adapter", "p2_bbox_head"})


def select_hashed_subset(
    image_paths: Iterable[Path],
    *,
    root: Path,
    fraction: float,
) -> list[Path]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    root = root.resolve()
    ranked = sorted(
        (Path(path) for path in image_paths),
        key=lambda path: (sha256(_relative_name(path, root).encode("utf-8")).digest(), _relative_name(path, root)),
    )
    if not ranked:
        return []
    count = max(1, floor(len(ranked) * fraction))
    return ranked[:count]


def subset_signature(paths: Iterable[Path], *, root: Path) -> str:
    root = root.resolve()
    digest = sha256()
    for path in paths:
        digest.update(_relative_name(Path(path), root).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def write_d2_subset(
    image_paths: Iterable[Path],
    *,
    root: Path,
    output: Path,
    fraction: float = 0.10,
) -> dict[str, int | float | str]:
    selected = select_hashed_subset(image_paths, root=root, fraction=fraction)
    content = "".join(f"{path.resolve()}\n" for path in selected)
    output = Path(output)
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace changed subset file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return {
        "count": len(selected),
        "fraction": fraction,
        "sha256": subset_signature(selected, root=root),
    }


def build_initial_state(
    control_state: Mapping[str, torch.Tensor],
    method_state: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    common_names = set(control_state)
    method_names = set(method_state)
    missing = common_names - method_names
    if missing:
        raise ValueError(f"method is missing common state: {sorted(missing)}")
    innovation_names = method_names - common_names
    unapproved = {name for name in innovation_names if not _is_innovation_name(name)}
    if unapproved:
        raise ValueError(f"unapproved innovation state: {sorted(unapproved)}")

    for name in sorted(common_names):
        if control_state[name].shape != method_state[name].shape:
            raise ValueError(f"common tensor shape mismatch: {name}")

    common = {name: value.detach().cpu().clone() for name, value in control_state.items()}
    innovation = {name: method_state[name].detach().cpu().clone() for name in sorted(innovation_names)}
    return {
        "format_version": 1,
        "common_state": common,
        "innovation_state": innovation,
        "metadata": dict(metadata),
        "fingerprints": {
            "common": state_fingerprint(common),
            "innovation": state_fingerprint(innovation),
        },
    }


def load_initial_state(model: nn.Module, artifact: Mapping[str, Any], *, include_innovation: bool) -> None:
    common = artifact["common_state"]
    innovation = artifact["innovation_state"]
    fingerprints = artifact["fingerprints"]
    if state_fingerprint(common) != fingerprints["common"]:
        raise ValueError("common initial-state fingerprint mismatch")
    if state_fingerprint(innovation) != fingerprints["innovation"]:
        raise ValueError("innovation initial-state fingerprint mismatch")

    expected = dict(common)
    if include_innovation:
        expected.update(innovation)
    model_names = set(model.state_dict())
    expected_names = set(expected)
    if model_names != expected_names:
        missing = sorted(model_names - expected_names)
        unexpected = sorted(expected_names - model_names)
        raise ValueError(f"initial-state keys do not match model: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(expected, strict=True)


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


def dataset_signature(dataset_root: Path) -> dict[str, int | str]:
    dataset_root = dataset_root.resolve()
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


def _relative_name(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _is_innovation_name(name: str) -> bool:
    return any(part in INNOVATION_MODULES for part in name.split("."))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

"""Immutable paired authority for the Ultralytics FDR-only experiment."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


DFINE_COMMIT = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
PRIVATE_SEED = 10_000
FORMAT_VERSION = 1

FDR_PROTOCOL: dict[str, Any] = {
    "design": "ultralytics-rtdetr-l-fdr-only-v1",
    "dfine_commit": DFINE_COMMIT,
    "reg_max": 32,
    "reg_scale": 4.0,
    "up": 0.5,
    "loss_weights": {"vfl": 1.0, "bbox": 5.0, "giou": 2.0, "fgl": 0.15},
    "excluded": ["DDF", "GO-LSD", "teacher", "LQE", "target_gating"],
    "environment": {
        "model": "Ultralytics RT-DETR-L",
        "ultralytics": "8.4.90",
        "gpu": "NVIDIA GeForce RTX 4090",
        "driver": "550.142",
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda": "12.1",
    },
    "dataset": {
        "name": "VisDrone",
        "train_images": 6471,
        "val_images": 548,
        "classes": 10,
        "sha256": "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB",
        "screen_train_images": 647,
        "screen_sha256": "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0",
    },
    "training": {
        "pretrained": False,
        "screen_schedule_epochs": 50,
        "screen_cutoff_epoch": 30,
        "formal_schedule_epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": 0,
        "amp": True,
        "amp_scale": 128.0,
        "seeds": [0],
        "deterministic": True,
        "cache": False,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 64,
        "cos_lr": False,
        "queries": 300,
        "max_det": 300,
        "nms": False,
    },
    "augmentation": {
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    },
    "initialization": {
        "public": "one shared seed0 state copied byte-for-byte",
        "private_seed": PRIVATE_SEED,
        "pre_bbox_head": "copied from stock decoder layer 0",
        "distribution_finals": "six zero-initialized linear layers",
    },
}


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize protocol data to the single accepted JSON representation."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        if value.is_quantized or value.layout is not torch.strided:
            raise TypeError(f"state entry {name!r} cannot be byte-fingerprinted")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest().upper()


def public_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor state byte-for-byte, or canonical protocol mappings."""
    if state and all(isinstance(value, torch.Tensor) for value in state.values()):
        return _tensor_state_sha256(state)  # type: ignore[arg-type]
    return hashlib.sha256(canonical_json_bytes(state)).hexdigest().upper()


FDR_PROTOCOL_SHA256 = public_state_sha256(FDR_PROTOCOL)


def partition_state_dicts(
    control_state: Mapping[str, torch.Tensor],
    fdr_state: Mapping[str, torch.Tensor],
    *,
    private_prefixes: Sequence[str],
    public_aliases: Mapping[str, str] | None = None,
    replaced_control_prefixes: Sequence[str] = (),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Validate exact shared tensors across a declared structural migration."""
    if not private_prefixes or any(not prefix for prefix in private_prefixes):
        raise ValueError("at least one non-empty private prefix is required")
    aliases = dict(public_aliases or {})
    replaced_prefixes = tuple(replaced_control_prefixes)
    if any(not prefix for prefix in replaced_prefixes):
        raise ValueError("replaced control prefixes must be non-empty")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("public alias targets must be one-to-one")
    if any(source == target for source, target in aliases.items()):
        raise ValueError("public aliases must describe renamed tensors")

    missing_alias_sources = sorted(set(aliases) - set(control_state))
    missing_alias_targets = sorted(set(aliases.values()) - set(fdr_state))
    if missing_alias_sources or missing_alias_targets:
        raise ValueError(
            "public alias endpoint is missing: "
            f"control={missing_alias_sources[:5]}, fdr={missing_alias_targets[:5]}"
        )

    public: dict[str, torch.Tensor] = {}
    consumed_fdr_names: set[str] = set()
    replaced_names: set[str] = set()
    for name, expected in control_state.items():
        if name in aliases:
            target_name = aliases[name]
            actual = fdr_state[target_name]
            difference = "alias tensor differs"
            consumed_fdr_names.add(target_name)
        elif any(name.startswith(prefix) for prefix in replaced_prefixes):
            replaced_names.add(name)
            continue
        elif any(name.startswith(prefix) for prefix in private_prefixes):
            raise ValueError(f"undeclared missing control replacement: {name}")
        elif name in fdr_state:
            actual = fdr_state[name]
            difference = "public tensor differs"
            consumed_fdr_names.add(name)
        else:
            raise ValueError(f"undeclared missing control tensor: {name}")
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(f"{difference} in shape or dtype: {name}")
        if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
            raise ValueError(f"{difference}: {name}")
        public[name] = expected.detach().cpu().clone()

    unused_replaced_prefixes = [
        prefix
        for prefix in replaced_prefixes
        if not any(name.startswith(prefix) for name in replaced_names)
    ]
    if unused_replaced_prefixes:
        raise ValueError(
            f"replaced control prefix matched no missing tensor: {unused_replaced_prefixes[:5]}"
        )

    private_names = sorted(set(fdr_state) - consumed_fdr_names)
    unapproved = [
        name for name in private_names if not any(name.startswith(prefix) for prefix in private_prefixes)
    ]
    if unapproved:
        raise ValueError(f"unapproved private tensors: {unapproved[:5]}")
    if not private_names:
        raise ValueError("FDR state contains no private tensors")
    private = {name: fdr_state[name].detach().cpu().clone() for name in private_names}
    return public, private


def copy_public_pre_head(source: nn.Module, target: nn.Module) -> str:
    """Copy the stock layer-0 preliminary box head exactly into the FDR path."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError("pre-head state keys differ")
    for name in source_state:
        if source_state[name].shape != target_state[name].shape:
            raise ValueError(f"pre-head tensor shape differs: {name}")
    target.load_state_dict(source_state, strict=True)
    source_hash = public_state_sha256(source.state_dict())
    if public_state_sha256(target.state_dict()) != source_hash:
        raise RuntimeError("pre-head public copy is not byte-exact")
    return source_hash


def _cuda_device_indices(module: nn.Module) -> list[int]:
    indices = {
        parameter.device.index
        for parameter in module.parameters()
        if parameter.is_cuda and parameter.device.index is not None
    }
    return sorted(indices)


def initialize_private_module(
    module: nn.Module,
    *,
    private_seed: int = PRIVATE_SEED,
    zero_final_layers: Sequence[nn.Module] = (),
) -> None:
    """Initialize private modules without advancing the caller's CPU/CUDA RNG streams."""
    devices = _cuda_device_indices(module)
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(private_seed)
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(private_seed)
        for child in module.modules():
            if child is module:
                continue
            reset = getattr(child, "reset_parameters", None)
            if callable(reset) and not any(True for _ in child.children()):
                reset()
        with torch.no_grad():
            for layer in zero_final_layers:
                weight = getattr(layer, "weight", None)
                bias = getattr(layer, "bias", None)
                if not isinstance(weight, torch.Tensor):
                    raise ValueError("distribution final layer has no weight tensor")
                weight.zero_()
                if isinstance(bias, torch.Tensor):
                    bias.zero_()


def build_fdr_initial_state(
    control_state: Mapping[str, torch.Tensor],
    fdr_state: Mapping[str, torch.Tensor],
    *,
    private_prefixes: Sequence[str],
    public_aliases: Mapping[str, str] | None = None,
    replaced_control_prefixes: Sequence[str] = (),
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the paired public/private initialization artifact."""
    aliases = dict(public_aliases or {})
    replaced_prefixes = tuple(replaced_control_prefixes)
    public, private = partition_state_dicts(
        control_state,
        fdr_state,
        private_prefixes=private_prefixes,
        public_aliases=aliases,
        replaced_control_prefixes=replaced_prefixes,
    )
    replaced_control = {
        name: value.detach().cpu().clone()
        for name, value in control_state.items()
        if name not in public
    }
    fdr_public = {
        aliases.get(name, name): fdr_state[aliases.get(name, name)].detach().cpu().clone()
        for name in public
    }
    control = {name: value.detach().cpu().clone() for name, value in control_state.items()}
    fdr = {name: value.detach().cpu().clone() for name, value in fdr_state.items()}
    return {
        "format_version": FORMAT_VERSION,
        "public_state": public,
        "fdr_public_state": fdr_public,
        "replaced_control_state": replaced_control,
        "private_state": private,
        "migration": {
            "public_aliases": aliases,
            "replaced_control_prefixes": list(replaced_prefixes),
            "approved_private_prefixes": list(private_prefixes),
        },
        "metadata": dict(metadata),
        "fingerprints": {
            "public": public_state_sha256(public),
            "fdr_public": public_state_sha256(fdr_public),
            "replaced_control": public_state_sha256(replaced_control),
            "private": public_state_sha256(private),
            "control": public_state_sha256(control),
            "fdr": public_state_sha256(fdr),
        },
    }


def validate_fdr_initial_state(artifact: Mapping[str, Any]) -> None:
    """Validate all public/private/migration partitions and their fingerprints."""
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("FDR initial-state format mismatch")
    public = artifact.get("public_state")
    fdr_public = artifact.get("fdr_public_state")
    replaced = artifact.get("replaced_control_state")
    private = artifact.get("private_state")
    migration = artifact.get("migration")
    fingerprints = artifact.get("fingerprints", {})
    if (
        not isinstance(public, Mapping)
        or not isinstance(fdr_public, Mapping)
        or not isinstance(replaced, Mapping)
        or not isinstance(private, Mapping)
        or not private
        or not isinstance(migration, Mapping)
    ):
        raise ValueError("FDR initial-state partition is invalid")
    if public_state_sha256(public) != fingerprints.get("public"):
        raise ValueError("FDR public fingerprint mismatch")
    if public_state_sha256(fdr_public) != fingerprints.get("fdr_public"):
        raise ValueError("FDR aliased public fingerprint mismatch")
    if public_state_sha256(replaced) != fingerprints.get("replaced_control"):
        raise ValueError("FDR replaced-control fingerprint mismatch")
    if public_state_sha256(private) != fingerprints.get("private"):
        raise ValueError("FDR private fingerprint mismatch")

    aliases = migration.get("public_aliases")
    replaced_prefixes = migration.get("replaced_control_prefixes")
    private_prefixes = migration.get("approved_private_prefixes")
    if (
        not isinstance(aliases, Mapping)
        or not isinstance(replaced_prefixes, list)
        or not isinstance(private_prefixes, list)
    ):
        raise ValueError("FDR structural migration manifest is invalid")
    expected_fdr_public = {aliases.get(name, name): value for name, value in public.items()}
    if set(expected_fdr_public) != set(fdr_public):
        raise ValueError("FDR aliased public keys mismatch")
    for name, expected in expected_fdr_public.items():
        actual = fdr_public[name]
        if expected.shape != actual.shape or expected.dtype != actual.dtype or not torch.equal(expected, actual):
            raise ValueError(f"FDR aliased public tensor mismatch: {name}")
    if any(
        not any(name.startswith(prefix) for prefix in replaced_prefixes)
        for name in replaced
    ):
        raise ValueError("FDR replaced-control state is not declared")
    if any(
        not any(name.startswith(prefix) for prefix in private_prefixes)
        for name in private
    ):
        raise ValueError("FDR private state is not approved")

    control = {**public, **replaced}
    fdr = {**fdr_public, **private}
    if public_state_sha256(control) != fingerprints.get("control"):
        raise ValueError("FDR full control fingerprint mismatch")
    if public_state_sha256(fdr) != fingerprints.get("fdr"):
        raise ValueError("FDR full method fingerprint mismatch")


def load_fdr_initial_state(model: nn.Module, artifact: Mapping[str, Any], *, variant: str) -> None:
    """Strictly load the shared control state or shared-plus-private FDR state."""
    if variant not in {"control", "fdr"}:
        raise ValueError(f"unknown FDR paired variant: {variant}")
    validate_fdr_initial_state(artifact)
    if variant == "control":
        expected = {
            **artifact["public_state"],
            **artifact["replaced_control_state"],
        }
    else:
        expected = {
            **artifact["fdr_public_state"],
            **artifact["private_state"],
        }
    if set(model.state_dict()) != set(expected):
        raise ValueError("FDR initial-state keys do not match target model")
    model.load_state_dict(expected, strict=True)


def validate_optimizer_coverage(model: nn.Module, optimizer: Any) -> dict[str, int]:
    """Require each trainable model parameter in exactly one optimizer group."""
    expected = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    counts: dict[int, int] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            counts[id(parameter)] = counts.get(id(parameter), 0) + 1
    duplicate = sorted(identifier for identifier, count in counts.items() if count != 1)
    if duplicate:
        raise ValueError(f"optimizer duplicate parameter entries: {len(duplicate)}")
    missing = set(expected) - set(counts)
    foreign = set(counts) - set(expected)
    if missing or foreign:
        raise ValueError(
            f"optimizer coverage mismatch: missing={len(missing)}, foreign={len(foreign)}"
        )
    return {
        "tensor_count": len(expected),
        "parameter_count": sum(parameter.numel() for parameter in expected.values()),
    }


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse and attributes & reparse)


def _reject_link_traversal(path: Path) -> None:
    for component in (*path.parents, path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if _is_link_or_reparse(metadata):
            raise ValueError("manifest path cannot traverse a symlink or reparse point")


def write_create_only_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create one canonical, fsynced JSON manifest and never replace it."""
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("manifest path must end in .json")
    _reject_link_traversal(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_traversal(destination)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    return destination


def build_run_identity(
    source_identity: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    """Bind one run to immutable source, protocol, arm, stage, and seed identities."""
    if stage not in {"screen", "formal"}:
        raise ValueError(f"unknown FDR stage: {stage}")
    if variant not in {"control", "fdr"}:
        raise ValueError(f"unknown FDR variant: {variant}")
    if seed != 0:
        raise ValueError("FDR protocol is frozen to seed0")
    source_sha256 = public_state_sha256(source_identity)
    run_id = (
        f"{variant}-{stage}-seed0-{source_sha256[:12].lower()}-"
        f"{FDR_PROTOCOL_SHA256[:12].lower()}"
    )
    return {
        "source_sha256": source_sha256,
        "protocol_sha256": FDR_PROTOCOL_SHA256,
        "run_id": run_id,
        "stage": stage,
        "variant": variant,
        "seed": seed,
    }


def validate_resume_authority(
    checkpoint_identity: Mapping[str, Any], expected_identity: Mapping[str, Any]
) -> None:
    """Reject a checkpoint unless every run-identity field is exactly bound."""
    required = ("source_sha256", "protocol_sha256", "run_id", "stage", "variant", "seed")
    for field in required:
        if field not in checkpoint_identity or checkpoint_identity[field] != expected_identity.get(field):
            raise ValueError(
                f"resume authority mismatch for {field}: "
                f"expected={expected_identity.get(field)!r}, actual={checkpoint_identity.get(field)!r}"
            )


__all__ = [
    "DFINE_COMMIT",
    "FDR_PROTOCOL",
    "FDR_PROTOCOL_SHA256",
    "FORMAT_VERSION",
    "PRIVATE_SEED",
    "build_fdr_initial_state",
    "build_run_identity",
    "canonical_json_bytes",
    "copy_public_pre_head",
    "initialize_private_module",
    "load_fdr_initial_state",
    "partition_state_dicts",
    "public_state_sha256",
    "validate_optimizer_coverage",
    "validate_fdr_initial_state",
    "validate_resume_authority",
    "write_create_only_manifest",
]

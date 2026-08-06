"""Immutable paired authority for the FDR versus SCADS experiment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.fdr_protocol import canonical_json_bytes, public_state_sha256


FORMAT_VERSION = 1
FDR_PRIVATE_SEED = 10_000
SCADS_PRIVATE_SEED = 20_000
DEFAULT_SCADS_PRIVATE_PREFIXES = (
    "model.28.decoder.support_router.",
    "model.28.decoder.adaptive_integral.",
)


SCADS_PROTOCOL: dict[str, Any] = {
    "design": "ultralytics-rtdetr-l-fdr-scads-v1",
    "comparison": ["fdr", "scads"],
    "dfine_commit": "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6",
    "fdr": {
        "reg_max": 32,
        "reg_scale": 4.0,
        "up": 0.5,
        "fgl_weight": 0.15,
        "preliminary_box": True,
        "cumulative": True,
    },
    "scads": {
        "support_up_values": [0.25, 0.5, 1.0],
        "router_hidden": 64,
        "temperature": 1.0,
        "route_weight": 0.05,
        "margin_ratio": 0.02,
        "shared_across_decoder_layers": True,
        "router_inputs_detached": True,
    },
    "environment": {
        "model": "Ultralytics RT-DETR-L",
        "ultralytics": "8.4.90",
        "gpu": "NVIDIA GeForce RTX 4090",
        "driver": "575.57.08",
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
        "shared_seed": 0,
        "fdr_private_seed": FDR_PRIVATE_SEED,
        "scads_private_seed": SCADS_PRIVATE_SEED,
        "scads_base_bias": [-4.0, 4.0, -4.0],
    },
}
SCADS_PROTOCOL_SHA256 = public_state_sha256(SCADS_PROTOCOL)


def build_scads_initial_state(
    fdr_state: Mapping[str, torch.Tensor],
    scads_state: Mapping[str, torch.Tensor],
    *,
    private_prefixes: Sequence[str] = DEFAULT_SCADS_PRIVATE_PREFIXES,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a byte-exact FDR/SCADS paired initialization artifact."""

    prefixes = tuple(private_prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("SCADS private prefixes must be non-empty")
    missing = sorted(set(fdr_state) - set(scads_state))
    if missing:
        raise ValueError(f"SCADS state is missing FDR tensors: {missing[:5]}")
    common: dict[str, torch.Tensor] = {}
    for name, expected in fdr_state.items():
        actual = scads_state[name]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(f"SCADS common tensor shape/dtype differs: {name}")
        if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
            raise ValueError(f"SCADS common tensor bytes differ: {name}")
        common[name] = expected.detach().cpu().clone()

    private_names = sorted(set(scads_state) - set(fdr_state))
    if not private_names:
        raise ValueError("SCADS state contains no private tensors")
    unapproved = [
        name
        for name in private_names
        if not any(name.startswith(prefix) for prefix in prefixes)
    ]
    if unapproved:
        raise ValueError(f"unapproved SCADS private tensors: {unapproved[:5]}")
    private = {
        name: scads_state[name].detach().cpu().clone()
        for name in private_names
    }
    full_scads = {**common, **private}
    return {
        "format_version": FORMAT_VERSION,
        "common_state": common,
        "scads_private_state": private,
        "migration": {"approved_private_prefixes": list(prefixes)},
        "metadata": dict(metadata),
        "fingerprints": {
            "common": public_state_sha256(common),
            "scads_private": public_state_sha256(private),
            "fdr": public_state_sha256(common),
            "scads": public_state_sha256(full_scads),
        },
    }


def validate_scads_initial_state(artifact: Mapping[str, Any]) -> None:
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("SCADS initial-state format mismatch")
    common = artifact.get("common_state")
    private = artifact.get("scads_private_state")
    migration = artifact.get("migration")
    fingerprints = artifact.get("fingerprints", {})
    if (
        not isinstance(common, Mapping)
        or not common
        or not isinstance(private, Mapping)
        or not private
        or not isinstance(migration, Mapping)
    ):
        raise ValueError("SCADS initial-state partitions are invalid")
    prefixes = migration.get("approved_private_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("SCADS private-prefix authority is invalid")
    if any(
        not any(name.startswith(prefix) for prefix in prefixes)
        for name in private
    ):
        raise ValueError("SCADS private state contains an unapproved tensor")
    if public_state_sha256(common) != fingerprints.get("common"):
        raise ValueError("SCADS common fingerprint mismatch")
    if public_state_sha256(private) != fingerprints.get("scads_private"):
        raise ValueError("SCADS private fingerprint mismatch")
    if public_state_sha256(common) != fingerprints.get("fdr"):
        raise ValueError("SCADS FDR-arm fingerprint mismatch")
    if public_state_sha256({**common, **private}) != fingerprints.get("scads"):
        raise ValueError("SCADS full-arm fingerprint mismatch")


def load_scads_initial_state(
    model: nn.Module,
    artifact: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    if variant not in {"fdr", "scads"}:
        raise ValueError(f"unknown SCADS paired variant: {variant}")
    validate_scads_initial_state(artifact)
    expected = dict(artifact["common_state"])
    if variant == "scads":
        expected.update(artifact["scads_private_state"])
    if set(model.state_dict()) != set(expected):
        raise ValueError("SCADS initial-state keys do not match target model")
    model.load_state_dict(expected, strict=True)


def build_run_identity(
    source_identity: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    if stage not in {"screen", "formal"}:
        raise ValueError(f"unknown SCADS stage: {stage}")
    if variant not in {"fdr", "scads"}:
        raise ValueError(f"unknown SCADS variant: {variant}")
    if seed != 0:
        raise ValueError("SCADS protocol is frozen to seed0")
    source_sha256 = public_state_sha256(source_identity)
    return {
        "source_sha256": source_sha256,
        "protocol_sha256": SCADS_PROTOCOL_SHA256,
        "run_id": (
            f"{variant}-{stage}-seed0-{source_sha256[:12].lower()}-"
            f"{SCADS_PROTOCOL_SHA256[:12].lower()}"
        ),
        "stage": stage,
        "variant": variant,
        "seed": seed,
    }


def validate_resume_authority(
    checkpoint_identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> None:
    required = ("source_sha256", "protocol_sha256", "run_id", "stage", "variant", "seed")
    for field in required:
        if checkpoint_identity.get(field) != expected_identity.get(field):
            raise ValueError(f"SCADS resume authority mismatch for {field}")


__all__ = [
    "DEFAULT_SCADS_PRIVATE_PREFIXES",
    "FDR_PRIVATE_SEED",
    "SCADS_PRIVATE_SEED",
    "SCADS_PROTOCOL",
    "SCADS_PROTOCOL_SHA256",
    "build_run_identity",
    "build_scads_initial_state",
    "canonical_json_bytes",
    "load_scads_initial_state",
    "public_state_sha256",
    "validate_resume_authority",
    "validate_scads_initial_state",
]

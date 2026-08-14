"""Frozen retrospective bridge protocol for the missing FDR+BPDD B arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from src.fdr_bpdd_ra_glgm_protocol import (
    BPDD_SOURCE_BLOB,
    BPDD_SOURCE_COMMIT,
)
from src.fdr_protocol import FDR_PROTOCOL, canonical_json_bytes, public_state_sha256
from src.ra_experiment_protocol import BASELINE_PARAMETERS


BRIDGE_VARIANT = "fdr_bpdd_bridge"
BRIDGE_STAGES = ("smoke", "formal")
BRIDGE_INITIAL_STATE_SHA256 = (
    "4990B008B267F8D63C3BA6D5E3DBABCFA716CB463C537C561318B29647B222AC"
)
RA_SOURCE_COMMIT = "69c188b2717499ac5ad54b5186299d3aaf1351ad"
COMBO_SOURCE_COMMIT = "5926ac7502ab355b8e50efc4f7af94a16b532de0"
RA_PROTOCOL_SHA256 = "3ABAF37EAEDB984F70846927ABA8BFF09C5804A11234419400612859A02A336B"
COMBO_PROTOCOL_SHA256 = (
    "3B82A7C8EDDC252D9B6C20546DC9B3B40388D4A7C7E9CCA516DE5B060840FE6D"
)
EXPECTED_GPU_UUID = "GPU-6906ed5f-079f-b24d-e58a-3e5fe0da7d6a"
DATASET_SHA256 = "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
IGNORE_SHA256 = "B59EB704C9225B46D94D6783D0C42F8E63635DBAE832C373181135E77E424085"

SHARED_TRAINING = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "nbs": 64,
    "seed": 0,
    "pretrained": False,
    "optimizer": "MuSGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "deterministic": True,
    "amp": True,
    "cache": False,
    "cos_lr": False,
    "max_det": 300,
}

BRIDGE_PROTOCOL: dict[str, Any] = {
    "design": "fdr-bpdd-ra-full100-seed0-retrospective-bridge",
    "evidence_role": (
        "complete the missing B arm against immutable A/C/D runs sharing the "
        "RA Full100 seed0 initial tensor authority"
    ),
    "claim_boundary": (
        "single-seed retrospective matched bridge; multi-seed confirmation remains required"
    ),
    "variant": BRIDGE_VARIANT,
    "components": {
        "fdr": "RA Full100 synchronized FDR graph and baseline state",
        "bpdd": {
            "source_commit": BPDD_SOURCE_COMMIT,
            "source_blob": BPDD_SOURCE_BLOB,
            "training_only": True,
            "parameters": 0,
            "weight": 0.5,
            "temperature": 0.5,
            "margin": 0.02,
            "eps": 1e-6,
            "assignment": "final ordinary stock Hungarian assignment only",
            "denoising_queries": "excluded",
        },
        "ra_glgm": "absent",
    },
    "reference_authorities": {
        "A_C_source_commit": RA_SOURCE_COMMIT,
        "A_C_protocol_sha256": RA_PROTOCOL_SHA256,
        "D_source_commit": COMBO_SOURCE_COMMIT,
        "D_protocol_sha256": COMBO_PROTOCOL_SHA256,
        "initial_state_sha256": BRIDGE_INITIAL_STATE_SHA256,
        "gpu_uuid": EXPECTED_GPU_UUID,
    },
    "dataset": {
        **FDR_PROTOCOL["dataset"],
        "train_images": 6471,
        "val_images": 548,
        "sha256": DATASET_SHA256,
        "ignore_sha256": IGNORE_SHA256,
    },
    "training": {
        **FDR_PROTOCOL["training"],
        **SHARED_TRAINING,
        "scratch": True,
        "inherit_checkpoint": False,
        "stages": {"smoke": 2, "formal": 100},
        "single_gpu": True,
        "ddp": False,
        "amp_initial_scale": 128.0,
        "milestone_period": {"smoke": 1, "formal": 5},
    },
    "augmentation": FDR_PROTOCOL["augmentation"],
    "evaluation": {
        "official_val_images": 548,
        "imgsz": 640,
        "conf": 0.001,
        "max_det": 300,
        "nms": False,
        "half": False,
        "primary_checkpoint": 100,
        "primary_tail": "epochs 96-100 online validation mean",
        "uniform_locked_reevaluation_required": True,
    },
    "engineering_gates": {
        "model_parameters_exact": BASELINE_PARAMETERS,
        "bpdd_parameters": 0,
        "ra_parameters": 0,
        "all_metrics_losses_gradients_finite": True,
        "amp_skipped_steps": 0,
        "bpdd_active_edge_ratio_ever_positive": True,
        "free_disk_hard_stop_gib": 8,
        "publish_pt": False,
    },
}

BRIDGE_PROTOCOL_SHA256 = (
    hashlib.sha256(canonical_json_bytes(BRIDGE_PROTOCOL)).hexdigest().upper()
)


def build_bridge_run_identity(
    source: Mapping[str, Any], *, stage: str, gpu_uuid: str
) -> dict[str, Any]:
    if stage not in BRIDGE_STAGES:
        raise ValueError(f"unknown bridge stage: {stage}")
    if gpu_uuid != EXPECTED_GPU_UUID:
        raise ValueError("bridge must use the A/C/D physical GPU")
    source_sha = public_state_sha256(source)
    return {
        "variant": BRIDGE_VARIANT,
        "stage": stage,
        "seed": 0,
        "source_sha256": source_sha,
        "protocol_sha256": BRIDGE_PROTOCOL_SHA256,
        "gpu_uuid": gpu_uuid,
        "run_id": (
            f"{BRIDGE_VARIANT}-{stage}-seed0-"
            f"{source_sha[:12].lower()}-{BRIDGE_PROTOCOL_SHA256[:12].lower()}"
        ),
    }


def reference_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset_authority", {})
    return {
        "source": payload.get("source"),
        "protocol_sha256": payload.get("protocol_sha256"),
        "run_identity": payload.get("run_identity"),
        "initial_state_sha256": payload.get("initial_state", {}).get("sha256"),
        "initial_fingerprints": payload.get("initial_state", {}).get("fingerprints"),
        "dataset_sha256": dataset.get("positive", {}).get("sha256"),
        "ignore_sha256": dataset.get("ignore", {}).get("sha256"),
        "gpu_uuid": payload.get("gpu_uuid"),
        "schedule_epochs": payload.get("schedule_epochs"),
        "model_parameters": payload.get("model_parameters"),
        "initialization_mode": payload.get("initialization_mode"),
        "parent_checkpoint": payload.get("parent_checkpoint"),
    }


def validate_reference_snapshots(
    a: Mapping[str, Any], c: Mapping[str, Any], d: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    snapshots = {
        name: reference_snapshot(value)
        for name, value in {"A": a, "C": c, "D": d}.items()
    }
    for name, snapshot in snapshots.items():
        if snapshot["initial_state_sha256"] != BRIDGE_INITIAL_STATE_SHA256:
            raise ValueError(f"{name} initial-state authority differs")
        if snapshot["dataset_sha256"] != DATASET_SHA256:
            raise ValueError(f"{name} positive dataset authority differs")
        if snapshot["ignore_sha256"] != IGNORE_SHA256:
            raise ValueError(f"{name} ignore authority differs")
        if snapshot["gpu_uuid"] != EXPECTED_GPU_UUID:
            raise ValueError(f"{name} GPU authority differs")
        if snapshot["schedule_epochs"] != 100:
            raise ValueError(f"{name} did not use the 100-epoch schedule")
        if snapshot["parent_checkpoint"] is not None:
            raise ValueError(f"{name} inherited a checkpoint")
    if snapshots["A"]["initial_fingerprints"] != snapshots["C"]["initial_fingerprints"]:
        raise ValueError("A/C initial tensor fingerprints differ")
    if snapshots["A"]["initial_fingerprints"] != snapshots["D"]["initial_fingerprints"]:
        raise ValueError("A/C/D initial tensor fingerprints differ")
    if snapshots["A"]["source"] != snapshots["C"]["source"]:
        raise ValueError("A/C source authority differs")
    if snapshots["A"]["protocol_sha256"] != RA_PROTOCOL_SHA256:
        raise ValueError("A protocol authority differs")
    if snapshots["C"]["protocol_sha256"] != RA_PROTOCOL_SHA256:
        raise ValueError("C protocol authority differs")
    if snapshots["D"]["protocol_sha256"] != COMBO_PROTOCOL_SHA256:
        raise ValueError("D protocol authority differs")
    if snapshots["A"]["run_identity"].get("variant") != "baseline":
        raise ValueError("A is not the RA Full100 synchronized baseline")
    if snapshots["C"]["run_identity"].get("variant") != "ra_glgm":
        raise ValueError("C is not the RA Full100 method arm")
    if snapshots["D"]["run_identity"].get("variant") != "fdr_bpdd_ra_glgm":
        raise ValueError("D is not the combination arm")
    pair_id = snapshots["A"]["run_identity"].get("pair_id")
    if not pair_id or pair_id != snapshots["C"]["run_identity"].get("pair_id"):
        raise ValueError("A/C pair identity differs")
    if snapshots["A"]["model_parameters"] != BASELINE_PARAMETERS:
        raise ValueError("A parameter count is not the synchronized FDR baseline")
    return snapshots


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_bridge_authority(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Fail closed if any source, reference, data, or initial-state byte drifts."""

    from src.lpr_protocol import dataset_signature
    from src.ra_experiment_protocol import (
        current_source_identity,
        ignore_sidecar_signature,
    )
    from src.ra_glgm_protocol import validate_ra_glgm_initial_state

    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("bridge authority format must be 1")
    recorded = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if recorded != actual:
        raise ValueError("bridge authority manifest SHA256 mismatch")
    if manifest.get("protocol") != BRIDGE_PROTOCOL:
        raise ValueError("bridge authority protocol payload differs")
    if manifest.get("protocol_sha256") != BRIDGE_PROTOCOL_SHA256:
        raise ValueError("bridge authority protocol SHA256 differs")

    source = manifest.get("source")
    current = current_source_identity(repository_root)
    if not isinstance(source, Mapping) or dict(source) != current:
        raise ValueError("checked-out source differs from bridge authority")
    if manifest.get("source_sha256") != public_state_sha256(source):
        raise ValueError("bridge source identity hash mismatch")

    gpu_uuid = str(manifest.get("gpu_uuid", ""))
    if gpu_uuid != EXPECTED_GPU_UUID:
        raise ValueError("bridge authority GPU UUID differs")
    identities = manifest.get("run_identities")
    if not isinstance(identities, Mapping):
        raise ValueError("bridge run identities are missing")
    for stage in BRIDGE_STAGES:
        expected = build_bridge_run_identity(source, stage=stage, gpu_uuid=gpu_uuid)
        if identities.get(stage) != expected:
            raise ValueError(f"bridge run identity mismatch: {stage}")

    initial = manifest.get("initial_state")
    if not isinstance(initial, Mapping):
        raise ValueError("bridge initial-state authority is missing")
    initial_path = Path(str(initial.get("path", ""))).resolve()
    if initial_path.is_symlink() or not initial_path.is_file():
        raise FileNotFoundError("bridge initial-state artifact is missing")
    if file_sha256(initial_path) != str(initial.get("sha256", "")).upper():
        raise ValueError("bridge initial-state SHA256 mismatch")
    artifact = torch.load(initial_path, map_location="cpu", weights_only=False)
    validate_ra_glgm_initial_state(artifact)
    if initial.get("fingerprints") != artifact.get("fingerprints"):
        raise ValueError("bridge initial-state fingerprints differ")

    references = manifest.get("reference_files")
    snapshots = manifest.get("reference_snapshots")
    if not isinstance(references, Mapping) or not isinstance(snapshots, Mapping):
        raise ValueError("bridge A/C/D references are missing")
    loaded_references: dict[str, dict[str, Any]] = {}
    for label in ("A", "C", "D"):
        record = references.get(label)
        if not isinstance(record, Mapping):
            raise ValueError(f"bridge {label} reference is missing")
        reference_path = Path(str(record.get("path", ""))).resolve()
        if file_sha256(reference_path) != str(record.get("sha256", "")).upper():
            raise ValueError(f"bridge {label} reference bytes differ")
        loaded_references[label] = json.loads(
            reference_path.read_text(encoding="utf-8")
        )
    if (
        validate_reference_snapshots(
            loaded_references["A"], loaded_references["C"], loaded_references["D"]
        )
        != snapshots
    ):
        raise ValueError("bridge A/C/D reference snapshots differ")

    argument_files = manifest.get("reference_argument_files")
    if not isinstance(argument_files, Mapping):
        raise ValueError("bridge A/C/D argument references are missing")
    for label in ("A", "C", "D"):
        record = argument_files.get(label)
        if not isinstance(record, Mapping):
            raise ValueError(f"bridge {label} argument reference is missing")
        argument_path = Path(str(record.get("path", ""))).resolve()
        if file_sha256(argument_path) != str(record.get("sha256", "")).upper():
            raise ValueError(f"bridge {label} argument bytes differ")

    dataset = manifest.get("dataset_authority")
    if not isinstance(dataset, Mapping):
        raise ValueError("bridge dataset authority is missing")
    root = Path(str(dataset.get("root", ""))).resolve()
    if dataset_signature(root) != dataset.get("positive"):
        raise ValueError("bridge positive dataset differs from authority")
    if ignore_sidecar_signature(root) != dataset.get("ignore"):
        raise ValueError("bridge ignore sidecars differ from authority")
    return manifest


__all__ = [
    "BRIDGE_INITIAL_STATE_SHA256",
    "BRIDGE_PROTOCOL",
    "BRIDGE_PROTOCOL_SHA256",
    "BRIDGE_STAGES",
    "BRIDGE_VARIANT",
    "EXPECTED_GPU_UUID",
    "build_bridge_run_identity",
    "file_sha256",
    "load_bridge_authority",
    "reference_snapshot",
    "validate_reference_snapshots",
]

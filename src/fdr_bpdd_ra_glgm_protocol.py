"""Frozen single-arm protocol for FDR+BPDD+RA-GLGM Formal100."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from src.fdr_protocol import FDR_PROTOCOL, canonical_json_bytes, public_state_sha256
from src.ra_experiment_protocol import (
    BASELINE_PARAMETERS,
    MAX_PEAK_VRAM_MIB,
)
from src.ra_glgm_protocol import RA_GLGM_PRIVATE_PARAMETERS


COMBO_VARIANT = "fdr_bpdd_ra_glgm"
COMBO_STAGES = ("smoke", "formal")
COMBO_PARAMETERS = BASELINE_PARAMETERS + RA_GLGM_PRIVATE_PARAMETERS
BPDD_SOURCE_COMMIT = "848f00cb7a40907e3884885ecd5bbd474450758a"
BPDD_SOURCE_BLOB = "dd536a7ed68763bfdb570727d0293b04e37bedd2"
RA_BASE_COMMIT = "69c188b2717499ac5ad54b5186299d3aaf1351ad"

COMBO_PROTOCOL: dict[str, Any] = {
    "design": "fdr-bpdd-ra-glgm-v1.1-single-arm-formal100",
    "evidence_role": (
        "single-arm combination validation against historical checkpoints; "
        "BPDD comparisons remain cross-authority until a fresh paired run exists"
    ),
    "components": {
        "fdr": "unchanged cumulative four-edge distribution regression and FGL",
        "bpdd": {
            "source_commit": BPDD_SOURCE_COMMIT,
            "source_blob": BPDD_SOURCE_BLOB,
            "training_only": True,
            "parameters": 0,
            "weight": 0.5,
            "temperature": 0.5,
            "margin": 0.02,
            "eps": 1.0e-6,
            "teacher": "detached reliability-weighted mixture of future decoder layers",
            "assignment": "final ordinary stock Hungarian assignment only",
            "denoising_queries": "excluded",
        },
        "ra_glgm": {
            "source_commit": RA_BASE_COMMIT,
            "version": "v1.1 hard three-scale gate",
            "insertion": "FDR decoder P3 only",
            "private_parameters": RA_GLGM_PRIVATE_PARAMETERS,
            "support_loss_weight": 0.05,
            "scale_loss_weight": 0.05,
            "identity_initialization": True,
            "private_input": "P3.detach()",
        },
    },
    "loss": (
        "sum(stock RT-DETR and FDR/FGL/pre-box losses) + loss_bpdd "
        "+ 0.05*loss_ra_support + 0.05*loss_ra_scale"
    ),
    "seed": 0,
    "device": "0",
    "dataset": {
        **FDR_PROTOCOL["dataset"],
        "train_images": 6471,
        "val_images": 548,
        "ignore_sidecar": {
            "files": {"train": 6471, "val": 548},
            "boxes": {"train": 10_343, "val": 1_410},
            "invalid_zero_area_rows_excluded": {"train": 2, "val": 0},
        },
    },
    "training": {
        **FDR_PROTOCOL["training"],
        "scratch": True,
        "inherit_checkpoint": False,
        "epochs": {"smoke": 2, "formal": 100},
        "imgsz": 640,
        "batch": 8,
        "nbs": 64,
        "workers": 8,
        "amp": True,
        "amp_initial_scale": 128.0,
        "deterministic": True,
        "single_gpu": True,
        "ddp": False,
        "internal_oom_retry": False,
        "last_checkpoint_every_epoch": True,
        "milestone_checkpoint_period": {"smoke": 1, "formal": 5},
        "lightweight_evidence_every_epoch": True,
    },
    "augmentation": FDR_PROTOCOL["augmentation"],
    "evaluation": {
        "official_val_images": 548,
        "imgsz": 640,
        "max_det": 300,
        "nms": False,
        "conf": 0.001,
        "half": False,
        "formal_milestones": list(range(5, 101, 5)),
        "primary_checkpoint": 100,
        "primary_tail": "epochs 96-100 standard validation mean",
        "best_checkpoint": "supplemental only",
        "metrics": [
            "precision",
            "recall",
            "map50",
            "map75",
            "map50_95",
            "ap_tiny",
            "ap_small",
            "class_ap_10",
        ],
        "same_locked_evaluator_for_all_historical_checkpoints": True,
    },
    "decision": {
        "arms": {
            "A": "historical FDR epoch100, uniformly re-evaluated",
            "B": "historical FDR+BPDD epoch100, uniformly re-evaluated",
            "C": "historical FDR+RA-GLGM epoch100, uniformly re-evaluated",
            "D": "new FDR+BPDD+RA-GLGM epoch100",
        },
        "historical_endpoint_screen": "D map50_95 - max(B,C) >= 0.001",
        "scientific_scope": (
            "cross-authority endpoint exploration only; synergy requires fresh comparable "
            "four-arm interaction contrast D+A>B+C and multi-seed confirmation"
        ),
        "endpoint_positive": [
            "D epoch100 map50_95 > max(B,C)",
            "D tail5 map50_95 > both corresponding historical tail5 means",
            "D AP75 >= max(B,C)",
            "D AP-tiny >= max(B,C)",
            "D AP-small >= max(B,C)",
            "D improves at least 7 of 10 class AP values over A",
        ],
        "fallback_labels": {
            "integration_positive": "D>A but D<=max(B,C); no synergy claim",
            "negative": "D<=A or engineering/scientific integrity gate fails",
        },
    },
    "engineering_gates": {
        "model_parameters_exact": COMBO_PARAMETERS,
        "bpdd_parameter_delta_from_ra": 0,
        "peak_vram_mib_max": MAX_PEAK_VRAM_MIB,
        "all_losses_metrics_gradients_finite": True,
        "amp_skipped_steps": 0,
        "bpdd_active_edge_ratio_ever_positive": True,
        "ra_private_gradient_each_epoch_positive": True,
        "checkpoint_scope": "local-only",
        "publish_pt": False,
        "free_disk_warning_gib": 12,
        "free_disk_hard_stop_gib": 8,
    },
}

COMBO_PROTOCOL_SHA256 = hashlib.sha256(
    canonical_json_bytes(COMBO_PROTOCOL)
).hexdigest().upper()


def build_combo_run_identity(
    source: Mapping[str, Any], *, stage: str, gpu_uuid: str
) -> dict[str, Any]:
    if stage not in COMBO_STAGES:
        raise ValueError(f"unknown combo stage: {stage}")
    if not gpu_uuid.startswith("GPU-") or any(character.isspace() for character in gpu_uuid):
        raise ValueError("gpu_uuid must be one NVIDIA GPU UUID token")
    source_sha = public_state_sha256(source)
    return {
        "variant": COMBO_VARIANT,
        "stage": stage,
        "seed": 0,
        "source_sha256": source_sha,
        "protocol_sha256": COMBO_PROTOCOL_SHA256,
        "gpu_uuid": gpu_uuid,
        "run_id": (
            f"{COMBO_VARIANT}-{stage}-seed0-"
            f"{source_sha[:12].lower()}-{COMBO_PROTOCOL_SHA256[:12].lower()}"
        ),
    }


def load_combo_authority(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Validate the complete combination authority against current bytes."""

    from src.ra_experiment_protocol import (
        current_source_identity,
        file_sha256,
        ignore_sidecar_signature,
    )
    from src.lpr_protocol import dataset_signature
    from src.ra_glgm_protocol import validate_ra_glgm_initial_state

    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("combo authority format must be 1")
    recorded = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if recorded != actual:
        raise ValueError("combo authority manifest SHA256 mismatch")
    if manifest.get("protocol") != COMBO_PROTOCOL:
        raise ValueError("combo authority protocol payload differs")
    if manifest.get("protocol_sha256") != COMBO_PROTOCOL_SHA256:
        raise ValueError("combo authority protocol SHA256 differs")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or dict(source) != current_source_identity(repository_root):
        raise ValueError("checked-out source differs from combo authority")
    if manifest.get("source_sha256") != public_state_sha256(source):
        raise ValueError("combo source identity hash mismatch")

    gpu_uuid = str(manifest.get("gpu_uuid", ""))
    identities = manifest.get("run_identities")
    if not isinstance(identities, Mapping):
        raise ValueError("combo run identities are missing")
    for stage in COMBO_STAGES:
        expected = build_combo_run_identity(source, stage=stage, gpu_uuid=gpu_uuid)
        if identities.get(stage) != expected:
            raise ValueError(f"combo run identity mismatch: {stage}")

    initial = manifest.get("initial_state")
    if not isinstance(initial, Mapping):
        raise ValueError("combo initial-state authority is missing")
    initial_path = Path(str(initial.get("path", ""))).resolve()
    if initial_path.is_symlink() or not initial_path.is_file():
        raise FileNotFoundError("combo initial-state artifact is missing")
    if file_sha256(initial_path) != str(initial.get("sha256", "")).upper():
        raise ValueError("combo initial-state SHA256 mismatch")
    artifact = torch.load(initial_path, map_location="cpu", weights_only=False)
    validate_ra_glgm_initial_state(artifact)
    if initial.get("fingerprints") != artifact.get("fingerprints"):
        raise ValueError("combo initial-state fingerprints differ")

    dataset = manifest.get("dataset_authority")
    if not isinstance(dataset, Mapping):
        raise ValueError("combo dataset authority is missing")
    root = Path(str(dataset.get("root", ""))).resolve()
    if dataset_signature(root) != dataset.get("positive"):
        raise ValueError("combo positive dataset differs from authority")
    if ignore_sidecar_signature(root) != dataset.get("ignore"):
        raise ValueError("combo ignore sidecars differ from authority")
    return manifest


__all__ = [
    "BPDD_SOURCE_BLOB",
    "BPDD_SOURCE_COMMIT",
    "COMBO_PARAMETERS",
    "COMBO_PROTOCOL",
    "COMBO_PROTOCOL_SHA256",
    "COMBO_STAGES",
    "COMBO_VARIANT",
    "RA_BASE_COMMIT",
    "build_combo_run_identity",
    "load_combo_authority",
]

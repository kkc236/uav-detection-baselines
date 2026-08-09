"""Frozen authority and scientific gates for FDR versus FDR+RA-GLGM."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.fdr_protocol import FDR_PROTOCOL, canonical_json_bytes, public_state_sha256


RA_VARIANTS = ("baseline", "ra_glgm")
RA_STAGES = ("smoke", "screen", "formal")
BASELINE_PARAMETERS = 33_156_614
MAX_PARAMETER_INCREASE_RATIO = 0.10
MAX_PEAK_VRAM_MIB = 22 * 1024

RA_EXPERIMENT_PROTOCOL: dict[str, Any] = {
    "design": "ra-glgm-on-fdr-v1",
    "baseline": "Ultralytics RT-DETR-L + FDR",
    "method": "Ultralytics RT-DETR-L + FDR + RA-GLGM(P3-only)",
    "seed": 0,
    "device": "0",
    "pairing": {
        "single_physical_gpu": True,
        "sequential_arms": True,
        "ddp": False,
        "scratch": True,
        "shared_public_initialization": "byte-identical",
        "private_seed": 20_000,
    },
    "dataset": {
        **FDR_PROTOCOL["dataset"],
        "ignore_sidecar": {
            "files": {"train": 6471, "val": 548},
            "boxes": {"train": 10_345, "val": 1_410},
            "source_rule": "VisDrone source confidence/score field equals zero",
        },
    },
    "training": {
        **FDR_PROTOCOL["training"],
        "smoke_epochs": 2,
        "screen_schedule_epochs": 50,
        "screen_cutoff_epoch": 30,
        "formal_schedule_epochs": 100,
        "save_period": 1,
    },
    "augmentation": FDR_PROTOCOL["augmentation"],
    "module": {
        "private_parameters": 812_817,
        "input": {
            "source": "FDR decoder P3 only",
            "shape": "[B,256,H,W]",
            "private_branch_input": "x.detach()",
        },
        "hidden_channels": 192,
        "reduction": {
            "operator": "1x1 Conv-BN-SiLU",
            "channels": "256->192",
            "bias": False,
        },
        "local_expert": {
            "operators": [
                "3x3 Conv-BN-SiLU",
                "3x3 Conv-BN-SiLU",
            ],
            "bias": False,
            "residual_source": "reduced",
        },
        "global_expert": {
            "operators": [
                "depthwise 7x7 Conv-BN-SiLU",
                "depthwise dilated 3x3 Conv-BN-SiLU",
            ],
            "dilated_kernel_dilation": 3,
            "pool_projection": {
                "operator": "1x1 Conv",
                "channels": "192->192",
                "bias": True,
                "batch_norm": False,
            },
        },
        "router": {
            "operator": "1x1 Conv",
            "channels": "192->16",
            "groups": 8,
            "bias": True,
            "initialization": "zeros",
            "input": "shared reduced feature",
            "competition": "per-position grouped two-expert softmax",
        },
        "support": {
            "operator": "1x1 Conv",
            "channels": "192->1",
            "bias": True,
            "activation": "sigmoid",
        },
        "output_projection": {
            "operator": "1x1 Conv",
            "channels": "192->256",
            "bias": False,
        },
        "alpha": {
            "shape": "[1,256,1,1]",
            "initialization": "zeros",
        },
        "output_equation": "X + 0.5*tanh(alpha)*O*tanh(Wo(U))",
        "residual_difficulty": {
            "prediction_source": "final ordinary decoder Query only",
            "excluded_predictions": ["encoder Query", "denoising Query"],
            "assignment": "reuse existing Hungarian assignment; no second matcher",
            "matched_equation": "clamp(0.7*(1-p)+0.3*(1-IoU),0.25,1.0)",
            "probability": "sigmoid(final target-class logit)",
            "unmatched_gt": 1.0,
            "target_generation": "FP32 detached",
        },
        "gaussian_target": {
            "sigma": "max(1,box_size_on_P3/8)",
            "truncate_sigma": 3,
            "overlap_reduction": "pixelwise maximum",
        },
        "auxiliary_focal": {
            "objective": "soft binary focal BCE",
            "alpha": 0.25,
            "gamma": 2.0,
            "reduction": "valid-pixel mean",
            "weight": 0.05,
        },
        "ignore_boxes": {
            "class_id": -1,
            "detection": "excluded",
            "target_generation": "excluded",
            "auxiliary_negative_supervision": "masked",
            "overlapping_positive_gaussian_pixels": "valid",
        },
        "identity_initialization": True,
        "parameter_budget_ratio": MAX_PARAMETER_INCREASE_RATIO,
        "peak_vram_mib_limit": MAX_PEAK_VRAM_MIB,
    },
    "evaluation": {
        "imgsz": 640,
        "max_det": 300,
        "nms": False,
        "conf": 0.001,
        "half": False,
        "tiny": "normalized box area at 640 below 16^2 pixels",
        "small": "normalized box area at 640 in [16^2,32^2) pixels",
        "screen_evaluated_epochs": [28, 29, 30],
        "formal_evaluated_epochs": [98, 99, 100],
    },
    "screen_gate": {
        "epoch30_map_delta_min": 0.005,
        "tail3_map_delta": ">0",
        "epoch30_recall_delta": ">0",
        "epoch30_ap50_delta": ">0",
        "epoch30_ap75_delta": ">=0",
        "epoch30_ap_tiny_delta": ">0",
        "epoch30_ap_small_delta": ">0",
        "class_ap_wins_min": 7,
        "classes": 10,
        "parameter_increase_ratio_max": MAX_PARAMETER_INCREASE_RATIO,
        "peak_vram_mib_max": MAX_PEAK_VRAM_MIB,
    },
    "advancement": {
        "formal_requires_screen_gate": True,
        "formal_initialization": "fresh paired scratch artifact; never Screen checkpoint",
        "primary_formal_evidence": ["epoch100", "tail3_mean"],
        "best_checkpoint": "supplemental only",
    },
    "publication": {
        "checkpoint_scope": "local-only",
        "publish_pt": False,
    },
}

RA_EXPERIMENT_PROTOCOL_SHA256 = hashlib.sha256(
    canonical_json_bytes(RA_EXPERIMENT_PROTOCOL)
).hexdigest().upper()


def build_ra_run_identity(
    source_identity: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int = 0,
    pair_id: str,
) -> dict[str, Any]:
    """Bind one arm to the frozen RA protocol and its paired launch."""

    if stage not in RA_STAGES:
        raise ValueError(f"unknown RA stage: {stage}")
    if variant not in RA_VARIANTS:
        raise ValueError(f"unknown RA variant: {variant}")
    if seed != 0:
        raise ValueError("RA-GLGM v1 is frozen to seed0")
    if not pair_id or any(character.isspace() for character in pair_id):
        raise ValueError("pair_id must be a non-empty token")
    source_sha256 = public_state_sha256(source_identity)
    run_id = (
        f"{variant}-{stage}-seed0-{source_sha256[:12].lower()}-"
        f"{RA_EXPERIMENT_PROTOCOL_SHA256[:12].lower()}"
    )
    return {
        "source_sha256": source_sha256,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "run_id": run_id,
        "pair_id": pair_id,
        "stage": stage,
        "variant": variant,
        "seed": seed,
    }


def finite_number(value: Any) -> bool:
    """Return true only for finite, non-boolean real values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def continuous_epochs(rows: Sequence[Mapping[str, Any]], expected: int) -> bool:
    try:
        epochs = [int(row["completed_epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return len(rows) == expected and epochs == list(range(1, expected + 1))


def validate_runtime_identity(
    manifest: Mapping[str, Any], *, variant: str, stage: str
) -> dict[str, Any]:
    """Fail closed on stage/arm/protocol identity drift."""
    if variant not in RA_VARIANTS or stage not in RA_STAGES:
        raise ValueError("unknown RA run identity")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("RA runtime manifest is missing run_identity")
    expected = {
        "variant": variant,
        "stage": stage,
        "seed": 0,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(
                f"RA runtime identity mismatch for {field}: expected={value!r}, actual={identity.get(field)!r}"
            )
    if not isinstance(identity.get("run_id"), str) or not identity["run_id"]:
        raise ValueError("RA runtime run_id is missing")
    return dict(identity)


def paired_manifests(
    baseline: Mapping[str, Any], method: Mapping[str, Any], *, stage: str
) -> bool:
    """Require both arms to share every authority except variant and run ID."""
    try:
        base_identity = validate_runtime_identity(baseline, variant="baseline", stage=stage)
        method_identity = validate_runtime_identity(method, variant="ra_glgm", stage=stage)
    except ValueError:
        return False
    shared_manifest_fields = (
        "format_version",
        "protocol_sha256",
        "source",
        "initial_state",
        "data",
        "dataset_authority",
        "learnability_report_sha256",
        "gpu_uuid",
        "schedule_epochs",
        "cutoff_epoch",
        "learnability_report_sha256",
    )
    shared_identity_fields = ("source_sha256", "protocol_sha256", "stage", "seed", "pair_id")
    return (
        all(baseline.get(field) == method.get(field) for field in shared_manifest_fields)
        and all(base_identity.get(field) == method_identity.get(field) for field in shared_identity_fields)
        and base_identity["run_id"] != method_identity["run_id"]
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ignore_sidecar_signature(dataset_root: str | Path) -> dict[str, Any]:
    """Hash every transformed VisDrone ignore box consumed by RA supervision."""

    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()
    splits: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        directory = root / "labels_ignore" / split
        if not directory.is_dir():
            raise FileNotFoundError(f"required ignore sidecar directory is missing: {directory}")
        files = sorted(path for path in directory.glob("*.txt") if path.is_file())
        expected_names = {
            path.with_suffix(".txt").name
            for path in (root / "images" / split).glob("*.jpg")
            if path.is_file()
        }
        actual_names = {path.name for path in files}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"ignore sidecar/image mismatch for {split}: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        boxes = 0
        nonempty_files = 0
        for path in files:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(content).hexdigest().upper().encode("ascii"))
            digest.update(b"\n")
            rows = [line for line in content.decode("ascii").splitlines() if line.strip()]
            nonempty_files += bool(rows)
            for line_number, line in enumerate(rows, 1):
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError(f"invalid ignore sidecar row at {path}:{line_number}")
                values = [float(value) for value in fields]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"non-finite ignore sidecar row at {path}:{line_number}")
                cx, cy, width, height = values
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"invalid normalized ignore box at {path}:{line_number}")
                boxes += 1
        splits[split] = {
            "files": len(files),
            "boxes": boxes,
            "nonempty_files": nonempty_files,
            "empty_files": len(files) - nonempty_files,
        }
    return {
        "files": sum(value["files"] for value in splits.values()),
        "boxes": sum(value["boxes"] for value in splits.values()),
        "nonempty_files": sum(value["nonempty_files"] for value in splits.values()),
        "empty_files": sum(value["empty_files"] for value in splits.values()),
        "splits": splits,
        "sha256": digest.hexdigest().upper(),
    }


__all__ = [
    "BASELINE_PARAMETERS",
    "MAX_PARAMETER_INCREASE_RATIO",
    "MAX_PEAK_VRAM_MIB",
    "RA_EXPERIMENT_PROTOCOL",
    "RA_EXPERIMENT_PROTOCOL_SHA256",
    "RA_STAGES",
    "RA_VARIANTS",
    "build_ra_run_identity",
    "continuous_epochs",
    "file_sha256",
    "finite_number",
    "ignore_sidecar_signature",
    "paired_manifests",
    "read_json",
    "read_jsonl",
    "validate_runtime_identity",
]

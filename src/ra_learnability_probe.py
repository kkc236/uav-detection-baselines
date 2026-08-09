"""Frozen low-cost learnability gate for RA-GLGM residual support targets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.fdr_protocol import canonical_json_bytes, public_state_sha256
from src.lpr_protocol import EXPECTED_SUBSET_SHA256, subset_signature
from src.ra_experiment_protocol import RA_EXPERIMENT_PROTOCOL_SHA256, file_sha256
from src.ra_glgm_protocol import RA_GLGM_PRIVATE_PREFIX


MATURE_FDR_CHECKPOINT_SHA256 = "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2"
SCREEN_SUBSET_SHA256 = EXPECTED_SUBSET_SHA256
PROBE_SEED = 73_091
PROBE_TRAIN_IMAGES = 518
PROBE_DEV_IMAGES = 129
PROBE_EPOCHS = 3
PROBE_BATCH = 8
PROBE_OPTIMIZER = {
    "name": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 5.0,
}
SUPPORT_EXCLUDED_SUFFIXES = ("alpha", "output_projection.weight")

LEARNABILITY_GATE = {
    "train_images": PROBE_TRAIN_IMAGES,
    "dev_images": PROBE_DEV_IMAGES,
    "epochs": PROBE_EPOCHS,
    "dev_loss_relative_reduction_min": 0.05,
    "train_loss_relative_reduction_min": 0.10,
    "target_mean_min": 0.0001,
    "target_mean_max": 0.20,
    "positive_pixel_fraction_min": 0.0005,
    "positive_pixel_fraction_max": 0.40,
    "difficulty_std_min": 0.05,
    "valid_fraction_min": 0.50,
    "batches_with_targets_fraction_min": 0.95,
}
LEARNABILITY_GATE_SHA256 = hashlib.sha256(
    canonical_json_bytes(LEARNABILITY_GATE)
).hexdigest().upper()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def deterministic_probe_split(
    screen_paths: Iterable[Path], *, root: Path
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """Split the frozen 647-image Screen authority without RNG dependence."""
    paths = [Path(path).resolve() for path in screen_paths]
    if len(paths) != PROBE_TRAIN_IMAGES + PROBE_DEV_IMAGES:
        raise ValueError("RA learnability probe requires exactly 647 Screen images")
    if len(set(paths)) != len(paths):
        raise ValueError("RA learnability Screen list contains duplicate images")
    actual_screen_sha = subset_signature(paths, root=root)
    if actual_screen_sha != SCREEN_SUBSET_SHA256:
        raise ValueError(
            "RA learnability Screen authority mismatch: "
            f"expected={SCREEN_SUBSET_SHA256}, actual={actual_screen_sha}"
        )
    ranked = sorted(
        paths,
        key=lambda path: (
            hashlib.sha256(
                f"ra-probe-v1:{PROBE_SEED}:{_relative(path, root)}".encode("utf-8")
            ).digest(),
            _relative(path, root),
        ),
    )
    dev = ranked[:PROBE_DEV_IMAGES]
    train = ranked[PROBE_DEV_IMAGES:]

    def signature(values: Sequence[Path]) -> str:
        digest = hashlib.sha256()
        for value in values:
            digest.update(_relative(value, root).encode("utf-8") + b"\n")
        return digest.hexdigest().upper()

    record = {
        "algorithm": "sha256('ra-probe-v1:<seed>:<relative-path>'), first 129 dev, remaining 518 train",
        "seed": PROBE_SEED,
        "screen_count": len(paths),
        "screen_sha256": actual_screen_sha,
        "train_count": len(train),
        "train_sha256": signature(train),
        "dev_count": len(dev),
        "dev_sha256": signature(dev),
        "disjoint": not bool(set(train) & set(dev)),
    }
    return train, dev, record


def public_fdr_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy every public FDR tensor while excluding all RA private state."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(RA_GLGM_PRIVATE_PREFIX)
    }


def freeze_for_support_probe(model: torch.nn.Module) -> dict[str, Any]:
    """Freeze FDR and RA detection-residual parameters; train support path only."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if not name.startswith(RA_GLGM_PRIVATE_PREFIX):
            continue
        private_name = name.removeprefix(RA_GLGM_PRIVATE_PREFIX)
        if private_name.endswith(SUPPORT_EXCLUDED_SUFFIXES):
            continue
        parameter.requires_grad_(True)
        trainable.append(name)
    if not trainable:
        raise RuntimeError("RA support probe found no trainable private parameters")
    violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            not name.startswith(RA_GLGM_PRIVATE_PREFIX)
            or name.endswith(SUPPORT_EXCLUDED_SUFFIXES)
        )
    ]
    if violations:
        raise RuntimeError(f"RA support probe exposed forbidden trainable parameters: {violations[:5]}")
    return {
        "trainable_names": sorted(trainable),
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "private_prefix": RA_GLGM_PRIVATE_PREFIX,
        "excluded": list(SUPPORT_EXCLUDED_SUFFIXES),
    }


def summarize_targets(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("RA target audit received no batches")
    required = (
        "target_sum",
        "target_pixels",
        "positive_pixels",
        "valid_pixels",
        "total_pixels",
        "difficulty_count",
        "difficulty_sum",
        "difficulty_square_sum",
    )
    if any(any(not math.isfinite(float(record[name])) for name in required) for record in records):
        raise FloatingPointError("NONFINITE_RA_LEARNABILITY_TARGET_AUDIT")
    totals = {name: sum(float(record[name]) for record in records) for name in required}
    count = int(totals["difficulty_count"])
    mean_difficulty = totals["difficulty_sum"] / count if count else 0.0
    variance = max(
        0.0,
        totals["difficulty_square_sum"] / count - mean_difficulty**2,
    ) if count else 0.0
    return {
        "batches": len(records),
        "batches_with_targets": sum(int(record["difficulty_count"]) > 0 for record in records),
        "batches_with_targets_fraction": sum(
            int(record["difficulty_count"]) > 0 for record in records
        ) / len(records),
        "difficulty_count": count,
        "difficulty_mean": mean_difficulty,
        "difficulty_std": math.sqrt(variance),
        "target_mean": totals["target_sum"] / max(1.0, totals["target_pixels"]),
        "positive_pixel_fraction": totals["positive_pixels"] / max(1.0, totals["target_pixels"]),
        "valid_fraction": totals["valid_pixels"] / max(1.0, totals["total_pixels"]),
    }


def evaluate_learnability_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered gate without accepting runtime threshold overrides."""
    checkpoint = evidence.get("mature_fdr_checkpoint", {})
    split = evidence.get("split", {})
    freeze = evidence.get("freeze", {})
    targets = evidence.get("targets", {})
    losses = evidence.get("losses", {})
    train_losses = losses.get("train_epochs", [])
    dev_losses = losses.get("dev", [])
    finite_losses = (
        isinstance(train_losses, list)
        and len(train_losses) == PROBE_EPOCHS
        and isinstance(dev_losses, list)
        and len(dev_losses) == PROBE_EPOCHS + 1
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in [*train_losses, *dev_losses])
    )
    train_reduction = (
        1.0 - float(train_losses[-1]) / float(train_losses[0])
        if finite_losses and float(train_losses[0]) > 0
        else float("-inf")
    )
    dev_reduction = (
        1.0 - float(dev_losses[-1]) / float(dev_losses[0])
        if finite_losses and float(dev_losses[0]) > 0
        else float("-inf")
    )

    def target_checks(name: str) -> bool:
        value = targets.get(name, {})
        return (
            isinstance(value, Mapping)
            and int(value.get("difficulty_count", 0)) > 0
            and LEARNABILITY_GATE["target_mean_min"]
            <= float(value.get("target_mean", -1))
            <= LEARNABILITY_GATE["target_mean_max"]
            and LEARNABILITY_GATE["positive_pixel_fraction_min"]
            <= float(value.get("positive_pixel_fraction", -1))
            <= LEARNABILITY_GATE["positive_pixel_fraction_max"]
            and float(value.get("difficulty_std", -1)) >= LEARNABILITY_GATE["difficulty_std_min"]
            and float(value.get("valid_fraction", -1)) >= LEARNABILITY_GATE["valid_fraction_min"]
            and float(value.get("batches_with_targets_fraction", -1))
            >= LEARNABILITY_GATE["batches_with_targets_fraction_min"]
        )

    trainable_names = freeze.get("trainable_names", [])
    private_only = (
        isinstance(trainable_names, list)
        and bool(trainable_names)
        and all(
            isinstance(name, str)
            and name.startswith(RA_GLGM_PRIVATE_PREFIX)
            and not name.endswith(SUPPORT_EXCLUDED_SUFFIXES)
            for name in trainable_names
        )
    )
    checks = {
        "experiment_protocol_sha256": evidence.get("authority", {}).get(
            "protocol_sha256"
        )
        == RA_EXPERIMENT_PROTOCOL_SHA256,
        "mature_fdr_checkpoint_sha256": str(checkpoint.get("sha256", "")).upper()
        == MATURE_FDR_CHECKPOINT_SHA256,
        "screen647_authority": (
            split.get("screen_count") == 647
            and split.get("screen_sha256") == SCREEN_SUBSET_SHA256
        ),
        "deterministic_disjoint_518_129_split": (
            split.get("train_count") == PROBE_TRAIN_IMAGES
            and split.get("dev_count") == PROBE_DEV_IMAGES
            and split.get("disjoint") is True
        ),
        "frozen_train_dev_path_manifests": (
            isinstance(split.get("path_manifests"), Mapping)
            and all(
                isinstance(split["path_manifests"].get(name), Mapping)
                and split["path_manifests"][name].get("count") == count
                and isinstance(split["path_manifests"][name].get("sha256"), str)
                and len(split["path_manifests"][name]["sha256"]) == 64
                for name, count in (
                    ("train", PROBE_TRAIN_IMAGES),
                    ("dev", PROBE_DEV_IMAGES),
                )
            )
        ),
        "support_private_parameters_only": private_only,
        "public_fdr_state_unchanged": (
            isinstance(freeze.get("public_sha256_before"), str)
            and freeze.get("public_sha256_before") == freeze.get("public_sha256_after")
        ),
        "train_targets_non_degenerate": target_checks("train"),
        "dev_targets_non_degenerate": target_checks("dev"),
        "finite_fixed_length_losses": finite_losses,
        "train_loss_reduction_at_least_10_percent": train_reduction
        >= LEARNABILITY_GATE["train_loss_relative_reduction_min"],
        "holdout_loss_reduction_at_least_5_percent": dev_reduction
        >= LEARNABILITY_GATE["dev_loss_relative_reduction_min"],
        "best_holdout_occurs_after_initialization": (
            finite_losses and min(range(len(dev_losses)), key=lambda index: dev_losses[index]) > 0
        ),
    }
    passed = all(checks.values())
    return {
        "gate_name": "RA-GLGM-mature-FDR-residual-learnability-v1",
        "gate_sha256": LEARNABILITY_GATE_SHA256,
        "criteria": LEARNABILITY_GATE,
        "checks": checks,
        "train_loss_relative_reduction": train_reduction if math.isfinite(train_reduction) else None,
        "dev_loss_relative_reduction": dev_reduction if math.isfinite(dev_reduction) else None,
        "passed": passed,
        "smoke2_eligible": passed,
        "scientific_scope": "support-target learnability only; not detector accuracy evidence",
    }


def validate_learnability_report(
    path: str | Path,
    *,
    protocol_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("RA learnability report must be a JSON object")
    if report.get("format_version") != 1 or report.get("gate_sha256") != LEARNABILITY_GATE_SHA256:
        raise ValueError("RA learnability report format/gate authority is invalid")
    evidence = report.get("evidence")
    gate = report.get("gate")
    if not isinstance(evidence, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("RA learnability report is incomplete")
    evidence_sha = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest().upper()
    if report.get("evidence_sha256") != evidence_sha:
        raise ValueError("RA learnability evidence SHA256 mismatch")
    implementation = report.get("implementation")
    root = Path(__file__).resolve().parents[1]
    expected_implementation = {
        "core": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
        "runner": hashlib.sha256(
            (root / "scripts" / "run_ra_learnability_probe.py").read_bytes()
        ).hexdigest().upper(),
    }
    if implementation != expected_implementation:
        raise ValueError("RA learnability implementation SHA256 mismatch")
    unhashed = dict(report)
    recorded_report_sha = unhashed.pop("report_sha256", None)
    actual_report_sha = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if recorded_report_sha != actual_report_sha:
        raise ValueError("RA learnability report SHA256 mismatch")
    expected = evaluate_learnability_gate(evidence)
    if dict(gate) != expected:
        raise ValueError("RA learnability report gate does not match frozen recomputation")
    if gate.get("smoke2_eligible") is not True:
        raise ValueError("RA learnability report did not authorize Smoke2")
    if protocol_manifest is not None:
        authority = evidence.get("authority")
        expected = {
            "protocol_sha256": protocol_manifest.get("protocol_sha256"),
            "source": protocol_manifest.get("source"),
            "dataset_authority": protocol_manifest.get("dataset_authority"),
            "gpu_uuid": protocol_manifest.get("gpu_uuid"),
        }
        if authority != expected:
            raise ValueError("RA learnability report differs from experiment authority")
        manifests = evidence.get("split", {}).get("path_manifests", {})
        for name, count in (("train", PROBE_TRAIN_IMAGES), ("dev", PROBE_DEV_IMAGES)):
            record = manifests.get(name, {}) if isinstance(manifests, Mapping) else {}
            manifest_path = Path(str(record.get("path", ""))).resolve()
            if (
                not manifest_path.is_file()
                or record.get("count") != count
                or str(record.get("sha256", "")).upper() != file_sha256(manifest_path)
            ):
                raise ValueError(f"RA learnability {name} path manifest differs from evidence")
    return report


__all__ = [
    "LEARNABILITY_GATE",
    "LEARNABILITY_GATE_SHA256",
    "MATURE_FDR_CHECKPOINT_SHA256",
    "PROBE_BATCH",
    "PROBE_DEV_IMAGES",
    "PROBE_EPOCHS",
    "PROBE_OPTIMIZER",
    "PROBE_SEED",
    "PROBE_TRAIN_IMAGES",
    "SCREEN_SUBSET_SHA256",
    "deterministic_probe_split",
    "evaluate_learnability_gate",
    "freeze_for_support_probe",
    "public_fdr_state",
    "summarize_targets",
    "validate_learnability_report",
]

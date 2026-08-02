"""Equal-capacity B0-B3 Probe training and frozen IBER-BE Gate-1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.iber_cache import EvidenceCache, image_rgb_for_probe
from src.iber_head import IBEROutput, IBERRefiner
from src.iber_loss import IBERBucketCounts, boundary_bucket_counts, iber_private_loss
from src.iber_protocol import (
    BOUNDARY_LOSS_CONTRACT,
    DESIGN_VERSION,
    PRIVATE_OPTIMIZER,
    PRIVATE_SEED,
    PROBES,
    PROBE_EPOCHS,
    file_sha256,
    module_state_sha256,
    write_immutable_report,
)
from src.itber_geometry import correction_targets, cxcywh_to_xyxy
from src.itber_metrics import aligned_iou, direction_accuracy, edge_area_bucket


ARM_ORDER = ("b0", "b1", "b2", "b3")
PRIVATE_BATCH = 8
AMP_AUTHORITY = {
    "enabled_on_cuda": True,
    "dtype": "float16",
    "init_scale": 128.0,
    "growth_interval": 2**31 - 1,
}
REQUIRED_METRICS = (
    "edge_mae",
    "stock_matched_iou",
    "refined_matched_iou",
    "matched_iou_delta",
    "tiny_direction_accuracy",
    "small_direction_accuracy",
    "gate_mean",
    "gate_p95",
    "residual_rms",
    "gradient_rms",
    "total_loss",
    "f3_boundary_rms",
    "rgb_boundary_rms",
)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _same_mapping(value: object, expected: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    candidate = dict(value)
    wanted = dict(expected)
    if set(candidate) != set(wanted):
        return False
    for key, expected_value in wanted.items():
        actual = candidate[key]
        if key == "betas" and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected_value or type(actual) is not type(expected_value):
            return False
    return True


def _valid_history(value: object) -> bool:
    if not isinstance(value, list) or len(value) != PROBE_EPOCHS:
        return False
    for epoch, row in enumerate(value, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != epoch:
            return False
        if not _finite_number(row.get("total_loss")):
            return False
        gradient = row.get("gradient_rms")
        if gradient is not None and not _finite_number(gradient):
            return False
    return True


def _valid_cache_authority(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    required = {
        "baseline_sha256",
        "dataset_sha256",
        "category_sha256",
        "subset_sha256",
        "source_commit",
        "runtime_amendment_sha256",
    }
    if set(value) != required:
        return False
    for name in required - {"source_commit"}:
        if re.fullmatch(r"[0-9A-F]{64}", str(value[name])) is None:
            return False
    return re.fullmatch(r"[0-9a-f]{40}", str(value["source_commit"])) is not None


def _valid_boundary_loss_authority(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "contract",
        "bucket_counts",
        "batches_per_epoch",
    }:
        return False
    contract = value["contract"]
    if not isinstance(contract, Mapping) or json.dumps(
        dict(contract), sort_keys=True
    ) != json.dumps(dict(BOUNDARY_LOSS_CONTRACT), sort_keys=True):
        return False
    counts = value["bucket_counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"direction", "margin"}:
        return False
    for values in counts.values():
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 3
            or any(type(item) is not int or item < 0 for item in values)
        ):
            return False
    return type(value["batches_per_epoch"]) is int and value["batches_per_epoch"] > 0


def evaluate_gate1(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply every pre-registered Gate-1 condition without rounding."""
    exact_arms = isinstance(reports, Mapping) and set(reports) == set(ARM_ORDER)
    engineering: dict[str, bool] = {"exact_arms": exact_arms}
    if not exact_arms:
        return {
            "design_version": DESIGN_VERSION,
            "stage": "gate1_decision",
            "status": "engineering_invalid",
            "arm_order": list(ARM_ORDER),
            "engineering": engineering,
            "conditions": {},
        }

    ordered = [reports[arm] for arm in ARM_ORDER]
    engineering.update(
        {
            "arm_identity": all(
                isinstance(report, Mapping) and report.get("arm") == arm
                for arm, report in zip(ARM_ORDER, ordered)
            ),
            "frozen_report_authority": all(
                report.get("design_version") == DESIGN_VERSION
                and report.get("stage") == "gate1_probe"
                and report.get("epochs") == PROBE_EPOCHS
                and report.get("evaluated_epoch") == PROBE_EPOCHS
                and report.get("checkpoint_epoch") == PROBE_EPOCHS
                and report.get("selection") == "epoch12_only"
                and report.get("private_seed") == PRIVATE_SEED
                and report.get("batch_size") == PRIVATE_BATCH
                for report in ordered
            ),
            "optimizer_authority": all(
                _same_mapping(report.get("optimizer"), PRIVATE_OPTIMIZER)
                for report in ordered
            ),
            "amp_authority": all(
                _same_mapping(report.get("amp"), AMP_AUTHORITY)
                for report in ordered
            ),
            "exact_history": all(
                _valid_history(report.get("history")) for report in ordered
            ),
        }
    )

    parameter_counts = [report.get("parameter_count") for report in ordered]
    fingerprints = [report.get("initialization_sha256") for report in ordered]
    cache_authorities = [report.get("cache_authority") for report in ordered]
    boundary_loss_authorities = [report.get("boundary_loss") for report in ordered]
    engineering["equal_capacity"] = (
        all(type(value) is int and value > 0 for value in parameter_counts)
        and len(set(parameter_counts)) == 1
    )
    engineering["equal_initialization"] = (
        all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9A-F]{64}", value) is not None
            for value in fingerprints
        )
        and len(set(fingerprints)) == 1
    )
    engineering["equal_cache_authority"] = (
        all(_valid_cache_authority(value) for value in cache_authorities)
        and all(dict(value) == dict(cache_authorities[0]) for value in cache_authorities)
    )
    engineering["equal_boundary_loss_authority"] = (
        all(_valid_boundary_loss_authority(value) for value in boundary_loss_authorities)
        and all(
            json.dumps(dict(value), sort_keys=True)
            == json.dumps(dict(boundary_loss_authorities[0]), sort_keys=True)
            for value in boundary_loss_authorities
        )
    )
    engineering["finite_metrics"] = all(
        isinstance(report.get("metrics"), Mapping)
        and all(
            name in report["metrics"] and _finite_number(report["metrics"][name])
            for name in REQUIRED_METRICS
        )
        for report in ordered
    )
    if not all(engineering.values()):
        return {
            "design_version": DESIGN_VERSION,
            "stage": "gate1_decision",
            "status": "engineering_invalid",
            "arm_order": list(ARM_ORDER),
            "engineering": engineering,
            "conditions": {},
        }

    metrics = {arm: reports[arm]["metrics"] for arm in ARM_ORDER}
    b0, b1, b3 = metrics["b0"], metrics["b1"], metrics["b3"]
    conditions = {
        "edge_over_b0": b3["edge_mae"] <= b0["edge_mae"] * 0.95,
        "edge_over_b1": b3["edge_mae"] <= b1["edge_mae"] * 0.985,
        "matched_iou": b3["matched_iou_delta"] >= 0.005,
        "tiny_direction": (
            b3["tiny_direction_accuracy"] - b0["tiny_direction_accuracy"] >= 0.03
        ),
        "small_direction": (
            b3["small_direction_accuracy"] - b0["small_direction_accuracy"] >= 0.03
        ),
        "b3_best_primary": (
            b3["edge_mae"]
            == min(value["edge_mae"] for value in metrics.values())
            and b3["matched_iou_delta"]
            == max(value["matched_iou_delta"] for value in metrics.values())
        ),
        "finite_activity": (
            all(_finite_number(value) for value in b3.values())
            and b3["gradient_rms"] > 0.0
            and b3["gate_mean"] > 1e-4
            and 1e-3 < b3["gate_p95"] < 0.999
            and b3["residual_rms"] > 1e-4
        ),
    }
    return {
        "design_version": DESIGN_VERSION,
        "stage": "gate1_decision",
        "status": "passed" if all(conditions.values()) else "scientific_failed",
        "arm_order": list(ARM_ORDER),
        "engineering": engineering,
        "conditions": conditions,
        "reports": {arm: dict(reports[arm]) for arm in ARM_ORDER},
    }


def _batches(
    records: Sequence[Mapping[str, Any]], size: int = PRIVATE_BATCH
) -> Sequence[Sequence[Mapping[str, Any]]]:
    return [records[start : start + size] for start in range(0, len(records), size)]


def _fixed_boundary_bucket_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    rho: float,
    image_size: int,
) -> IBERBucketCounts:
    """Aggregate cache-global counts from immutable stock assignments."""
    total = IBERBucketCounts(direction=(0, 0, 0), margin=(0, 0, 0))
    for record in records:
        source = record["match_source"].to(dtype=torch.long)
        destination = record["match_target"].to(dtype=torch.long)
        if not len(source):
            continue
        stock_edges = cxcywh_to_xyxy(record["stock_boxes"].float())[source]
        target_edges = record["target_edges"].float()[destination]
        total = total + boundary_bucket_counts(
            stock_edges,
            target_edges,
            rho=rho,
            image_size=image_size,
        )
    return total


def _move_batch(
    records: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    if not records:
        raise ValueError("Probe batch cannot be empty")
    evidence = {
        name: torch.stack([record[name] for record in records])
        .to(device=device, dtype=torch.float32, non_blocking=True)
        for name in ("hidden", "stock_boxes", "stock_scores", "f3")
    }
    evidence["image_rgb"] = torch.stack(
        [image_rgb_for_probe(record) for record in records]
    ).to(device=device, dtype=torch.float32, non_blocking=True)

    targets: list[torch.Tensor] = []
    matches: list[tuple[torch.Tensor, torch.Tensor]] = []
    offset = 0
    for record in records:
        target = record["target_edges"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        targets.append(target)
        source = record["match_source"].to(device=device, dtype=torch.long)
        destination = record["match_target"].to(device=device, dtype=torch.long)
        matches.append((source, destination + offset))
        offset += len(target)
    target_edges = (
        torch.cat(targets, dim=0)
        if targets
        else torch.empty((0, 4), device=device, dtype=torch.float32)
    )
    return evidence, target_edges, matches


def _tensor_rms(values: list[torch.Tensor], *, device: torch.device) -> float:
    if not values:
        return 0.0
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for value in values:
        detached = value.detach().to(device=device, dtype=torch.float64)
        total += detached.square().sum()
        count += detached.numel()
    return float((total / max(count, 1)).sqrt().cpu())


def _gradient_rms(module: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("NONFINITE_IBER_PROBE_GRADIENT")
        total += gradient.square().sum()
        count += gradient.numel()
    if count == 0:
        return 0.0
    return float((total / count).sqrt())


@torch.no_grad()
def evaluate_probe_records(
    refiner: IBERRefiner,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    gradient_rms: float,
    total_loss: float,
) -> dict[str, float]:
    """Evaluate only the final Probe checkpoint over immutable val evidence."""
    if not records:
        raise ValueError("Probe validation cache is empty")
    edge_errors: list[torch.Tensor] = []
    stock_ious: list[torch.Tensor] = []
    refined_ious: list[torch.Tensor] = []
    predicted_directions: list[torch.Tensor] = []
    target_directions: list[torch.Tensor] = []
    target_buckets: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    f3_features: list[torch.Tensor] = []
    rgb_features: list[torch.Tensor] = []

    refiner.eval()
    for selected in _batches(records):
        evidence, targets, matches = _move_batch(selected, device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = refiner(**evidence)
        gates.append(output.gates.float())
        residuals.append(output.residuals.float())
        f3_features.append(output.f3_boundary_features.float())
        rgb_features.append(output.rgb_boundary_features.float())
        for image_index, (source, destination) in enumerate(matches):
            if not len(source):
                continue
            stock = output.stock_edges[image_index, source]
            refined = output.refined_edges[image_index, source]
            target = targets[destination]
            _, _, normalized = correction_targets(
                stock, target, rho=refiner.rho
            )
            edge_errors.append((refined - target).abs().float())
            stock_ious.append(aligned_iou(stock, target).float())
            refined_ious.append(aligned_iou(refined, target).float())
            predicted_directions.append(
                output.effective_correction[image_index, source].float()
            )
            target_directions.append(normalized.float())
            target_buckets.append(
                edge_area_bucket(target.float(), image_size=refiner.image_size)
            )

    if not edge_errors:
        raise ValueError("Probe validation cache has no matched targets")
    edge_values = torch.cat([value.reshape(-1) for value in edge_errors])
    stock_values = torch.cat(stock_ious)
    refined_values = torch.cat(refined_ious)
    predicted = torch.cat(predicted_directions)
    target = torch.cat(target_directions)
    buckets = torch.cat(target_buckets)
    gate_values = torch.cat([value.reshape(-1) for value in gates])
    residual_values = torch.cat([value.reshape(-1) for value in residuals])
    metrics = {
        "edge_mae": float(edge_values.mean().cpu()),
        "stock_matched_iou": float(stock_values.mean().cpu()),
        "refined_matched_iou": float(refined_values.mean().cpu()),
        "matched_iou_delta": float(
            (refined_values.mean() - stock_values.mean()).cpu()
        ),
        "tiny_direction_accuracy": float(
            direction_accuracy(predicted, target, mask=buckets == 0).cpu()
        ),
        "small_direction_accuracy": float(
            direction_accuracy(predicted, target, mask=buckets == 1).cpu()
        ),
        "gate_mean": float(gate_values.mean().cpu()),
        "gate_p95": float(torch.quantile(gate_values, 0.95).cpu()),
        "residual_rms": float(residual_values.square().mean().sqrt().cpu()),
        "gradient_rms": float(gradient_rms),
        "total_loss": float(total_loss),
        "f3_boundary_rms": _tensor_rms(f3_features, device=device),
        "rgb_boundary_rms": _tensor_rms(rgb_features, device=device),
    }
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid:
        raise FloatingPointError("NONFINITE_IBER_PROBE_METRIC: " + ", ".join(invalid))
    return metrics


def _save_checkpoint_immutable(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Probe checkpoint: {path}")
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        # Windows requires a writable descriptor for FlushFileBuffers/fsync.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def train_probe_arm(
    cache: EvidenceCache,
    *,
    arm: str,
    output_root: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Fresh-train one equal-capacity arm for exactly 12 epochs."""
    if arm not in PROBES or arm not in ARM_ORDER:
        raise ValueError(f"unknown IBER Probe arm: {arm}")
    if not cache.records["train"] or not cache.records["val"]:
        raise ValueError("Probe cache train and val splits must be non-empty")

    first = cache.records["train"][0]
    torch.manual_seed(PRIVATE_SEED)
    refiner = IBERRefiner(
        hidden_dim=int(first["hidden"].shape[-1]),
        f3_channels=int(first["f3"].shape[0]),
        private_seed=PRIVATE_SEED,
        probe=arm,
        image_size=640,
        rho=0.05,
    ).to(device)
    initialization_sha256 = module_state_sha256(refiner)
    parameter_count = sum(parameter.numel() for parameter in refiner.parameters())
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=float(PRIVATE_OPTIMIZER["lr"]),
        weight_decay=float(PRIVATE_OPTIMIZER["weight_decay"]),
        betas=tuple(PRIVATE_OPTIMIZER["betas"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=float(AMP_AUTHORITY["init_scale"]),
        growth_interval=int(AMP_AUTHORITY["growth_interval"]),
    )
    history: list[dict[str, float | int]] = []
    bucket_counts = _fixed_boundary_bucket_counts(
        cache.records["train"], rho=refiner.rho, image_size=refiner.image_size
    )
    batches_per_epoch = len(_batches(cache.records["train"]))
    refiner.train()
    for epoch in range(1, PROBE_EPOCHS + 1):
        losses: list[float] = []
        gradients: list[float] = []
        for selected in _batches(cache.records["train"]):
            evidence, target_edges, matches = _move_batch(selected, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output: IBEROutput = refiner(**evidence)
                private_losses = iber_private_loss(
                    output,
                    target_edges=target_edges,
                    match_indices=matches,
                    rho=refiner.rho,
                    image_size=refiner.image_size,
                    boundary_supervision=arm != "b0",
                    bucket_counts=bucket_counts,
                    batches_per_epoch=batches_per_epoch,
                )
            scaler.scale(private_losses.total).backward()
            scaler.unscale_(optimizer)
            gradient_value = _gradient_rms(refiner)
            if gradient_value <= 0.0:
                raise FloatingPointError("ZERO_IBER_PROBE_GRADIENT")
            torch.nn.utils.clip_grad_norm_(
                refiner.parameters(), max_norm=float(PRIVATE_OPTIMIZER["clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda" and scaler.get_scale() != AMP_AUTHORITY["init_scale"]:
                raise FloatingPointError("IBER Probe AMP scale drifted from 128")
            losses.append(float(private_losses.total.detach().float().cpu()))
            gradients.append(gradient_value)
        history.append(
            {
                "epoch": epoch,
                "total_loss": math.fsum(losses) / len(losses),
                "gradient_rms": math.fsum(gradients) / len(gradients),
            }
        )

    final = history[-1]
    metrics = evaluate_probe_records(
        refiner,
        cache.records["val"],
        device=device,
        gradient_rms=float(final["gradient_rms"]),
        total_loss=float(final["total_loss"]),
    )
    output_root = Path(output_root)
    checkpoint = output_root / f"{arm}-epoch-0012.pt"
    _save_checkpoint_immutable(
        checkpoint,
        {
            "format_version": 1,
            "design_version": DESIGN_VERSION,
            "stage": "gate1_probe",
            "arm": arm,
            "epoch": PROBE_EPOCHS,
            "private_seed": PRIVATE_SEED,
            "batch_size": PRIVATE_BATCH,
            "optimizer_authority": dict(PRIVATE_OPTIMIZER),
            "amp_authority": dict(AMP_AUTHORITY),
            "cache_authority": cache.manifest.authority,
            "initialization_sha256": initialization_sha256,
            "refiner": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "boundary_loss": {
                "contract": dict(BOUNDARY_LOSS_CONTRACT),
                "bucket_counts": bucket_counts.as_dict(),
                "batches_per_epoch": batches_per_epoch,
            },
        },
    )
    report = {
        "design_version": DESIGN_VERSION,
        "stage": "gate1_probe",
        "arm": arm,
        "epochs": PROBE_EPOCHS,
        "evaluated_epoch": PROBE_EPOCHS,
        "checkpoint_epoch": PROBE_EPOCHS,
        "selection": "epoch12_only",
        "private_seed": PRIVATE_SEED,
        "batch_size": PRIVATE_BATCH,
        "optimizer": dict(PRIVATE_OPTIMIZER),
        "amp": dict(AMP_AUTHORITY),
        "parameter_count": parameter_count,
        "initialization_sha256": initialization_sha256,
        "cache_authority": dict(cache.manifest.authority),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
        },
        "history": history,
        "boundary_loss": {
            "contract": dict(BOUNDARY_LOSS_CONTRACT),
            "bucket_counts": bucket_counts.as_dict(),
            "batches_per_epoch": batches_per_epoch,
        },
        "metrics": metrics,
    }
    write_immutable_report(output_root / f"{arm}-report.json", report)
    return report


__all__ = [
    "AMP_AUTHORITY",
    "ARM_ORDER",
    "PRIVATE_BATCH",
    "REQUIRED_METRICS",
    "evaluate_gate1",
    "evaluate_probe_records",
    "train_probe_arm",
]

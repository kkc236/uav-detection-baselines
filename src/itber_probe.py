"""Equal-capacity P0-P3 Probe training and immutable Gate 1 decision."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.itber_cache import EvidenceCache
from src.itber_geometry import correction_targets
from src.itber_head import ITBERRefiner, PROBES
from src.itber_loss import itber_private_loss
from src.itber_metrics import aligned_iou, direction_accuracy, edge_area_bucket
from src.itber_protocol import module_state_sha256, write_immutable_report


PROBE_EPOCHS = 12
PRIVATE_SEED = 10_000
PRIVATE_BATCH = 8


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_gate1(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply pre-registered Gate 1 thresholds without rounding inputs."""
    engineering: dict[str, bool] = {
        "all_probes_present": set(reports) == set(PROBES),
    }
    if not engineering["all_probes_present"]:
        return {"status": "engineering_invalid", "engineering": engineering, "conditions": {}}
    ordered = [reports[name] for name in ("p0", "p1", "p2", "p3")]
    engineering.update(
        {
            "exact_epochs": all(report.get("epochs") == PROBE_EPOCHS for report in ordered),
            "equal_capacity": len({report.get("parameter_count") for report in ordered}) == 1,
            "equal_initialization": len({report.get("initialization_sha256") for report in ordered}) == 1,
            "probe_identity": all(report.get("probe") == name for name, report in zip(("p0", "p1", "p2", "p3"), ordered)),
        }
    )
    required_metrics = (
        "edge_mae",
        "matched_iou_delta",
        "tiny_direction_accuracy",
        "small_direction_accuracy",
        "gate_mean",
        "gate_p95",
        "residual_rms",
    )
    engineering["finite_metrics"] = all(
        report.get("metrics", {}).get("finite") is True
        and all(_finite_number(report.get("metrics", {}).get(name)) for name in required_metrics)
        for report in ordered
    )
    if not all(engineering.values()):
        return {"status": "engineering_invalid", "engineering": engineering, "conditions": {}}

    metrics = {name: reports[name]["metrics"] for name in PROBES}
    p0, p2, p3 = metrics["p0"], metrics["p2"], metrics["p3"]
    conditions = {
        "edge_over_p0": p3["edge_mae"] <= p0["edge_mae"] * 0.95,
        "edge_over_p2": p3["edge_mae"] <= p2["edge_mae"] * 0.985,
        "matched_iou": p3["matched_iou_delta"] >= 0.005,
        "tiny_direction": p3["tiny_direction_accuracy"] - p0["tiny_direction_accuracy"] >= 0.03,
        "small_direction": p3["small_direction_accuracy"] - p0["small_direction_accuracy"] >= 0.03,
        "p3_best_primary": (
            p3["edge_mae"] == min(value["edge_mae"] for value in metrics.values())
            and p3["matched_iou_delta"] == max(value["matched_iou_delta"] for value in metrics.values())
        ),
        "finite_activity": (
            p3["gate_mean"] > 1e-4
            and 1e-3 < p3["gate_p95"] < 0.999
            and p3["residual_rms"] > 1e-4
        ),
    }
    return {
        "status": "passed" if all(conditions.values()) else "scientific_failed",
        "engineering": engineering,
        "conditions": conditions,
        "reports": {name: dict(reports[name]) for name in sorted(reports)},
    }


def _batches(records: Sequence[Mapping[str, Any]], size: int = PRIVATE_BATCH) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _move_batch(records: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    evidence = {
        name: torch.stack([record[name] for record in records]).to(device, non_blocking=True)
        for name in ("hidden", "box_l2", "box_l1", "stock_boxes", "stock_scores", "f3")
    }
    targets: list[torch.Tensor] = []
    matches: list[tuple[torch.Tensor, torch.Tensor]] = []
    offset = 0
    for record in records:
        target = record["target_edges"].to(device, non_blocking=True)
        targets.append(target)
        matches.append(
            (
                record["match_source"].to(device=device, dtype=torch.long),
                record["match_target"].to(device=device, dtype=torch.long) + offset,
            )
        )
        offset += len(target)
    target_edges = torch.cat(targets) if offset else torch.empty(0, 4, device=device)
    return evidence, target_edges, matches


@torch.no_grad()
def evaluate_probe_records(
    refiner: ITBERRefiner,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, float | bool]:
    edge_errors: list[torch.Tensor] = []
    stock_ious: list[torch.Tensor] = []
    refined_ious: list[torch.Tensor] = []
    predicted_directions: list[torch.Tensor] = []
    target_directions: list[torch.Tensor] = []
    target_buckets: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    for selected in _batches(records):
        evidence, targets, matches = _move_batch(selected, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = refiner(**evidence)
        gates.append(output.gates.float().cpu().reshape(-1))
        residuals.append(output.residuals.float().cpu().reshape(-1))
        for image_index, (source, destination) in enumerate(matches):
            if not len(source):
                continue
            stock = output.stock_edges[image_index, source]
            refined = output.refined_edges[image_index, source]
            target = targets[destination]
            magnitude, direction, _ = correction_targets(stock, target, rho=refiner.rho)
            edge_errors.append((refined - target).abs().float().cpu().reshape(-1))
            stock_ious.append(aligned_iou(stock, target).float().cpu())
            refined_ious.append(aligned_iou(refined, target).float().cpu())
            predicted_directions.append(output.effective_correction[image_index, source].float().cpu())
            target_directions.append((magnitude * direction).float().cpu())
            target_buckets.append(edge_area_bucket(target.float(), image_size=refiner.image_size).cpu())

    if not edge_errors:
        raise ValueError("Probe validation cache has no matched targets")
    edge_values = torch.cat(edge_errors)
    stock_values = torch.cat(stock_ious)
    refined_values = torch.cat(refined_ious)
    predicted = torch.cat(predicted_directions)
    target = torch.cat(target_directions)
    buckets = torch.cat(target_buckets)
    gate_values = torch.cat(gates)
    residual_values = torch.cat(residuals)
    metrics: dict[str, float | bool] = {
        "edge_mae": float(edge_values.mean()),
        "stock_matched_iou": float(stock_values.mean()),
        "refined_matched_iou": float(refined_values.mean()),
        "matched_iou_delta": float(refined_values.mean() - stock_values.mean()),
        "tiny_direction_accuracy": float(direction_accuracy(predicted, target, mask=buckets == 0)),
        "small_direction_accuracy": float(direction_accuracy(predicted, target, mask=buckets == 1)),
        "gate_mean": float(gate_values.mean()),
        "gate_p95": float(torch.quantile(gate_values, 0.95)),
        "residual_rms": float(residual_values.square().mean().sqrt()),
    }
    metrics["finite"] = all(math.isfinite(value) for value in metrics.values())
    return metrics


def train_probe_arm(
    cache: EvidenceCache,
    *,
    probe: str,
    output_root: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Fresh-train one fixed Probe arm and use only epoch 12 metrics."""
    if probe not in PROBES:
        raise ValueError(f"unknown probe: {probe}")
    first = cache.records["train"][0]
    refiner = ITBERRefiner(
        hidden_dim=first["hidden"].shape[-1],
        f3_channels=first["f3"].shape[0],
        private_seed=PRIVATE_SEED,
        probe=probe,
        image_size=640,
        rho=0.05,
    ).to(device)
    initial_sha = module_state_sha256(refiner)
    parameter_count = sum(parameter.numel() for parameter in refiner.parameters())
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=128.0,
        growth_interval=2**31 - 1,
    )
    history: list[dict[str, float | int]] = []
    refiner.train()
    for epoch in range(1, PROBE_EPOCHS + 1):
        totals: list[float] = []
        for selected in _batches(cache.records["train"]):
            evidence, target_edges, matches = _move_batch(selected, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = refiner(**evidence)
                losses = itber_private_loss(
                    output,
                    target_edges=target_edges,
                    match_indices=matches,
                    rho=refiner.rho,
                )
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            totals.append(float(losses.total.detach().float().cpu()))
        history.append({"epoch": epoch, "loss": sum(totals) / len(totals)})

    refiner.eval()
    metrics = evaluate_probe_records(refiner, cache.records["val"], device=device)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / f"{probe}-epoch-0012.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "design_version": "itber-v1.1",
            "stage": "probe",
            "probe": probe,
            "epoch": PROBE_EPOCHS,
            "private_seed": PRIVATE_SEED,
            "refiner": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cache_authority": cache.manifest.authority,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    report = {
        "design_version": "itber-v1.1",
        "stage": "probe",
        "probe": probe,
        "epochs": PROBE_EPOCHS,
        "private_seed": PRIVATE_SEED,
        "parameter_count": parameter_count,
        "initialization_sha256": initial_sha,
        "checkpoint": str(checkpoint),
        "history": history,
        "metrics": metrics,
    }
    write_immutable_report(output_root / f"{probe}-report.json", report)
    return report

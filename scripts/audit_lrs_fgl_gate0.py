"""Audit frozen mature Clean FDR for usable LRS-FGL signal on train10 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import Tensor
from ultralytics.utils.metrics import bbox_iou

from src.fdr_loss import (
    layerwise_reliability_shrinkage,
    representable_fgl_targets,
)
from src.fdr_math import (
    REG_MAX,
    REG_SCALE,
    UP,
    bbox2distance,
    cxcywh_to_xyxy,
    distance2bbox,
)
from src.rtdetr_fdr import FDRRTDETRDetectionModel


ALPHA0 = 0.25
BATCH_LIMIT = 16
LOW_Q_THRESHOLD = 0.2
SUM_TOLERANCE = 2e-6
MAX_BENEFICIARY_WEIGHT_TO_COUNT_RATIO = 0.90
QUALITY_BIN_EDGES = tuple(index / 10.0 for index in range(11))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def quality_bin_statistics(
    quality: Tensor, target_indices: Tensor
) -> list[dict[str, float | int]]:
    """Count boxes and saturated target edges in fixed 0.1-wide IoU bins."""

    if quality.ndim != 1:
        raise ValueError("quality must be one-dimensional")
    if target_indices.ndim != 1 or target_indices.numel() != quality.numel() * 4:
        raise ValueError("target_indices must contain four edges per quality value")
    boundaries = torch.tensor(
        QUALITY_BIN_EDGES[1:-1], dtype=quality.dtype, device=quality.device
    )
    bin_indices = torch.bucketize(quality, boundaries)
    saturated = (target_indices <= 0.0) | (target_indices >= REG_MAX - 1)
    rows: list[dict[str, float | int]] = []
    for index, (lower, upper) in enumerate(
        zip(QUALITY_BIN_EDGES[:-1], QUALITY_BIN_EDGES[1:])
    ):
        box_mask = bin_indices == index
        edge_mask = box_mask.repeat_interleave(4)
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "boxes": int(box_mask.sum()),
                "edges": int(edge_mask.sum()),
                "saturated_edges": int((saturated & edge_mask).sum()),
            }
        )
    return rows


def _metric_rows(layers: Sequence[Mapping[str, float | int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in layers:
        row = dict(source)
        matches = int(row["matches"])
        recoverable = int(row["recoverable_matches"])
        beneficiary = int(row["beneficiary_matches"])
        recoverable_quality = float(row["recoverable_quality_sum"])
        beneficiary_quality = float(row["beneficiary_quality_sum"])
        beneficiary_fraction = beneficiary / matches if matches else 0.0
        count_share = beneficiary / recoverable if recoverable else 0.0
        quality_share = (
            beneficiary_quality / recoverable_quality
            if recoverable_quality > 0.0
            else 0.0
        )
        share_ratio = quality_share / count_share if count_share > 0.0 else math.inf
        row.update(
            {
                "beneficiary_fraction": beneficiary_fraction,
                "beneficiary_count_share": count_share,
                "beneficiary_weight_share": quality_share,
                "beneficiary_weight_to_count_ratio": share_ratio,
            }
        )
        rows.append(row)
    return rows


def decide_gate0(
    layers: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    """Apply the frozen pre-training LRS signal checks."""

    rows = _metric_rows(layers)
    support = [row for row in rows if float(row["beneficiary_fraction"]) >= 0.25]
    eligible = [
        row
        for row in support
        if float(row["beneficiary_weight_to_count_ratio"])
        <= MAX_BENEFICIARY_WEIGHT_TO_COUNT_RATIO
    ]
    beneficiary_edges = sum(int(row["beneficiary_edges"]) for row in support)
    saturated_edges = sum(
        int(row["beneficiary_saturated_edges"]) for row in support
    )
    saturation = saturated_edges / beneficiary_edges if beneficiary_edges else 1.0
    checks = {
        "three_shallow_layers_have_recoverable_beneficiaries": len(support) >= 3,
        "beneficiaries_are_underweighted": len(eligible) == len(support)
        and len(eligible) >= 3,
        "beneficiaries_have_no_saturated_edges": saturated_edges == 0,
        "per_image_sum_conserved": max(
            (float(row["max_sum_error"]) for row in rows), default=math.inf
        )
        <= SUM_TOLERANCE,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "eligible_layer_indices": [int(row["layer_index"]) for row in eligible],
        "beneficiary_edge_saturation": saturation,
        "layers": rows,
    }


def _checkpoint_model(path: Path) -> torch.nn.Module:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    source: Any = artifact
    if isinstance(artifact, Mapping):
        source = artifact.get("ema")
        if source is None:
            source = artifact.get("model")
    if not isinstance(source, torch.nn.Module):
        raise TypeError("Clean FDR checkpoint contains no loadable model or EMA")
    return source.float()


def _load_clean_model(checkpoint: Path, device: torch.device) -> FDRRTDETRDetectionModel:
    source = _checkpoint_model(checkpoint)
    model = FDRRTDETRDetectionModel(
        ROOT / "configs" / "rtdetr-l-clean-fdr.yaml",
        ch=3,
        nc=10,
        verbose=False,
        private_seed=10_000,
    )
    result = model.load_state_dict(source.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict Clean FDR checkpoint loading was not exact")
    return model.to(device)


def _empty_layer(layer_index: int) -> dict[str, Any]:
    return {
        "layer_index": layer_index,
        "matches": 0,
        "recoverable_matches": 0,
        "beneficiary_matches": 0,
        "recoverable_quality_sum": 0.0,
        "beneficiary_quality_sum": 0.0,
        "beneficiary_edges": 0,
        "beneficiary_saturated_edges": 0,
        "max_sum_error": 0.0,
        "quality_bins": quality_bin_statistics(
            torch.empty(0), torch.empty(0)
        ),
    }


def _update_layer(
    row: dict[str, Any],
    *,
    layer_index: int,
    predicted_boxes: Tensor,
    reference_boxes: Tensor,
    targets: Tensor,
    matches: list[tuple[Tensor, Tensor]],
    criterion: Any,
) -> None:
    predicted_index, target_index = criterion._get_index(matches)
    if target_index.numel() == 0:
        return
    matched_targets = targets[target_index]
    quality = bbox_iou(
        predicted_boxes[predicted_index].detach(), matched_targets, xywh=True
    ).squeeze(-1).float()
    batch_indices = predicted_index[0].to(device=quality.device)
    target_indices, _right, _left = bbox2distance(
        reference_boxes[predicted_index].detach(),
        cxcywh_to_xyxy(matched_targets),
        REG_MAX,
        REG_SCALE,
        UP,
    )
    recoverable = representable_fgl_targets(target_indices)
    beneficiary = torch.zeros_like(recoverable)
    for batch_index in torch.unique(batch_indices):
        image_recoverable = (batch_indices == batch_index) & recoverable
        if int(image_recoverable.sum()) <= 1:
            continue
        image_mean = quality[image_recoverable].mean()
        beneficiary |= image_recoverable & (quality < image_mean)
    weights = layerwise_reliability_shrinkage(
        quality,
        batch_indices,
        layer_index=layer_index,
        num_layers=6,
        alpha0=ALPHA0,
        eligible_mask=recoverable,
    )
    max_error = 0.0
    for batch_index in torch.unique(batch_indices):
        mask = batch_indices == batch_index
        error = abs(
            float(weights[mask].double().sum() - quality[mask].double().sum())
        )
        max_error = max(max_error, error)
    beneficiary_edges = beneficiary.repeat_interleave(4)
    saturated = (target_indices <= 0.0) | (target_indices >= REG_MAX - 1)
    batch_bins = quality_bin_statistics(quality, target_indices)
    for total, current in zip(row["quality_bins"], batch_bins):
        total["boxes"] += current["boxes"]
        total["edges"] += current["edges"]
        total["saturated_edges"] += current["saturated_edges"]
    row["matches"] = int(row["matches"]) + int(quality.numel())
    row["recoverable_matches"] = int(row["recoverable_matches"]) + int(
        recoverable.sum()
    )
    row["beneficiary_matches"] = int(row["beneficiary_matches"]) + int(
        beneficiary.sum()
    )
    row["recoverable_quality_sum"] = float(row["recoverable_quality_sum"]) + float(
        quality[recoverable].sum()
    )
    row["beneficiary_quality_sum"] = float(row["beneficiary_quality_sum"]) + float(
        quality[beneficiary].sum()
    )
    row["beneficiary_edges"] = int(row["beneficiary_edges"]) + int(
        beneficiary_edges.sum()
    )
    row["beneficiary_saturated_edges"] = int(
        row["beneficiary_saturated_edges"]
    ) + int(
        (saturated & beneficiary_edges).sum()
    )
    row["max_sum_error"] = max(float(row["max_sum_error"]), max_error)


def run(args: Namespace) -> dict[str, Any]:
    if args.device != "0" or not torch.cuda.is_available():
        raise RuntimeError("Gate0 requires CUDA device 0")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(f"Clean FDR checkpoint is missing or unsafe: {checkpoint}")
    checkpoint_sha = _file_sha256(checkpoint)
    report_root = args.report_root.resolve()
    report_root.mkdir(parents=True, exist_ok=False)

    from src.fdr_runtime_preflight import _build_loader, _move_batch

    context = SimpleNamespace(
        dataset_root=args.dataset_root.resolve(), report_root=report_root
    )
    loader, subset_sha = _build_loader(context, augment=False)
    device = torch.device("cuda:0")
    model = _load_clean_model(checkpoint, device)
    model.train()
    layers: list[dict[str, Any]] = [_empty_layer(index) for index in range(5)]
    completed_batches = 0
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= int(args.batch_limit):
                break
            batch = _move_batch(raw_batch, device)
            model.loss(batch)
            evidence = model.last_fdr_evidence
            if evidence is None:
                raise RuntimeError("Clean FDR training evidence was not captured")
            assignments = model.criterion.normal_assignment_snapshot()[1:]
            if len(assignments) != 6:
                raise RuntimeError("Gate0 requires six normal decoder assignments")
            predicted_boxes = distance2bbox(
                evidence.references,
                model.fdr.integral(evidence.corner_logits),
                model.fdr.reg_scale,
            )
            for layer_index in range(5):
                _update_layer(
                    layers[layer_index],
                    layer_index=layer_index,
                    predicted_boxes=predicted_boxes[layer_index],
                    reference_boxes=evidence.references[0],
                    targets=batch["bboxes"],
                    matches=assignments[layer_index],
                    criterion=model.criterion,
                )
            completed_batches += 1
    if completed_batches != int(args.batch_limit):
        raise RuntimeError("Gate0 did not complete the frozen batch count")
    decision = decide_gate0(layers)
    if _file_sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("Clean FDR checkpoint changed during Gate0")
    payload = {
        "format_version": 1,
        "status": "passed" if decision["passed"] else "scientific_failed",
        "formal100_eligible": bool(decision["passed"]),
        "purpose": "lrs_fgl_pretraining_signal_gate",
        "data_scope": "fixed_train10_only",
        "official_val_opened": False,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "runtime": {
            "device": "cuda:0",
            "gpu": torch.cuda.get_device_name(0),
            "batch_limit": int(args.batch_limit),
            "batch": 8,
            "imgsz": 640,
            "augment": False,
        },
        "fixed_subset_sha256": subset_sha,
        "decision": decision,
    }
    report = report_root / "lrs-fgl-gate0.json"
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-limit", type=int, default=BATCH_LIMIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(build_parser().parse_args(argv))
    return 0 if payload["formal100_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

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

from src.fdr_loss import layerwise_reliability_shrinkage
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _metric_rows(layers: Sequence[Mapping[str, float | int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in layers:
        row = dict(source)
        matches = int(row["matches"])
        low_matches = int(row["low_matches"])
        quality_sum = float(row["quality_sum"])
        low_quality_sum = float(row["low_quality_sum"])
        low_fraction = low_matches / matches if matches else 0.0
        count_share = low_fraction
        quality_share = low_quality_sum / quality_sum if quality_sum > 0.0 else 0.0
        share_ratio = quality_share / count_share if count_share > 0.0 else math.inf
        row.update(
            {
                "low_q_fraction": low_fraction,
                "low_q_count_share": count_share,
                "low_q_weight_share": quality_share,
                "low_q_weight_to_count_ratio": share_ratio,
            }
        )
        rows.append(row)
    return rows


def decide_gate0(
    layers: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    """Apply the frozen pre-training LRS signal checks."""

    rows = _metric_rows(layers)
    support = [row for row in rows if float(row["low_q_fraction"]) >= 0.25]
    eligible = [
        row
        for row in support
        if float(row["low_q_weight_to_count_ratio"]) <= 0.5
    ]
    low_edges = sum(int(row["low_edges"]) for row in support)
    saturated_edges = sum(int(row["saturated_low_edges"]) for row in support)
    saturation = saturated_edges / low_edges if low_edges else 1.0
    checks = {
        "three_shallow_layers_have_low_q_support": len(support) >= 3,
        "supported_layers_are_underweighted": len(eligible) == len(support)
        and len(eligible) >= 3,
        "low_q_edge_saturation_below_half": saturation < 0.5,
        "per_image_sum_conserved": max(
            (float(row["max_sum_error"]) for row in rows), default=math.inf
        )
        <= SUM_TOLERANCE,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "eligible_layer_indices": [int(row["layer_index"]) for row in eligible],
        "low_q_edge_saturation": saturation,
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


def _empty_layer(layer_index: int) -> dict[str, float | int]:
    return {
        "layer_index": layer_index,
        "matches": 0,
        "low_matches": 0,
        "quality_sum": 0.0,
        "low_quality_sum": 0.0,
        "low_edges": 0,
        "saturated_low_edges": 0,
        "max_sum_error": 0.0,
    }


def _update_layer(
    row: dict[str, float | int],
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
    batch_indices = predicted_index[0]
    low = quality < LOW_Q_THRESHOLD
    weights = layerwise_reliability_shrinkage(
        quality,
        batch_indices,
        layer_index=layer_index,
        num_layers=6,
        alpha0=ALPHA0,
    )
    max_error = 0.0
    for batch_index in torch.unique(batch_indices):
        mask = batch_indices == batch_index
        error = abs(float(weights[mask].sum() - quality[mask].sum()))
        max_error = max(max_error, error)
    target_indices, _right, _left = bbox2distance(
        reference_boxes[predicted_index].detach(),
        cxcywh_to_xyxy(matched_targets),
        REG_MAX,
        REG_SCALE,
        UP,
    )
    low_edges = low.repeat_interleave(4)
    saturated = (target_indices <= 0.0) | (target_indices >= REG_MAX - 1)
    row["matches"] = int(row["matches"]) + int(quality.numel())
    row["low_matches"] = int(row["low_matches"]) + int(low.sum())
    row["quality_sum"] = float(row["quality_sum"]) + float(quality.sum())
    row["low_quality_sum"] = float(row["low_quality_sum"]) + float(
        quality[low].sum()
    )
    row["low_edges"] = int(row["low_edges"]) + int(low_edges.sum())
    row["saturated_low_edges"] = int(row["saturated_low_edges"]) + int(
        (saturated & low_edges).sum()
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
    layers = [_empty_layer(index) for index in range(5)]
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

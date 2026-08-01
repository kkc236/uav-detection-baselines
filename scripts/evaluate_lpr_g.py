"""Independently evaluate stock/refined outputs from one LPR-G v2 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from ultralytics.models.rtdetr.val import RTDETRValidator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rtdetr_lpr_g import FROZEN_PROTOCOL
from src.github_checkpoint_sync import checkpoint_metadata
from src.lpr_g_loss import MatchRecordingRTDETRDetectionLoss
from src.lpr_protocol import (
    EXPECTED_DATASET_SHA256,
    current_environment,
    dataset_signature,
    environment_violations,
    source_violations,
)
from src.rtdetr_lpr_g import LPRGRTDETRDetectionModel


def _model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    source = checkpoint.get("ema")
    if source is None:
        source = checkpoint.get("model")
    if isinstance(source, torch.nn.Module):
        state = source.float().state_dict()
    elif isinstance(source, Mapping):
        state = source.get("state_dict", source)
    else:
        raise ValueError("checkpoint has no loadable EMA/model state")
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint model state is not a tensor state dictionary")
    return dict(state)


def load_lpr_g_checkpoint(path: Path) -> tuple[LPRGRTDETRDetectionModel, dict]:
    """Load a resumable method checkpoint strictly and reject control weights."""
    metadata = checkpoint_metadata(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _model_state(checkpoint)
    if not any("lpr_g_refiner." in name for name in state):
        raise ValueError("independent LPR-G evaluation refuses control checkpoints")
    model = LPRGRTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False, private_seed=10_000
    )
    model.load_state_dict(state, strict=True)
    return model, {
        "completed_epoch": metadata.completed_epoch,
        "bytes": metadata.bytes,
        "sha256": metadata.sha256,
    }


def _targets(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    image = batch["img"]
    batch_index = batch["batch_idx"]
    batch_size = image.shape[0]
    return {
        "cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
        "bboxes": batch["bboxes"].to(device=image.device),
        "batch_idx": batch_index.to(image.device, dtype=torch.long).view(-1),
        "gt_groups": [
            (batch_index == index).sum().item() for index in range(batch_size)
        ],
    }


def _matched_localization(
    model: LPRGRTDETRDetectionModel,
    validator: RTDETRValidator,
    *,
    mode: str,
) -> dict[str, float]:
    """Accumulate matched stock/refined box losses using the stock assignment."""
    criterion = MatchRecordingRTDETRDetectionLoss(nc=10, use_vfl=True)
    totals = {"l1": 0.0, "giou": 0.0}
    normalization = 0
    model.set_refinement_output(mode)
    model.eval()
    with torch.inference_mode():
        for raw_batch in validator.dataloader:
            batch = validator.preprocess(raw_batch)
            model.predict(batch["img"])
            decoder = model.model[-1].decoder
            stock = decoder.last_stock_bboxes
            refined = decoder.last_refined_bboxes
            scores = decoder.last_stock_scores
            if stock is None or refined is None or scores is None:
                raise RuntimeError("LPR-G decoder did not expose matched localization outputs")
            targets = _targets(batch)
            criterion((stock.unsqueeze(0), scores.unsqueeze(0)), targets)
            selected = stock if mode == "stock" else refined
            losses = criterion.refinement_loss(selected, targets)
            weight = max(int(targets["bboxes"].shape[0]), 1)
            totals["l1"] += float(losses["loss_bbox_refine"].detach().float().cpu()) * weight
            totals["giou"] += float(losses["loss_giou_refine"].detach().float().cpu()) * weight
            normalization += weight
    if normalization == 0:
        raise RuntimeError("validation set produced no localization normalization")
    return {name: value / normalization for name, value in totals.items()}


def validate_with_matched_localization(
    model: LPRGRTDETRDetectionModel,
    *,
    data: str | Path,
    mode: str,
) -> tuple[Any, dict[str, float]]:
    """Run frozen AP validation and a second no-augmentation matched pass."""
    model.set_refinement_output(mode)
    validator = RTDETRValidator(
        args={
            "model": "rtdetr-l.yaml",
            "data": str(data),
            "imgsz": 640,
            "batch": 8,
            "workers": 8,
            "device": "0",
            "max_det": 300,
            "nms": False,
            "cache": False,
            "plots": False,
            "save_json": False,
            "verbose": False,
            "task": "detect",
            "mode": "val",
            "split": "val",
            "rect": False,
        }
    )
    validator(model=model)
    localization = _matched_localization(model, validator, mode=mode)
    return validator.metrics, localization


def _distribution(prefix: str, tensor: torch.Tensor | None) -> dict[str, float]:
    if tensor is None or tensor.numel() == 0:
        raise RuntimeError(f"missing {prefix} distribution after validation")
    values = tensor.detach().float().reshape(-1).cpu()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"non-finite {prefix} distribution")
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_rms": float(values.square().mean().sqrt()),
        f"{prefix}_p05": float(torch.quantile(values, 0.05)),
        f"{prefix}_p50": float(torch.quantile(values, 0.50)),
        f"{prefix}_p95": float(torch.quantile(values, 0.95)),
    }


def validate_evaluation_authority(
    checkpoint: Path,
    runtime: dict,
    *,
    stage: str,
) -> dict:
    if runtime.get("variant") != "lprg":
        raise ValueError("independent LPR-G evaluation requires a method runtime manifest")
    if runtime.get("stage") != stage or runtime.get("seed") != 0:
        raise ValueError("evaluation stage/seed does not match the checkpoint runtime")
    if runtime.get("protocol") != FROZEN_PROTOCOL:
        raise ValueError("checkpoint training protocol is not frozen LPR-G v2")
    authority = runtime.get("authority", {})
    if authority.get("format_version") != 2 or authority.get("seed") != 0:
        raise ValueError("checkpoint authority is not format-v2 seed0")
    violations = environment_violations(current_environment())
    if violations:
        raise ValueError(f"evaluation environment does not match authority: {violations}")
    drift = source_violations()
    if drift:
        raise ValueError(f"evaluation Ultralytics source drift: {drift}")
    dataset = dataset_signature(Path(authority["dataset_root"]))
    expected_dataset = {"file_count": 14038, "sha256": EXPECTED_DATASET_SHA256}
    if dataset != expected_dataset or authority.get("dataset") != expected_dataset:
        raise ValueError("evaluation dataset does not match frozen authority")
    expected_epochs = 50 if stage == "screen" else 100
    metadata = checkpoint_metadata(checkpoint)
    if metadata.completed_epoch != expected_epochs:
        raise ValueError(
            f"evaluation requires completed epoch {expected_epochs}, got {metadata.completed_epoch}"
        )
    return authority


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    stage: str,
    runtime_manifest: Path,
) -> dict:
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    authority = validate_evaluation_authority(checkpoint, runtime, stage=stage)
    model, checkpoint_record = load_lpr_g_checkpoint(checkpoint)
    report: dict[str, Any] = {
        "design_version": "lpr-g-v2",
        "variant": "lprg",
        "stage": stage,
        "seed": 0,
        "checkpoint": checkpoint_record,
        "source_commit": authority.get("git_commit"),
        "environment": current_environment(),
        "protocol": runtime,
    }
    for mode in ("stock", "refined"):
        metrics, localization = validate_with_matched_localization(
            model,
            data=authority["data"][stage]["path"],
            mode=mode,
        )
        report[mode] = {
            "map": float(metrics.box.map),
            "map50": float(metrics.box.map50),
            "ap75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "matched_l1": float(localization["l1"]),
            "matched_giou": float(localization["giou"]),
        }
    refiner = model.model[-1].decoder.lpr_g_refiner
    report["activity"] = {
        **_distribution("quality", refiner.last_quality),
        **_distribution("gate", refiner.last_gate),
        **_distribution("residual", refiner.last_residual),
    }
    if not all(
        math.isfinite(float(value))
        for section in ("stock", "refined", "activity")
        for value in report[section].values()
    ):
        raise FloatingPointError("independent evaluation produced non-finite evidence")
    return report


def _write_immutable(path: Path, report: dict) -> None:
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed evaluation report: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently evaluate one frozen LPR-G v2 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    runtime_manifest = (
        args.runtime_manifest.resolve()
        if args.runtime_manifest is not None
        else checkpoint.parent.parent / "lpr_g_protocol.json"
    )
    report = evaluate_checkpoint(
        checkpoint,
        stage=args.stage,
        runtime_manifest=runtime_manifest,
    )
    _write_immutable(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

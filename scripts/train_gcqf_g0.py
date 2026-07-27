"""Train only the GCQF module from sealed frozen-detector evidence."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from src.gcqf import GCQF
from src.gcqf_cache import VerifiedEvidenceCache
from src.gcqf_cache import SRPEG_CACHE_SCHEMA_VERSION
from src.gcqf_loss import compute_gcqf_loss
from src.gcqf_training import (
    GCQF_BATCH_SIZE,
    GCQF_EPOCHS,
    GCQF_FIXED_AMP_SCALE,
    GCQF_LR,
    GCQF_LRF,
    GCQF_MOMENTUM,
    GCQF_WARMUP_EPOCHS,
    GCQF_WARMUP_MOMENTUM,
    build_module_optimizer,
    collate_evidence_records,
    compute_positive_weights,
    split_seed0_records,
)


MODULE_ARTIFACT_SCHEMA = "gcte-gcqf-module/v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed seed0-only SR-PEG module screen."
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--epochs", type=int, default=GCQF_EPOCHS)
    parser.add_argument("--batch", type=int, default=GCQF_BATCH_SIZE)
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp-scale",
        type=float,
        default=GCQF_FIXED_AMP_SCALE,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def schedule_values(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> tuple[float, float]:
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("training step is outside the schedule")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be inside training")
    if step <= warmup_steps:
        fraction = step / max(warmup_steps, 1)
        return (
            GCQF_LR * fraction,
            GCQF_WARMUP_MOMENTUM
            + (GCQF_MOMENTUM - GCQF_WARMUP_MOMENTUM) * fraction,
        )
    progress = (step - warmup_steps) / max(
        total_steps - 1 - warmup_steps,
        1,
    )
    learning_rate = GCQF_LR * (
        (1.0 - progress) + GCQF_LRF * progress
    )
    return learning_rate, GCQF_MOMENTUM


def build_module_artifact(
    module: GCQF,
    *,
    seed: int,
    epoch: int,
    train_cache_sha256: str,
    source_commit: str,
    train_image_ids: tuple[str, ...],
    calibration_image_ids: tuple[str, ...],
    positive_weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": MODULE_ARTIFACT_SCHEMA,
        "module": "GCQF",
        "config": {
            "query_dim": module.query_dim,
            "num_classes": module.num_classes,
            "num_heads": module.query_interaction.attention.num_heads,
            "num_views": module.num_views,
            "residual_cap": (
                module.geometry_projector.residual_cap
            ),
            "residual_eta": (
                module.sr_peg.residual_eta
            ),
        },
        "seed": int(seed),
        "epoch": int(epoch),
        "train_cache_sha256": train_cache_sha256.upper(),
        "source_commit": source_commit.lower(),
        "train_image_ids": list(train_image_ids),
        "calibration_image_ids": list(calibration_image_ids),
        "positive_weights": dict(positive_weights),
        "module_state": {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        },
    }


def validate_training_protocol(args: argparse.Namespace) -> None:
    """Fail closed on any deviation from the approved seed0 diagnostic."""

    if (
        args.seed != 0
        or args.epochs != GCQF_EPOCHS
        or args.batch != GCQF_BATCH_SIZE
        or args.optimizer != "MuSGD"
        or args.device != "0"
        or args.amp_scale != GCQF_FIXED_AMP_SCALE
        or not args.amp
    ):
        raise ValueError("GCQF G0 training protocol drift")
    commit = str(args.source_commit).lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("source commit must be an exact 40-character Git SHA")


def _seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loss_for_batch(
    module: GCQF,
    batch,
    *,
    amp: bool,
    positive_weights: dict[str, float],
):
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp,
    ):
        output = module(
            batch.global_evidence,
            batch.local_evidence,
            batch.geometry,
            anchor_mask=batch.anchor_mask,
            residual_enabled=True,
        )
        loss = compute_gcqf_loss(
            adjusted_scores=output.adjusted_local_scores,
            quality_targets=batch.quality_targets,
            canonical_queries=output.canonical_local.queries,
            equivariance_pairs=batch.equivariance_pairs,
            score_residual=output.score_residual,
            valid_mask=batch.geometry.valid_mask,
            anchor_mask=batch.anchor_mask,
            tiny_utility_logits=output.tiny_utility_logits,
            tiny_utility_targets=batch.local_tiny_utility_targets,
            non_tiny_risk_logits=output.non_tiny_risk_logits,
            non_tiny_risk_targets=batch.local_non_tiny_risk_targets,
            global_retain_logits=output.global_retain_logits,
            global_retain_targets=batch.global_retain_targets,
            positive_weights=positive_weights,
        )
    return output, loss


def _epoch(
    *,
    module: GCQF,
    loader,
    device: torch.device,
    amp: bool,
    optimizer=None,
    scaler=None,
    step_offset: int = 0,
    total_steps: int = 1,
    warmup_steps: int = 0,
    positive_weights: dict[str, float],
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    module.train(training)
    totals = {
        "total": 0.0,
        "quality": 0.0,
        "equivariance": 0.0,
        "residual": 0.0,
        "tiny_utility": 0.0,
        "non_tiny_risk": 0.0,
        "global_retain": 0.0,
    }
    count = 0
    step = step_offset
    for records in loader:
        batch = collate_evidence_records(
            records,
            require_sr_peg_targets=True,
        ).to(device)
        if training:
            learning_rate, momentum = schedule_values(
                step=step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
                group["momentum"] = momentum
            optimizer.zero_grad(set_to_none=True)
            _output, loss = _loss_for_batch(
                module,
                batch,
                amp=amp,
                positive_weights=positive_weights,
            )
            scaler.scale(loss.total).backward()
            if float(scaler.get_scale()) != GCQF_FIXED_AMP_SCALE:
                raise FloatingPointError("GCQF AMP scale drift before step")
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) != GCQF_FIXED_AMP_SCALE:
                raise FloatingPointError("GCQF AMP scale drift after step")
            step += 1
        else:
            with torch.no_grad():
                _output, loss = _loss_for_batch(
                    module,
                    batch,
                    amp=amp,
                    positive_weights=positive_weights,
                )
        batch_size = batch.global_evidence.batch_size
        count += batch_size
        totals["total"] += float(loss.total.detach()) * batch_size
        totals["quality"] += float(loss.quality.detach()) * batch_size
        totals["equivariance"] += (
            float(loss.equivariance.detach()) * batch_size
        )
        totals["residual"] += (
            float(loss.residual_regularization.detach()) * batch_size
        )
        totals["tiny_utility"] += (
            float(loss.tiny_utility.detach()) * batch_size
        )
        totals["non_tiny_risk"] += (
            float(loss.non_tiny_risk.detach()) * batch_size
        )
        totals["global_retain"] += (
            float(loss.global_retain.detach()) * batch_size
        )
    if count <= 0:
        raise RuntimeError("GCQF epoch received no records")
    return {
        name: value / count for name, value in totals.items()
    }, step


def train(args: argparse.Namespace) -> Path:
    validate_training_protocol(args)
    if not torch.cuda.is_available():
        raise RuntimeError("GCQF G0 training requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _seed_everything(args.seed)
    train_cache = VerifiedEvidenceCache(args.train_cache)
    if train_cache.manifest["schema_version"] != SRPEG_CACHE_SCHEMA_VERSION:
        raise ValueError("training requires a supervised v2 train cache")
    all_records = list(train_cache.iter_records())
    train_records, calibration_records = split_seed0_records(all_records)
    positive_weights = compute_positive_weights(train_records)
    first = train_records[0]
    module = GCQF(
        query_dim=first.global_evidence.query_dim,
        num_classes=first.global_evidence.num_classes,
        num_heads=8,
        num_views=4,
        residual_cap=0.2,
        residual_eta=0.2,
    ).to(torch.device("cuda:0"))
    optimizer = build_module_optimizer(module)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=GCQF_FIXED_AMP_SCALE,
        growth_interval=2**31 - 1,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_records,
        batch_size=args.batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda values: values,
    )
    calibration_loader = torch.utils.data.DataLoader(
        calibration_records,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda values: values,
    )
    steps_per_epoch = math.ceil(len(train_records) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(GCQF_WARMUP_EPOCHS * steps_per_epoch))
    results_path = output / "losses.csv"
    best_loss = float("inf")
    best_path = output / "best-module.pt"
    last_path = output / "last-module.pt"
    step = 0
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "epoch",
            "train_total",
            "train_quality",
            "train_equivariance",
            "train_residual",
            "train_tiny_utility",
            "train_non_tiny_risk",
            "train_global_retain",
            "calibration_total",
            "calibration_quality",
            "calibration_equivariance",
            "calibration_residual",
            "calibration_tiny_utility",
            "calibration_non_tiny_risk",
            "calibration_global_retain",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics, step = _epoch(
                module=module,
                loader=train_loader,
                device=torch.device("cuda:0"),
                amp=args.amp,
                optimizer=optimizer,
                scaler=scaler,
                step_offset=step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                positive_weights=positive_weights,
            )
            calibration_metrics, _ = _epoch(
                module=module,
                loader=calibration_loader,
                device=torch.device("cuda:0"),
                amp=args.amp,
                positive_weights=positive_weights,
            )
            row = {
                "epoch": epoch,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.items()
                },
                **{
                    f"calibration_{name}": value
                    for name, value in calibration_metrics.items()
                },
            }
            writer.writerow(row)
            handle.flush()
            artifact = build_module_artifact(
                module,
                seed=args.seed,
                epoch=epoch,
                train_cache_sha256=_sha256_file(
                    Path(args.train_cache)
                ),
                source_commit=args.source_commit,
                train_image_ids=tuple(
                    record.image_id for record in train_records
                ),
                calibration_image_ids=tuple(
                    record.image_id for record in calibration_records
                ),
                positive_weights=positive_weights,
            )
            torch.save(artifact, last_path)
            if calibration_metrics["total"] < best_loss:
                best_loss = calibration_metrics["total"]
                torch.save(artifact, best_path)
            print(
                f"GCQF_TRAIN epoch={epoch}/{args.epochs} "
                f"train={train_metrics['total']:.8f} "
                f"calibration={calibration_metrics['total']:.8f}",
                flush=True,
            )
    manifest = {
        "schema_version": "gcte-gcqf-training/v2",
        "source_commit": args.source_commit.lower(),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "amp_scale": args.amp_scale,
        "baseline_sha256": train_cache.manifest["baseline_sha256"],
        "train_cache_manifest_sha256": _sha256_file(
            Path(args.train_cache)
        ),
        "train_image_ids": [
            record.image_id for record in train_records
        ],
        "calibration_image_ids": [
            record.image_id for record in calibration_records
        ],
        "positive_weights": positive_weights,
        "best_module_sha256": _sha256_file(best_path),
        "last_module_sha256": _sha256_file(last_path),
        "losses_sha256": _sha256_file(results_path),
        "best_calibration_loss": best_loss,
        "parameter_count": sum(
            parameter.numel() for parameter in module.parameters()
        ),
        "optimizer_groups": [
            {
                "param_group": group.get("param_group"),
                "use_muon": bool(group.get("use_muon", False)),
                "parameter_count": sum(
                    parameter.numel() for parameter in group["params"]
                ),
                "weight_decay": float(group["weight_decay"]),
            }
            for group in optimizer.param_groups
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"GCQF_TRAIN_COMPLETE seed={args.seed} "
        f"manifest_sha256={_sha256_file(manifest_path)}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    print(train(build_parser().parse_args()))


if __name__ == "__main__":
    main()

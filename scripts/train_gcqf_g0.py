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
)


MODULE_ARTIFACT_SCHEMA = "gcte-gcqf-module/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the two-seed module-only GCQF G0 screen."
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
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
    val_cache_sha256: str,
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
                module.residual_fusion.residual_eta
            ),
        },
        "seed": int(seed),
        "epoch": int(epoch),
        "train_cache_sha256": train_cache_sha256.upper(),
        "val_cache_sha256": val_cache_sha256.upper(),
        "module_state": {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        },
    }


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
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    module.train(training)
    totals = {
        "total": 0.0,
        "quality": 0.0,
        "equivariance": 0.0,
        "residual": 0.0,
    }
    count = 0
    step = step_offset
    for records in loader:
        batch = collate_evidence_records(records).to(device)
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
            _output, loss = _loss_for_batch(module, batch, amp=amp)
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
                _output, loss = _loss_for_batch(module, batch, amp=amp)
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
    if count <= 0:
        raise RuntimeError("GCQF epoch received no records")
    return {
        name: value / count for name, value in totals.items()
    }, step


def train(args: argparse.Namespace) -> Path:
    if (
        args.epochs != GCQF_EPOCHS
        or args.batch != GCQF_BATCH_SIZE
        or args.optimizer != "MuSGD"
        or args.device != "0"
        or args.amp_scale != GCQF_FIXED_AMP_SCALE
        or not args.amp
    ):
        raise ValueError("GCQF G0 training protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("GCQF G0 training requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _seed_everything(args.seed)
    train_cache = VerifiedEvidenceCache(args.train_cache)
    val_cache = VerifiedEvidenceCache(args.val_cache)
    if (
        train_cache.manifest["baseline_sha256"]
        != val_cache.manifest["baseline_sha256"]
    ):
        raise ValueError("train and val cache baseline authorities differ")
    train_records = list(train_cache.iter_records())
    val_records = list(val_cache.iter_records())
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
    val_loader = torch.utils.data.DataLoader(
        val_records,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda values: values,
    )
    steps_per_epoch = math.ceil(len(train_records) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(GCQF_WARMUP_EPOCHS * steps_per_epoch))
    results_path = output / "results.csv"
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
            "val_total",
            "val_quality",
            "val_equivariance",
            "val_residual",
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
            )
            val_metrics, _ = _epoch(
                module=module,
                loader=val_loader,
                device=torch.device("cuda:0"),
                amp=args.amp,
            )
            row = {
                "epoch": epoch,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.items()
                },
                **{
                    f"val_{name}": value
                    for name, value in val_metrics.items()
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
                val_cache_sha256=_sha256_file(Path(args.val_cache)),
            )
            torch.save(artifact, last_path)
            if val_metrics["total"] < best_loss:
                best_loss = val_metrics["total"]
                torch.save(artifact, best_path)
            print(
                f"GCQF_TRAIN epoch={epoch}/{args.epochs} "
                f"train={train_metrics['total']:.8f} "
                f"val={val_metrics['total']:.8f}",
                flush=True,
            )
    manifest = {
        "schema_version": "gcte-gcqf-training/v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "amp_scale": args.amp_scale,
        "baseline_sha256": train_cache.manifest["baseline_sha256"],
        "train_cache_manifest_sha256": _sha256_file(
            Path(args.train_cache)
        ),
        "val_cache_manifest_sha256": _sha256_file(Path(args.val_cache)),
        "best_module_sha256": _sha256_file(best_path),
        "last_module_sha256": _sha256_file(last_path),
        "results_sha256": _sha256_file(results_path),
        "best_val_loss": best_loss,
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

"""Train one frozen-detector I-TBER screen or formal private stage."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.itber_metrics import correction_rms  # noqa: E402
from src.itber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    assert_detector_frozen,
    module_state_sha256,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    dataset_signature,
    file_sha256,
    select_hashed_subset,
)
from src.rtdetr_itber import FrozenITBERAdapter  # noqa: E402


TRAINING_CONSTANTS = {
    "seed": 0,
    "private_seed": 10000,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "amp": True,
    "amp_scale": 128.0,
    "save_period": 1,
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "betas": (0.9, 0.999),
    "clip_grad_norm": 10.0,
    "on_the_fly_evidence": True,
}
AUGMENTATION = {
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


@dataclass(frozen=True)
class StageProtocol:
    name: str
    train_images: int
    val_images: int
    epochs: int
    use_subset: bool
    fresh_private_initialization: bool = True
    allow_prior_stage_checkpoint: bool = False


def stage_protocol(stage: str) -> StageProtocol:
    if stage == "screen":
        return StageProtocol("screen", 647, 548, 12, True)
    if stage == "formal":
        return StageProtocol("formal", 6471, 548, 30, False)
    raise ValueError(f"unknown I-TBER stage: {stage}")


def build_private_optimizer(module: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        module.parameters(),
        lr=TRAINING_CONSTANTS["lr"],
        betas=TRAINING_CONSTANTS["betas"],
        weight_decay=TRAINING_CONSTANTS["weight_decay"],
    )


def validate_resume_checkpoint(artifact: dict[str, Any], *, stage: str) -> None:
    expected = {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "stage": stage,
        "probe": "p3",
        "seed": 0,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
    }
    violations = {
        name: {"expected": value, "actual": artifact.get(name)}
        for name, value in expected.items()
        if artifact.get(name) != value
    }
    protocol = stage_protocol(stage)
    epoch = artifact.get("epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= protocol.epochs:
        violations["epoch"] = {"expected": f"1..{protocol.epochs}", "actual": epoch}
    if violations:
        raise ValueError("invalid I-TBER resume " + ", ".join(sorted(violations)))


def atomic_save_private_checkpoint(path: str | Path, artifact: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, destination)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_train_loader(dataset_root: Path, output_root: Path, protocol: StageProtocol):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import RTDETRDataset

    image_paths = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    if protocol.use_subset:
        image_paths = select_hashed_subset(image_paths, root=dataset_root, fraction=0.10)
        subset_path = output_root / "fixed-train647.txt"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text("\n".join(str(path.resolve()) for path in image_paths) + "\n", encoding="utf-8")
        image_source = str(subset_path)
    else:
        image_source = str((dataset_root / "images" / "train").resolve())
    if len(image_paths) != protocol.train_images:
        raise ValueError(f"{protocol.name} train count mismatch: {len(image_paths)}")
    val_count = len(list((dataset_root / "images" / "val").glob("*.jpg")))
    if val_count != protocol.val_images:
        raise ValueError(f"validation image count mismatch: {val_count}")

    overrides = {
        "task": "detect",
        "mode": "train",
        "imgsz": TRAINING_CONSTANTS["imgsz"],
        "batch": TRAINING_CONSTANTS["batch"],
        "workers": TRAINING_CONSTANTS["workers"],
        "cache": False,
        "rect": False,
        "single_cls": False,
        "classes": None,
        "fraction": 1.0,
        "deterministic": True,
        **AUGMENTATION,
    }
    cfg = get_cfg(overrides=overrides)
    data = {
        "path": str(dataset_root.resolve()),
        "train": image_source,
        "val": str((dataset_root / "images" / "val").resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": len(CATEGORY_NAMES),
        "channels": 3,
    }
    dataset = RTDETRDataset(
        img_path=image_source,
        imgsz=TRAINING_CONSTANTS["imgsz"],
        batch_size=TRAINING_CONSTANTS["batch"],
        augment=True,
        hyp=cfg,
        rect=False,
        cache=None,
        single_cls=False,
        prefix=f"itber-{protocol.name}: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    loader = build_dataloader(
        dataset,
        batch=TRAINING_CONSTANTS["batch"],
        workers=TRAINING_CONSTANTS["workers"],
        shuffle=True,
        rank=-1,
        drop_last=False,
    )
    return loader, cfg


def _move_training_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {
        name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    moved["img"] = moved["img"].float().div_(255)
    return moved


def _correction_diagnostics(adapter: FrozenITBERAdapter) -> tuple[float, float, dict[str, float]]:
    output = adapter.last_output
    matches = adapter.last_match_indices
    if output is None or matches is None:
        raise RuntimeError("missing I-TBER output diagnostics")
    batch, queries = output.effective_correction.shape[:2]
    matched = torch.zeros((batch, queries), device=output.effective_correction.device, dtype=torch.bool)
    for image_index, (source, _target) in enumerate(matches):
        matched[image_index, source.to(device=matched.device, dtype=torch.long)] = True
    correction = output.effective_correction.float()
    matched_correction_rms = float(correction_rms(correction, matched).detach().cpu())
    unmatched_correction_rms = float(correction_rms(correction, ~matched).detach().cpu())
    activity = {
        "gate_mean": float(output.gates.float().mean().detach().cpu()),
        "gate_p95": float(torch.quantile(output.gates.float(), 0.95).detach().cpu()),
        "residual_mean": float(output.residuals.float().mean().detach().cpu()),
        "residual_rms": float(output.residuals.float().square().mean().sqrt().detach().cpu()),
    }
    return matched_correction_rms, unmatched_correction_rms, activity


def _restore_rng(artifact: dict[str, Any]) -> None:
    random.setstate(artifact["rng"]["python"])
    np.random.set_state(artifact["rng"]["numpy"])
    torch.set_rng_state(artifact["rng"]["torch"])
    if torch.cuda.is_available() and artifact["rng"].get("cuda") is not None:
        torch.cuda.set_rng_state_all(artifact["rng"]["cuda"])


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    args = _parse_args()
    protocol = stage_protocol(args.stage)
    _seed_everything(TRAINING_CONSTANTS["seed"])
    baseline_sha = file_sha256(args.baseline_checkpoint)
    dataset_sha = str(dataset_signature(args.dataset_root)["sha256"])
    if baseline_sha != EXPECTED_BASELINE_SHA256 or dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError("I-TBER training authority mismatch")

    from ultralytics import RTDETR

    device = torch.device(f"cuda:{args.device}")
    detector = RTDETR(str(args.baseline_checkpoint)).model.to(device).eval()
    detector.requires_grad_(False)
    adapter = FrozenITBERAdapter.from_detector(
        detector,
        private_seed=TRAINING_CONSTANTS["private_seed"],
        probe="p3",
        image_size=TRAINING_CONSTANTS["imgsz"],
        rho=0.05,
    ).to(device).train()
    detector.eval()
    detector.requires_grad_(False)
    assert_detector_frozen(detector)
    detector_sha_before = module_state_sha256(detector)
    optimizer = build_private_optimizer(adapter.refiner)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=TRAINING_CONSTANTS["amp_scale"],
        growth_interval=2**31 - 1,
    )
    start_epoch = 1
    if args.resume_checkpoint is not None:
        artifact = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        validate_resume_checkpoint(artifact, stage=args.stage)
        if artifact.get("detector_sha_after") != detector_sha_before:
            raise ValueError("detector_sha_after resume authority mismatch")
        adapter.refiner.load_state_dict(artifact["refiner"], strict=True)
        optimizer.load_state_dict(artifact["optimizer"])
        scaler.load_state_dict(artifact["scaler"])
        _restore_rng(artifact)
        start_epoch = artifact["epoch"] + 1
    loader, dataset_hyp = _build_train_loader(args.dataset_root, args.output_root, protocol)
    diagnostics_path = args.output_root / "diagnostics.jsonl"
    checkpoint_root = args.output_root / "checkpoints"

    for epoch in range(start_epoch, protocol.epochs + 1):
        if epoch - 1 == protocol.epochs - AUGMENTATION["close_mosaic"]:
            loader.dataset.close_mosaic(hyp=copy.copy(dataset_hyp))
        loss_accumulator: dict[str, list[float]] = {}
        matched_values: list[float] = []
        unmatched_values: list[float] = []
        activity_values: dict[str, list[float]] = {}
        for raw_batch in loader:
            batch = _move_training_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                losses = adapter.training_step(batch)
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                adapter.refiner.parameters(), max_norm=TRAINING_CONSTANTS["clip_grad_norm"]
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite I-TBER private gradient")
            if any(parameter.grad is not None for parameter in detector.parameters()):
                raise RuntimeError("frozen detector received a gradient")
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) != TRAINING_CONSTANTS["amp_scale"]:
                raise RuntimeError("fixed I-TBER amp_scale changed")
            for name, value in vars(losses).items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    loss_accumulator.setdefault(name, []).append(float(value.detach().float().cpu()))
            matched_rms, unmatched_rms, activity = _correction_diagnostics(adapter)
            matched_values.append(matched_rms)
            unmatched_values.append(unmatched_rms)
            for name, value in activity.items():
                activity_values.setdefault(name, []).append(value)

        detector_sha_after = module_state_sha256(detector)
        if detector_sha_after != detector_sha_before:
            raise RuntimeError("frozen detector state changed during private training")
        diagnostic = {
            "epoch": epoch,
            "stage": protocol.name,
            "losses": {name: sum(values) / len(values) for name, values in loss_accumulator.items()},
            "matched_correction_rms": sum(matched_values) / len(matched_values),
            "unmatched_correction_rms": sum(unmatched_values) / len(unmatched_values),
            "activity": {name: sum(values) / len(values) for name, values in activity_values.items()},
            "detector_sha_before": detector_sha_before,
            "detector_sha_after": detector_sha_after,
            "amp_scale": float(scaler.get_scale()),
        }
        artifact = {
            "format_version": 1,
            "design_version": "itber-v1.1",
            "stage": protocol.name,
            "probe": "p3",
            "seed": TRAINING_CONSTANTS["seed"],
            "private_seed": TRAINING_CONSTANTS["private_seed"],
            "epoch": epoch,
            "baseline_sha256": baseline_sha,
            "dataset_sha256": dataset_sha,
            "detector_sha_before": detector_sha_before,
            "detector_sha_after": detector_sha_after,
            "refiner": adapter.refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _rng_state(),
            "diagnostic": diagnostic,
            "training_constants": TRAINING_CONSTANTS,
            "augmentation": AUGMENTATION,
        }
        checkpoint = atomic_save_private_checkpoint(
            checkpoint_root / f"epoch-{epoch:04d}.pt", artifact
        )
        last_temporary = checkpoint_root / "last.pt.tmp"
        shutil.copy2(checkpoint, last_temporary)
        os.replace(last_temporary, checkpoint_root / "last.pt")
        _append_jsonl(diagnostics_path, diagnostic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

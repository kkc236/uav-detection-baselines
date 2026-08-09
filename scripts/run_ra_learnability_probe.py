"""Run the frozen mature-FDR residual support learnability probe before Smoke2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader  # noqa: E402

from src.btd_se_dataset import BTDSEVisDroneDataset  # noqa: E402
from src.fdr_protocol import public_state_sha256  # noqa: E402
from src.fdr_protocol import canonical_json_bytes  # noqa: E402
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    dataset_signature,
    select_hashed_subset,
)
from src.ra_glgm_loss import (  # noqa: E402
    ResidualDifficultyTargets,
    build_residual_difficulty_targets,
    residual_support_focal_loss,
)
from src.ra_learnability_probe import (  # noqa: E402
    LEARNABILITY_GATE,
    LEARNABILITY_GATE_SHA256,
    MATURE_FDR_CHECKPOINT_SHA256,
    PROBE_BATCH,
    PROBE_EPOCHS,
    PROBE_OPTIMIZER,
    PROBE_SEED,
    deterministic_probe_split,
    evaluate_learnability_gate,
    freeze_for_support_probe,
    public_fdr_state,
    summarize_targets,
)
from src.ra_experiment_protocol import (  # noqa: E402
    file_sha256,
    ignore_sidecar_signature,
)
from src.rtdetr_btdse import filter_detection_batch  # noqa: E402
from src.rtdetr_fdr import FDRRTDETRDetectionModel  # noqa: E402
from src.rtdetr_ra_glgm import RAGLGMDetectionModel  # noqa: E402
from scripts.train_rtdetr_ra_glgm import load_authority, validate_source  # noqa: E402


def _seed_everything() -> None:
    random.seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PROBE_SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)


def read_screen_list(path: str | Path, *, dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).resolve()
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"frozen Screen list is missing: {source}")
    images: list[Path] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        image = Path(line.strip()).resolve()
        try:
            image.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Screen image escapes dataset root at line {line_number}") from error
        if image.is_symlink() or not image.is_file():
            raise FileNotFoundError(f"Screen image is missing at line {line_number}: {image}")
        images.append(image)
    return images


def fixed_screen_paths(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).resolve()
    images = sorted((root / "images" / "train").glob("*.jpg"))
    if len(images) != 6471:
        raise ValueError(f"RA learnability requires 6471 train images, got {len(images)}")
    selected = select_hashed_subset(images, root=root, fraction=0.10)
    if len(selected) != 647:
        raise ValueError(f"RA learnability requires 647 Screen images, got {len(selected)}")
    return selected


def _gpu_uuid() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise ValueError("RA learnability requires exactly one visible physical GPU")
    return values[0]


def load_mature_fdr_public_state(model: RAGLGMDetectionModel, checkpoint: Path) -> dict[str, Any]:
    """Load every FDR public tensor while retaining deterministic RA-private initialization."""
    checkpoint = checkpoint.resolve()
    digest = file_sha256(checkpoint)
    if digest != MATURE_FDR_CHECKPOINT_SHA256:
        raise ValueError(
            "mature FDR checkpoint SHA256 mismatch: "
            f"expected={MATURE_FDR_CHECKPOINT_SHA256}, actual={digest}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("mature FDR checkpoint is not a mapping")
    source_model = payload.get("ema") or payload.get("model")
    if not isinstance(source_model, nn.Module):
        raise ValueError("mature FDR checkpoint has no loadable EMA/model module")
    source_state = source_model.float().state_dict()
    target_state = model.state_dict()
    public_names = set(public_fdr_state(model))
    missing = sorted(public_names - set(source_state))
    unexpected_private = sorted(name for name in source_state if ".ra_glgm." in name)
    if missing or unexpected_private:
        raise ValueError(
            "mature checkpoint is not the public FDR authority: "
            f"missing={missing[:5]}, unexpected_ra={unexpected_private[:5]}"
        )
    for name in sorted(public_names):
        source = source_state[name]
        target = target_state[name]
        if source.shape != target.shape:
            raise ValueError(f"mature FDR public tensor shape differs: {name}")
        target_state[name] = source.to(dtype=target.dtype)
    model.load_state_dict(target_state, strict=True)
    return {
        "path": str(checkpoint),
        "sha256": digest,
        "bytes": checkpoint.stat().st_size,
        "public_tensors_loaded": len(public_names),
    }


def _dataset_args() -> Any:
    return get_cfg(
        overrides={
            "imgsz": 640,
            "batch": PROBE_BATCH,
            "workers": 8,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
            "rect": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "bgr": 0.0,
            "auto_augment": None,
            "erasing": 0.0,
        }
    )


def write_probe_path_manifest(path: str | Path, paths: Sequence[Path]) -> dict[str, Any]:
    """Write one create-only Ultralytics image list and bind its exact bytes."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = [Path(value).resolve() for value in paths]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("RA learnability path manifest must be non-empty and unique")
    missing = [value for value in resolved if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"RA learnability image is missing: {missing[0]}")
    payload = "".join(f"{value}\n" for value in resolved)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace RA learnability path manifest: {destination}"
        ) from error
    return {
        "path": str(destination),
        "count": len(resolved),
        "sha256": file_sha256(destination),
    }


def build_probe_loader(
    paths: Sequence[Path],
    *,
    path_manifest: Path,
    workers: int,
    shuffle: bool,
):
    args = _dataset_args()
    data = {
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": len(CATEGORY_NAMES),
        "channels": 3,
    }
    # augment=True is required only so the audited ignore sidecar is appended;
    # every stochastic geometry/color probability above is frozen to zero.
    dataset = BTDSEVisDroneDataset(
        # Ultralytics 8.4.90 treats every file passed directly in a list as a
        # text image-list file. Pass one frozen text manifest instead.
        img_path=str(path_manifest.resolve()),
        imgsz=640,
        batch_size=PROBE_BATCH,
        augment=True,
        hyp=args,
        rect=False,
        cache=None,
        single_cls=False,
        prefix="ra-learnability: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    return build_dataloader(
        dataset,
        batch=PROBE_BATCH,
        workers=workers,
        shuffle=shuffle,
        rank=-1,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = {
        name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    result["img"] = result["img"].float().div_(255.0)
    return result


def _set_probe_mode(model: RAGLGMDetectionModel, *, support_training: bool) -> None:
    # Decoder training mode is required to expose the normal Hungarian
    # assignment.  Public BN/Dropout modules remain frozen and deterministic.
    model.train()
    ra_modules = {id(module) for module in model.ra_glgm.modules()}
    for module in model.modules():
        if id(module) not in ra_modules and isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
            module.eval()
    model.ra_glgm.train(support_training)


def support_objective(
    model: RAGLGMDetectionModel,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ResidualDifficultyTargets]:
    """Reuse the mature FDR final assignment and return only support supervision."""
    detection = filter_detection_batch(dict(batch))
    image = batch["img"]
    batch_index = detection["batch_idx"]
    decoder_targets = {
        "cls": detection["cls"].to(image.device, dtype=torch.long).view(-1),
        "bboxes": detection["bboxes"].to(image.device),
        "batch_idx": batch_index.to(image.device, dtype=torch.long).view(-1),
        "gt_groups": [
            int((batch_index == index).sum().item()) for index in range(int(image.shape[0]))
        ],
    }
    preds = model.predict(image, batch=decoder_targets)
    # Invoke the unchanged FDR criterion solely to record its final normal
    # assignment.  Its detector loss is intentionally excluded from backward.
    FDRRTDETRDetectionModel.loss(model, detection, preds=preds)
    if not isinstance(preds, tuple) or len(preds) != 5:
        raise RuntimeError("RA learnability prediction contract changed")
    dec_bboxes, dec_scores, _, _, dn_meta = preds
    if dn_meta is not None:
        partition = dn_meta.get("dn_num_split")
        if not isinstance(partition, (list, tuple)) or len(partition) != 2:
            raise ValueError("RA learnability denoising partition is invalid")
        _, dec_bboxes = torch.split(dec_bboxes, tuple(map(int, partition)), dim=2)
        _, dec_scores = torch.split(dec_scores, tuple(map(int, partition)), dim=2)
    assignment = model.criterion.last_normal_decoder_assignment
    support = model.ra_glgm.last_support_map
    if assignment is None or support is None:
        raise RuntimeError("RA learnability assignment/support diagnostics are unavailable")
    targets = build_residual_difficulty_targets(
        pred_bboxes=dec_bboxes[-1],
        pred_scores=dec_scores[-1],
        detection_bboxes=detection["bboxes"],
        detection_classes=detection["cls"],
        detection_batch_idx=detection["batch_idx"],
        match_indices=assignment,
        all_bboxes=batch["bboxes"],
        all_classes=batch["cls"],
        all_batch_idx=batch["batch_idx"],
        height=int(support.shape[-2]),
        width=int(support.shape[-1]),
    )
    loss = residual_support_focal_loss(support, targets)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("NONFINITE_RA_LEARNABILITY_LOSS")
    return loss, targets


def _target_record(targets: ResidualDifficultyTargets) -> dict[str, float | int]:
    heatmap = targets.heatmap.detach().float()
    difficulty = targets.difficulty.detach().float()
    return {
        "target_sum": float(heatmap.sum().cpu()),
        "target_pixels": int(heatmap.numel()),
        "positive_pixels": int((heatmap > 0).sum().cpu()),
        "valid_pixels": int(targets.valid_mask.sum().cpu()),
        "total_pixels": int(targets.valid_mask.numel()),
        "difficulty_count": int(difficulty.numel()),
        "difficulty_sum": float(difficulty.sum().cpu()),
        "difficulty_square_sum": float(difficulty.square().sum().cpu()),
    }


@torch.no_grad()
def evaluate_dev(
    model: RAGLGMDetectionModel, loader: Any, device: torch.device
) -> tuple[float, list[dict[str, float | int]]]:
    _set_probe_mode(model, support_training=False)
    losses: list[float] = []
    target_records: list[dict[str, float | int]] = []
    for raw in loader:
        batch = _move_batch(raw, device)
        loss, targets = support_objective(model, batch)
        losses.append(float(loss.cpu()))
        target_records.append(_target_record(targets))
    if not losses:
        raise RuntimeError("RA learnability dev loader produced no batches")
    return sum(losses) / len(losses), target_records


def train_epoch(
    model: RAGLGMDetectionModel,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, list[dict[str, float | int]]]:
    _set_probe_mode(model, support_training=True)
    losses: list[float] = []
    target_records: list[dict[str, float | int]] = []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for raw in loader:
        batch = _move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        loss, targets = support_objective(model, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(PROBE_OPTIMIZER["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("NONFINITE_RA_LEARNABILITY_GRADIENT")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        target_records.append(_target_record(targets))
    if not losses:
        raise RuntimeError("RA learnability train loader produced no batches")
    return sum(losses) / len(losses), target_records


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace RA learnability evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("RA mature-FDR learnability probe requires the authorized RTX 4090")
    _seed_everything()
    manifest = load_authority(args.protocol_manifest.resolve())
    validate_source(manifest)
    dataset_root = args.dataset_root.resolve()
    dataset_authority = manifest.get("dataset_authority")
    if not isinstance(dataset_authority, Mapping):
        raise ValueError("RA learnability manifest has no dataset authority")
    if dataset_root != Path(str(dataset_authority.get("root", ""))).resolve():
        raise ValueError("RA learnability dataset root differs from experiment authority")
    if dataset_signature(dataset_root) != dataset_authority.get("positive"):
        raise ValueError("RA learnability positive dataset differs from experiment authority")
    if ignore_sidecar_signature(dataset_root) != dataset_authority.get("ignore"):
        raise ValueError("RA learnability ignore sidecar differs from experiment authority")
    gpu_uuid = _gpu_uuid()
    if gpu_uuid != manifest.get("gpu_uuid"):
        raise ValueError("RA learnability GPU differs from experiment authority")
    screen_paths = fixed_screen_paths(dataset_root)
    train_paths, dev_paths, split = deterministic_probe_split(screen_paths, root=dataset_root)
    output = args.output.resolve()
    train_manifest = write_probe_path_manifest(
        output.with_name(f"{output.stem}-train-images.txt"), train_paths
    )
    dev_manifest = write_probe_path_manifest(
        output.with_name(f"{output.stem}-dev-images.txt"), dev_paths
    )
    split["path_manifests"] = {"train": train_manifest, "dev": dev_manifest}
    checkpoint = args.mature_fdr_checkpoint.resolve()
    device = torch.device("cuda:0")
    model = RAGLGMDetectionModel(nc=10, verbose=False).to(device)
    checkpoint_record = load_mature_fdr_public_state(model, checkpoint)
    public_before = public_state_sha256(public_fdr_state(model))
    freeze = freeze_for_support_probe(model)
    freeze["public_sha256_before"] = public_before
    train_loader = build_probe_loader(
        train_paths,
        path_manifest=Path(train_manifest["path"]),
        workers=args.workers,
        shuffle=True,
    )
    dev_loader = build_probe_loader(
        dev_paths,
        path_manifest=Path(dev_manifest["path"]),
        workers=args.workers,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(PROBE_OPTIMIZER["lr"]),
        weight_decay=float(PROBE_OPTIMIZER["weight_decay"]),
    )

    initial_dev, dev_targets = evaluate_dev(model, dev_loader, device)
    train_losses: list[float] = []
    dev_losses = [initial_dev]
    train_target_records: list[dict[str, float | int]] | None = None
    for _epoch in range(PROBE_EPOCHS):
        train_loss, target_records = train_epoch(model, train_loader, optimizer, device)
        if train_target_records is None:
            train_target_records = target_records
        train_losses.append(train_loss)
        dev_loss, _ = evaluate_dev(model, dev_loader, device)
        dev_losses.append(dev_loss)
    freeze["public_sha256_after"] = public_state_sha256(public_fdr_state(model))
    evidence = {
        "format_version": 1,
        "authority": {
            "protocol_sha256": manifest["protocol_sha256"],
            "source": manifest["source"],
            "dataset_authority": dataset_authority,
            "gpu_uuid": gpu_uuid,
        },
        "mature_fdr_checkpoint": checkpoint_record,
        "split": split,
        "freeze": freeze,
        "probe": {
            "seed": PROBE_SEED,
            "epochs": PROBE_EPOCHS,
            "batch": PROBE_BATCH,
            "optimizer": PROBE_OPTIMIZER,
            "amp": False,
            "augmentation": "all stochastic geometry/color disabled; ignore sidecar retained",
        },
        "targets": {
            "train": summarize_targets(train_target_records or []),
            "dev": summarize_targets(dev_targets),
        },
        "losses": {"train_epochs": train_losses, "dev": dev_losses},
    }
    gate = evaluate_learnability_gate(evidence)
    report = {
        "format_version": 1,
        "gate_sha256": LEARNABILITY_GATE_SHA256,
        "evidence_sha256": hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest().upper(),
        "implementation": {
            "core": file_sha256(ROOT / "src" / "ra_learnability_probe.py"),
            "runner": file_sha256(Path(__file__).resolve()),
        },
        "evidence": evidence,
        "gate": gate,
    }
    report["report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest().upper()
    _write_create_only_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--mature-fdr-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    report = run_probe(build_parser().parse_args())
    if report["gate"]["smoke2_eligible"] is not True:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

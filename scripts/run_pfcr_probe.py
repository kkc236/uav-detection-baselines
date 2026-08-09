"""Run the source-bound Protected Frequency Candidate Rescue learnability probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pfcr import (  # noqa: E402
    PFCRGate,
    RESCUE_SLOT_GRID,
    one_to_one_union_teacher,
    pfcr_boundary_loss,
    pfcr_features,
    protected_merge,
    stock_predictions,
)
from src.pfcr_cache import PFCRCacheWriter, load_pfcr_cache  # noqa: E402
from src.iber_evaluation import compute_detection_metrics  # noqa: E402
from src.rtdetr_complementarity_oracle import (  # noqa: E402
    candidate_iou_matrix,
    load_paired_cache,
    one_to_one_same_class_assignment,
    visdrone_size_bucket,
)
from scripts.run_rtdetr_complementarity_oracle import (  # noqa: E402
    FDR_CHECKPOINT_SHA256,
    FDR_INDEPENDENT_EVALUATOR_AUTHORITY,
    FREQUENCYCM_CHECKPOINT_SHA256,
    _batch_targets,
    _dataset_authority,
    _device,
    _execution_environment,
    _extract_decoder_batch,
    _load_detector,
    _model_state_sha256,
    _verify_checkpoint,
)


IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 8
CONFIDENCE = 0.001
MAX_DET = 300
NMS = False
NUM_CLASSES = 10
TRAIN_COUNT = 6471
VAL_COUNT = 548
DEV_MODULUS = 5
PROBE_EPOCHS = 20
PROBE_SEED = 0
RESCUE_BUDGETS = tuple(value for value in RESCUE_SLOT_GRID if value)
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
GRADIENT_NORM_CAP = 1.0
INTERNAL_MAP_BUFFER = 0.0020
NEAR_BEST_MAP = 0.0002

PFCR_FEATURE_NAMES = (
    "cm_class_logit", "cm_class_probability", "cm_query_max_probability",
    "cm_top_two_margin", "cm_normalized_entropy", "cm_flattened_rank",
    "cm_cx", "cm_cy", "cm_width", "cm_height", "cm_log_area",
    "cm_log_aspect", "fdr_class_logit", "fdr_class_probability",
    "fdr_query_max_probability", "fdr_top_two_margin", "fdr_normalized_entropy",
    "fdr_flattened_rank", "cross_box_iou", "cross_center_dx",
    "cross_center_dy", "cross_log_width_ratio", "cross_log_height_ratio",
    "cross_class_score_delta", "cross_query_max_score_delta",
    *(f"class_{index}" for index in range(NUM_CLASSES)),
)
PFCR_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    (json.dumps(PFCR_FEATURE_NAMES, separators=(",", ":")) + "\n").encode("utf-8")
).hexdigest().upper()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-checkpoint", type=Path, required=True)
    parser.add_argument("--frequencycm-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-cache-root", type=Path, required=True)
    parser.add_argument("--val-cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0", choices=("0",))
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True, text=True, check=True, timeout=30,
    )
    value = result.stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("source commit is invalid")
    return value


def _cache_authority(
    *,
    fdr_sha256: str,
    frequencycm_sha256: str,
    dataset_sha256: str,
    evaluator_sha256: str,
    source_commit: str,
) -> dict[str, str]:
    return {
        "fdr_sha256": fdr_sha256.upper(),
        "frequencycm_sha256": frequencycm_sha256.upper(),
        "dataset_sha256": dataset_sha256.upper(),
        "evaluator_sha256": evaluator_sha256.upper(),
        "feature_schema_sha256": PFCR_FEATURE_SCHEMA_SHA256,
        "source_commit": source_commit.lower(),
    }


def _freeze_randomness() -> None:
    random.seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PROBE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_probe_optimizer(gate: PFCRGate) -> torch.optim.AdamW:
    if not isinstance(gate, PFCRGate):
        raise TypeError("optimizer accepts only PFCRGate")
    return torch.optim.AdamW(
        tuple(gate.parameters()), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY
    )


def select_internal_checkpoint(history: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    dev = [row for row in history if row.get("split") == "dev"]
    if not dev:
        raise ValueError("internal history has no development rows")
    for row in dev:
        if row.get("slots") not in RESCUE_BUDGETS or type(row.get("epoch")) is not int:
            raise ValueError("internal history schema mismatch")
        for name in ("map", "ap75", "ap50"):
            if not isinstance(row.get(name), (int, float)):
                raise ValueError(f"internal history {name} is invalid")
    best_by_budget: dict[int, Mapping[str, Any]] = {}
    for row in dev:
        budget = int(row["slots"])
        current = best_by_budget.get(budget)
        key = (float(row["map"]), float(row["ap75"]), float(row["ap50"]), -int(row["epoch"]))
        if current is None or key > (
            float(current["map"]), float(current["ap75"]),
            float(current["ap50"]), -int(current["epoch"]),
        ):
            best_by_budget[budget] = row
    best_map = max(float(row["map"]) for row in best_by_budget.values())
    eligible = [
        row for row in best_by_budget.values()
        if best_map - float(row["map"]) <= NEAR_BEST_MAP + 1e-15
    ]
    selected = min(eligible, key=lambda row: int(row["slots"]))
    return {"epoch": int(selected["epoch"]), "slots": int(selected["slots"])}


def decide_internal(
    c0: Mapping[str, float],
    c1: Mapping[str, float],
    candidate: Mapping[str, float],
    tiny_small: Mapping[str, float],
) -> dict[str, Any]:
    for arm in (c0, c1, candidate):
        if not {"map", "ap75", "ap50"}.issubset(arm):
            raise ValueError("internal metric schema mismatch")
    if not {"c0", "candidate"}.issubset(tiny_small):
        raise ValueError("tiny/small metric schema mismatch")
    passed = (
        float(candidate["map"]) - max(float(c0["map"]), float(c1["map"]))
        >= INTERNAL_MAP_BUFFER
        and float(candidate["ap75"]) > max(float(c0["ap75"]), float(c1["ap75"]))
        and float(candidate["ap50"]) > float(c0["ap50"])
        and float(tiny_small["candidate"]) >= float(tiny_small["c0"])
    )
    return {
        "status": "passed" if passed else "scientific_failed",
        "observed": {"c0": dict(c0), "c1": dict(c1), "candidate": dict(candidate)},
        "tiny_small": dict(tiny_small),
        "thresholds": {
            "map_over_better_control": INTERNAL_MAP_BUFFER,
            "ap75_over_both": "strict",
            "ap50_over_c0": "strict",
            "tiny_small_recall50_vs_c0": "nonnegative",
        },
    }


def decide_official(fdr: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, Any]:
    if not {"map", "ap75"}.issubset(fdr) or not {"map", "ap75"}.issubset(candidate):
        raise ValueError("official metric schema mismatch")
    eligible = (
        float(candidate["map"]) > float(fdr["map"])
        and float(candidate["ap75"]) >= float(fdr["ap75"])
    )
    return {
        "eligible": eligible,
        "status": "eligible_for_network_integration" if eligible else "scientific_failed",
        "fdr": dict(fdr),
        "candidate": dict(candidate),
        "thresholds": {"map": "strictly_positive", "ap75": "nonnegative"},
    }


def load_official_val_cache(path: Path) -> tuple[dict[str, Any], ...]:
    root = Path(path)
    manifest_path = root / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    authority = manifest.get("authority") if isinstance(manifest, dict) else None
    if not isinstance(authority, dict):
        raise RuntimeError("official val cache authority is missing")
    if authority.get("fdr_sha256") != FDR_CHECKPOINT_SHA256:
        raise RuntimeError("official val cache FDR authority mismatch")
    if authority.get("frequencycm_sha256") != FREQUENCYCM_CHECKPOINT_SHA256:
        raise RuntimeError("official val cache FrequencyCM authority mismatch")
    records = load_paired_cache(root, authority)
    if len(records) != VAL_COUNT:
        raise RuntimeError("official val cache record count mismatch")
    return records


def advance_after_internal(
    decision: Mapping[str, Any], val_cache_root: Path
) -> tuple[dict[str, Any], ...] | None:
    if decision.get("status") != "passed":
        return None
    return load_official_val_cache(Path(val_cache_root))


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def publish_reports(report_root: Path, reports: Mapping[str, Any]) -> None:
    root = Path(report_root)
    if os.path.lexists(root):
        raise FileExistsError(f"report root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    published = False
    try:
        payloads: dict[str, bytes] = {}
        for name, value in sorted(reports.items()):
            if Path(name).name != name or not name.endswith((".json", ".csv", ".md")):
                raise ValueError(f"invalid report name: {name}")
            content = value if isinstance(value, bytes) else _canonical_json(value)
            payloads[name] = content
            with (staging / name).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        sums = "".join(
            f"{hashlib.sha256(content).hexdigest().upper()}  {name}\n"
            for name, content in sorted(payloads.items())
        ).encode("ascii")
        with (staging / "SHA256SUMS.txt").open("xb") as stream:
            stream.write(sums)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staging, root)
        published = True
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)


def extract_train_cache(
    fdr: torch.nn.Module,
    frequencycm: torch.nn.Module,
    loader: Any,
    validator: Any,
    writer: PFCRCacheWriter,
    *,
    expected_count: int,
) -> int:
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be positive")
    for label, detector in (("FDR", fdr), ("FrequencyCM", frequencycm)):
        if any(parameter.requires_grad for parameter in detector.parameters()):
            raise RuntimeError(f"{label} detector must be frozen")
        if any(parameter.grad is not None for parameter in detector.parameters()):
            raise RuntimeError(f"{label} detector contains gradients")
    before = (_model_state_sha256(fdr), _model_state_sha256(frequencycm))
    completed = set(writer.completed_image_ids)
    count = len(completed)
    for raw_batch in loader:
        batch = validator.preprocess(raw_batch)
        images = batch["img"]
        _, fdr_boxes, fdr_logits = _extract_decoder_batch(fdr, images)
        _, cm_boxes, cm_logits = _extract_decoder_batch(frequencycm, images)
        image_ids = batch.get("im_file")
        original_shapes = batch.get("ori_shape")
        resized_shapes = batch.get("resized_shape")
        for value, label in (
            (image_ids, "identifiers"), (original_shapes, "original shapes"),
            (resized_shapes, "resized shapes"),
        ):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != images.shape[0]:
                raise RuntimeError(f"validator image {label} are invalid")
        records = []
        for image_index, image_id in enumerate(image_ids):
            normalized_id = Path(str(image_id)).name
            if normalized_id in completed:
                continue
            original_shape = tuple(int(value) for value in original_shapes[image_index])
            resized_shape = tuple(int(value) for value in resized_shapes[image_index])
            if len(original_shape) != 2 or min(original_shape) <= 0:
                raise RuntimeError("validator original shape is invalid")
            if resized_shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError("validator resized shape is not frozen 640 square")
            target_boxes, target_classes = _batch_targets(batch, image_index)
            records.append({
                "image_id": normalized_id,
                "original_shape": original_shape,
                "resized_shape": resized_shape,
                "fdr_boxes": fdr_boxes[image_index].cpu().contiguous().clone(),
                "fdr_logits": fdr_logits[image_index].cpu().contiguous().clone(),
                "frequencycm_boxes": cm_boxes[image_index].cpu().contiguous().clone(),
                "frequencycm_logits": cm_logits[image_index].cpu().contiguous().clone(),
                "target_boxes": target_boxes.clone(),
                "target_classes": target_classes.clone(),
            })
            completed.add(normalized_id)
        writer.append_many(records)
        count += len(records)
    after = (_model_state_sha256(fdr), _model_state_sha256(frequencycm))
    if after != before:
        raise RuntimeError("detector state changed during paired extraction")
    if count != expected_count:
        raise RuntimeError(f"paired evidence count mismatch: expected={expected_count}, actual={count}")
    return count


def _build_train_loader(
    dataset_root: Path,
    checkpoint: Path,
    device: torch.device,
    *,
    save_dir: Path,
):
    from ultralytics.models.rtdetr.val import RTDETRValidator

    root = Path(dataset_root).resolve()
    from src.lpr_protocol import CATEGORY_NAMES

    data = {
        "path": str(root),
        "train": str((root / "images" / "train").resolve()),
        "val": str((root / "images" / "val").resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": NUM_CLASSES,
        "channels": 3,
    }
    validator = RTDETRValidator(
        save_dir=Path(save_dir),
        args={
            "model": str(Path(checkpoint).resolve()),
            "data": data,
            "task": "detect",
            "mode": "val",
            "split": "train",
            "imgsz": IMAGE_SIZE,
            "batch": BATCH_SIZE,
            "workers": WORKERS,
            "device": "0",
            "max_det": MAX_DET,
            "nms": NMS,
            "cache": False,
            "conf": CONFIDENCE,
            "half": False,
            "rect": False,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
        },
    )
    validator.data = data
    validator.device = device
    loader = validator.get_dataloader(data["train"], BATCH_SIZE)
    if len(loader.dataset) != TRAIN_COUNT:
        raise RuntimeError(
            f"training loader count mismatch: expected={TRAIN_COUNT}, actual={len(loader.dataset)}"
        )
    return loader, validator


def adjusted_frequencycm_logits(
    gate: PFCRGate, record: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Adjust only detached FrequencyCM logits while retaining gate gradients."""

    if set(("features", "frequencycm_logits")) - set(record):
        raise ValueError("prepared PFCR record is incomplete")
    parameter = next(gate.parameters())
    features = record["features"].detach().to(
        device=parameter.device, dtype=parameter.dtype
    )
    logits = record["frequencycm_logits"].detach().to(
        device=parameter.device, dtype=parameter.dtype
    )
    return logits + gate(features)


def prepare_records(
    records: Sequence[Mapping[str, Any]], *, device: torch.device, progress: bool = False
) -> tuple[dict[str, Any], ...]:
    """Materialize expensive detached features and one-to-one teachers once."""

    prepared: list[dict[str, Any]] = []
    # no_grad keeps prepared tensors usable as constant inputs to a trainable gate;
    # inference_mode tensors cannot be saved for the Linear weight backward pass.
    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            fdr_boxes = record["fdr_boxes"].detach().float().to(device)
            fdr_logits = record["fdr_logits"].detach().float().to(device)
            cm_boxes = record["frequencycm_boxes"].detach().float().to(device)
            cm_logits = record["frequencycm_logits"].detach().float().to(device)
            target_boxes = record["target_boxes"].detach().float().to(device)
            target_classes = record["target_classes"].detach().long().to(device)
            features = pfcr_features(fdr_boxes, fdr_logits, cm_boxes, cm_logits)
            teacher = one_to_one_union_teacher(
                fdr_boxes, fdr_logits, cm_boxes, cm_logits,
                target_boxes, target_classes,
            )
            prepared.append({
                "image_id": record.get("image_id", ""),
                "original_shape": tuple(record.get("original_shape", (IMAGE_SIZE, IMAGE_SIZE))),
                "resized_shape": tuple(record.get("resized_shape", (IMAGE_SIZE, IMAGE_SIZE))),
                "features": features.cpu().contiguous(),
                "fdr_boxes": fdr_boxes.cpu().contiguous(),
                "fdr_logits": fdr_logits.cpu().contiguous(),
                "frequencycm_boxes": cm_boxes.cpu().contiguous(),
                "frequencycm_logits": cm_logits.cpu().contiguous(),
                "target_boxes": target_boxes.cpu().contiguous(),
                "target_classes": target_classes.cpu().contiguous(),
                "fdr_teacher": teacher.fdr.cpu().contiguous(),
                "frequencycm_teacher": teacher.frequencycm.cpu().contiguous(),
            })
            if progress and (index % 64 == 0 or index == len(records)):
                print(
                    json.dumps(
                        {"stage": "prepare", "completed": index, "total": len(records)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return tuple(prepared)


def _batch_training_loss(
    gate: PFCRGate, records: Sequence[Mapping[str, torch.Tensor]]
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    parameter = next(gate.parameters())
    for record in records:
        adjusted = adjusted_frequencycm_logits(gate, record)
        fdr_logits = record["fdr_logits"].detach().to(
            device=parameter.device, dtype=parameter.dtype
        )
        fdr_teacher = record["fdr_teacher"].detach().to(
            device=parameter.device, dtype=parameter.dtype
        )
        cm_teacher = record["frequencycm_teacher"].detach().to(
            device=parameter.device, dtype=parameter.dtype
        )
        losses.extend(
            pfcr_boundary_loss(
                adjusted,
                cm_teacher,
                fdr_logits,
                fdr_teacher,
                rescue_slots=slots,
            )
            for slots in RESCUE_BUDGETS
        )
    if not losses:
        raise ValueError("PFCR training batch is empty")
    result = torch.stack(losses).mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("PFCR training loss is non-finite")
    return result


def evaluate_prepared(
    gate: PFCRGate,
    records: Sequence[Mapping[str, Any]],
    slots: int,
) -> dict[str, float]:
    """Evaluate one PFCR budget; implemented below the training primitive."""

    return _evaluate_records(gate, records, slots=slots)


def _prediction_record(postprocessed: torch.Tensor) -> dict[str, torch.Tensor]:
    selected = postprocessed.detach().float().cpu()
    selected = selected[selected[:, 4] > CONFIDENCE]
    return {
        "boxes": selected[:, :4].contiguous(),
        "scores": selected[:, 4].contiguous(),
        "classes": selected[:, 5].long().contiguous(),
    }


def _tiny_small_recall50(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    records: Sequence[Mapping[str, Any]],
) -> float:
    matched = 0
    total = 0
    for prediction, record in zip(predictions, records, strict=True):
        target_boxes = record["target_boxes"].detach().float().cpu()
        target_classes = record["target_classes"].detach().long().cpu()
        height, width = (int(value) for value in record["original_shape"])
        resized_height, resized_width = (int(value) for value in record["resized_shape"])
        if (resized_height, resized_width) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError("PFCR resized geometry differs from 640 square")
        gain = min(resized_height / height, resized_width / width)
        small_mask = torch.tensor(
            [
                visdrone_size_bucket(
                    float(box[2]) * resized_width / gain,
                    float(box[3]) * resized_height / gain,
                )
                in {"tiny", "small"}
                for box in target_boxes
            ],
            dtype=torch.bool,
        )
        total += int(small_mask.sum())
        if not target_boxes.numel() or not prediction["boxes"].numel():
            continue
        assignment = one_to_one_same_class_assignment(
            candidate_iou_matrix(prediction["boxes"], target_boxes),
            prediction["classes"],
            target_classes,
        )
        if assignment.target_indices.numel():
            accepted = small_mask[assignment.target_indices] & (assignment.ious >= 0.5)
            matched += int(accepted.sum())
    return float(matched / total) if total else 0.0


def _evaluate_records(
    gate: PFCRGate | None,
    records: Sequence[Mapping[str, Any]],
    *,
    slots: int,
) -> dict[str, float]:
    if not records:
        raise ValueError("PFCR evaluation records are empty")
    if slots not in RESCUE_SLOT_GRID:
        raise ValueError(f"slots must be one of {RESCUE_SLOT_GRID}")
    if gate is None and slots == 0:
        parameter = None
    elif gate is None:
        parameter = None
    else:
        gate.eval()
        parameter = next(gate.parameters())

    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            adjusted_batch: torch.Tensor | None = None
            if gate is not None:
                features = torch.stack([record["features"] for record in batch]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                cm_logits = torch.stack(
                    [record["frequencycm_logits"] for record in batch]
                ).to(device=parameter.device, dtype=parameter.dtype)
                adjusted_batch = (cm_logits.detach() + gate(features)).cpu()
            for offset, record in enumerate(batch):
                fdr_boxes = record["fdr_boxes"].detach().float().cpu()
                fdr_logits = record["fdr_logits"].detach().float().cpu()
                if slots == 0:
                    postprocessed = stock_predictions(fdr_boxes, fdr_logits)
                else:
                    cm_boxes = record["frequencycm_boxes"].detach().float().cpu()
                    cm_logits = (
                        adjusted_batch[offset]
                        if adjusted_batch is not None
                        else record["frequencycm_logits"].detach().float().cpu()
                    )
                    postprocessed = protected_merge(
                        fdr_boxes,
                        fdr_logits,
                        cm_boxes,
                        cm_logits,
                        rescue_slots=slots,
                    )
                predictions.append(_prediction_record(postprocessed))
                targets.append({
                    "boxes": record["target_boxes"].detach().float().cpu(),
                    "classes": record["target_classes"].detach().long().cpu(),
                })
    metrics = compute_detection_metrics(predictions, targets, image_size=IMAGE_SIZE)
    metrics["tiny_small_recall50"] = _tiny_small_recall50(predictions, records)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("PFCR evaluation contains non-finite metrics")
    return {name: float(value) for name, value in metrics.items()}


def evaluate_control(
    records: Sequence[Mapping[str, Any]], *, slots: int
) -> dict[str, float]:
    """Evaluate exact FDR stock at zero slots or raw-score protected union otherwise."""

    return _evaluate_records(None, records, slots=slots)


def _write_json_create_only(path: Path, payload: Any) -> None:
    content = _canonical_json(payload)
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _save_checkpoint_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(f"checkpoint already exists: {target}")
    temporary = target.parent / f".{target.name}.staging-{os.getpid()}"
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, target)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _completed_training_history(run_root: Path) -> tuple[list[dict[str, Any]], int]:
    root = Path(run_root)
    checkpoints = sorted((root / "checkpoints").glob("epoch-*.pt"))
    metrics = sorted((root / "metrics").glob("epoch-*.json"))
    if len(checkpoints) != len(metrics):
        raise RuntimeError("PFCR checkpoint/metric journal mismatch")
    history: list[dict[str, Any]] = []
    for expected_epoch, (checkpoint, metric_path) in enumerate(
        zip(checkpoints, metrics, strict=True), start=1
    ):
        expected_name = f"epoch-{expected_epoch:02d}"
        if checkpoint.stem != expected_name or metric_path.stem != expected_name:
            raise RuntimeError("PFCR epoch journal is not contiguous")
        payload = json.loads(metric_path.read_text("utf-8"))
        if not isinstance(payload, dict) or payload.get("epoch") != expected_epoch:
            raise RuntimeError("PFCR epoch metric schema mismatch")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(RESCUE_BUDGETS):
            raise RuntimeError("PFCR epoch metric rows are invalid")
        history.extend(rows)
    return history, len(checkpoints)


def train_gate(
    records: Mapping[str, Sequence[Mapping[str, torch.Tensor]]],
    run_root: Path,
    *,
    epochs: int = PROBE_EPOCHS,
    resume: bool = False,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Train only PFCRGate and maintain a create-only resumable epoch journal."""

    if type(epochs) is not int or epochs <= 0 or epochs > PROBE_EPOCHS:
        raise ValueError(f"epochs must be in [1, {PROBE_EPOCHS}]")
    if set(records) != {"train", "dev"} or not records["train"] or not records["dev"]:
        raise ValueError("PFCR train/dev records must both be nonempty")
    root = Path(run_root)
    if os.path.lexists(root):
        if not resume:
            raise FileExistsError(f"PFCR run already exists: {root}")
        if {path.name for path in root.iterdir()} != {"checkpoints", "metrics"}:
            raise RuntimeError("PFCR run contents mismatch")
    else:
        root.mkdir(parents=True)
        (root / "checkpoints").mkdir()
        (root / "metrics").mkdir()

    _freeze_randomness()
    gate = PFCRGate().to(device)
    optimizer = build_probe_optimizer(gate)
    history, completed = _completed_training_history(root)
    if completed:
        checkpoint_path = root / "checkpoints" / f"epoch-{completed:02d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("epoch") != completed:
            raise RuntimeError("PFCR checkpoint epoch mismatch")
        gate.load_state_dict(checkpoint["gate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    if completed >= epochs:
        return history

    train_records = records["train"]
    for epoch in range(completed + 1, epochs + 1):
        gate.train()
        generator = torch.Generator().manual_seed(PROBE_SEED + epoch)
        order = torch.randperm(len(train_records), generator=generator).tolist()
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(order), BATCH_SIZE):
            selected = [train_records[index] for index in order[start : start + BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_training_loss(gate, selected)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), GRADIENT_NORM_CAP)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("PFCR gradient norm is non-finite")
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        gate.eval()
        rows: list[dict[str, Any]] = []
        for slots in RESCUE_BUDGETS:
            metrics = evaluate_prepared(gate, records["dev"], slots)
            rows.append({
                "epoch": epoch,
                "split": "dev",
                "slots": slots,
                **{name: float(value) for name, value in metrics.items()},
                "train_loss": epoch_loss / max(steps, 1),
            })
        checkpoint_payload = {
            "format_version": 1,
            "epoch": epoch,
            "gate": gate.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        _save_checkpoint_create_only(
            root / "checkpoints" / f"epoch-{epoch:02d}.pt", checkpoint_payload
        )
        _write_json_create_only(
            root / "metrics" / f"epoch-{epoch:02d}.json",
            {"epoch": epoch, "rows": rows},
        )
        history.extend(rows)
    return history


def _history_csv(history: Sequence[Mapping[str, Any]]) -> bytes:
    if not history:
        raise ValueError("PFCR history is empty")
    fieldnames = tuple(history[0])
    if any(tuple(row) != fieldnames for row in history):
        raise ValueError("PFCR history row schema drift")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    return stream.getvalue().encode("utf-8")


def _verify_official_stock(metrics: Mapping[str, float]) -> None:
    for name, expected in FDR_INDEPENDENT_EVALUATOR_AUTHORITY.items():
        if name not in metrics or abs(float(metrics[name]) - float(expected)) > 1e-12:
            raise RuntimeError(
                f"official FDR stock reproduction mismatch for {name}: "
                f"expected={expected}, actual={metrics.get(name)}"
            )


def _report_markdown(
    internal: Mapping[str, Any], official: Mapping[str, Any] | None
) -> bytes:
    selected = internal["selection"]
    candidate = internal["decision"]["observed"]["candidate"]
    lines = [
        "# PFCR v1 Learnability Probe",
        "",
        "> This is detached two-checkpoint design-selection evidence, not the final YAML detector.",
        "",
        f"- Internal status: `{internal['decision']['status']}`",
        f"- Selected epoch / rescue slots: `{selected['epoch']}` / `{selected['slots']}`",
        f"- Internal mAP: `{candidate['map']}`",
        f"- Internal AP75: `{candidate['ap75']}`",
    ]
    if official is None:
        lines.append("- Official val: `not opened because the internal Gate failed`")
    else:
        lines.extend(
            (
                f"- Official eligibility: `{official['decision']['eligible']}`",
                f"- Official FDR mAP: `{official['fdr']['map']}`",
                f"- Official PFCR mAP: `{official['candidate']['map']}`",
                f"- Official FDR AP75: `{official['fdr']['ap75']}`",
                f"- Official PFCR AP75: `{official['candidate']['ap75']}`",
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _run(args: argparse.Namespace) -> int:
    fdr_checkpoint = Path(args.fdr_checkpoint).resolve()
    cm_checkpoint = Path(args.frequencycm_checkpoint).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    train_cache_root = Path(os.path.abspath(args.train_cache_root))
    val_cache_root = Path(os.path.abspath(args.val_cache_root))
    run_root = Path(os.path.abspath(args.run_root))
    report_root = Path(os.path.abspath(args.report_root))

    fdr_sha256 = _verify_checkpoint(fdr_checkpoint, FDR_CHECKPOINT_SHA256)
    cm_sha256 = _verify_checkpoint(cm_checkpoint, FREQUENCYCM_CHECKPOINT_SHA256)
    source_commit = _source_commit()
    dataset = _dataset_authority(dataset_root)
    environment = _execution_environment()
    authority = _cache_authority(
        fdr_sha256=fdr_sha256,
        frequencycm_sha256=cm_sha256,
        dataset_sha256=str(dataset["sha256"]),
        evaluator_sha256=_file_sha256(REPOSITORY_ROOT / "src" / "iber_evaluation.py"),
        source_commit=source_commit,
    )
    device = _device(args.device)
    _freeze_randomness()
    print(json.dumps({"stage": "authority", "source_commit": source_commit}), flush=True)

    if (train_cache_root / "manifest.json").is_file():
        raw = load_pfcr_cache(train_cache_root, authority)
    else:
        writer = PFCRCacheWriter(train_cache_root, authority, shard_size=64)
        loader, validator = _build_train_loader(
            dataset_root,
            fdr_checkpoint,
            device,
            save_dir=report_root.parent / f".{report_root.name}-train-validator",
        )
        fdr = _load_detector(fdr_checkpoint, device)
        frequencycm = _load_detector(cm_checkpoint, device)
        extract_train_cache(
            fdr,
            frequencycm,
            loader,
            validator,
            writer,
            expected_count=TRAIN_COUNT,
        )
        writer.finalize()
        del fdr, frequencycm, loader, validator
        torch.cuda.empty_cache()
        raw = load_pfcr_cache(train_cache_root, authority)
    if sum(len(values) for values in raw.values()) != TRAIN_COUNT:
        raise RuntimeError("PFCR verified train cache count mismatch")
    print(
        json.dumps(
            {"stage": "train_cache", "train": len(raw["train"]), "dev": len(raw["dev"])},
            sort_keys=True,
        ),
        flush=True,
    )

    prepared = {
        split: prepare_records(values, device=device, progress=True)
        for split, values in raw.items()
    }
    del raw
    torch.cuda.empty_cache()
    controls = {
        "c0": evaluate_control(prepared["dev"], slots=0),
        "c1": {
            str(slots): evaluate_control(prepared["dev"], slots=slots)
            for slots in RESCUE_BUDGETS
        },
    }
    resume = run_root.exists()
    history = train_gate(
        prepared,
        run_root,
        resume=resume,
        device=device,
    )
    selection = select_internal_checkpoint(history)
    selected_checkpoint = (
        run_root / "checkpoints" / f"epoch-{selection['epoch']:02d}.pt"
    )
    checkpoint = torch.load(selected_checkpoint, map_location=device, weights_only=True)
    gate = PFCRGate().to(device).eval()
    gate.load_state_dict(checkpoint["gate"])
    candidate = evaluate_prepared(gate, prepared["dev"], selection["slots"])
    c1 = controls["c1"][str(selection["slots"])]
    internal_decision = decide_internal(
        controls["c0"],
        c1,
        candidate,
        {
            "c0": controls["c0"]["tiny_small_recall50"],
            "candidate": candidate["tiny_small_recall50"],
        },
    )
    internal = {
        "selection": selection,
        "selected_checkpoint": {
            "path": str(selected_checkpoint),
            "sha256": _file_sha256(selected_checkpoint),
        },
        "controls": controls,
        "decision": internal_decision,
    }
    print(json.dumps({"stage": "internal", "decision": internal_decision}), flush=True)

    official: dict[str, Any] | None = None
    val_records = advance_after_internal(internal_decision, val_cache_root)
    if val_records is not None:
        prepared_val = prepare_records(val_records, device=device, progress=True)
        fdr_metrics = evaluate_control(prepared_val, slots=0)
        _verify_official_stock(fdr_metrics)
        official_candidate = evaluate_prepared(gate, prepared_val, selection["slots"])
        official = {
            "fdr": fdr_metrics,
            "candidate": official_candidate,
            "decision": decide_official(fdr_metrics, official_candidate),
        }
        print(json.dumps({"stage": "official", **official}), flush=True)

    reports = {
        "authority.json": {
            "source_commit": source_commit,
            "cache_authority": authority,
            "train_cache_manifest_sha256": _file_sha256(train_cache_root / "manifest.json"),
            "fdr_checkpoint": {"path": str(fdr_checkpoint), "sha256": fdr_sha256},
            "frequencycm_checkpoint": {"path": str(cm_checkpoint), "sha256": cm_sha256},
            "dataset": dataset,
            "environment": environment,
            "protocol": {
                "epochs": PROBE_EPOCHS,
                "seed": PROBE_SEED,
                "batch": BATCH_SIZE,
                "optimizer": "AdamW",
                "lr": PROBE_LR,
                "weight_decay": PROBE_WEIGHT_DECAY,
                "gradient_norm_cap": GRADIENT_NORM_CAP,
                "rescue_budgets": list(RESCUE_BUDGETS),
                "official_val_opened": official is not None,
            },
        },
        "internal-history.csv": _history_csv(history),
        "internal-selection.json": internal,
        "internal-decision.json": internal_decision,
        "official-metrics.json": official or {"status": "not_opened"},
        "pfcr-decision.json": (
            official["decision"] if official is not None else internal_decision
        ),
        "PFCR_REPORT.md": _report_markdown(internal, official),
    }
    publish_reports(report_root, reports)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

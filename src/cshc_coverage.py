"""Extraction of frozen tiny misses and pre-Top-300 CSHC candidate records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from src.cshc import CSHCRTDDETRDecoder, SparseCandidates
from src.rtdetr_cshc import CSHCDetectionModel


def resolve_cshc_decoder(model: Any, decoder_type: type = CSHCRTDDETRDecoder):
    """Find the final CSHC decoder through the optional framework wrapper used by Validator."""
    current = model
    for _ in range(4):
        layers = getattr(current, "model", None)
        if layers is None or layers is current:
            break
        try:
            final_layer = layers[-1]
        except (TypeError, KeyError, IndexError):
            current = layers
            continue
        if isinstance(final_layer, decoder_type):
            return final_layer
        current = layers
    raise TypeError("coverage export requires a final CSHCRTDDETRDecoder")


def load_frozen_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Load the immutable BQP image ledger without treating its old 7.43% gate as a CSHC metric."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "image_id" not in row or "tiny_targets" not in row:
                raise ValueError(f"frozen ledger {path}:{line_number} lacks image_id or tiny_targets")
            rows.append(row)
    names = [str(row["image_id"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("frozen ledger contains duplicate image identifiers")
    return rows


def frozen_image_names(ledger: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["image_id"]) for row in ledger}


def _misses_by_image(ledger: Iterable[dict[str, Any]]) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for row in ledger:
        image_id = str(row["image_id"])
        targets = row.get("tiny_targets")
        if not isinstance(targets, list):
            raise ValueError(f"ledger entry {image_id} has invalid tiny_targets")
        image_misses: dict[int, int] = {}
        for target in targets:
            if not isinstance(target, dict):
                raise ValueError(f"ledger entry {image_id} has invalid tiny target")
            if str(target.get("status")) == "covered":
                continue
            index = int(target["gt_index"])
            class_id = int(target["gt_class"])
            if index in image_misses:
                raise ValueError(f"ledger entry {image_id} duplicates GT index {index}")
            image_misses[index] = class_id
        result[image_id] = image_misses
    return result


def _xywh_to_records_box(boxes_xywh: Tensor) -> list[list[float]]:
    xyxy = torch.cat((boxes_xywh[:, :2] - boxes_xywh[:, 2:] / 2, boxes_xywh[:, :2] + boxes_xywh[:, 2:] / 2), dim=-1)
    return [[round(float(value), 6) for value in row] for row in xyxy.detach().float().cpu().tolist()]


def frozen_miss_records(
    *,
    image_files: list[str],
    batch_idx: Tensor,
    classes: Tensor,
    boxes_xywh: Tensor,
    ledger: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert only the frozen non-covered GT indices from a validator batch into xyxy records."""
    if boxes_xywh.ndim != 2 or boxes_xywh.shape[-1] != 4:
        raise ValueError("boxes_xywh must have shape (N, 4)")
    if batch_idx.numel() != boxes_xywh.shape[0] or classes.numel() != boxes_xywh.shape[0]:
        raise ValueError("batch_idx, classes and boxes_xywh must have matching counts")
    misses_by_image = _misses_by_image(ledger)
    records: list[dict[str, Any]] = []
    flat_classes = classes.view(-1).to(dtype=torch.long)
    flat_batch_idx = batch_idx.view(-1).to(dtype=torch.long)
    for image_index, image_file in enumerate(image_files):
        image_id = Path(image_file).name
        if image_id not in misses_by_image:
            raise ValueError(f"validator produced image not present in frozen ledger: {image_id}")
        mask = (flat_batch_idx == image_index) & (flat_classes >= 0)
        image_classes = flat_classes[mask]
        image_boxes = boxes_xywh[mask]
        expected = misses_by_image[image_id]
        if expected and max(expected) >= len(image_boxes):
            raise ValueError(f"ledger GT index is out of range for {image_id}")
        for gt_index, expected_class in sorted(expected.items()):
            actual_class = int(image_classes[gt_index])
            if actual_class != expected_class:
                raise ValueError(
                    f"frozen ledger class mismatch for {image_id} GT {gt_index}: "
                    f"expected {expected_class}, received {actual_class}"
                )
            records.append(
                {
                    "image_id": image_id,
                    "class_id": actual_class,
                    "box": _xywh_to_records_box(image_boxes[gt_index : gt_index + 1])[0],
                }
            )
    return records


def c2_candidate_records(image_files: list[str], candidates: SparseCandidates) -> list[dict[str, Any]]:
    """Convert every C2 candidate before combined Top-300 selection into a class-aware xyxy record."""
    batch, count, _ = candidates.anchor_logits.shape
    if len(image_files) != batch or candidates.class_logits.shape[:2] != (batch, count):
        raise ValueError("image files and C2 candidate tensors have incompatible batch dimensions")
    if candidates.indices.shape != (batch, count):
        raise ValueError("candidate indices must align with anchor logits")
    boxes = candidates.anchor_logits.sigmoid().reshape(batch * count, 4)
    classes = candidates.class_logits.argmax(dim=-1).reshape(-1).detach().cpu().tolist()
    records: list[dict[str, Any]] = []
    for index, box in enumerate(_xywh_to_records_box(boxes)):
        image_id = Path(image_files[index // count]).name
        records.append({"image_id": image_id, "class_id": int(classes[index]), "box": box})
    return records


def build_coverage_validator(ledger: list[dict[str, Any]]):
    """Build a validator that uses exact RT-DETR val transforms and exports no final detections."""
    from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
    from ultralytics.utils import colorstr

    frozen_names = frozen_image_names(ledger)

    class CSHCCoverageValidator(RTDETRValidator):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.frozen_misses: list[dict[str, Any]] = []
            self.c2_candidates: list[dict[str, Any]] = []
            self.audited_names: list[str] = []
            self.decoder: CSHCRTDDETRDecoder | None = None

        def build_dataset(self, img_path, mode="val", batch=None):
            dataset = RTDETRDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=False,
                hyp=self.args,
                rect=False,
                cache=self.args.cache or None,
                prefix=colorstr(f"{mode}: "),
                data=self.data,
            )
            selected = [index for index, file in enumerate(dataset.im_files) if Path(file).name in frozen_names]
            selected_names = {Path(dataset.im_files[index]).name for index in selected}
            if selected_names != frozen_names:
                missing = sorted(frozen_names - selected_names)
                raise RuntimeError(f"dataset does not contain every frozen ledger image: {missing[:3]}")
            dataset.im_files = [dataset.im_files[index] for index in selected]
            dataset.labels = [dataset.labels[index] for index in selected]
            return dataset

        def init_metrics(self, model) -> None:
            self.decoder = resolve_cshc_decoder(model)

        def update_metrics(self, preds, batch) -> None:
            del preds
            if self.decoder is None or self.decoder.last_candidates is None:
                raise RuntimeError("C2 candidates were not populated during validation forward pass")
            image_files = list(batch["im_file"])
            self.frozen_misses.extend(
                frozen_miss_records(
                    image_files=image_files,
                    batch_idx=batch["batch_idx"],
                    classes=batch["cls"],
                    boxes_xywh=batch["bboxes"],
                    ledger=ledger,
                )
            )
            self.c2_candidates.extend(c2_candidate_records(image_files, self.decoder.last_candidates))
            self.audited_names.extend(Path(image_file).name for image_file in image_files)

        def gather_stats(self) -> None:
            return None

        def get_stats(self) -> dict[str, float]:
            return {}

        def finalize_metrics(self) -> None:
            return None

        def print_results(self) -> None:
            return None

    return CSHCCoverageValidator


def export_from_checkpoint(
    *,
    checkpoint: str | Path,
    config: str | Path,
    data: str | Path,
    ledger: list[dict[str, Any]],
    split: str = "train",
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 8,
    device: str = "0",
):
    """Run CSHC on exactly the frozen ledger image set and return only raw C2 records plus frozen misses."""
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = CSHCDetectionModel(config, nc=10, verbose=False)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load(loaded, verbose=False)
    model.eval()
    validator_type = build_coverage_validator(ledger)
    validator = validator_type(
        args={
            "model": str(checkpoint),
            "data": str(Path(data).resolve()),
            "split": split,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "device": device,
            "plots": False,
            "save_json": False,
            "verbose": False,
            "project": str(checkpoint.parent / "_cshc_coverage_validator"),
            "name": "frozen-ledger",
            "exist_ok": True,
        }
    )
    validator(model=model)
    expected_names = frozen_image_names(ledger)
    if set(validator.audited_names) != expected_names or len(validator.audited_names) != len(expected_names):
        raise RuntimeError("coverage export did not process the frozen ledger image set exactly once")
    expected_misses = sum(len(items) for items in _misses_by_image(ledger).values())
    if len(validator.frozen_misses) != expected_misses:
        raise RuntimeError(
            f"coverage export expected {expected_misses} frozen misses, got {len(validator.frozen_misses)}"
        )
    return validator.frozen_misses, validator.c2_candidates

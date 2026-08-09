"""Run the frozen FDR/FrequencyCM candidate-complementarity upper-bound oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.iber_evaluation import compute_detection_metrics
from src.lpr_protocol import CATEGORY_NAMES
from src.rtdetr_complementarity_oracle import (
    build_matched_quality_arm,
    candidate_iou_matrix,
    coverage_summary,
    decide_complementarity,
    one_to_one_same_class_assignment,
    visdrone_size_bucket,
)
from src.rtdetr_quality_oracle import flattened_topk


IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 8
CONFIDENCE = 0.001
MAX_DET = 300
NMS = False
VAL_COUNT = 548
NUM_CLASSES = 10
FDR_CHECKPOINT_SHA256 = (
    "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2"
)
FREQUENCYCM_CHECKPOINT_SHA256 = (
    "2BBCD6057FEFED5792F786A18E603F8FECA3EC426A6F68938F5F8ADA1603A141"
)
FREQUENCYCM_SOURCE_COMMIT = "d3655b14c17a3c8ca14e1888517b6fde4e059766"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-checkpoint", type=Path, required=True)
    parser.add_argument("--frequencycm-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _verify_checkpoint(path: Path, expected_sha256: str) -> str:
    checkpoint = Path(path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RuntimeError(f"checkpoint is not a regular file: {checkpoint}")
    expected = str(expected_sha256).upper()
    if len(expected) != 64 or any(character not in "0123456789ABCDEF" for character in expected):
        raise ValueError("expected checkpoint SHA-256 must be 64 hexadecimal characters")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest().upper()
    if actual != expected:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def _extract_decoder_batch(
    detector: Any, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must have shape [B,C,H,W]")
    head = detector.model[-1]
    original_export = head.export
    try:
        head.export = False
        with torch.inference_mode():
            result = detector.predict(images)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("RT-DETR prediction must contain stock and auxiliary outputs")
            stock_output, auxiliary = result
            if not isinstance(auxiliary, tuple) or len(auxiliary) != 5:
                raise RuntimeError("RT-DETR auxiliary decoder tuple is invalid")
            decoder_boxes, decoder_logits, _, _, _ = auxiliary
            boxes = decoder_boxes[-1].detach().float()
            logits = decoder_logits[-1].detach().float()
            reconstructed = head.postprocess(boxes, logits.sigmoid())
    finally:
        head.export = original_export
    if not torch.equal(reconstructed, stock_output):
        raise RuntimeError("decoder reconstruction differs from stock RT-DETR output")
    expected_batch = images.shape[0]
    if boxes.shape != (expected_batch, MAX_DET, 4):
        raise RuntimeError(f"decoder box shape mismatch: {tuple(boxes.shape)}")
    if logits.shape != (expected_batch, MAX_DET, NUM_CLASSES):
        raise RuntimeError(f"decoder logit shape mismatch: {tuple(logits.shape)}")
    if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():
        raise RuntimeError("decoder evidence contains non-finite values")
    if boxes.requires_grad or logits.requires_grad or stock_output.requires_grad:
        raise RuntimeError("decoder evidence is attached to gradients")
    if any(parameter.grad is not None for parameter in detector.parameters()):
        raise RuntimeError("detector parameters contain gradients after inference")
    return stock_output, boxes, logits


def _model_state_sha256(detector: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(detector.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def _batch_targets(
    batch: Mapping[str, Any], image_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["batch_idx"].view(-1).long() == image_index
    boxes = batch["bboxes"][mask].detach().float().cpu().contiguous()
    classes = batch["cls"][mask].view(-1).detach().long().cpu().contiguous()
    return boxes, classes


def _extract_paired_records(
    fdr: torch.nn.Module,
    frequencycm: torch.nn.Module,
    loader: Any,
    validator: Any,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    for label, detector in (("FDR", fdr), ("FrequencyCM", frequencycm)):
        if any(parameter.requires_grad for parameter in detector.parameters()):
            raise RuntimeError(f"{label} detector must be frozen")
        if any(parameter.grad is not None for parameter in detector.parameters()):
            raise RuntimeError(f"{label} detector contains gradients")
    states_before = (_model_state_sha256(fdr), _model_state_sha256(frequencycm))
    records: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = validator.preprocess(raw_batch)
        images = batch["img"]
        _, fdr_boxes, fdr_logits = _extract_decoder_batch(fdr, images)
        _, frequencycm_boxes, frequencycm_logits = _extract_decoder_batch(
            frequencycm, images
        )
        image_ids = batch.get("im_file")
        original_shapes = batch.get("ori_shape")
        if (
            not isinstance(image_ids, Sequence)
            or isinstance(image_ids, (str, bytes))
            or len(image_ids) != images.shape[0]
        ):
            raise RuntimeError("validator image identifiers are invalid")
        if (
            not isinstance(original_shapes, Sequence)
            or isinstance(original_shapes, (str, bytes))
            or len(original_shapes) != images.shape[0]
        ):
            raise RuntimeError("validator original shapes are invalid")
        for image_index, image_id in enumerate(image_ids):
            shape = tuple(int(value) for value in original_shapes[image_index])
            if len(shape) != 2 or any(value <= 0 for value in shape):
                raise RuntimeError("validator original shape must be positive height-width")
            target_boxes, target_classes = _batch_targets(batch, image_index)
            records.append(
                {
                    "image_id": Path(str(image_id)).name,
                    "original_shape": shape,
                    "fdr_boxes": fdr_boxes[image_index].cpu().contiguous().clone(),
                    "fdr_logits": fdr_logits[image_index].cpu().contiguous().clone(),
                    "frequencycm_boxes": frequencycm_boxes[image_index]
                    .cpu()
                    .contiguous()
                    .clone(),
                    "frequencycm_logits": frequencycm_logits[image_index]
                    .cpu()
                    .contiguous()
                    .clone(),
                    "target_boxes": target_boxes.clone(),
                    "target_classes": target_classes.clone(),
                }
            )
    if len(records) != expected_count:
        raise RuntimeError(
            f"paired evidence count mismatch: expected={expected_count}, actual={len(records)}"
        )
    states_after = (_model_state_sha256(fdr), _model_state_sha256(frequencycm))
    if states_after != states_before:
        raise RuntimeError("detector state changed during paired evidence extraction")
    return records


def _prediction_record(postprocessed: torch.Tensor) -> dict[str, torch.Tensor]:
    selected = postprocessed.detach().float().cpu()
    selected = selected[selected[:, 4] > CONFIDENCE]
    return {
        "boxes": selected[:, :4].contiguous(),
        "scores": selected[:, 4].contiguous(),
        "classes": selected[:, 5].long().contiguous(),
    }


def _matched_target_iou(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    boxes = boxes.detach().float().cpu()
    target_boxes = target_boxes.detach().float().cpu()
    target_classes = target_classes.detach().long().cpu()
    candidate_boxes = boxes.repeat_interleave(num_classes, dim=0)
    candidate_classes = torch.arange(num_classes).repeat(boxes.shape[0])
    assignment = one_to_one_same_class_assignment(
        candidate_iou_matrix(candidate_boxes, target_boxes),
        candidate_classes,
        target_classes,
    )
    result = torch.zeros(target_boxes.shape[0], dtype=torch.float32)
    result[assignment.target_indices] = assignment.ious.float()
    return result


def _stock_utility(
    prediction: Mapping[str, torch.Tensor],
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> float:
    boxes = prediction["boxes"].detach().float().cpu()
    classes = prediction["classes"].detach().long().cpu()
    assignment = one_to_one_same_class_assignment(
        candidate_iou_matrix(boxes, target_boxes.detach().float().cpu()),
        classes,
        target_classes.detach().long().cpu(),
    )
    return float(assignment.ious.sum())


def _target_scales(record: Mapping[str, Any]) -> tuple[str, ...]:
    height, width = record["original_shape"]
    return tuple(
        visdrone_size_bucket(float(box[2]) * width, float(box[3]) * height)
        for box in record["target_boxes"]
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=tuple(fieldnames), extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _coverage_rows(
    coverage: Mapping[str, Any], *, group_key: str, label_key: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("raw", "one_to_one"):
        groups = coverage[mode][group_key]
        for label, thresholds in groups.items():
            for threshold_key in ("iou50", "iou75"):
                values = thresholds[threshold_key]
                rows.append(
                    {
                        "mode": mode,
                        label_key: label,
                        "threshold": values["threshold"],
                        "total": values["total"],
                        "fdr": values["fdr"],
                        "frequencycm": values["frequencycm"],
                        "union": values["union"],
                        "union_gain": values["union_gain"],
                        "fdr_rate": values["fdr_rate"],
                        "frequencycm_rate": values["frequencycm_rate"],
                        "union_rate": values["union_rate"],
                        "union_gain_rate": values["union_gain_rate"],
                    }
                )
    return rows


def _missed_rows(coverage: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for grouping, groups in (
        ("overall", {"overall": coverage["one_to_one"]["overall"]}),
        ("scale", coverage["one_to_one"]["by_scale"]),
        ("class", coverage["one_to_one"]["by_class"]),
    ):
        for group, thresholds in groups.items():
            for threshold_key in ("iou50", "iou75"):
                values = thresholds[threshold_key]
                rows.append(
                    {
                        "grouping": grouping,
                        "group": group,
                        "threshold": values["threshold"],
                        "total": values["total"],
                        "both": values["both"],
                        "fdr_only": values["fdr_only"],
                        "frequencycm_only": values["frequencycm_only"],
                        "neither": values["neither"],
                    }
                )
    return rows


def _write_report_bundle(
    report_root: Path,
    payload: Mapping[str, Any],
    *,
    scale_rows: Sequence[Mapping[str, Any]],
    class_rows: Sequence[Mapping[str, Any]],
    missed_rows: Sequence[Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
) -> None:
    root = Path(report_root)
    artifact_payloads = {
        "oracle-summary.json": _canonical_json(payload),
        "coverage-by-scale.csv": _csv_bytes(
            scale_rows,
            (
                "mode", "scale", "threshold", "total", "fdr", "frequencycm",
                "union", "union_gain", "fdr_rate", "frequencycm_rate",
                "union_rate", "union_gain_rate",
            ),
        ),
        "coverage-by-class.csv": _csv_bytes(
            class_rows,
            (
                "mode", "class_id", "threshold", "total", "fdr", "frequencycm",
                "union", "union_gain", "fdr_rate", "frequencycm_rate",
                "union_rate", "union_gain_rate",
            ),
        ),
        "missed-target-categories.csv": _csv_bytes(
            missed_rows,
            (
                "grouping", "group", "threshold", "total", "both", "fdr_only",
                "frequencycm_only", "neither",
            ),
        ),
        "oracle-arms.csv": _csv_bytes(
            arm_rows,
            ("arm", "map", "ap50", "ap75", "ap_tiny", "ap_small", "precision", "recall"),
        ),
    }
    decision = payload["decision"]["decision"]
    markdown = (
        "# FrequencyCM Complementarity Upper-Bound Oracle\n\n"
        "> **Non-deployable design-selection evidence.** This diagnostic uses "
        "ground truth on the official validation set and is not a detector gain.\n\n"
        f"- Decision: `{decision}`\n"
        f"- Candidate-oracle mAP delta: `{payload['oracle']['candidate_map_delta']}`\n"
        f"- Tiny/small one-to-one recall@0.50 delta: "
        f"`{payload['coverage']['tiny_small_recall50_delta']}`\n"
        f"- Duplicate-FDR neutral control: "
        f"`{payload['reproduction']['duplicate_fdr_neutral']}`\n"
    ).encode("utf-8")
    artifact_payloads["frequencycm-complementarity-report.md"] = markdown
    for name in artifact_payloads:
        path = root / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable report output already exists: {path}")
    sums_path = root / "SHA256SUMS.txt"
    if sums_path.exists() or sums_path.is_symlink():
        raise FileExistsError(f"immutable report output already exists: {sums_path}")
    for name, content in artifact_payloads.items():
        _write_create_only(root / name, content)
    sums = "".join(
        f"{hashlib.sha256(content).hexdigest().upper()}  {name}\n"
        for name, content in sorted(artifact_payloads.items())
    ).encode("ascii")
    _write_create_only(sums_path, sums)


def run_from_records(
    records: Sequence[Mapping[str, Any]], report_root: Path
) -> dict[str, Any]:
    """Evaluate a frozen paired cache and write all non-deployable oracle evidence."""

    if not records:
        raise ValueError("complementarity oracle requires paired records")
    predictions: dict[str, list[dict[str, torch.Tensor]]] = {
        name: []
        for name in (
            "fdr_stock",
            "frequencycm_stock",
            "selector_stock",
            "fdr_oracle",
            "frequencycm_oracle",
            "duplicate_fdr_oracle",
            "union_oracle",
        )
    }
    targets: list[dict[str, torch.Tensor]] = []
    all_fdr_best: list[torch.Tensor] = []
    all_frequencycm_best: list[torch.Tensor] = []
    all_fdr_matched: list[torch.Tensor] = []
    all_frequencycm_matched: list[torch.Tensor] = []
    all_union_matched: list[torch.Tensor] = []
    all_scales: list[str] = []
    all_classes: list[torch.Tensor] = []
    source_ranks = torch.arange(MAX_DET, dtype=torch.long)

    with torch.inference_mode():
        for record in records:
            fdr_boxes = record["fdr_boxes"].detach().float().cpu()
            fdr_logits = record["fdr_logits"].detach().float().cpu()
            frequencycm_boxes = record["frequencycm_boxes"].detach().float().cpu()
            frequencycm_logits = record["frequencycm_logits"].detach().float().cpu()
            target_boxes = record["target_boxes"].detach().float().cpu()
            target_classes = record["target_classes"].detach().long().cpu()
            target = {"boxes": target_boxes, "classes": target_classes}
            targets.append(target)

            fdr_stock = _prediction_record(
                flattened_topk(
                    fdr_boxes[None], fdr_logits.sigmoid()[None], NUM_CLASSES, MAX_DET
                )[0]
            )
            frequencycm_stock = _prediction_record(
                flattened_topk(
                    frequencycm_boxes[None],
                    frequencycm_logits.sigmoid()[None],
                    NUM_CLASSES,
                    MAX_DET,
                )[0]
            )
            predictions["fdr_stock"].append(fdr_stock)
            predictions["frequencycm_stock"].append(frequencycm_stock)

            fdr_arm = build_matched_quality_arm(
                fdr_boxes,
                fdr_logits.sigmoid(),
                source_ranks,
                target_boxes,
                target_classes,
                MAX_DET,
            )
            frequencycm_arm = build_matched_quality_arm(
                frequencycm_boxes,
                frequencycm_logits.sigmoid(),
                source_ranks,
                target_boxes,
                target_classes,
                MAX_DET,
            )
            duplicate_arm = build_matched_quality_arm(
                torch.cat((fdr_boxes, fdr_boxes)),
                torch.cat((fdr_logits.sigmoid(), fdr_logits.sigmoid())),
                torch.arange(MAX_DET * 2, dtype=torch.long),
                target_boxes,
                target_classes,
                MAX_DET,
            )
            union_arm = build_matched_quality_arm(
                torch.cat((fdr_boxes, frequencycm_boxes)),
                torch.cat((fdr_logits.sigmoid(), frequencycm_logits.sigmoid())),
                torch.arange(MAX_DET * 2, dtype=torch.long),
                target_boxes,
                target_classes,
                MAX_DET,
            )
            predictions["fdr_oracle"].append(_prediction_record(fdr_arm))
            predictions["frequencycm_oracle"].append(
                _prediction_record(frequencycm_arm)
            )
            predictions["duplicate_fdr_oracle"].append(
                _prediction_record(duplicate_arm)
            )
            predictions["union_oracle"].append(_prediction_record(union_arm))
            fdr_utility = _stock_utility(fdr_stock, target_boxes, target_classes)
            frequencycm_utility = _stock_utility(
                frequencycm_stock, target_boxes, target_classes
            )
            predictions["selector_stock"].append(
                frequencycm_stock if frequencycm_utility > fdr_utility else fdr_stock
            )

            fdr_iou = candidate_iou_matrix(fdr_boxes, target_boxes)
            frequencycm_iou = candidate_iou_matrix(
                frequencycm_boxes, target_boxes
            )
            all_fdr_best.append(
                fdr_iou.amax(dim=0) if target_boxes.shape[0] else torch.empty(0)
            )
            all_frequencycm_best.append(
                frequencycm_iou.amax(dim=0)
                if target_boxes.shape[0]
                else torch.empty(0)
            )
            fdr_matched = _matched_target_iou(
                fdr_boxes, target_boxes, target_classes
            )
            frequencycm_matched = _matched_target_iou(
                frequencycm_boxes, target_boxes, target_classes
            )
            union_matched = _matched_target_iou(
                torch.cat((fdr_boxes, frequencycm_boxes)),
                target_boxes,
                target_classes,
            )
            all_fdr_matched.append(fdr_matched)
            all_frequencycm_matched.append(frequencycm_matched)
            all_union_matched.append(union_matched)
            all_scales.extend(_target_scales(record))
            all_classes.append(target_classes)

    arm_metrics = {
        name: compute_detection_metrics(values, targets, image_size=IMAGE_SIZE)
        for name, values in predictions.items()
    }
    duplicate_neutral = all(
        abs(arm_metrics["duplicate_fdr_oracle"][name] - arm_metrics["fdr_oracle"][name])
        <= 1e-12
        for name in arm_metrics["fdr_oracle"]
    )
    if not duplicate_neutral:
        raise RuntimeError("duplicated-FDR oracle control is not neutral")

    fdr_best = torch.cat(all_fdr_best)
    frequencycm_best = torch.cat(all_frequencycm_best)
    fdr_matched = torch.cat(all_fdr_matched)
    frequencycm_matched = torch.cat(all_frequencycm_matched)
    union_matched = torch.cat(all_union_matched)
    target_classes = torch.cat(all_classes)
    coverage = coverage_summary(
        fdr_best,
        frequencycm_best,
        fdr_matched_iou=fdr_matched,
        frequencycm_matched_iou=frequencycm_matched,
        union_matched_iou=union_matched,
        target_scales=tuple(all_scales),
        target_classes=target_classes,
    )
    tiny_small = torch.tensor(
        [scale in {"tiny", "small"} for scale in all_scales], dtype=torch.bool
    )
    tiny_small_total = int(tiny_small.sum())
    if tiny_small_total:
        fdr_recall = float(((fdr_matched >= 0.5) & tiny_small).sum()) / tiny_small_total
        frequencycm_recall = (
            float(((frequencycm_matched >= 0.5) & tiny_small).sum())
            / tiny_small_total
        )
        union_recall = (
            float(((union_matched >= 0.5) & tiny_small).sum()) / tiny_small_total
        )
        tiny_small_delta = union_recall - max(fdr_recall, frequencycm_recall)
    else:
        fdr_recall = frequencycm_recall = union_recall = tiny_small_delta = 0.0

    candidate_map_delta = arm_metrics["union_oracle"]["map"] - max(
        arm_metrics["fdr_oracle"]["map"],
        arm_metrics["frequencycm_oracle"]["map"],
    )
    candidate_ap50_delta = arm_metrics["union_oracle"]["ap50"] - max(
        arm_metrics["fdr_oracle"]["ap50"],
        arm_metrics["frequencycm_oracle"]["ap50"],
    )
    candidate_ap75_delta = arm_metrics["union_oracle"]["ap75"] - max(
        arm_metrics["fdr_oracle"]["ap75"],
        arm_metrics["frequencycm_oracle"]["ap75"],
    )
    decision = decide_complementarity(candidate_map_delta, tiny_small_delta)
    payload: dict[str, Any] = {
        "format_version": 1,
        "interpretation": "non_deployable_design_selection_evidence",
        "stock": {
            "fdr": arm_metrics["fdr_stock"],
            "frequencycm": arm_metrics["frequencycm_stock"],
            "image_selector": arm_metrics["selector_stock"],
        },
        "oracle_arms": {
            name: metrics
            for name, metrics in arm_metrics.items()
            if name.endswith("oracle")
        },
        "reproduction": {"duplicate_fdr_neutral": duplicate_neutral},
        "oracle": {
            "candidate_map_delta": candidate_map_delta,
            "candidate_ap50_delta": candidate_ap50_delta,
            "candidate_ap75_delta": candidate_ap75_delta,
        },
        "coverage": {
            "tiny_small_total": tiny_small_total,
            "tiny_small_fdr_recall50": fdr_recall,
            "tiny_small_frequencycm_recall50": frequencycm_recall,
            "tiny_small_union_recall50": union_recall,
            "tiny_small_recall50_delta": tiny_small_delta,
            "details": coverage,
        },
        "decision": decision,
    }
    scale_rows = _coverage_rows(
        coverage, group_key="by_scale", label_key="scale"
    )
    class_rows = _coverage_rows(
        coverage, group_key="by_class", label_key="class_id"
    )
    missed_rows = _missed_rows(coverage)
    arm_rows = [{"arm": name, **metrics} for name, metrics in arm_metrics.items()]
    _write_report_bundle(
        report_root,
        payload,
        scale_rows=scale_rows,
        class_rows=class_rows,
        missed_rows=missed_rows,
        arm_rows=arm_rows,
    )
    return payload


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable report output already exists: {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_summary(report_root: Path, payload: Mapping[str, Any]) -> None:
    root = Path(report_root)
    summary_path = root / "oracle-summary.json"
    markdown_path = root / "frequencycm-complementarity-report.md"
    sums_path = root / "SHA256SUMS.txt"
    for path in (summary_path, markdown_path, sums_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable report output already exists: {path}")

    normalized = dict(payload)
    normalized["interpretation"] = "non_deployable_design_selection_evidence"
    summary = _canonical_json(normalized)
    decision = str(normalized.get("decision", {}).get("decision", "unknown"))
    candidate_delta = normalized.get("oracle", {}).get("candidate_map_delta", "unknown")
    coverage_delta = normalized.get("coverage", {}).get(
        "tiny_small_recall50_delta", "unknown"
    )
    markdown = (
        "# FrequencyCM Complementarity Oracle\n\n"
        "> This report uses ground truth and is non-deployable design-selection evidence.\n\n"
        f"- Decision: `{decision}`\n"
        f"- Candidate-oracle mAP delta: `{candidate_delta}`\n"
        f"- Tiny/small recall@0.50 delta: `{coverage_delta}`\n"
    ).encode("utf-8")
    _write_create_only(summary_path, summary)
    _write_create_only(markdown_path, markdown)
    sums = "".join(
        f"{hashlib.sha256(content).hexdigest().upper()}  {name}\n"
        for name, content in (
            (summary_path.name, summary),
            (markdown_path.name, markdown),
        )
    ).encode("ascii")
    _write_create_only(sums_path, sums)


def main(argv: Sequence[str] | None = None) -> int:
    _parse_args(argv)
    raise RuntimeError("full complementarity execution is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())

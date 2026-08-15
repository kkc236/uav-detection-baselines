"""Evaluate stock and refined outputs from one frozen IBER-BE checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_iber_stock import (  # noqa: E402
    BASELINE_REFERENCE_ENVIRONMENT,
    current_execution_environment,
)
from scripts.train_iber import (  # noqa: E402
    AUGMENTATION,
    TRAINING_CONSTANTS,
    _validate_gate1_decision,
    highest_contiguous_verified_epoch,
    validate_resume_checkpoint,
)
from src.iber_evaluation import (  # noqa: E402
    EVALUATION_CONSTANTS,
    assert_repeated_evaluations,
    compute_detection_metrics,
    compute_refinement_diagnostics,
    evaluate_gate2,
)
from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PRIVATE_SEED,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    execution_environment,
    file_sha256,
    module_state_sha256,
    write_immutable_report,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
)
from src.rtdetr_iber import FrozenIBERAdapter  # noqa: E402


EXPECTED_CATEGORY_SHA256 = (
    "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--private-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gate1-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _seed_evaluation() -> None:
    seed = EVALUATION_CONSTANTS["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _source_commit() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("IBER-BE source commit is invalid")
    return value


def _build_validation_loader(
    dataset_root: Path,
    baseline_checkpoint: Path,
    device: torch.device,
    *,
    save_dir: Path,
):
    """Use the exact Ultralytics RT-DETR validation preprocessing and classes."""
    from ultralytics.models.rtdetr.val import RTDETRValidator

    data = {
        "path": str(dataset_root.resolve()),
        "train": str((dataset_root / "images" / "train").resolve()),
        "val": str((dataset_root / "images" / "val").resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": len(CATEGORY_NAMES),
        "channels": 3,
    }
    validator = RTDETRValidator(
        save_dir=save_dir,
        args={
            "model": str(baseline_checkpoint.resolve()),
            "data": data,
            "task": "detect",
            "mode": "val",
            "split": "val",
            "imgsz": EVALUATION_CONSTANTS["imgsz"],
            "batch": EVALUATION_CONSTANTS["batch"],
            "workers": EVALUATION_CONSTANTS["workers"],
            "device": EVALUATION_CONSTANTS["device"],
            "max_det": EVALUATION_CONSTANTS["max_det"],
            "nms": EVALUATION_CONSTANTS["nms"],
            "cache": EVALUATION_CONSTANTS["cache"],
            "conf": EVALUATION_CONSTANTS["conf"],
            "half": EVALUATION_CONSTANTS["half"],
            "rect": False,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
        }
    )
    validator.data = data
    validator.device = device
    loader = validator.get_dataloader(data["val"], EVALUATION_CONSTANTS["batch"])
    if len(loader.dataset) != 548:
        raise ValueError(
            f"IBER-BE validation image count mismatch: {len(loader.dataset)}"
        )
    return loader, validator


def _batch_targets(
    batch: Mapping[str, Any], image_index: int
) -> dict[str, torch.Tensor]:
    mask = batch["batch_idx"].view(-1).long() == image_index
    return {
        "boxes": batch["bboxes"][mask].detach().float().cpu(),
        "classes": batch["cls"][mask].view(-1).detach().long().cpu(),
    }


def _batch_predictions(
    postprocessed: torch.Tensor, image_index: int
) -> dict[str, torch.Tensor]:
    prediction = postprocessed[image_index].detach().float().cpu()
    prediction = prediction[prediction[:, 4] > EVALUATION_CONSTANTS["conf"]]
    return {
        "boxes": prediction[:, :4],
        "scores": prediction[:, 4],
        "classes": prediction[:, 5].long(),
    }


def _assert_shared_prediction_scores(
    stock_postprocessed: torch.Tensor,
    refined_postprocessed: torch.Tensor,
) -> None:
    """Reject any postprocess drift outside the box coordinates."""
    if (
        stock_postprocessed.shape != refined_postprocessed.shape
        or stock_postprocessed.ndim != 3
        or stock_postprocessed.shape[-1] < 6
        or not torch.equal(
            stock_postprocessed[..., 4:6], refined_postprocessed[..., 4:6]
        )
    ):
        raise RuntimeError(
            "IBER-BE stock/refined postprocess did not preserve shared scores/classes"
        )


def _globalize_match_indices(
    matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    groups: Sequence[int],
    prior_target_count: int,
    query_count: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert matcher batch-global GT indices to evaluation-global indices."""
    if len(matches) != len(groups) or prior_target_count < 0 or query_count < 1:
        raise ValueError("IBER-BE matcher batch schema is invalid")
    converted: list[tuple[torch.Tensor, torch.Tensor]] = []
    batch_target_offset = 0
    for image_index, ((source, destination), group) in enumerate(
        zip(matches, groups, strict=True)
    ):
        if type(group) is not int or group < 0:
            raise ValueError("IBER-BE matcher target group is invalid")
        source = source.detach().long().cpu().view(-1)
        destination = destination.detach().long().cpu().view(-1)
        if source.numel() != destination.numel():
            raise ValueError("IBER-BE matcher source/target lengths differ")
        if source.numel() and (
            int(source.min()) < 0 or int(source.max()) >= query_count
        ):
            raise ValueError("IBER-BE matcher query index is out of range")
        target_end = batch_target_offset + group
        if destination.numel() and (
            int(destination.min()) < batch_target_offset
            or int(destination.max()) >= target_end
        ):
            raise ValueError(
                f"IBER-BE matcher target crosses image boundary at image {image_index}"
            )
        converted.append((source, destination + prior_target_count))
        batch_target_offset = target_end
    return converted


def _evaluate_once(
    adapter: FrozenIBERAdapter,
    loader: Any,
    validator: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _seed_evaluation()
    adapter.eval()
    stock_predictions: list[dict[str, torch.Tensor]] = []
    refined_predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    stock_parts: list[torch.Tensor] = []
    refined_parts: list[torch.Tensor] = []
    correction_parts: list[torch.Tensor] = []
    gate_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    f3_parts: list[torch.Tensor] = []
    rgb_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    target_group_sizes: list[int] = []
    all_matches: list[tuple[torch.Tensor, torch.Tensor]] = []
    global_target_offset = 0

    with torch.inference_mode():
        for raw_batch in loader:
            batch = validator.preprocess(raw_batch)
            output = adapter.forward_evidence(batch["img"])
            head = adapter.detector.model[-1]
            shared_scores = output.stock_scores.sigmoid()
            stock_post = head.postprocess(
                output.stock_boxes, shared_scores
            )
            refined_post = head.postprocess(
                output.refined_boxes, shared_scores
            )
            _assert_shared_prediction_scores(stock_post, refined_post)
            target_boxes = batch["bboxes"].detach().to(
                device=device, dtype=output.stock_boxes.dtype
            )
            target_classes = (
                batch["cls"].detach().to(device=device, dtype=torch.long).view(-1)
            )
            batch_index = (
                batch["batch_idx"]
                .detach()
                .to(device=device, dtype=torch.long)
                .view(-1)
            )
            groups = [
                int((batch_index == index).sum())
                for index in range(batch["img"].shape[0])
            ]
            matches = adapter.criterion.matcher(
                output.stock_boxes.detach(),
                output.stock_scores.detach(),
                target_boxes,
                target_classes,
                groups,
            )
            all_matches.extend(
                _globalize_match_indices(
                    matches,
                    groups=groups,
                    prior_target_count=global_target_offset,
                    query_count=output.stock_boxes.shape[1],
                )
            )
            target_group_sizes.extend(groups)
            global_target_offset += sum(groups)

            stock_parts.append(output.stock_boxes.detach().float().cpu())
            refined_parts.append(output.refined_boxes.detach().float().cpu())
            correction_parts.append(
                output.effective_correction.detach().float().cpu()
            )
            gate_parts.append(output.gates.detach().float().cpu())
            residual_parts.append(output.residuals.detach().float().cpu())
            f3_parts.append(output.f3_boundary_features.detach().float().cpu())
            rgb_parts.append(output.rgb_boundary_features.detach().float().cpu())
            target_parts.append(target_boxes.detach().float().cpu())
            for image_index, group in enumerate(groups):
                targets.append(_batch_targets(batch, image_index))
                stock_predictions.append(
                    _batch_predictions(stock_post, image_index)
                )
                refined_predictions.append(
                    _batch_predictions(refined_post, image_index)
                )

    if len(targets) != 548:
        raise RuntimeError(
            f"IBER-BE evaluation processed {len(targets)} images instead of 548"
        )
    stock_metrics = compute_detection_metrics(
        stock_predictions,
        targets,
        image_size=EVALUATION_CONSTANTS["imgsz"],
    )
    refined_metrics = compute_detection_metrics(
        refined_predictions,
        targets,
        image_size=EVALUATION_CONSTANTS["imgsz"],
    )
    diagnostics = compute_refinement_diagnostics(
        torch.cat(stock_parts),
        torch.cat(refined_parts),
        torch.cat(target_parts),
        all_matches,
        torch.cat(correction_parts),
        torch.cat(gate_parts),
        torch.cat(residual_parts),
        torch.cat(f3_parts),
        torch.cat(rgb_parts),
        target_group_sizes=target_group_sizes,
    )
    return {
        "stock": stock_metrics,
        "refined": refined_metrics,
        "diagnostics": diagnostics,
    }


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("IBER-BE result history row must be a mapping")
            rows.append(row)
    return rows


def _last5_history(
    output_path: Path,
    current: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    results_path = output_path.resolve().parent.parent / "results.jsonl"
    ledger_path = output_path.resolve().parent.parent / "publication-ledger.jsonl"
    rows = _json_rows(results_path)
    ledger_rows = _json_rows(ledger_path)
    if len(rows) != 29 or len(ledger_rows) != 29:
        raise ValueError(
            "IBER-BE epoch30 decision requires exactly 29 published prior rows"
        )
    if highest_contiguous_verified_epoch(ledger_rows) != 29:
        raise ValueError("IBER-BE epoch30 publication ledger tip is not epoch29")
    if current.get("design_version") != DESIGN_VERSION or current.get("epoch") != 30:
        raise ValueError("IBER-BE current epoch30 evaluation authority mismatch")
    prior: list[Mapping[str, Any]] = []
    for expected_epoch, (row, publication) in enumerate(
        zip(rows, ledger_rows, strict=True), start=1
    ):
        if (
            row.get("design_version") != DESIGN_VERSION
            or row.get("stage") != "screen"
            or row.get("probe") != "b3"
            or row.get("seed") != 0
            or row.get("epoch") != expected_epoch
        ):
            raise ValueError("IBER-BE result history epochs are not contiguous")
        evaluation = row.get("evaluation")
        if (
            not isinstance(evaluation, Mapping)
            or evaluation.get("design_version") != DESIGN_VERSION
            or evaluation.get("epoch") != expected_epoch
        ):
            raise ValueError("IBER-BE result history authority mismatch")
        private_checkpoint = evaluation.get("private_checkpoint")
        published_checkpoint = publication.get("checkpoint")
        if (
            not isinstance(private_checkpoint, Mapping)
            or not isinstance(published_checkpoint, Mapping)
            or str(private_checkpoint.get("sha256", "")).lower()
            != str(published_checkpoint.get("sha256", "")).lower()
        ):
            raise ValueError(
                f"IBER-BE result history checkpoint differs at epoch {expected_epoch}"
            )
        if expected_epoch >= 26:
            prior.append(evaluation)
    if len(prior) != 4:
        raise ValueError("IBER-BE epoch30 decision lacks epochs 26-29")
    evaluations = [*prior, current]
    try:
        last5_stock_map = [float(value["stock"]["map"]) for value in evaluations]
        last5_refined_map = [float(value["refined"]["map"]) for value in evaluations]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("IBER-BE epoch30 last5 metric schema is invalid") from error
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in [*last5_stock_map, *last5_refined_map]
    ):
        raise ValueError("IBER-BE epoch30 last5 metrics are invalid")
    return last5_stock_map, last5_refined_map


def evaluate_checkpoint(
    *,
    baseline_checkpoint: Path,
    private_checkpoint: Path,
    dataset_root: Path,
    gate1_decision: Path,
    output_path: Path,
) -> dict[str, Any]:
    source_commit = _source_commit()
    baseline_sha = file_sha256(baseline_checkpoint)
    dataset_sha = str(dataset_signature(dataset_root)["sha256"])
    category_sha = category_mapping_sha256(CATEGORY_NAMES)
    if (baseline_sha, dataset_sha, category_sha) != (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_DATASET_SHA256,
        EXPECTED_CATEGORY_SHA256,
    ):
        raise ValueError("IBER-BE evaluation artifact authority mismatch")
    artifact = torch.load(
        private_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(artifact, Mapping):
        raise ValueError("IBER-BE private checkpoint schema mismatch")
    validate_resume_checkpoint(
        artifact,
        source_commit=source_commit,
        highest_verified_epoch=int(artifact.get("epoch", -1)),
    )
    gate1_sha = _validate_gate1_decision(
        gate1_decision, source_commit=source_commit
    )
    if (
        artifact.get("training_constants") != TRAINING_CONSTANTS
        or artifact.get("augmentation") != AUGMENTATION
        or artifact.get("protocol_sha256") != PROTOCOL_SHA256
        or artifact.get("runtime_amendment_sha256")
        != RUNTIME_AMENDMENT_SHA256
        or artifact.get("subset_sha256") != EXPECTED_SUBSET_SHA256
        or artifact.get("gate1_decision_sha256") != gate1_sha
        or artifact.get("execution_environment") != execution_environment()
    ):
        raise ValueError("IBER-BE checkpoint training protocol mismatch")
    measured_environment = current_execution_environment()
    if measured_environment != execution_environment():
        raise ValueError("IBER-BE evaluation execution environment mismatch")

    from ultralytics import RTDETR

    device = torch.device("cuda:0")
    detector = RTDETR(str(baseline_checkpoint)).model.to(device).eval()
    detector.requires_grad_(False)
    with FrozenIBERAdapter.from_detector(
        detector,
        private_seed=PRIVATE_SEED,
        probe="b3",
        image_size=EVALUATION_CONSTANTS["imgsz"],
        rho=0.05,
    ).to(device).eval() as adapter:
        adapter.refiner.load_state_dict(artifact["refiner"], strict=True)
        detector_sha_before = module_state_sha256(detector)
        if artifact.get("detector_sha_after") != detector_sha_before:
            raise ValueError("IBER-BE private checkpoint detector authority mismatch")
        loader, validator = _build_validation_loader(
            dataset_root,
            baseline_checkpoint,
            device,
            save_dir=output_path.resolve().parent / "validator",
        )
        repeats = [
            _evaluate_once(adapter, loader, validator, device=device)
            for _ in range(EVALUATION_CONSTANTS["repeats"])
        ]
        accepted = assert_repeated_evaluations(repeats)
        detector_sha_after = module_state_sha256(detector)
    accepted["diagnostics"].update(
        {
            "detector_sha_before": detector_sha_before,
            "detector_sha_after": detector_sha_after,
        }
    )
    epoch = int(artifact["epoch"])
    report: dict[str, Any] = {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "epoch": epoch,
        "checkpoint_epoch": epoch,
        "source_commit": source_commit,
        "baseline_checkpoint": {
            "path": str(baseline_checkpoint.resolve()),
            "bytes": baseline_checkpoint.stat().st_size,
            "sha256": baseline_sha,
        },
        "private_checkpoint": {
            "path": str(private_checkpoint.resolve()),
            "bytes": private_checkpoint.stat().st_size,
            "sha256": file_sha256(private_checkpoint),
        },
        "dataset_sha256": dataset_sha,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "category_sha256": category_sha,
        "gate1_decision_sha256": gate1_sha,
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime_amendment": dict(RUNTIME_AMENDMENT),
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "baseline_reference_environment": dict(BASELINE_REFERENCE_ENVIRONMENT),
        "execution_environment": execution_environment(),
        "measured_environment": measured_environment,
        "evaluation_constants": dict(EVALUATION_CONSTANTS),
        "repeat_count": len(repeats),
        "repeat_exact": True,
        **accepted,
    }
    if epoch == 30:
        last5_stock_map, last5_refined_map = _last5_history(
            output_path, report
        )
        report["decision"] = evaluate_gate2(
            report["stock"],
            report["refined"],
            report["diagnostics"],
            repeats=repeats,
            last5_stock_map=last5_stock_map,
            last5_refined_map=last5_refined_map,
            checkpoint_epoch=epoch,
        )
    else:
        report["decision"] = {
            "status": "pending_epoch30",
            "required_epoch": 30,
            "actual_epoch": epoch,
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve()
    report = evaluate_checkpoint(
        baseline_checkpoint=args.baseline_checkpoint.resolve(),
        private_checkpoint=args.private_checkpoint.resolve(),
        dataset_root=args.dataset_root.resolve(),
        gate1_decision=args.gate1_decision.resolve(),
        output_path=output,
    )
    write_immutable_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate stock and refined outputs from one frozen I-TBER checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_itber import (  # noqa: E402
    AUGMENTATION,
    TRAINING_CONSTANTS,
    stage_protocol,
    validate_resume_checkpoint,
)
from src.itber_evaluation import (  # noqa: E402
    EVALUATION_CONSTANTS,
    assert_repeated_evaluations,
    compute_detection_metrics,
    compute_refinement_diagnostics,
    evaluate_gate2,
    write_immutable_report,
)
from src.itber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
    module_state_sha256,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    current_environment,
    dataset_signature,
    file_sha256,
)
from src.rtdetr_itber import FrozenITBERAdapter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--private-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _seed_evaluation() -> None:
    seed = EVALUATION_CONSTANTS["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _build_validation_loader(dataset_root: Path, baseline_checkpoint: Path, device: torch.device):
    """Use the exact Ultralytics RT-DETR validation dataset and preprocessing."""
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
        raise ValueError(f"I-TBER validation image count mismatch: {len(loader.dataset)}")
    return loader, validator


def _batch_targets(batch: dict[str, Any], image_index: int) -> dict[str, torch.Tensor]:
    mask = batch["batch_idx"].view(-1).long() == image_index
    return {
        "boxes": batch["bboxes"][mask].detach().float().cpu(),
        "classes": batch["cls"][mask].view(-1).detach().long().cpu(),
    }


def _batch_predictions(postprocessed: torch.Tensor, image_index: int) -> dict[str, torch.Tensor]:
    prediction = postprocessed[image_index].detach().float().cpu()
    selected = prediction[:, 4] > EVALUATION_CONSTANTS["conf"]
    prediction = prediction[selected]
    return {
        "boxes": prediction[:, :4],
        "scores": prediction[:, 4],
        "classes": prediction[:, 5].long(),
    }


def _evaluate_once(
    adapter: FrozenITBERAdapter,
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
    target_parts: list[torch.Tensor] = []
    all_matches: list[tuple[torch.Tensor, torch.Tensor]] = []
    global_target_offset = 0

    with torch.inference_mode():
        for raw_batch in loader:
            batch = validator.preprocess(raw_batch)
            output = adapter.forward_evidence(batch["img"])
            decoder = adapter.detector.model[-1].decoder
            scores = decoder.last_stock_scores
            if scores is None:
                raise RuntimeError("I-TBER evaluation did not capture stock scores")
            head = adapter.detector.model[-1]
            stock_post = head.postprocess(output.stock_boxes, scores.sigmoid())
            refined_post = head.postprocess(output.refined_boxes, scores.sigmoid())
            target_boxes = batch["bboxes"].detach().to(
                device=device, dtype=output.stock_boxes.dtype
            )
            target_classes = batch["cls"].detach().to(device=device, dtype=torch.long).view(-1)
            batch_index = batch["batch_idx"].detach().to(device=device, dtype=torch.long).view(-1)
            groups = [int((batch_index == index).sum()) for index in range(batch["img"].shape[0])]
            matches = adapter.criterion.matcher(
                output.stock_boxes.detach(),
                scores.detach(),
                target_boxes,
                target_classes,
                groups,
            )

            stock_parts.append(output.stock_boxes.detach().float().cpu())
            refined_parts.append(output.refined_boxes.detach().float().cpu())
            correction_parts.append(output.effective_correction.detach().float().cpu())
            gate_parts.append(output.gates.detach().float().cpu())
            residual_parts.append(output.residuals.detach().float().cpu())
            target_parts.append(target_boxes.detach().float().cpu())
            batch_target_offset = 0
            for image_index, group in enumerate(groups):
                target_record = _batch_targets(batch, image_index)
                targets.append(target_record)
                stock_predictions.append(_batch_predictions(stock_post, image_index))
                refined_predictions.append(_batch_predictions(refined_post, image_index))
                source, destination = matches[image_index]
                local_destination = destination.long().cpu() - batch_target_offset
                all_matches.append((source.long().cpu(), local_destination + global_target_offset))
                batch_target_offset += group
                global_target_offset += group

    if len(targets) != 548:
        raise RuntimeError(f"I-TBER evaluation processed {len(targets)} images instead of 548")
    stock_metrics = compute_detection_metrics(
        stock_predictions, targets, image_size=EVALUATION_CONSTANTS["imgsz"]
    )
    refined_metrics = compute_detection_metrics(
        refined_predictions, targets, image_size=EVALUATION_CONSTANTS["imgsz"]
    )
    diagnostics = compute_refinement_diagnostics(
        torch.cat(stock_parts),
        torch.cat(refined_parts),
        torch.cat(target_parts),
        all_matches,
        torch.cat(correction_parts),
        torch.cat(gate_parts),
        torch.cat(residual_parts),
    )
    return {"stock": stock_metrics, "refined": refined_metrics, "diagnostics": diagnostics}


def evaluate_checkpoint(
    *,
    stage: str,
    baseline_checkpoint: Path,
    private_checkpoint: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    baseline_sha = file_sha256(baseline_checkpoint)
    dataset_sha = str(dataset_signature(dataset_root)["sha256"])
    category_sha = category_mapping_sha256(CATEGORY_NAMES)
    if (baseline_sha, dataset_sha, category_sha) != (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_DATASET_SHA256,
        EXPECTED_CATEGORY_SHA256,
    ):
        raise ValueError("I-TBER evaluation authority mismatch")
    artifact = torch.load(private_checkpoint, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(artifact, stage=stage)
    if artifact.get("training_constants") != TRAINING_CONSTANTS or artifact.get("augmentation") != AUGMENTATION:
        raise ValueError("I-TBER checkpoint training protocol mismatch")

    from ultralytics import RTDETR

    device = torch.device("cuda:0")
    detector = RTDETR(str(baseline_checkpoint)).model.to(device).eval()
    detector.requires_grad_(False)
    adapter = FrozenITBERAdapter.from_detector(
        detector,
        private_seed=TRAINING_CONSTANTS["private_seed"],
        probe="p3",
        image_size=EVALUATION_CONSTANTS["imgsz"],
        rho=0.05,
    ).to(device).eval()
    adapter.refiner.load_state_dict(artifact["refiner"], strict=True)
    detector_sha_before = module_state_sha256(detector)
    if artifact.get("detector_sha_after") != detector_sha_before:
        raise ValueError("I-TBER private checkpoint detector authority mismatch")
    loader, validator = _build_validation_loader(dataset_root, baseline_checkpoint, device)
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
    report: dict[str, Any] = {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "stage": stage,
        "seed": 0,
        "epoch": int(artifact["epoch"]),
        "source_commit": _source_commit(),
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
        "category_sha256": category_sha,
        "environment": current_environment(),
        "evaluation_constants": EVALUATION_CONSTANTS,
        "repeat_count": len(repeats),
        "repeat_exact": True,
        **accepted,
    }
    if artifact["epoch"] == stage_protocol(stage).epochs:
        if stage == "screen":
            report["decision"] = evaluate_gate2(
                report["stock"], report["refined"], report["diagnostics"]
            )
        else:
            report["decision"] = {
                "status": "pending_tail5_history",
                "reason": "formal decision requires immutable epochs 26-30 evaluation reports",
            }
    else:
        report["decision"] = {
            "status": "not_final_epoch",
            "required_epoch": stage_protocol(stage).epochs,
        }
    return report


def main() -> int:
    args = _parse_args()
    report = evaluate_checkpoint(
        stage=args.stage,
        baseline_checkpoint=args.baseline_checkpoint.resolve(),
        private_checkpoint=args.private_checkpoint.resolve(),
        dataset_root=args.dataset_root.resolve(),
    )
    write_immutable_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

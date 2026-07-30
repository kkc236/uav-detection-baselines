from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_sqda_geometry_branches import (
    _build_loader,
    _device,
    _predictions_to_coco,
    _state_sha256,
    _write_json,
)
from scripts.evaluate_sqda_small_ap import build_coco_dataset, evaluate_predictions
from src.rtdetr_sqda_sgc import (
    BASELINE_SHA256,
    SQDASGCDetectionModel,
    load_mature_baseline,
    load_trained_geometry_adapter,
    sha256_file,
)
from src.sqda_geometry_checkpoint_selection import (
    select_earliest_passing_candidate,
    select_trainable_candidates,
)
from src.sqda_error_audit import precision_recall_f1_curve, summarize_detection_errors
from src.sqda_geometry_gate_decision import decide_g1_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a completed SQDA geometry-gate G1 against frozen retained-G2 evidence."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-images", type=int, default=548)
    parser.add_argument("--expected-annotations", type=int, default=38759)
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    baseline = args.checkpoint.expanduser().resolve()
    if args.candidate_checkpoint is None:
        raise ValueError("single-checkpoint evaluation requires --candidate-checkpoint")
    candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve()
    diagnosis_path = args.diagnosis.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    images = args.images.expanduser().resolve()
    labels = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if sha256_file(baseline) != BASELINE_SHA256:
        raise ValueError("mature baseline SHA256 mismatch")
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    full = diagnosis.get("branches", {}).get("full")
    if not isinstance(full, dict):
        raise ValueError("read-only diagnosis is missing the retained-G2 full branch")
    threshold = float(full["fixed_baseline_threshold"]["confidence_threshold"])
    dataset = build_coco_dataset(images, labels)
    if len(dataset["images"]) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(dataset['images'])}")
    if len(dataset["annotations"]) != args.expected_annotations:
        raise RuntimeError(
            f"expected {args.expected_annotations} annotations, found {len(dataset['annotations'])}"
        )
    data, loader = _build_loader(data_yaml, args.workers)
    device = _device(args.device)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = SQDASGCDetectionModel("rtdetr-l.yaml", nc=int(data["nc"]), verbose=False)
    load_mature_baseline(model, baseline, expected_sha256=BASELINE_SHA256)
    trained_metadata = load_trained_geometry_adapter(model, candidate_checkpoint)
    model = model.to(device).eval()
    adapter_before = _state_sha256(model.sqda_sgc)
    records: list[dict[str, Any]] = []
    gate_count = 0
    gate_sum = 0.0
    gate_min = float("inf")
    gate_max = float("-inf")
    lower_count = 0
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=device.type == "cuda",
    ):
        for batch in loader:
            images_tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
            prediction = model(images_tensor)
            decoded = prediction[0] if isinstance(prediction, tuple) else prediction
            records.extend(
                _predictions_to_coco(
                    decoded,
                    list(batch["im_file"]),
                    list(batch["ori_shape"]),
                    image_size=640,
                )
            )
            budget = model.last_sqda_diagnostics["geometry_budget"].detach().float()
            gate_count += budget.numel()
            gate_sum += float(budget.sum().cpu())
            gate_min = min(gate_min, float(budget.min().cpu()))
            gate_max = max(gate_max, float(budget.max().cpu()))
            lower_count += int((budget <= 0.805).sum().cpu())
    adapter_after = _state_sha256(model.sqda_sgc)
    if adapter_before != adapter_after:
        raise AssertionError("read-only G1 evaluation mutated the candidate adapter")
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.json"
    _write_json(prediction_path, records)
    metrics = evaluate_predictions(dataset, prediction_path)
    candidate = {
        "training_signal": False,
        "coco": metrics,
        "fixed_baseline_threshold": {
            "confidence_threshold": threshold,
            "iou_threshold": 0.50,
            "error": summarize_detection_errors(
                dataset,
                records,
                confidence_threshold=threshold,
                iou_threshold=0.50,
            ),
        },
        "pr_f1_curve": precision_recall_f1_curve(dataset, records, iou_threshold=0.50),
        "gate": {
            "count": gate_count,
            "min": gate_min,
            "mean": gate_sum / gate_count,
            "max": gate_max,
            "lower_bound_fraction": lower_count / gate_count,
        },
        "prediction": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "entries": len(records),
        },
        "adapter_tensor_audit": {"before": adapter_before, "after": adapter_after},
        "trained_adapter": trained_metadata,
    }
    decision = decide_g1_result(full, candidate)
    decision.update(
        {
            "candidate_checkpoint": str(candidate_checkpoint),
            "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
            "diagnosis": str(diagnosis_path),
            "diagnosis_sha256": sha256_file(diagnosis_path),
        }
    )
    _write_json(output / "candidate-evaluation.json", candidate)
    _write_json(output / "final-gate-decision.json", decision)
    return decision


def run_checkpoint_inventory(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate every actual SMGT update and select only a strictly passing snapshot."""
    if args.weights_dir is None:
        raise ValueError("checkpoint inventory requires --weights-dir")
    candidates = select_trainable_candidates(args.weights_dir)
    if not candidates:
        raise RuntimeError("checkpoint inventory contains no updated SMGT snapshots")
    output = args.output.expanduser().resolve()
    records: list[tuple[Path, dict[str, Any]]] = []
    for checkpoint in candidates:
        child_args = argparse.Namespace(**vars(args))
        child_args.candidate_checkpoint = checkpoint
        child_args.weights_dir = None
        child_args.output = output / checkpoint.stem
        decision = run_evaluation(child_args)
        records.append((checkpoint, decision))
    selected = select_earliest_passing_candidate(records)
    summary = {
        "schema": 1,
        "training_signal": False,
        "weights_dir": str(args.weights_dir.expanduser().resolve()),
        "candidates": [
            {
                "checkpoint": str(checkpoint),
                "output": str(output / checkpoint.stem),
                "passed": bool(decision["passed"]),
                "criteria": decision["criteria"],
                "deltas": decision["deltas"],
            }
            for checkpoint, decision in records
        ],
        "selected_checkpoint": str(selected) if selected is not None else None,
    }
    _write_json(output / "candidate-inventory.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if (args.candidate_checkpoint is None) == (args.weights_dir is None):
        raise ValueError("provide exactly one of --candidate-checkpoint or --weights-dir")
    runner = run_checkpoint_inventory if args.weights_dir is not None else run_evaluation
    print(json.dumps(runner(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

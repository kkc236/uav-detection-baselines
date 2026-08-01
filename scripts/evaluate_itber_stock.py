"""Create the immutable current-environment stock baseline authority."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_itber import (  # noqa: E402
    _batch_predictions,
    _batch_targets,
    _build_validation_loader,
    _seed_evaluation,
    _source_commit,
)
from scripts.train_itber import TRAINING_CONSTANTS  # noqa: E402
from src.itber_evaluation import (  # noqa: E402
    EVALUATION_CONSTANTS,
    assert_repeated_evaluations,
    compute_detection_metrics,
    write_immutable_report,
)
from src.itber_protocol import (  # noqa: E402
    BASELINE_REFERENCE_ENVIRONMENT,
    EXECUTION_ENVIRONMENT,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    current_execution_environment,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    file_sha256,
)
from src.rtdetr_itber import FrozenITBERAdapter  # noqa: E402


AMENDED_GATE_STATUS = "passed_with_runtime_amendment"


def build_stock_authority_report(
    *,
    repeats: Sequence[Mapping[str, Any]],
    baseline_path: Path,
    baseline_bytes: int,
    baseline_sha256: str,
    dataset_sha256: str,
    category_sha256: str,
    execution_environment: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Validate repeated stock metrics and bind them to the amended runtime."""
    stock = assert_repeated_evaluations(repeats)
    actual_artifacts = (
        baseline_sha256.upper(),
        dataset_sha256.upper(),
        category_sha256.upper(),
    )
    expected_artifacts = (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_DATASET_SHA256,
        EXPECTED_CATEGORY_SHA256,
    )
    if actual_artifacts != expected_artifacts:
        raise ValueError("I-TBER stock authority artifact mismatch")
    if dict(execution_environment) != EXECUTION_ENVIRONMENT:
        raise ValueError("I-TBER stock authority execution environment mismatch")
    if len(source_commit) != 40:
        raise ValueError("I-TBER stock authority source commit is invalid")
    return {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "status": AMENDED_GATE_STATUS,
        "source_commit": source_commit.lower(),
        "baseline_checkpoint": {
            "path": str(Path(baseline_path).resolve()),
            "bytes": int(baseline_bytes),
            "sha256": baseline_sha256.upper(),
        },
        "dataset_sha256": dataset_sha256.upper(),
        "category_sha256": category_sha256.upper(),
        "baseline_reference_environment": dict(BASELINE_REFERENCE_ENVIRONMENT),
        "execution_environment": dict(EXECUTION_ENVIRONMENT),
        "runtime_amendment": dict(RUNTIME_AMENDMENT),
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "evaluation_constants": dict(EVALUATION_CONSTANTS),
        "repeat_count": len(repeats),
        "repeat_exact": True,
        "stock": stock,
    }


def _evaluate_stock_once(
    adapter: FrozenITBERAdapter,
    loader: Any,
    validator: Any,
    *,
    device: torch.device,
) -> dict[str, float]:
    _seed_evaluation()
    adapter.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = validator.preprocess(raw_batch)
            output = adapter.forward_evidence(batch["img"])
            if not torch.equal(output.stock_boxes, output.refined_boxes):
                raise RuntimeError("zero-initialized I-TBER changed stock boxes")
            decoder = adapter.detector.model[-1].decoder
            scores = decoder.last_stock_scores
            if scores is None:
                raise RuntimeError("I-TBER stock authority did not capture scores")
            postprocessed = adapter.detector.model[-1].postprocess(
                output.stock_boxes, scores.sigmoid()
            )
            groups = [
                int((batch["batch_idx"].view(-1).long() == index).sum())
                for index in range(batch["img"].shape[0])
            ]
            for image_index, _group in enumerate(groups):
                predictions.append(_batch_predictions(postprocessed, image_index))
                targets.append(_batch_targets(batch, image_index))
    if len(targets) != 548:
        raise RuntimeError(
            f"I-TBER stock authority processed {len(targets)} images instead of 548"
        )
    return compute_detection_metrics(
        predictions,
        targets,
        image_size=EVALUATION_CONSTANTS["imgsz"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline = args.baseline_checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    from ultralytics import RTDETR

    device = torch.device("cuda:0")
    detector = RTDETR(str(baseline)).model.to(device).eval()
    detector.requires_grad_(False)
    adapter = FrozenITBERAdapter.from_detector(
        detector,
        private_seed=TRAINING_CONSTANTS["private_seed"],
        probe="p3",
        image_size=EVALUATION_CONSTANTS["imgsz"],
        rho=0.05,
    ).to(device).eval()
    loader, validator = _build_validation_loader(dataset_root, baseline, device)
    repeats = [
        _evaluate_stock_once(adapter, loader, validator, device=device)
        for _ in range(EVALUATION_CONSTANTS["repeats"])
    ]
    report = build_stock_authority_report(
        repeats=repeats,
        baseline_path=baseline,
        baseline_bytes=baseline.stat().st_size,
        baseline_sha256=file_sha256(baseline),
        dataset_sha256=str(dataset_signature(dataset_root)["sha256"]),
        category_sha256=category_mapping_sha256(CATEGORY_NAMES),
        execution_environment=current_execution_environment(),
        source_commit=_source_commit(),
    )
    write_immutable_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the frozen FDR/FrequencyCM candidate-complementarity upper-bound oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


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

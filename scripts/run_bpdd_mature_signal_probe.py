"""Probe frozen mature FDR for usable BPDD teacher signal on train10 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.bpdd_runtime_preflight import summarize_assignment_continuity
from src.rtdetr_fdr_bpdd import BPDD_MODEL_CFG, FDRBPDDDetectionModel


FDR_EPOCH100_SHA256 = (
    "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2"
)
BATCH_LIMIT = 16
OFFICIAL_VAL_OPENED = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _create_only(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite mature probe evidence: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_report(root: Path, payload: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    report = root / "mature-signal.json"
    sums = root / "SHA256SUMS.txt"
    if report.exists() or sums.exists():
        raise FileExistsError(f"mature probe report already exists: {root}")
    encoded = _canonical(dict(payload))
    _create_only(report, encoded)
    digest = hashlib.sha256(encoded).hexdigest().upper()
    _create_only(sums, f"{digest}  {report.name}\n".encode("ascii"))


def _verify_checkpoint(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError("frozen mature FDR checkpoint is missing or unsafe")
    actual = _file_sha256(resolved)
    if actual != FDR_EPOCH100_SHA256:
        raise RuntimeError(
            "checkpoint SHA256 mismatch: "
            f"expected {FDR_EPOCH100_SHA256}, received {actual}"
        )
    return actual


def _checkpoint_model(path: Path) -> torch.nn.Module:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    source: Any = artifact
    if isinstance(artifact, Mapping):
        source = artifact.get("ema")
        if source is None:
            source = artifact.get("model")
    if not isinstance(source, torch.nn.Module):
        raise TypeError("mature FDR checkpoint contains no loadable model or EMA")
    return source.float()


def _load_model(checkpoint: Path, device: torch.device) -> FDRBPDDDetectionModel:
    source = _checkpoint_model(checkpoint)
    model = FDRBPDDDetectionModel(
        BPDD_MODEL_CFG,
        ch=3,
        nc=10,
        verbose=False,
        private_seed=10_000,
    )
    result = model.load_state_dict(source.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict mature FDR state loading was not exact")
    return model.to(device)


def _aggregate(
    probes: Sequence[Mapping[str, float]],
    continuity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(probes) != BATCH_LIMIT or len(continuity) != BATCH_LIMIT:
        raise RuntimeError(f"mature probe requires exactly {BATCH_LIMIT} batches")
    finite = all(math.isfinite(float(value)) for row in probes for value in row.values())
    eligible = [max(int(row["eligible_edges"]), 0) for row in probes]
    total_edges = sum(eligible)

    def weighted(name: str) -> float:
        if total_edges == 0:
            return 0.0
        return float(
            sum(float(row[name]) * weight for row, weight in zip(probes, eligible))
            / total_edges
        )

    matches = sum(int(row["matched_queries"]) for row in probes)
    final_matches = sum(int(row["final_matched_queries"]) for row in continuity)
    continuity_weight = sum(
        int(row["final_matched_queries"]) * max(len(row["layers"]), 1)
        for row in continuity
    )

    def weighted_continuity(name: str) -> float:
        if continuity_weight == 0:
            return 0.0
        return float(
            sum(
                float(row[name])
                * int(row["final_matched_queries"])
                * max(len(row["layers"]), 1)
                for row in continuity
            )
            / continuity_weight
        )

    return {
        "batches": len(probes),
        "matched_queries": matches,
        "final_matched_queries": final_matches,
        "eligible_edges": total_edges,
        "statistics_finite": finite,
        "active_edge_ratio_mean": weighted("active_edge_ratio"),
        "active_edge_ratio_max": max(float(row["active_edge_ratio"]) for row in probes),
        "mean_reliability": weighted("mean_reliability"),
        "mean_teacher_improvement": weighted("mean_teacher_improvement"),
        "mean_teacher_improvement_max": max(
            float(row["mean_teacher_improvement"]) for row in probes
        ),
        "mixture_beats_final_ratio_mean": weighted("mixture_beats_final_ratio"),
        "mean_mixture_advantage_over_final": weighted(
            "mean_mixture_advantage_over_final"
        ),
        "query_support_rate": weighted_continuity("overall_query_support_rate"),
        "same_target_rate": weighted_continuity("overall_same_target_rate"),
    }


def _decide(summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "all_batches_completed": int(summary.get("batches", 0)) == BATCH_LIMIT,
        "matched_normal_queries": int(summary.get("final_matched_queries", 0)) > 0,
        "statistics_finite": bool(summary.get("statistics_finite", False)),
        "mature_teacher_active": float(summary.get("active_edge_ratio_max", 0.0)) > 0.0,
        "teacher_improvement_positive": float(
            summary.get("mean_teacher_improvement_max", 0.0)
        ) > 0.0,
        "mixture_beats_final_on_majority": float(
            summary.get("mixture_beats_final_ratio_mean", 0.0)
        ) > 0.5,
        "mixture_mean_advantage_positive": float(
            summary.get("mean_mixture_advantage_over_final", 0.0)
        ) > 0.0,
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "scientific_failed",
        "screen30_eligible": passed,
        "checks": checks,
    }


def run(args: Namespace) -> dict[str, Any]:
    if args.device != "0":
        raise ValueError("the frozen mature probe permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = _verify_checkpoint(checkpoint)
    report_root = args.report_root.resolve()
    if report_root.exists():
        raise FileExistsError(f"mature probe report root already exists: {report_root}")
    report_root.mkdir(parents=True, exist_ok=False)

    from src.fdr_runtime_preflight import _build_loader, _move_batch

    context = SimpleNamespace(
        dataset_root=args.dataset_root.resolve(),
        report_root=report_root,
    )
    loader, subset_sha = _build_loader(context, augment=False)
    model = _load_model(checkpoint, device)
    model.train()
    probes: list[dict[str, float]] = []
    continuity: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= BATCH_LIMIT:
                break
            batch = _move_batch(raw_batch, device)
            model.loss(batch)
            probes.append(
                {
                    name: float(value.detach().float().cpu())
                    for name, value in model.last_bpdd_statistics.items()
                }
            )
            assignments = model.criterion.normal_assignment_snapshot()
            continuity.append(summarize_assignment_continuity(assignments[1:]))

    summary = _aggregate(probes, continuity)
    decision = _decide(summary)
    checkpoint_after = _file_sha256(checkpoint)
    if checkpoint_after != checkpoint_sha:
        raise RuntimeError("frozen FDR checkpoint changed during the mature probe")
    payload = {
        "format_version": 1,
        **decision,
        "purpose": "bpdd_learnability_and_final_teacher_ablation_probe",
        "data_scope": "fixed_train10_only",
        "official_val_opened": OFFICIAL_VAL_OPENED,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size,
        },
        "runtime": {
            "device": "cuda:0",
            "gpu": torch.cuda.get_device_name(0),
            "batch": 8,
            "imgsz": 640,
            "augment": False,
            "batch_limit": BATCH_LIMIT,
        },
        "fixed_subset_sha256": subset_sha,
        "summary": summary,
        "per_batch": probes,
    }
    _write_report(report_root, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(_parse_args(argv))
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if payload["screen30_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

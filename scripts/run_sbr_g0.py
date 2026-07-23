#!/usr/bin/env python3
"""Strict zero-training SBR-RTDETR runner.

Only operational arguments are exposed.  Scientific constants live in
``src.sbr_g0.FrozenSBRProtocol`` and cannot be overridden from the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image

from src.sbr_artifacts import (
    atomic_write_json,
    atomic_write_jsonl_gz,
    canonical_json_bytes,
    ensure_empty_output,
    environment_info,
    git_provenance,
    load_dataset,
    protocol_signature,
    sha256_file,
    write_checksums,
)
from src.sbr_g0 import FrozenSBRProtocol, assemble_arm, assemble_paired_arms, collect_raw_views
from src.sbr_metrics import evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run audited SBR-RTDETR smoke/G0 evaluation")
    p.add_argument("mode", choices=("s0", "g0-a", "g0-b", "g0-c"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--data", required=True)
    p.add_argument("--dataset-root", help="Operational dataset root override for YAMLs with relocatable paths")
    p.add_argument("--split", default="val")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--smoke-manifest", "--image-list", dest="image_list")
    p.add_argument("--gate", help="Prior G0-A gate JSON for g0-b/g0-c")
    p.add_argument("--evidence", help="Existing evidence directory (defaults to --output)")
    return p


def validate_prior_gate(path: Path | str, expected: Mapping[str, str]) -> dict[str, Any]:
    try:
        gate = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read prior gate: {path}") from exc
    if gate.get("status") != "SBR_G0A_PASS":
        raise ValueError("g0-b/c require status SBR_G0A_PASS")
    for key in ("source_hash", "checkpoint_hash", "dataset_signature", "protocol_hash"):
        if str(gate.get(key, "")) != str(expected.get(key, "")):
            raise ValueError(f"prior gate mismatch: {key}")
    gate_path = Path(path)
    evidence = gate_path.parent
    adjudication = evidence / "independent_adjudication.json"
    if not adjudication.exists():
        raise ValueError("independent adjudication artifact is required")
    try:
        adj = json.loads(adjudication.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("invalid independent adjudication artifact") from exc
    if adj.get("status") != "SBR_G0A_INDEPENDENT_PASS" or adj.get("decision") != "PASS":
        raise ValueError("independent adjudication must explicitly PASS")
    for key in ("source_hash", "checkpoint_hash", "dataset_signature", "protocol_hash"):
        if str(adj.get(key, "")).lower() != str(expected.get(key, "")).lower():
            raise ValueError(f"adjudication provenance mismatch: {key}")
    checksums = evidence / "checksums.sha256"
    if not checksums.exists():
        raise ValueError("checksums.sha256 is required")
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("invalid checksums.sha256 line") from exc
        target = evidence / rel
        if not target.is_file() or sha256_file(target).lower() != digest.lower():
            raise ValueError(f"artifact checksum mismatch: {rel}")
    return gate


def arm_a_effective_gain(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return min(640.0 / float(width), 640.0 / float(height), 1.0)


def evaluate_g0bc_gate(
    c_metrics: Mapping[str, Any],
    d_metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, float], str]:
    """Apply frozen D-C gates; metrics are fractions, deltas are fractions."""
    deltas = {
        "mAP50-95": float(d_metrics.get("mAP50-95", 0.0)) - float(c_metrics.get("mAP50-95", 0.0)),
        "AP-tiny-SBR": float(d_metrics.get("AP-tiny-SBR", 0.0)) - float(c_metrics.get("AP-tiny-SBR", 0.0)),
        "AP-large-SBR": float(d_metrics.get("AP-large-SBR", 0.0)) - float(c_metrics.get("AP-large-SBR", 0.0)),
        "boundary_target_recall": float(diagnostics.get("boundary_target_recall", {}).get("D", 0.0)) - float(diagnostics.get("boundary_target_recall", {}).get("C", 0.0)),
    }
    def reduction(name: str) -> float | None:
        pair = diagnostics.get(name, {})
        c, d = float(pair.get("C", 0.0)), float(pair.get("D", 0.0))
        if c == 0:
            if d > 0:
                return -float("inf")
            return None
        return 1.0 - d / c
    fp_red, dup_red = reduction("internal_boundary_fp"), reduction("duplicate_detections")
    singleton = float(diagnostics.get("singleton_preservation", 0.0))
    passed = (
        all(np.isfinite(v) for v in deltas.values())
        and deltas["mAP50-95"] >= -0.001 - 1e-12
        and deltas["AP-tiny-SBR"] >= -0.001 - 1e-12
        and deltas["AP-large-SBR"] >= -0.002 - 1e-12
        and deltas["boundary_target_recall"] >= -0.002 - 1e-12
        and singleton == 1.0
        and ((fp_red is not None and fp_red >= 0.15) or (dup_red is not None and dup_red >= 0.15))
    )
    deltas.update({"internal_boundary_fp_reduction": fp_red if fp_red is not None else 0.0, "duplicate_reduction": dup_red if dup_red is not None else 0.0, "singleton_preservation": singleton})
    return deltas, "SBR_G0BC_PASS" if passed else "SBR_G0BC_FAIL"


def evaluate_g0a_gate(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, float], str]:
    """Apply the frozen C-A effectiveness gate (metrics are fractions)."""
    if "A" not in metrics or "C" not in metrics:
        raise ValueError("A and C metrics are required for G0-A gate")
    keys = ("AP-tiny-SBR", "mAP50-95", "AP75", "AP-large-SBR")
    deltas = {k: float(metrics["C"].get(k, 0.0)) - float(metrics["A"].get(k, 0.0)) for k in keys}
    tiny_recall = float(metrics["C"].get("tiny_recall", 0.0)) - float(metrics["A"].get("tiny_recall", 0.0))
    deltas["tiny_recall"] = tiny_recall
    finite = all(np.isfinite(v) for v in deltas.values())
    passed = finite and deltas["AP-tiny-SBR"] >= 0.01 and deltas["mAP50-95"] >= 0.003 and tiny_recall >= 0.02 and deltas["AP75"] >= -0.002 and deltas["AP-large-SBR"] >= -0.005
    return deltas, "SBR_G0A_PASS" if passed else "SBR_G0A_FAIL"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _load_predictor(checkpoint: Path, device: str):
    try:
        from ultralytics import RTDETR
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for SBR inference") from exc
    model = RTDETR(str(checkpoint))

    def predict(square: np.ndarray, imgsz: int):
        result = model.predict(
            source=square,
            imgsz=imgsz,
            conf=FrozenSBRProtocol().conf,
            max_det=FrozenSBRProtocol().max_det,
            device=device,
            augment=False,
            verbose=False,
        )
        return result

    return predict


def _rows_for_metrics(arm_result: Mapping[str, Any], image: Mapping[str, Any]) -> dict[str, Any]:
    preds = list(arm_result.get("predictions", ()))
    return {
        "pred_boxes": [getattr(p, "global_xyxy", getattr(p, "box", ())) for p in preds],
        "pred_scores": [float(getattr(p, "score", 0.0)) for p in preds],
        "pred_classes": [int(getattr(p, "class_id", 0)) for p in preds],
        "pred_source": [int(getattr(p, "source_order", 0)) for p in preds],
        "pred_query": [int(getattr(p, "query_index", i)) for i, p in enumerate(preds)],
        "gt_boxes": image["gt_boxes"],
        "gt_classes": image["gt_classes"],
        "ignore_boxes": image["ignore_boxes"],
        "effective_gain": arm_a_effective_gain(int(image["width"]), int(image["height"])),
    }


def run(args: argparse.Namespace) -> int:
    protocol = FrozenSBRProtocol()
    output = ensure_empty_output(args.output)
    if args.mode in {"g0-b", "g0-c"} and (
        not args.gate or not args.evidence or Path(args.evidence).resolve() != Path(args.gate).resolve().parent
    ):
        raise ValueError("--evidence must match the directory containing --gate")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    if args.checkpoint_sha256 and checkpoint_hash.lower() != args.checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA256 mismatch")
    dataset = load_dataset(args.data, split=args.split, root_override=args.dataset_root)
    if args.mode == "g0-a" and dataset["image_count"] != 548:
        raise ValueError("g0-a requires exactly 548 images")
    if args.mode == "s0":
        selected = dataset["image_list"]
        if args.image_list:
            p = Path(args.image_list)
            selected = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not 8 <= len(selected) <= 16:
            raise ValueError("s0 requires a deterministic 8-16 image list")
        wanted = set(selected)
        dataset["images"] = [r for r in dataset["images"] if r["relative_path"] in wanted]
        dataset["image_list"] = [r["relative_path"] for r in dataset["images"]]
        dataset["image_count"] = len(dataset["images"])
    source = git_provenance(Path.cwd())
    if not source.get("clean_tracked", False) or source.get("untracked", False):
        raise ValueError("source worktree must be clean with no untracked files before inference")
    source_hash = source.get("commit", "")
    proto_hash = protocol_signature(protocol.__dict__)
    expected = {"source_hash": source_hash, "checkpoint_hash": checkpoint_hash, "dataset_signature": dataset["dataset_signature"], "protocol_hash": proto_hash}
    if args.mode in {"g0-b", "g0-c"}:
        if not args.gate:
            raise ValueError("g0-b/c require --gate")
        if not args.evidence or Path(args.evidence).resolve() != Path(args.gate).resolve().parent:
            raise ValueError("--evidence must match the directory containing --gate")
        validate_prior_gate(args.gate, expected)
        raise ValueError("g0-b/c consume adjudicated G0-A evidence; independent adjudicator required before execution")
    predict = _load_predictor(checkpoint, args.device)
    # Cache inference by exact square bytes so shared views (B/C/D and full
    # views) execute once while each arm receives independent raw metadata.
    raw_rows, arm_rows = [], {a: [] for a in "ABCDEF"}
    metric_rows = {a: [] for a in "ABCDEF"}
    started = time.time()
    for image in dataset["images"]:
        predict_cache: dict[tuple[int, bytes], Any] = {}
        def cached_predict(square: np.ndarray, imgsz: int):
            key = (int(imgsz), np.ascontiguousarray(square).tobytes())
            if key not in predict_cache:
                predict_cache[key] = predict(square, imgsz)
            return predict_cache[key]
        with Image.open(image["path"]) as im:
            array = np.asarray(im.convert("RGB"))
        image_id = image["relative_path"]
        raw_by_arm = {}
        for arm in "ABCEF":
            raw, manifest = collect_raw_views(array, arm, cached_predict, image_id=image_id, return_manifest=True)
            raw_by_arm[arm] = (raw, manifest)
            raw_rows.extend([dict(r.to_dict(), view_manifest=manifest) for r in raw])
        assembled = {a: assemble_arm(raw_by_arm[a][0], a, width=image["width"], height=image["height"], view_manifest=raw_by_arm[a][1]) for a in "ABEF"}
        assembled.update(assemble_paired_arms(raw_by_arm["C"][0], width=image["width"], height=image["height"], view_manifest=raw_by_arm["C"][1]))
        for arm, result in assembled.items():
            arm_rows[arm].append({"image_id": image_id, "predictions": _jsonable(result["predictions"]), "records": result["records"]})
            metric_rows[arm].append(_rows_for_metrics(result, image))
    metrics = {arm: evaluate_dataset(rows) for arm, rows in metric_rows.items()}
    deltas, gate_status = evaluate_g0a_gate(metrics)
    atomic_write_json(output / "g0_manifest.json", {"mode": args.mode, "source": source, "source_hash": source_hash, "checkpoint": str(checkpoint), "checkpoint_hash": checkpoint_hash, "dataset_signature": dataset["dataset_signature"], "image_count": dataset["image_count"], "image_list": dataset["image_list"], "protocol": protocol.__dict__, "protocol_hash": proto_hash, "environment": environment_info()})
    atomic_write_jsonl_gz(output / "raw_views.jsonl.gz", raw_rows)
    atomic_write_jsonl_gz(output / "arm_predictions.jsonl.gz", [x for rows in arm_rows.values() for x in rows])
    atomic_write_json(output / "g0_metrics.json", metrics)
    atomic_write_json(output / "g0_deltas.json", deltas)
    adjudication_status = "NOT_RUN"
    status = "SBR_S0_COMPLETE" if args.mode == "s0" else ("SBR_G0A_PASS" if gate_status == "SBR_G0A_PASS" and adjudication_status != "NOT_RUN" else "SBR_G0A_FAIL")
    atomic_write_json(output / "g0_gate.json", {**expected, "status": status})
    atomic_write_json(output / "runtime.json", {"seconds": time.time() - started, "device": args.device, "workers": args.workers})
    atomic_write_json(output / "independent_adjudication.json", {"status": "NOT_RUN"})
    atomic_write_json(output / "README.md", {"mode": args.mode, "status": status, "note": "Metrics are evidence only; independent adjudicator pending."})
    files = [p for p in output.iterdir() if p.is_file() and p.name != "checksums.sha256"]
    write_checksums(output / "checksums.sha256", files, root=output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return run(args)
    except Exception as exc:
        print(f"SBR_G0_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

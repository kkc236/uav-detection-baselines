"""Evaluate the completed FDR/SCADS screen pair and write one immutable Gate report."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_math import cxcywh_to_xyxy  # noqa: E402
from src.iber_evaluation import compute_detection_metrics  # noqa: E402
from src.lpr_protocol import CATEGORY_NAMES  # noqa: E402
from src.scads import build_support_projects, continuous_edge_offsets  # noqa: E402
from src.scads_evaluation import (  # noqa: E402
    gate_decision,
    metric_delta,
    summarize_representation,
)


EXPECTED_EPOCHS = 30
EVALUATION = {
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "conf": 0.001,
    "max_det": 300,
    "nms": False,
    "half": False,
    "seed": 0,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} is not an object: {path}")
        rows.append(value)
    return rows


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def load_arm(run: Path, variant: str) -> dict[str, Any]:
    run = Path(run).resolve()
    manifest = read_json(run / "scads-run.json")
    evidence = read_jsonl(run / "scads-epochs.jsonl")
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        results = list(csv.DictReader(stream))
    identity = manifest.get("run_identity", {})
    checks = {
        "manifest_variant": identity.get("variant") == variant,
        "manifest_stage": identity.get("stage") == "screen",
        "manifest_seed": identity.get("seed") == 0,
        "manifest_cutoff": manifest.get("screen_cutoff_epoch") == EXPECTED_EPOCHS,
        "evidence_30": len(evidence) == EXPECTED_EPOCHS,
        "results_30": len(results) == EXPECTED_EPOCHS,
        "continuous_epochs": [int(row.get("completed_epoch", -1)) for row in evidence]
        == list(range(1, EXPECTED_EPOCHS + 1)),
        "row_authority": all(
            row.get("variant") == variant
            and row.get("stage") == "screen"
            and row.get("run_id") == identity.get("run_id")
            for row in evidence
        ),
    }
    for row in evidence:
        for name in ("map", "map50", "map75", "precision", "recall"):
            _finite(row[name], f"{variant}.{name}")
    if not all(checks.values()):
        raise ValueError(f"{variant} arm evidence failed: {checks}")
    checkpoint = run / "weights" / "epoch29.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final verified checkpoint is unavailable: {checkpoint}")
    return {
        "run": str(run),
        "manifest": manifest,
        "evidence": evidence,
        "results": results,
        "checkpoint": checkpoint,
        "checkpoint_sha256": file_sha256(checkpoint),
        "checks": checks,
    }


def _window(rows: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> dict[str, float]:
    fields = ("map", "map50", "map75", "precision", "recall")
    return {
        name: statistics.fmean(_finite(rows[index][name], name) for index in indices)
        for name in fields
    }


def training_comparison(fdr: Mapping[str, Any], scads: Mapping[str, Any]) -> dict[str, Any]:
    fdr_rows, scads_rows = fdr["evidence"], scads["evidence"]
    windows = {
        "final": [EXPECTED_EPOCHS - 1],
        "tail3": list(range(EXPECTED_EPOCHS - 3, EXPECTED_EPOCHS)),
    }
    result = {}
    for name, indices in windows.items():
        left, right = _window(fdr_rows, indices), _window(scads_rows, indices)
        result[name] = {
            "epochs": [index + 1 for index in indices],
            "fdr": left,
            "scads": right,
            "delta": {field: right[field] - left[field] for field in left},
        }
    result["best"] = {}
    for label, rows in (("fdr", fdr_rows), ("scads", scads_rows)):
        row = max(rows, key=lambda item: _finite(item["map"], "map"))
        result["best"][label] = {
            "epoch": int(row["completed_epoch"]),
            **_window([row], [0]),
        }
    return result


def paired_authority(fdr: Mapping[str, Any], scads: Mapping[str, Any]) -> bool:
    left, right = fdr["manifest"], scads["manifest"]
    shared = ("format_version", "protocol_sha256", "source", "initial_state", "data", "screen_cutoff_epoch")
    left_id, right_id = left["run_identity"], right["run_identity"]
    identity_shared = ("source_sha256", "protocol_sha256", "stage", "seed")
    return (
        all(left.get(name) == right.get(name) for name in shared)
        and all(left_id.get(name) == right_id.get(name) for name in identity_shared)
        and left_id.get("variant") == "fdr"
        and right_id.get("variant") == "scads"
        and left_id.get("run_id") != right_id.get("run_id")
    )


def publication_checks(ledger_path: Path, arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ledger = read_jsonl(ledger_path)
    detail = {}
    for arm in arms:
        identity = arm["manifest"]["run_identity"]
        rows = [row for row in ledger if row.get("run_id") == identity["run_id"]]
        epochs = sorted(int(row.get("completed_epoch", -1)) for row in rows)
        detail[identity["variant"]] = {
            "verified_rows": len(rows),
            "continuous": epochs == list(range(1, EXPECTED_EPOCHS + 1)),
            "all_verified": all(row.get("status") == "published-verified" for row in rows),
        }
    return {"complete": all(all(item.values()) for item in detail.values()), "arms": detail}


def _data_payload(dataset_root: Path) -> dict[str, Any]:
    return {
        "path": str(dataset_root.resolve()),
        "train": str((dataset_root / "images" / "train").resolve()),
        "val": str((dataset_root / "images" / "val").resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": len(CATEGORY_NAMES),
        "channels": 3,
    }


def build_loader(
    dataset_root: Path,
    image_source: Path,
    checkpoint: Path,
    *,
    device: torch.device,
    save_dir: Path,
):
    from ultralytics.models.rtdetr.val import RTDETRValidator

    data = _data_payload(dataset_root)
    validator = RTDETRValidator(
        save_dir=save_dir,
        args={
            "model": str(checkpoint),
            "data": data,
            "task": "detect",
            "mode": "val",
            "split": "val",
            **EVALUATION,
            "device": "0",
            "cache": False,
            "rect": False,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
        },
    )
    validator.data = data
    validator.device = device
    return validator.get_dataloader(str(image_source), EVALUATION["batch"]), validator


def load_checkpoint_model(path: Path, *, device: torch.device):
    import src.rtdetr_fdr  # noqa: F401
    import src.rtdetr_scads  # noqa: F401

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    model = artifact.get("ema")
    if model is None:
        model = artifact.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"checkpoint has no model: {path}")
    return model.float().to(device)


def seed_evaluation(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_targets(batch: Mapping[str, Any], image_index: int) -> dict[str, torch.Tensor]:
    mask = batch["batch_idx"].view(-1).long() == image_index
    return {
        "boxes": batch["bboxes"][mask].detach().float().cpu(),
        "classes": batch["cls"][mask].view(-1).detach().long().cpu(),
    }


def _batch_predictions(predictions: torch.Tensor, image_index: int) -> dict[str, torch.Tensor]:
    value = predictions[image_index].detach().float().cpu()
    value = value[value[:, 4] > EVALUATION["conf"]]
    return {"boxes": value[:, :4], "scores": value[:, 4], "classes": value[:, 5].long()}


def exact_metrics(
    checkpoint: Path,
    dataset_root: Path,
    *,
    device: torch.device,
    save_dir: Path,
) -> dict[str, float]:
    seed_evaluation()
    model = load_checkpoint_model(checkpoint, device=device).eval()
    model.requires_grad_(False)
    loader, validator = build_loader(
        dataset_root,
        dataset_root / "images" / "val",
        checkpoint,
        device=device,
        save_dir=save_dir,
    )
    predictions, targets = [], []
    with torch.inference_mode():
        for raw in loader:
            batch = validator.preprocess(raw)
            output = model(batch["img"])
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[-1] != 6:
                raise RuntimeError(f"RT-DETR evaluation output contract changed: {type(output)}")
            for index in range(output.shape[0]):
                predictions.append(_batch_predictions(output, index))
                targets.append(_batch_targets(batch, index))
    if len(targets) != 548:
        raise RuntimeError(f"exact evaluation processed {len(targets)} images")
    metrics = compute_detection_metrics(predictions, targets, image_size=EVALUATION["imgsz"])
    del model, loader, validator, predictions, targets
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _target_keys(batch: Mapping[str, Any]) -> list[str]:
    batch_indices = batch["batch_idx"].view(-1).long().detach().cpu().tolist()
    files = [Path(value).name for value in batch["im_file"]]
    seen: dict[int, int] = {}
    keys = []
    for image_index in batch_indices:
        local = seen.get(image_index, 0)
        seen[image_index] = local + 1
        keys.append(f"{files[image_index]}:{local}")
    return keys


def mechanism_records(
    checkpoint: Path,
    dataset_root: Path,
    screen_list: Path,
    *,
    variant: str,
    device: torch.device,
    save_dir: Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor | None]:
    model = load_checkpoint_model(checkpoint, device=device).train()
    model.requires_grad_(False)
    loader, validator = build_loader(
        dataset_root,
        screen_list,
        checkpoint,
        device=device,
        save_dir=save_dir,
    )
    records: dict[str, dict[str, torch.Tensor]] = {}
    project_bank = None
    for batch_number, raw in enumerate(loader):
        seed_evaluation(EVALUATION["seed"] + batch_number)
        batch = validator.preprocess(raw)
        with torch.no_grad():
            total, _displayed = model.loss(batch)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite {variant} mechanism loss")
        criterion = model.criterion
        assignments = criterion._to_layer_order(criterion._recorded_assignments.get("", []))
        if len(assignments) < 2:
            raise RuntimeError(f"{variant} mechanism assignments are missing")
        predicted_index, target_index = criterion._get_index(assignments[-1])
        evidence = model.last_fdr_evidence
        if evidence is None:
            raise RuntimeError(f"{variant} mechanism evidence is missing")
        references = evidence.pre_boxes[predicted_index].detach()
        boxes = batch["bboxes"][target_index].detach()
        offsets = continuous_edge_offsets(references, cxcywh_to_xyxy(boxes))
        keys = _target_keys(batch)
        for row, target in enumerate(target_index.detach().cpu().tolist()):
            key = keys[target]
            if key in records:
                raise ValueError(f"duplicate matched target key: {key}")
            record = {
                "offsets": offsets[row].float().cpu(),
                "box": boxes[row].float().cpu(),
            }
            if variant == "scads":
                record.update(
                    {
                        "project": evidence.support_projects[predicted_index][row].detach().float().cpu(),
                        "route_weights": evidence.support_weights[predicted_index][row].detach().float().cpu(),
                    }
                )
                current_bank = criterion.support_project_bank.detach().float().cpu()
                if project_bank is None:
                    project_bank = current_bank
                else:
                    torch.testing.assert_close(project_bank, current_bank, rtol=0, atol=0)
            records[key] = record
    del model, loader, validator
    gc.collect()
    torch.cuda.empty_cache()
    return records, project_bank


def representation_report(
    fdr_checkpoint: Path,
    scads_checkpoint: Path,
    dataset_root: Path,
    screen_list: Path,
    *,
    device: torch.device,
    save_dir: Path,
) -> dict[str, Any]:
    fdr, _ = mechanism_records(
        fdr_checkpoint,
        dataset_root,
        screen_list,
        variant="fdr",
        device=device,
        save_dir=save_dir / "fdr",
    )
    scads, project_bank = mechanism_records(
        scads_checkpoint,
        dataset_root,
        screen_list,
        variant="scads",
        device=device,
        save_dir=save_dir / "scads",
    )
    common = sorted(set(fdr) & set(scads))
    if not common or project_bank is None:
        raise RuntimeError("FDR/SCADS have no common mechanism target records")
    dropped = {"fdr_only": len(set(fdr) - set(scads)), "scads_only": len(set(scads) - set(fdr))}
    report = summarize_representation(
        fdr_offsets=torch.stack([fdr[key]["offsets"] for key in common]),
        scads_offsets=torch.stack([scads[key]["offsets"] for key in common]),
        target_boxes=torch.stack([scads[key]["box"] for key in common]),
        base_project=build_support_projects((0.25, 0.5, 1.0))[1],
        scads_projects=torch.stack([scads[key]["project"] for key in common]),
        route_weights=torch.stack([scads[key]["route_weights"] for key in common]),
        project_bank=project_bank,
        support_ups=(0.25, 0.5, 1.0),
        margin_ratio=0.02,
    )
    return {"matched_key_policy": "intersection-by-image-and-target-order", "dropped": dropped, **report}


def latency_report(checkpoint: Path, *, device: torch.device) -> dict[str, Any]:
    model = load_checkpoint_model(checkpoint, device=device).eval()
    model.requires_grad_(False)
    sample = torch.zeros(1, 3, EVALUATION["imgsz"], EVALUATION["imgsz"], device=device)
    with torch.inference_mode():
        for _ in range(20):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(sample)
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(100):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(sample)
            end.record()
            torch.cuda.synchronize()
            elapsed.append(float(start.elapsed_time(end)))
    elapsed.sort()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "batch": 1,
        "imgsz": EVALUATION["imgsz"],
        "amp": True,
        "warmup": 20,
        "repeats": 100,
        "latency_ms_median": statistics.median(elapsed),
        "latency_ms_p95": elapsed[94],
        "fps_from_median": 1000.0 / statistics.median(elapsed),
        "parameters": parameters,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    del model, sample
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    seed_evaluation()
    device = torch.device("cuda:0")
    fdr = load_arm(args.fdr_run, "fdr")
    scads = load_arm(args.scads_run, "scads")
    authority = paired_authority(fdr, scads)
    publication = publication_checks(args.ledger, (fdr, scads))
    training = training_comparison(fdr, scads)
    exact = {
        "fdr": exact_metrics(
            fdr["checkpoint"], args.dataset_root, device=device, save_dir=args.work_dir / "exact-fdr"
        ),
        "scads": exact_metrics(
            scads["checkpoint"], args.dataset_root, device=device, save_dir=args.work_dir / "exact-scads"
        ),
    }
    exact["delta"] = metric_delta(exact["fdr"], exact["scads"])
    representation = representation_report(
        fdr["checkpoint"],
        scads["checkpoint"],
        args.dataset_root,
        args.screen_list,
        device=device,
        save_dir=args.work_dir / "representation",
    )
    efficiency = {
        "fdr": latency_report(fdr["checkpoint"], device=device),
        "scads": latency_report(scads["checkpoint"], device=device),
    }
    efficiency["delta"] = {
        name: float(efficiency["scads"][name]) - float(efficiency["fdr"][name])
        for name in ("latency_ms_median", "latency_ms_p95", "parameters", "checkpoint_bytes")
    }
    engineering_complete = authority and publication["complete"]
    gate = gate_decision(
        final_delta=training["final"]["delta"],
        tail3_delta=training["tail3"]["delta"],
        exact_delta=exact["delta"],
        representation=representation,
        engineering_complete=engineering_complete,
    )
    report = {
        "format_version": 1,
        "gate_name": "SCADS-FDR-paired-screen-seed0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation": EVALUATION,
        "checkpoints": {
            "fdr": {"path": str(fdr["checkpoint"]), "sha256": fdr["checkpoint_sha256"]},
            "scads": {"path": str(scads["checkpoint"]), "sha256": scads["checkpoint_sha256"]},
        },
        "engineering": {
            "complete": engineering_complete,
            "paired_authority": authority,
            "publication": publication,
            "arm_checks": {"fdr": fdr["checks"], "scads": scads["checks"]},
        },
        "training_metrics": training,
        "exact_metrics": exact,
        "representation": representation,
        "efficiency": efficiency,
        "gate": gate,
        "formal100_eligible": gate["passed"],
    }
    write_create_only(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-run", type=Path, required=True)
    parser.add_argument("--scads-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--screen-list", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("fdr_run", "scads_run", "dataset_root", "screen_list", "ledger", "work_dir", "output"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    report = execute(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

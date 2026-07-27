"""Calibrate three SR-PEG thresholds on the sealed train10 holdout."""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.evaluate_gcqf_g0 import (
    _dataset_key,
    _metric_row,
    _stringify_mapping_keys,
    load_gcqf_module,
    metric_deltas,
)
from scripts.train_gcqf_g0 import MODULE_ARTIFACT_SCHEMA
from src.gcqf_cache import SRPEG_CACHE_SCHEMA_VERSION, VerifiedEvidenceCache
from src.gcqf_routing import route_gcqf_record
from src.gcqf_training import collate_evidence_records
from src.sbr_artifacts import atomic_write_json, load_dataset
from src.sbr_metrics import evaluate_dataset
from src.sr_peg_routing import SRPEGThresholds, route_sr_peg_record


MEDIUM_BUDGET = -0.002
LARGE_BUDGET = -0.005


def threshold_grid() -> tuple[tuple[float, float, float], ...]:
    return tuple(itertools.product((0.4, 0.5, 0.6), repeat=3))


def select_calibration(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best budget-safe row with one deterministic tie break."""

    eligible = [
        row
        for row in rows
        if float(row["deltas"].get("AP-medium-SBR", -1.0))
        >= MEDIUM_BUDGET
        and float(row["deltas"].get("AP-large-SBR", -1.0))
        >= LARGE_BUDGET
    ]
    if not eligible:
        raise ValueError("no calibration threshold satisfies scale budgets")

    def key(row: dict[str, Any]) -> tuple[float, ...]:
        delta = row["deltas"]
        thresholds = row["thresholds"]
        return (
            float(delta.get("mAP50-95", -1.0)),
            float(delta.get("AP-tiny-SBR", -1.0)),
            float(delta.get("tiny_recall", -1.0)),
            -float(thresholds["tiny_utility"]),
            float(thresholds["non_tiny_risk"]),
            -float(thresholds["global_retain"]),
        )

    return max(eligible, key=key)


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate SR-PEG only on the 129-image train10 holdout."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    return parser


def calibrate(args: argparse.Namespace) -> Path:
    if args.device != "0" or args.batch != 8:
        raise ValueError("SR-PEG calibration protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("SR-PEG calibration requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    cache = VerifiedEvidenceCache(args.cache)
    if (
        cache.manifest["schema_version"] != SRPEG_CACHE_SCHEMA_VERSION
        or cache.manifest["record_count"] != 647
    ):
        raise ValueError("calibration requires the sealed 647-record v2 cache")
    all_records = list(cache.iter_records())
    record_by_id = {record.image_id: record for record in all_records}
    if len(record_by_id) != 647:
        raise ValueError("train10 cache image identities are not unique")

    module_payload = torch.load(
        Path(args.module).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(module_payload, dict)
        or module_payload.get("schema_version") != MODULE_ARTIFACT_SCHEMA
    ):
        raise ValueError("module artifact schema mismatch")
    calibration_ids = module_payload.get("calibration_image_ids")
    train_ids = module_payload.get("train_image_ids")
    if (
        not isinstance(calibration_ids, list)
        or len(calibration_ids) != 129
        or len(set(calibration_ids)) != 129
        or not isinstance(train_ids, list)
        or len(train_ids) != 518
        or set(train_ids) & set(calibration_ids)
        or set(train_ids) | set(calibration_ids) != set(record_by_id)
    ):
        raise ValueError("module artifact seed0 split identities drift")
    records = [record_by_id[image_id] for image_id in calibration_ids]

    dataset = load_dataset(args.data, split="train")
    image_by_id = {
        image["relative_path"]: image for image in dataset["images"]
    }
    images: dict[str, Mapping[str, Any]] = {}
    for record in records:
        key = _dataset_key(record.image_id)
        if key not in image_by_id:
            raise ValueError(f"calibration image missing from dataset: {key}")
        images[record.image_id] = image_by_id[key]

    module = load_gcqf_module(
        args.module,
        device=torch.device("cuda:0"),
    )
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    loader = torch.utils.data.DataLoader(
        records,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda values: values,
    )
    for values in loader:
        batch = collate_evidence_records(
            values,
            require_sr_peg_targets=True,
        ).to("cuda:0")
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            prediction = module(
                batch.global_evidence,
                batch.local_evidence,
                batch.geometry,
                anchor_mask=batch.anchor_mask,
                residual_enabled=True,
            )
        for index, image_id in enumerate(batch.image_ids):
            outputs[image_id] = {
                "score_residual": prediction.score_residual[
                    index : index + 1
                ].detach().cpu(),
                "tiny_utility": prediction.tiny_utility_logits[
                    index : index + 1
                ].sigmoid().detach().cpu(),
                "non_tiny_risk": prediction.non_tiny_risk_logits[
                    index : index + 1
                ].sigmoid().detach().cpu(),
                "global_retain": prediction.global_retain_logits[
                    index : index + 1
                ].sigmoid().detach().cpu(),
            }
    if set(outputs) != set(calibration_ids):
        raise RuntimeError("calibration inference coverage drift")

    global_rows = []
    for record in records:
        anchor = route_gcqf_record(record, score_residual=None)
        global_rows.append(_metric_row(images[record.image_id], anchor.control))
    global_metrics = evaluate_dataset(global_rows)

    candidate_rows = []
    for tiny_threshold, risk_threshold, retain_threshold in threshold_grid():
        thresholds = SRPEGThresholds(
            tiny_threshold,
            risk_threshold,
            retain_threshold,
        )
        method_rows = []
        protected_exact = True
        for record in records:
            prediction = outputs[record.image_id]
            routed = route_sr_peg_record(
                record,
                **prediction,
                thresholds=thresholds,
                residual_enabled=True,
            )
            method_rows.append(
                _metric_row(images[record.image_id], routed.output)
            )
            protected_exact = protected_exact and bool(
                routed.invariants.get("protected_identity_exact")
                and routed.invariants.get("max_det_respected")
            )
        metrics = evaluate_dataset(method_rows)
        deltas = metric_deltas(global_metrics, metrics)
        candidate_rows.append(
            {
                "thresholds": {
                    "tiny_utility": tiny_threshold,
                    "non_tiny_risk": risk_threshold,
                    "global_retain": retain_threshold,
                },
                "metrics": metrics,
                "deltas": deltas,
                "protected_global_exact": protected_exact,
            }
        )
    if len(candidate_rows) != 27:
        raise RuntimeError("calibration grid coverage drift")
    selected = select_calibration(candidate_rows)
    result = {
        "schema_version": "gcte-sr-peg-calibration/v1",
        "cache_manifest_sha256": _sha256_file(args.cache),
        "module_sha256": _sha256_file(args.module),
        "source_commit": module_payload.get("source_commit"),
        "image_count": len(calibration_ids),
        "image_ids": calibration_ids,
        "global_metrics": global_metrics,
        "budgets": {
            "AP-medium-SBR": MEDIUM_BUDGET,
            "AP-large-SBR": LARGE_BUDGET,
        },
        "candidates": candidate_rows,
        "selected": selected,
        "selected_thresholds": selected["thresholds"],
    }
    atomic_write_json(output, _stringify_mapping_keys(result))
    print(
        "SRPEG_CALIBRATION_COMPLETE "
        f"thresholds={selected['thresholds']} output={output}",
        flush=True,
    )
    return output


def main() -> None:
    print(calibrate(build_parser().parse_args()))


if __name__ == "__main__":
    main()

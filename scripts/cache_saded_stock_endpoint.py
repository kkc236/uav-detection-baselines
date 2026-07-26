#!/usr/bin/env python3
"""Seal one fresh-stock five-view cache without loading GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_saded_endpoint import (  # noqa: E402
    _predict_image_views,
    _raw_detection,
    _view_manifest_is_complete,
)
from src.saded_stage import prediction_payload  # noqa: E402
from src.saded_stock_evaluation_protocol import (  # noqa: E402
    postprocess_source_state,
    reject_forbidden,
    validate_evaluation_protocol,
    verify_image_authority,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl_gz,
    sha256_file,
    write_checksums,
)
from src.sbr_g0 import FrozenSBRProtocol, assemble_paired_arms  # noqa: E402


CACHE_SCHEMA = "saded-fresh-stock-cache/v1"
CACHE_ARTIFACTS = {
    "cache_manifest.json",
    "predictions.jsonl.gz",
    "raw_views.jsonl.gz",
    "view_manifests.jsonl.gz",
    "cache_invariants.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal fresh-stock SADED five-view cache."
    )
    parser.add_argument(
        "--evaluation-protocol",
        required=True,
        type=Path,
    )
    parser.add_argument("--device", default="0")
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_endpoint(args: argparse.Namespace) -> Path:
    reject_forbidden(vars(args))
    if str(args.device) != "0":
        raise ValueError("fresh SADED cache requires device 0")
    protocol_path = args.evaluation_protocol.resolve()
    protocol = validate_evaluation_protocol(
        protocol_path,
        repo_root=REPO_ROOT,
        verify_images=True,
    )
    output = Path(protocol["outputs"]["cache"]).resolve()
    anchor = output.parent / "cache_anchor.json"
    if output.exists() or anchor.exists():
        raise FileExistsError("fresh SADED cache target exists")
    image_list_path = Path(
        protocol["protocol_artifacts"]["image_list"]["path"]
    ).resolve()
    authority_path = Path(
        protocol["protocol_artifacts"]["image_authority"]["path"]
    ).resolve()
    image_list = _read_json(image_list_path)
    authority = _read_json(authority_path)
    checkpoint = Path(
        protocol["training"]["checkpoint"]["path"]
    ).resolve()
    source_before = postprocess_source_state(REPO_ROOT)
    input_before = {
        "evaluation_protocol": sha256_file(protocol_path),
        "training_protocol": protocol["training"]["protocol"]["sha256"],
        "training_summary": protocol["training"]["summary"]["sha256"],
        "checkpoint": sha256_file(checkpoint),
        "image_list": sha256_file(image_list_path),
        "image_authority": sha256_file(authority_path),
    }

    try:
        from ultralytics import RTDETR
    except Exception as error:
        raise RuntimeError("Ultralytics RTDETR is required") from error
    import src.rtdetr_tascv  # noqa: F401

    model = RTDETR(str(checkpoint))
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    started = time.time()
    image_root = Path(protocol["dataset"]["image_root"]).resolve()
    for index, image_id in enumerate(image_list, start=1):
        image_path = (image_root / image_id).resolve()
        if image_root not in image_path.parents or not image_path.is_file():
            raise ValueError(f"fresh SADED val image missing: {image_id}")
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        height, width = image.shape[:2]
        raw, view_manifest = _predict_image_views(
            model,
            image,
            image_id=image_id,
            device=str(args.device),
        )
        if not _view_manifest_is_complete(view_manifest):
            raise ValueError("fresh SADED exact five-view execution drift")
        view_rows.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "view_manifest": view_manifest,
            }
        )
        raw_rows.extend(record.to_dict() for record in raw)
        full = tuple(
            _raw_detection(record)
            for record in raw
            if record.source_order == 0
        )
        local = assemble_paired_arms(
            raw,
            width=width,
            height=height,
            view_manifest=view_manifest,
        )["C"]["predictions"]
        rows.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "full_predictions": [
                    prediction_payload(item) for item in full
                ],
                "local_fused_predictions": [
                    prediction_payload(item) for item in local
                ],
            }
        )
        if index % 25 == 0 or index == len(image_list):
            print(
                f"SADED_FRESH_CACHE_PROGRESS {index}/{len(image_list)}",
                flush=True,
            )
    verify_image_authority(
        authority,
        image_list,
        verify_bytes=True,
    )
    protocol_after = validate_evaluation_protocol(
        protocol_path,
        repo_root=REPO_ROOT,
        verify_images=False,
    )
    source_after = postprocess_source_state(REPO_ROOT)
    input_after = {
        "evaluation_protocol": sha256_file(protocol_path),
        "training_protocol": protocol_after["training"]["protocol"][
            "sha256"
        ],
        "training_summary": protocol_after["training"]["summary"]["sha256"],
        "checkpoint": sha256_file(checkpoint),
        "image_list": sha256_file(image_list_path),
        "image_authority": sha256_file(authority_path),
    }
    invariants = {
        "image_count": len(rows),
        "expected_image_count": 548,
        "image_order_exact": [row["image_id"] for row in rows]
        == image_list,
        "five_views_per_image_executed": all(
            _view_manifest_is_complete(row["view_manifest"])
            for row in view_rows
        ),
        "prediction_max_det_respected": all(
            len(row["full_predictions"]) <= FrozenSBRProtocol().max_det
            and len(row["local_fused_predictions"])
            <= FrozenSBRProtocol().max_det
            for row in rows
        ),
        "source_unchanged": source_before == source_after,
        "inputs_unchanged": input_before == input_after,
        "gt_module_absent": "src.sbr_metrics" not in sys.modules,
    }
    invariants["passed"] = (
        invariants["image_count"] == invariants["expected_image_count"]
        and all(
            value is True
            for key, value in invariants.items()
            if key not in {"image_count", "expected_image_count"}
        )
    )
    if not invariants["passed"]:
        raise ValueError("fresh SADED cache invariants failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".cache-staging-",
            dir=output.parent,
        )
    )
    try:
        predictions_path = atomic_write_jsonl_gz(
            staging / "predictions.jsonl.gz",
            rows,
        )
        raw_path = atomic_write_jsonl_gz(
            staging / "raw_views.jsonl.gz",
            raw_rows,
        )
        views_path = atomic_write_jsonl_gz(
            staging / "view_manifests.jsonl.gz",
            view_rows,
        )
        invariants_path = atomic_write_json(
            staging / "cache_invariants.json",
            invariants,
        )
        manifest_path = atomic_write_json(
            staging / "cache_manifest.json",
            {
                "schema_version": CACHE_SCHEMA,
                "evaluation_protocol": {
                    "path": protocol_path.as_posix(),
                    "sha256": sha256_file(protocol_path),
                },
                "source": source_after,
                "training": protocol["training"],
                "checkpoint": protocol["training"]["checkpoint"],
                "dataset": protocol["dataset"],
                "image_list_sha256": sha256_file(image_list_path),
                "image_authority_sha256": sha256_file(authority_path),
                "route_contract": protocol["route_contract"],
                "artifacts": {
                    "predictions_sha256": sha256_file(predictions_path),
                    "raw_views_sha256": sha256_file(raw_path),
                    "view_manifests_sha256": sha256_file(views_path),
                    "invariants_sha256": sha256_file(invariants_path),
                },
                "runtime": {
                    "seconds": time.time() - started,
                    "device": str(args.device),
                },
                "required_artifacts": sorted(
                    CACHE_ARTIFACTS | {"checksums.sha256"}
                ),
            },
        )
        checksums = write_checksums(
            staging / "checksums.sha256",
            [
                manifest_path,
                predictions_path,
                raw_path,
                views_path,
                invariants_path,
            ],
            root=staging,
        )
        staging.rename(output)
        atomic_write_json(
            anchor,
            {
                "schema_version": "saded-fresh-stock-cache-anchor/v1",
                "cache_checksums_sha256": sha256_file(
                    output / checksums.name
                ),
                "cache_manifest_sha256": sha256_file(
                    output / manifest_path.name
                ),
                "predictions_sha256": sha256_file(
                    output / predictions_path.name
                ),
                "evaluation_protocol_sha256": sha256_file(protocol_path),
            },
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if anchor.exists():
            anchor.unlink()
        raise
    return output


def main() -> None:
    args = build_parser().parse_args()
    print(cache_endpoint(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Seal one GT-free full-plus-four-view SADED endpoint cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.saded_stage import prediction_payload  # noqa: E402
from src.saded_stage_protocol import stage_source_state  # noqa: E402
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl_gz,
    sha256_file,
    write_checksums,
)
from src.sbr_fusion import Detection  # noqa: E402
from src.sbr_g0 import (  # noqa: E402
    FrozenSBRProtocol,
    RawViewRecord,
    _letterbox,
    assemble_paired_arms,
    build_arm_views,
    collect_raw_views,
)
from src.tascv_protocol import (  # noqa: E402
    FROZEN_OPTIMIZER_OBSERVATION,
    FROZEN_STAGE_CONTRACT,
    reject_forbidden_path,
    validate_runtime_manifest,
)


CACHE_SCHEMA_VERSION = "saded-endpoint-cache/v1"
CACHE_ARTIFACTS = (
    "cache_manifest.json",
    "predictions.jsonl.gz",
    "raw_views.jsonl.gz",
    "view_manifests.jsonl.gz",
    "cache_invariants.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal one frozen GT-free SADED endpoint cache."
    )
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--training-source-repo", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _same_sha(path: Path, expected: object) -> bool:
    return sha256_file(path) == str(expected).lower()


def _expected_canaries(stage: str) -> list[tuple[int, int]]:
    return {
        "PREFLIGHT_1": [(0, 1)],
        "TINY_MECHANISM_500": [(0, 1), (0, 2), (1, 82)],
        "SCREEN_10": [(0, 1), (0, 2), (1, 82)],
        "FORMAL_100": [(0, 1), (0, 2), (1, 810)],
    }[stage]


def _replay_training_predecessor(
    *,
    training_repo: Path,
    summary: dict[str, Any],
) -> None:
    predecessor = summary.get("predecessor_evidence")
    if not isinstance(predecessor, dict):
        raise ValueError("training predecessor evidence is missing")
    path = reject_forbidden_path(
        predecessor.get("path", ""),
        context="SADED endpoint training predecessor",
    )
    if (
        not path.is_file()
        or not _same_sha(path, predecessor.get("sha256"))
    ):
        raise ValueError("training predecessor checksum drift")
    stage = summary["stage"]
    seed = int(summary["seed"])
    if stage == "SCREEN_10" and seed == 0:
        module = "src.tascv_adjudicator"
        function = "replay_mechanism_gate"
    elif stage == "SCREEN_10":
        module = "src.saded_adjudicator"
        function = "replay_screen_seed0_gate"
    elif stage == "FORMAL_100" and seed == 0:
        module = "src.saded_adjudicator"
        function = "replay_screen_three_seed_gate"
    elif stage == "FORMAL_100":
        module = "src.saded_adjudicator"
        function = "replay_formal_seed0_gate"
    else:
        raise ValueError("unsupported SADED endpoint stage")
    code = (
        "import json;"
        f"from {module} import {function};"
        f"{function}(json.load(open({str(path)!r},encoding='utf-8')))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(training_repo)
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=training_repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def validate_completed_endpoint(
    *,
    protocol_path: Path,
    training_repo: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol, protocol_sha = validate_runtime_manifest(
        protocol_path,
        repo_root=training_repo,
    )
    summary = _read_json(summary_path)
    stage = summary.get("stage")
    seed = summary.get("seed")
    arm = summary.get("arm")
    if (
        stage not in {"SCREEN_10", "FORMAL_100"}
        or seed not in FROZEN_STAGE_CONTRACT[stage]["seeds"]
        or arm not in {"control", "tascv"}
        or summary.get("schema_version")
        != "tascv-training-summary/v1"
    ):
        raise ValueError("training summary identity drift")
    if arm == "control":
        endpoint = protocol["control_allowlist"]["slots"][
            f"B:{stage}:{seed}"
        ]
        if endpoint.get("resolution") != "RUN_FRESH":
            raise ValueError("cache requires a fresh control endpoint")
        target = endpoint["fresh_target"]
    else:
        target = protocol["treatment_endpoints"][f"T:{stage}:{seed}"]
    expected_summary = (
        Path(target["target_dir"]).resolve()
        / "tascv_training_summary.json"
    )
    if summary_path != expected_summary:
        raise ValueError("training summary fixed endpoint drift")
    initial = protocol["initial_states"][str(seed)]
    data_key = (
        "train_only_yaml"
        if FROZEN_STAGE_CONTRACT[stage]["uses_hashed_subset"]
        else "full_train_only_yaml"
    )
    contract = FROZEN_STAGE_CONTRACT[stage]
    exact = {
        "protocol_manifest": protocol_path.as_posix(),
        "protocol_manifest_sha256": protocol_sha,
        "protocol_source_commit": protocol["runtime_source"]["commit"],
        "source_repo_bundle_sha256": protocol["runtime_source"][
            "repo_bundle_sha256"
        ],
        "source_upstream_bundle_sha256": protocol["runtime_source"][
            "upstream_bundle_sha256"
        ],
        "approved_tascv_parent": protocol["approved_tascv_parent"],
        "r0_evaluation_anchor_sha256": protocol["r0_authority"][
            "evaluation_anchor_sha256"
        ],
        "initial_state_sha256": initial["sha256"],
        "initial_state": initial["path"],
        "initial_state_common_fingerprint": initial[
            "common_fingerprint"
        ],
        "data_sha256": protocol[data_key]["sha256"],
        "data": protocol[data_key]["path"],
        "subset_binding": {
            key: protocol["subset"][key]
            for key in ("count", "semantic_sha256", "file_sha256")
        },
        "batch": 8,
        "observed_tensor_batch_sizes": contract[
            "allowed_observed_tensor_batch_sizes"
        ],
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": contract["expected_successful_batches"],
        "optimizer_attempts": contract["expected_optimizer_attempts"],
        "expected_successful_batches": contract[
            "expected_successful_batches"
        ],
        "expected_optimizer_attempts": contract[
            "expected_optimizer_attempts"
        ],
        "workers": 8,
        "loader": {
            "trainer_batch_size": 8,
            "per_rank_batch_size": 8,
            "loader_batch_size": 8,
            "loader_num_workers": 8,
        },
        "optimizer": FROZEN_OPTIMIZER_OBSERVATION,
        "local_bn_preserved_batches": (
            contract["expected_successful_batches"]
            if arm == "tascv"
            else 0
        ),
        "internal_validation_bypass_count": 1,
        "test_loader_is_none": True,
        "auxiliary_non_tiny_pair_count": 0,
        "hardware": {
            "device": "cuda:0",
            "gpu": "NVIDIA GeForce RTX 4090",
        },
        "mechanism_summary": None,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise ValueError(f"completed endpoint drift: {key}")
    histogram = summary.get("local_forward_call_histogram")
    if arm == "control":
        if (
            summary.get("local_forward_calls") != 0
            or histogram != {"1": 0, "2": 0}
        ):
            raise ValueError("control executed a local forward")
    elif (
        not isinstance(histogram, dict)
        or set(histogram) != {"1", "2"}
        or sum(histogram.values())
        != contract["expected_successful_batches"]
        or summary.get("local_forward_calls")
        != histogram["1"] + 2 * histogram["2"]
    ):
        raise ValueError("treatment local-forward closure drift")
    canaries = summary.get("batch_canaries")
    if (
        not isinstance(canaries, list)
        or [
            (record.get("epoch"), record.get("batch"))
            for record in canaries
        ]
        != _expected_canaries(stage)
        or any(
            not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            for record in canaries
        )
    ):
        raise ValueError("completed endpoint canary drift")
    checkpoint = summary.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("kind") != "last.pt"
    ):
        raise ValueError("completed endpoint checkpoint schema drift")
    checkpoint_path = reject_forbidden_path(
        checkpoint.get("path", ""),
        context="SADED endpoint checkpoint",
    )
    if (
        checkpoint_path
        != Path(target["target_dir"]).resolve() / "weights/last.pt"
        or not checkpoint_path.is_file()
        or not _same_sha(checkpoint_path, checkpoint.get("sha256"))
    ):
        raise ValueError("completed endpoint checkpoint drift")
    _replay_training_predecessor(
        training_repo=training_repo,
        summary=summary,
    )
    return protocol, summary, protocol_sha


def _sealed_image_list(
    *,
    protocol: dict[str, Any],
    image_list_path: Path,
) -> list[str]:
    route_anchor_path = Path(
        protocol["r0_authority"]["route_anchor"]
    ).resolve()
    route_anchor = _read_json(route_anchor_path)
    route_manifest_path = (
        route_anchor_path.parent / "route/route_manifest.json"
    )
    if (
        not route_manifest_path.is_file()
        or sha256_file(route_manifest_path)
        != str(route_anchor["route_manifest_sha256"]).lower()
    ):
        raise ValueError("R0 route manifest closure drift")
    route_manifest = _read_json(route_manifest_path)
    if (
        sha256_file(image_list_path)
        != str(
            route_manifest["input_file_sha256"]["image_list"]
        ).lower()
    ):
        raise ValueError("SADED image list checksum drift")
    image_list = json.loads(image_list_path.read_text(encoding="utf-8"))
    if (
        not isinstance(image_list, list)
        or len(image_list) != 548
        or any(not isinstance(item, str) or not item for item in image_list)
        or len(set(image_list)) != len(image_list)
    ):
        raise ValueError("SADED image list identity drift")
    return image_list


def _raw_detection(record: RawViewRecord) -> Detection:
    return Detection(
        box=record.global_xyxy,
        global_xyxy=record.global_xyxy,
        score=record.score,
        class_id=record.class_id,
        source_order=record.source_order,
        query_index=record.query_index,
        view_xyxy=record.view_xyxy,
        network_xyxy=record.network_xyxy,
        tile_bounds=record.tile_bounds,
        transform=record.transform,
        tile_index=(
            record.source_order - 1
            if record.tile_bounds is not None
            else None
        ),
    )


def _view_manifest_is_complete(
    manifest: list[dict[str, Any]],
) -> bool:
    expected = [
        ("full", 0, True),
        ("TL", 1, True),
        ("TR", 2, True),
        ("BL", 3, True),
        ("BR", 4, True),
    ]
    try:
        observed = [
            (
                str(record["view_id"]),
                int(record["source_order"]),
                record["executed"],
            )
            for record in manifest
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return observed == expected


def _predict_image_views(
    model,
    image: np.ndarray,
    *,
    image_id: str,
    device: str,
):
    views = build_arm_views("C", image.shape[1], image.shape[0])
    squares = [
        _letterbox(
            image
            if view.tile is None
            else image[
                view.tile.top : view.tile.bottom,
                view.tile.left : view.tile.right,
            ],
            view.imgsz,
        )
        for view in views
    ]
    results = model.predict(
        source=squares,
        imgsz=FrozenSBRProtocol().imgsz,
        conf=FrozenSBRProtocol().conf,
        max_det=FrozenSBRProtocol().max_det,
        device=device,
        augment=False,
        verbose=False,
        nms=False,
        batch=len(squares),
    )
    if len(results) != len(squares):
        raise ValueError("SADED batched view inference count drift")
    position = 0

    def cached(square: np.ndarray, _imgsz: int):
        nonlocal position
        if (
            position >= len(squares)
            or not np.array_equal(square, squares[position])
        ):
            raise ValueError("SADED ordered view cache lookup drift")
        result = results[position]
        position += 1
        return result

    collected = collect_raw_views(
        image,
        "C",
        cached,
        image_id=image_id,
        return_manifest=True,
    )
    if position != len(squares):
        raise ValueError("SADED ordered view cache was not exhausted")
    return collected


def cache_endpoint(args: argparse.Namespace) -> Path:
    for value in (
        args.protocol_manifest,
        args.training_source_repo,
        args.training_summary,
        args.image_list,
        args.output,
    ):
        reject_forbidden_path(value, context="SADED endpoint cache")
    if str(args.device) != "0":
        raise ValueError("SADED endpoint cache requires device 0")
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError("SADED cache output already exists")
    training_repo = args.training_source_repo.resolve()
    protocol_path = args.protocol_manifest.resolve()
    summary_path = args.training_summary.resolve()
    protocol, summary, protocol_sha = validate_completed_endpoint(
        protocol_path=protocol_path,
        training_repo=training_repo,
        summary_path=summary_path,
    )
    image_list_path = args.image_list.resolve()
    image_list = _sealed_image_list(
        protocol=protocol,
        image_list_path=image_list_path,
    )
    evaluation_source = stage_source_state(REPO_ROOT)
    common_model_files = (
        "src/ascv_loc.py",
        "src/ascv_loc_protocol.py",
        "src/rtdetr_tascv.py",
        "src/tascv.py",
        "src/tascv_diagnostics.py",
        "src/tascv_stage.py",
    )
    if any(
        evaluation_source["files"][relative].lower()
        != protocol["runtime_source"]["repo_files"][relative].lower()
        for relative in common_model_files
    ):
        raise ValueError(
            "SADED training/evaluation model source bridge drift"
        )
    dataset_root = Path(protocol["dataset"]["root"]).resolve()
    image_root = dataset_root / "images/val"
    checkpoint = Path(summary["checkpoint"]["path"]).resolve()
    try:
        from ultralytics import RTDETR
    except Exception as error:
        raise RuntimeError("Ultralytics RTDETR is required") from error
    # Register the custom checkpoint class before torch unpickles treatment
    # endpoints. It is inference-identical to stock RT-DETR.
    import src.rtdetr_tascv  # noqa: F401

    model = RTDETR(str(checkpoint))
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    view_manifests: list[list[dict[str, Any]]] = []
    started = time.time()
    for index, image_id in enumerate(image_list, start=1):
        image_path = (image_root / image_id).resolve()
        if (
            image_root not in image_path.parents
            or not image_path.is_file()
        ):
            raise ValueError(f"SADED val image is missing: {image_id}")
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
            raise ValueError("SADED exact five-view execution drift")
        view_manifests.append(view_manifest)
        view_rows.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "view_manifest": view_manifest,
            }
        )
        for record in raw:
            raw_rows.append(record.to_dict())
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
                    prediction_payload(detection) for detection in full
                ],
                "local_fused_predictions": [
                    prediction_payload(detection) for detection in local
                ],
            }
        )
        if index % 25 == 0 or index == len(image_list):
            print(
                f"SADED_CACHE_PROGRESS {index}/{len(image_list)}",
                flush=True,
            )
    invariants = {
        "image_count": len(rows),
        "expected_image_count": 548,
        "image_order_exact": [row["image_id"] for row in rows]
        == image_list,
        "five_views_per_image_executed": all(
            _view_manifest_is_complete(manifest)
            for manifest in view_manifests
        ),
        "prediction_max_det_respected": all(
            len(row["full_predictions"]) <= FrozenSBRProtocol().max_det
            and len(row["local_fused_predictions"])
            <= FrozenSBRProtocol().max_det
            for row in rows
        ),
        "gt_module_absent": "src.sbr_metrics" not in sys.modules,
    }
    invariants["passed"] = (
        invariants["image_count"] == invariants["expected_image_count"]
        and invariants["image_order_exact"]
        and invariants["five_views_per_image_executed"]
        and invariants["prediction_max_det_respected"]
        and invariants["gt_module_absent"]
    )
    if not invariants["passed"]:
        raise ValueError("SADED endpoint cache invariants failed")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.cache-staging-",
            dir=output_root.parent,
        )
    )
    try:
        cache_dir = staging / "cache"
        cache_dir.mkdir()
        predictions_path = atomic_write_jsonl_gz(
            cache_dir / "predictions.jsonl.gz",
            rows,
        )
        raw_path = atomic_write_jsonl_gz(
            cache_dir / "raw_views.jsonl.gz",
            raw_rows,
        )
        view_manifests_path = atomic_write_jsonl_gz(
            cache_dir / "view_manifests.jsonl.gz",
            view_rows,
        )
        invariants_path = atomic_write_json(
            cache_dir / "cache_invariants.json",
            invariants,
        )
        manifest_path = atomic_write_json(
            cache_dir / "cache_manifest.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "training_protocol": {
                    "path": protocol_path.as_posix(),
                    "sha256": protocol_sha,
                    "source_commit": protocol["runtime_source"]["commit"],
                    "source_repo_bundle_sha256": protocol[
                        "runtime_source"
                    ]["repo_bundle_sha256"],
                },
                "evaluation_source": evaluation_source,
                "training_summary": {
                    "path": summary_path.as_posix(),
                    "sha256": sha256_file(summary_path),
                },
                "checkpoint": summary["checkpoint"],
                "identity": {
                    "stage": summary["stage"],
                    "seed": summary["seed"],
                    "arm": summary["arm"],
                },
                "dataset": {
                    "root": dataset_root.as_posix(),
                    "image_list": image_list_path.as_posix(),
                    "image_list_sha256": sha256_file(image_list_path),
                    "image_count": len(image_list),
                },
                "protocol": FrozenSBRProtocol().__dict__,
                "artifacts": {
                    "predictions_sha256": sha256_file(predictions_path),
                    "raw_views_sha256": sha256_file(raw_path),
                    "view_manifests_sha256": sha256_file(
                        view_manifests_path
                    ),
                    "invariants_sha256": sha256_file(invariants_path),
                },
                "runtime": {
                    "seconds": time.time() - started,
                    "device": str(args.device),
                },
                "required_artifacts": list(CACHE_ARTIFACTS)
                + ["checksums.sha256"],
            },
        )
        checksums = write_checksums(
            cache_dir / "checksums.sha256",
            [
                manifest_path,
                predictions_path,
                raw_path,
                view_manifests_path,
                invariants_path,
            ],
            root=cache_dir,
        )
        atomic_write_json(
            staging / "cache_anchor.json",
            {
                "schema_version": "saded-endpoint-cache-anchor/v1",
                "cache_manifest_sha256": sha256_file(manifest_path),
                "cache_checksums_sha256": sha256_file(checksums),
                "training_protocol_sha256": protocol_sha,
                "training_source_commit": protocol[
                    "runtime_source"
                ]["commit"],
                "evaluation_source_commit": evaluation_source["commit"],
            },
        )
        shutil.move(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def main() -> None:
    args = build_parser().parse_args()
    print(cache_endpoint(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Seal exactly nine confirmation prediction files after formal GO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_saded_endpoint import (  # noqa: E402
    _predict_image_views,
    _raw_detection,
)
from src.saded_adjudicator import (  # noqa: E402
    replay_formal_three_seed_gate,
)
from src.saded_stage import (  # noqa: E402
    ROUTE_ARMS,
    prediction_payload,
    route_paired_caches,
)
from src.saded_stage_protocol import stage_source_state  # noqa: E402
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    write_checksums,
)
from src.sbr_g0 import assemble_paired_arms  # noqa: E402
from src.tascv_protocol import (  # noqa: E402
    FROZEN_CONFIRMATION_CONTRACT,
    reject_forbidden_path,
    validate_runtime_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal the one authorized SADED confirmation batch."
    )
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument(
        "--formal-three-seed-gate",
        type=Path,
        required=True,
    )
    parser.add_argument("--formal-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser


def _endpoint_row(
    model,
    image: np.ndarray,
    *,
    image_id: str,
    device: str,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    raw, view_manifest = _predict_image_views(
        model,
        image,
        image_id=image_id,
        device=device,
    )
    full = [
        prediction_payload(_raw_detection(record))
        for record in raw
        if record.source_order == 0
    ]
    local = [
        prediction_payload(detection)
        for detection in assemble_paired_arms(
            raw,
            width=width,
            height=height,
            view_manifest=view_manifest,
        )["C"]["predictions"]
    ]
    return {
        "image_id": image_id,
        "width": width,
        "height": height,
        "full_predictions": full,
        "local_fused_predictions": local,
    }


def _checkpoint_bindings(
    gate: dict[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in range(3):
        evaluation_root = Path(
            gate["evaluation_bindings"][str(seed)]["root"]
        )
        evaluation_manifest = json.loads(
            (
                evaluation_root
                / "evaluation/evaluation_manifest.json"
            ).read_text(encoding="utf-8")
        )
        route_root = Path(
            evaluation_manifest["route"]["root"]
        )
        route_manifest = json.loads(
            (route_root / "route/route_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if route_manifest["identity"] != {
            "stage": "FORMAL_100",
            "seed": seed,
        }:
            raise ValueError("confirmation formal route identity drift")
        result[seed] = {
            "baseline": route_manifest["cache_inputs"]["baseline"][
                "checkpoint"
            ],
            "treatment": route_manifest["cache_inputs"][
                "treatment"
            ]["checkpoint"],
        }
    return result


def seal_predictions(args: argparse.Namespace) -> Path:
    for value in (
        args.protocol_manifest,
        args.formal_three_seed_gate,
        args.output,
    ):
        reject_forbidden_path(value, context="SADED confirmation sealer")
    if str(args.device) != "0":
        raise ValueError("SADED confirmation requires device 0")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("confirmation prediction output exists")
    source = stage_source_state(REPO_ROOT)
    protocol_path = args.protocol_manifest.resolve()
    protocol, protocol_sha = validate_runtime_manifest(
        protocol_path,
        repo_root=REPO_ROOT,
    )
    if (
        protocol["scientific_contract"]["confirmation"]
        != FROZEN_CONFIRMATION_CONTRACT
    ):
        raise ValueError("confirmation contract drift")
    gate_path = args.formal_three_seed_gate.resolve()
    expected_gate_sha = str(args.formal_gate_sha256).lower()
    if (
        not gate_path.is_file()
        or len(expected_gate_sha) != 64
        or sha256_file(gate_path) != expected_gate_sha
    ):
        raise ValueError("formal three-seed gate checksum drift")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    replay_formal_three_seed_gate(gate, recompute_metrics=False)
    if (
        gate["protocol_manifest_sha256"] != protocol_sha
        or gate["protocol_source_commit"]
        != protocol["runtime_source"]["commit"]
    ):
        raise ValueError("confirmation gate/protocol source drift")
    # This path is fixed by the frozen contract and is not user-selectable.
    parts = FROZEN_CONFIRMATION_CONTRACT[
        "image_root_derivation"
    ]["relative_parts"]
    image_root = (
        Path(protocol["dataset"]["root"])
        / parts[0]
        / f"{parts[1]}-{parts[2]}"
    ).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    image_list = sorted(
        path.relative_to(image_root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_list or len(set(image_list)) != len(image_list):
        raise ValueError("confirmation image set is empty or duplicated")
    checkpoints = _checkpoint_bindings(gate)
    for endpoint_by_seed in checkpoints.values():
        for checkpoint in endpoint_by_seed.values():
            path = Path(checkpoint["path"]).resolve()
            if (
                checkpoint.get("kind") != "last.pt"
                or not path.is_file()
                or sha256_file(path)
                != str(checkpoint["sha256"]).lower()
            ):
                raise ValueError(
                    "confirmation checkpoint binding drift"
                )
    try:
        from ultralytics import RTDETR
    except Exception as error:
        raise RuntimeError("Ultralytics RTDETR is required") from error
    import src.rtdetr_tascv  # noqa: F401

    prediction_sets: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in FROZEN_CONFIRMATION_CONTRACT[
            "prediction_files"
        ]
    }
    route_invariants: dict[str, list[dict[str, Any]]] = {
        str(seed): [] for seed in range(3)
    }
    for seed in range(3):
        baseline_model = RTDETR(
            str(checkpoints[seed]["baseline"]["path"])
        )
        treatment_model = RTDETR(
            str(checkpoints[seed]["treatment"]["path"])
        )
        for index, image_id in enumerate(image_list, start=1):
            image_path = (image_root / image_id).resolve()
            if image_root not in image_path.parents:
                raise ValueError("confirmation image path escaped root")
            with Image.open(image_path) as handle:
                image = np.asarray(handle.convert("RGB"))
            baseline = _endpoint_row(
                baseline_model,
                image,
                image_id=image_id,
                device=str(args.device),
            )
            treatment = _endpoint_row(
                treatment_model,
                image,
                image_id=image_id,
                device=str(args.device),
            )
            routed, invariants = route_paired_caches(
                [baseline],
                [treatment],
            )
            if invariants.get("passed") is not True:
                raise ValueError("confirmation route invariants failed")
            row = routed[0]
            for system in ROUTE_ARMS:
                filename = f"seed{seed}_{system}.json"
                prediction_sets[filename].append(
                    {
                        "image_id": image_id,
                        "width": row["width"],
                        "height": row["height"],
                        "predictions": row["arms"][system],
                    }
                )
            route_invariants[str(seed)].append(row["invariants"])
            if index % 25 == 0 or index == len(image_list):
                print(
                    "SADED_CONFIRMATION_PROGRESS "
                    f"seed={seed} {index}/{len(image_list)}",
                    flush=True,
                )
        del baseline_model, treatment_model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    if "src.sbr_metrics" in sys.modules:
        raise ValueError("confirmation sealer imported GT metrics")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.predictions-staging-",
            dir=output.parent,
        )
    )
    try:
        prediction_paths = {
            name: atomic_write_json(staging / name, prediction_sets[name])
            for name in FROZEN_CONFIRMATION_CONTRACT[
                "prediction_files"
            ]
        }
        manifest_path = atomic_write_json(
            staging / "prediction_manifest.json",
            {
                "schema_version":
                    "saded-confirmation-predictions/v1",
                "protocol": {
                    "path": protocol_path.as_posix(),
                    "sha256": protocol_sha,
                    "source_commit": protocol[
                        "runtime_source"
                    ]["commit"],
                },
                "formal_gate": {
                    "path": gate_path.as_posix(),
                    "sha256": expected_gate_sha,
                },
                "source": source,
                "image_root": image_root.as_posix(),
                "image_list": image_list,
                "image_count": len(image_list),
                "checkpoints": checkpoints,
                "prediction_files": {
                    name: sha256_file(path)
                    for name, path in prediction_paths.items()
                },
                "route_invariants": route_invariants,
                "annotation_inputs_opened": False,
                "exact_nine_sealed": True,
            },
        )
        checksums_path = write_checksums(
            staging / "checksums.sha256",
            [manifest_path, *prediction_paths.values()],
            root=staging,
        )
        atomic_write_json(
            staging / "prediction_anchor.json",
            {
                "schema_version":
                    "saded-confirmation-prediction-anchor/v1",
                "prediction_manifest_sha256": sha256_file(
                    manifest_path
                ),
                "prediction_checksums_sha256": sha256_file(
                    checksums_path
                ),
                "formal_gate_sha256": expected_gate_sha,
                "source_commit": source["commit"],
                "prediction_file_count": 9,
                "exact_nine_sealed": True,
            },
        )
        shutil.move(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    print(seal_predictions(build_parser().parse_args()))


if __name__ == "__main__":
    main()

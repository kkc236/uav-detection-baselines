#!/usr/bin/env python3
"""Freeze the fresh-stock SADED evaluation authority before val inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.saded_stock_evaluation_protocol import (  # noqa: E402
    EXPECTED_DATASET_SIGNATURE,
    EXPECTED_DATASET_YAML,
    EXPECTED_DATASET_YAML_SHA256,
    EXPECTED_IMAGE_COUNT,
    EXPECTED_IMAGE_LIST_SHA256,
    EXPECTED_IMAGE_ROOT,
    SCHEMA_VERSION,
    build_image_authority,
    canonical_image_list_bytes,
    frozen_route_contract,
    postprocess_source_state,
    reject_forbidden,
    validate_completed_stock_endpoint,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    write_checksums,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze fresh-stock SADED evaluation protocol."
    )
    parser.add_argument("--training-protocol", required=True, type=Path)
    parser.add_argument("--training-source-repo", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--protocol-parent", required=True, type=Path)
    parser.add_argument("--evidence-parent", required=True, type=Path)
    return parser


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def prepare(args: argparse.Namespace) -> Path:
    for value in vars(args).values():
        reject_forbidden(value)
    source = postprocess_source_state(REPO_ROOT)
    run_id = f"final-saded-fresh-eval-{source['commit'][:8]}"
    protocol_parent = args.protocol_parent.resolve()
    evidence_parent = args.evidence_parent.resolve()
    output = protocol_parent / run_id
    external_anchor = protocol_parent / f"{run_id}_protocol_anchor.json"
    evidence_root = evidence_parent / run_id
    if output.exists() or external_anchor.exists() or evidence_root.exists():
        raise FileExistsError("fresh evaluation protocol target exists")

    training_protocol_path = args.training_protocol.resolve()
    training_summary_path = args.training_summary.resolve()
    training_repo = args.training_source_repo.resolve()
    stock_protocol = _read_json(training_protocol_path)
    checkpoint_path = (
        Path(stock_protocol["endpoint"]["target_dir"]).resolve()
        / "weights"
        / "last.pt"
    )
    training = {
        "source_repo": training_repo.as_posix(),
        "source_commit": stock_protocol["runtime_source"]["commit"],
        "protocol": {
            "path": training_protocol_path.as_posix(),
            "sha256": sha256_file(training_protocol_path),
        },
        "summary": {
            "path": training_summary_path.as_posix(),
            "sha256": sha256_file(training_summary_path),
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
        },
    }
    provisional = {"training": training}
    summary = validate_completed_stock_endpoint(
        provisional,
        training_repo=training_repo,
    )

    image_root = Path(EXPECTED_IMAGE_ROOT).resolve()
    image_list = sorted(
        item.name for item in image_root.iterdir() if item.is_file()
    )
    if (
        len(image_list) != EXPECTED_IMAGE_COUNT
        or __import__("hashlib")
        .sha256(canonical_image_list_bytes(image_list))
        .hexdigest()
        .upper()
        != EXPECTED_IMAGE_LIST_SHA256
    ):
        raise ValueError("sealed dev-val image list authority drift")
    image_authority = build_image_authority(image_root, image_list)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{run_id}.protocol-staging-",
            dir=output.parent,
        )
    )
    try:
        image_list_path = staging / "image_list.json"
        image_list_path.write_bytes(canonical_image_list_bytes(image_list))
        image_authority_path = atomic_write_json(
            staging / "image_authority.json",
            image_authority,
        )
        endpoint_anchor_path = atomic_write_json(
            staging / "endpoint_anchor.json",
            {
                "schema_version": "saded-stock-endpoint-anchor/v1",
                "decision": "ENDPOINT_VALID",
                "training": training,
                "checkpoint_metadata": {
                    "epoch": summary["checkpoint"]["epoch"],
                    "completed_epochs": summary["completed_epochs"],
                    "successful_batches": summary["successful_batches"],
                    "optimizer_attempts": summary["optimizer_attempts"],
                    "amp_scale": summary["amp_scale"],
                },
            },
        )
        final_paths = {
            "image_list": output / "image_list.json",
            "image_authority": output / "image_authority.json",
            "endpoint_anchor": output / "endpoint_anchor.json",
        }
        outputs = {
            "cache": (evidence_root / "cache").as_posix(),
            "route": (evidence_root / "route").as_posix(),
            "evaluation": (evidence_root / "evaluation").as_posix(),
            "adjudication": (evidence_root / "adjudication").as_posix(),
            "evaluation_claim": (
                evidence_root / "evaluation_claim.json"
            ).as_posix(),
        }
        protocol_path = atomic_write_json(
            staging / "protocol_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "source": source,
                "training": training,
                "dataset": {
                    "root": "/mnt/uav/datasets/VisDrone",
                    "sha256": (
                        "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
                    ),
                    "signature": EXPECTED_DATASET_SIGNATURE,
                    "yaml": EXPECTED_DATASET_YAML,
                    "yaml_sha256": EXPECTED_DATASET_YAML_SHA256,
                    "image_root": EXPECTED_IMAGE_ROOT,
                    "image_count": EXPECTED_IMAGE_COUNT,
                    "image_list_sha256": EXPECTED_IMAGE_LIST_SHA256,
                },
                "protocol_artifacts": {
                    key: {
                        "path": path.resolve().as_posix(),
                        "sha256": sha256_file(
                            staging / path.name
                        ),
                    }
                    for key, path in final_paths.items()
                },
                "route_contract": frozen_route_contract(),
                "outputs": outputs,
                "evaluation_consumption": {
                    "claim_creation": "O_CREAT|O_EXCL",
                    "one_shot": True,
                },
            },
        )
        checksums_path = write_checksums(
            staging / "checksums.sha256",
            [
                endpoint_anchor_path,
                image_authority_path,
                image_list_path,
                protocol_path,
            ],
            root=staging,
        )
        staging.rename(output)
        atomic_write_json(
            external_anchor,
            {
                "schema_version": (
                    "saded-fresh-stock-evaluation-protocol-anchor/v1"
                ),
                "manifest_sha256": sha256_file(
                    output / "protocol_manifest.json"
                ),
                "checksums_sha256": sha256_file(
                    output / checksums_path.name
                ),
                "endpoint_anchor_sha256": sha256_file(
                    output / "endpoint_anchor.json"
                ),
            },
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if external_anchor.exists():
            external_anchor.unlink()
        raise
    return output / "protocol_manifest.json"


def main() -> None:
    args = build_parser().parse_args()
    print(prepare(args))


if __name__ == "__main__":
    main()

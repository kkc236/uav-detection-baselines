#!/usr/bin/env python3
"""Consume the sealed confirmation split exactly once for all nine systems."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_saded_stage import (  # noqa: E402
    _jsonable,
    _metric_row,
)
from src.saded_confirmation import (  # noqa: E402
    adjudicate_confirmation_metrics,
    verify_confirmation_predictions,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    write_checksums,
)
from src.tascv_protocol import (  # noqa: E402
    FROZEN_CONFIRMATION_CONTRACT,
    reject_forbidden_path,
)


RESULT_FILES = (
    "result_manifest.json",
    "metrics.json",
    "gate.json",
    "evaluation_invariants.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the sealed SADED confirmation exactly once."
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--prediction-anchor-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _create_claim(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def evaluate_once(args: argparse.Namespace) -> Path:
    prediction_root = reject_forbidden_path(
        args.prediction_root,
        context="SADED confirmation prediction root",
    )
    output = reject_forbidden_path(
        args.output,
        context="SADED confirmation result output",
    )
    if output.exists():
        raise FileExistsError("confirmation result output exists")
    verified = verify_confirmation_predictions(
        prediction_root,
        anchor_sha256=args.prediction_anchor_sha256,
    )
    protocol = json.loads(
        Path(
            verified["manifest"]["protocol"]["path"]
        ).read_text(encoding="utf-8")
    )
    formal_project = Path(
        protocol["treatment_endpoints"]["T:FORMAL_100:0"][
            "project"
        ]
    ).resolve()
    run_root = formal_project.parents[2]
    claim_path = (
        run_root
        / "confirmation"
        / FROZEN_CONFIRMATION_CONTRACT["claim_file"]
    )
    _create_claim(
        claim_path,
        {
            "schema_version": "saded-confirmation-open-claim/v1",
            "state": "CONSUMED",
            "prediction_root": prediction_root.as_posix(),
            "prediction_anchor_sha256": verified["anchor_sha256"],
            "prediction_snapshot": verified["snapshot"],
            "retry_permitted": False,
        },
    )
    claim_sha = sha256_file(claim_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.result-staging-",
            dir=output.parent,
        )
    )
    try:
        # All confirmation GT-aware imports and reads occur after the
        # immutable O_EXCL claim above. A crash from this point is consumed.
        import yaml
        from src.sbr_artifacts import load_dataset
        from src.sbr_metrics import evaluate_dataset

        manifest = verified["manifest"]
        parts = FROZEN_CONFIRMATION_CONTRACT[
            "image_root_derivation"
        ]["relative_parts"]
        split_name = f"{parts[1]}-{parts[2]}"
        full_yaml = yaml.safe_load(
            Path(protocol["dataset"]["full_yaml"]).read_text(
                encoding="utf-8"
            )
        )
        dataset_yaml = staging / "confirmation_dataset.yaml"
        dataset_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": protocol["dataset"]["root"],
                    split_name: f"{parts[0]}/{split_name}",
                    "names": full_yaml["names"],
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
            newline="\n",
        )
        dataset = load_dataset(
            dataset_yaml,
            split=split_name,
            root_override=Path(protocol["dataset"]["root"]),
        )
        if (
            dataset["image_list"] != manifest["image_list"]
            or dataset["image_count"] != manifest["image_count"]
        ):
            raise ValueError("confirmation dataset identity drift")
        image_by_id = {
            image["relative_path"]: image
            for image in dataset["images"]
        }
        metrics_by_seed: dict[str, dict[str, Any]] = {
            str(seed): {} for seed in range(3)
        }
        for seed in range(3):
            for system in FROZEN_CONFIRMATION_CONTRACT["systems"]:
                filename = f"seed{seed}_{system}.json"
                metric_rows = []
                for row in verified["rows"][filename]:
                    image = image_by_id[row["image_id"]]
                    if (
                        int(row["width"]) != int(image["width"])
                        or int(row["height"]) != int(image["height"])
                    ):
                        raise ValueError(
                            "confirmation prediction dimension drift"
                        )
                    metric_rows.append(
                        _metric_row(image, row["predictions"])
                    )
                metrics_by_seed[str(seed)][system] = _jsonable(
                    evaluate_dataset(metric_rows)
                )
        gate = adjudicate_confirmation_metrics(metrics_by_seed)
        prediction_snapshot_after = {
            name: sha256_file(prediction_root / name)
            for name in sorted(verified["snapshot"])
        }
        invariants = {
            "claim_created_before_gt_import": True,
            "claim_is_immutable_consumed": (
                claim_path.is_file()
                and sha256_file(claim_path) == claim_sha
            ),
            "prediction_snapshot_unchanged": (
                prediction_snapshot_after == verified["snapshot"]
            ),
            "exact_nine_metric_sets": (
                set(metrics_by_seed) == {"0", "1", "2"}
                and all(
                    set(value)
                    == set(FROZEN_CONFIRMATION_CONTRACT["systems"])
                    for value in metrics_by_seed.values()
                )
            ),
            "single_process_evaluation": True,
            "retry_permitted": False,
        }
        invariants["passed"] = all(invariants.values())
        if not invariants["passed"]:
            raise ValueError("confirmation result invariants failed")
        metrics_path = atomic_write_json(
            staging / "metrics.json",
            metrics_by_seed,
        )
        gate_path = atomic_write_json(staging / "gate.json", gate)
        invariants_path = atomic_write_json(
            staging / "evaluation_invariants.json",
            invariants,
        )
        result_manifest_path = atomic_write_json(
            staging / "result_manifest.json",
            {
                "schema_version": "saded-confirmation-result/v1",
                "prediction_root": prediction_root.as_posix(),
                "prediction_anchor_sha256": verified[
                    "anchor_sha256"
                ],
                "prediction_snapshot": verified["snapshot"],
                "claim": {
                    "path": claim_path.as_posix(),
                    "sha256": claim_sha,
                    "retry_permitted": False,
                },
                "dataset_signature": dataset["dataset_signature"],
                "image_count": dataset["image_count"],
                "artifacts": {
                    "metrics_sha256": sha256_file(metrics_path),
                    "gate_sha256": sha256_file(gate_path),
                    "invariants_sha256": sha256_file(
                        invariants_path
                    ),
                },
                "decision": gate["decision"],
                "terminal": True,
            },
        )
        checksums_path = write_checksums(
            staging / "checksums.sha256",
            [
                result_manifest_path,
                metrics_path,
                gate_path,
                invariants_path,
            ],
            root=staging,
        )
        atomic_write_json(
            staging / "result_anchor.json",
            {
                "schema_version":
                    "saded-confirmation-result-anchor/v1",
                "result_manifest_sha256": sha256_file(
                    result_manifest_path
                ),
                "result_checksums_sha256": sha256_file(
                    checksums_path
                ),
                "prediction_anchor_sha256": verified[
                    "anchor_sha256"
                ],
                "claim_sha256": claim_sha,
                "decision": gate["decision"],
                "terminal": True,
            },
        )
        shutil.move(staging, output)
    except BaseException as error:
        # Preserve a terminal consumed record. The immutable claim is never
        # removed, so a second GT-aware evaluation cannot start.
        if not output.exists():
            invalid = Path(
                tempfile.mkdtemp(
                    prefix=f".{output.name}.invalid-staging-",
                    dir=output.parent,
                )
            )
            atomic_write_json(
                invalid / "invalid_consumed.json",
                {
                    "schema_version":
                        "saded-confirmation-invalid-consumed/v1",
                    "state": "INVALID_CONSUMED",
                    "claim": {
                        "path": claim_path.as_posix(),
                        "sha256": claim_sha,
                    },
                    "prediction_anchor_sha256": verified[
                        "anchor_sha256"
                    ],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry_permitted": False,
                },
            )
            shutil.move(invalid, output)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    print(evaluate_once(build_parser().parse_args()))


if __name__ == "__main__":
    main()

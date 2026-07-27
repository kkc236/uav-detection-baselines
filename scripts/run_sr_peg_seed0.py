"""Fail-closed, resumable runner for the single approved SR-PEG seed0 screen."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

import torch

from src.ascv_loc_protocol import subset_signature
from src.gcqf_cache import (
    CACHE_SCHEMA_VERSION,
    SRPEG_CACHE_SCHEMA_VERSION,
    VerifiedEvidenceCache,
)


STAGES = ("TRAIN_CACHE", "TRAIN_SEED0", "CALIBRATE", "EVALUATE")
EXPECTED_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEF"
    "CF3AFEF6C174C6E4F3B1EF810C883099B"
)
EXPECTED_TRAIN10_SIGNATURE = (
    "52660F55552FFD953E2EE26F55FD0A1C"
    "B14217DBBEA0F5F3B981C3514F8D93A0"
)
EXPECTED_VAL_SIGNATURE = (
    "A9A0C00DC640BCAAEFE9360F5E3B5538"
    "2E74E169B5AEEF15EB1F0AE2A571228A"
)
EXPECTED_RUNTIME = {
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "ultralytics": "8.4.90",
    "cuda": "12.1",
    "gpu": "NVIDIA GeForce RTX 4090",
}
MIN_FREE_BYTES = 1024**3


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one fail-closed SR-PEG seed0 diagnostic."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--anchor-reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    return parser


def validate_runtime_environment(observed: Mapping[str, str]) -> None:
    for key, expected in EXPECTED_RUNTIME.items():
        if observed.get(key) != expected:
            raise ValueError(
                f"runtime {key} drift: expected={expected!r} "
                f"actual={observed.get(key)!r}"
            )


def _observed_runtime() -> dict[str, str]:
    try:
        import ultralytics
    except Exception as error:
        raise RuntimeError("Ultralytics 8.4.90 is required") from error
    if not torch.cuda.is_available():
        raise RuntimeError("SR-PEG seed0 requires CUDA")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
    }


def _validate_source_commit(source: Path, commit: str) -> str:
    normalized = str(commit).lower()
    if len(normalized) != 40 or any(
        value not in "0123456789abcdef" for value in normalized
    ):
        raise ValueError("source commit must be an exact 40-character Git SHA")
    if normalized[:8] not in source.name.lower():
        raise ValueError("source directory is not bound to source commit")
    return normalized


def build_stage_commands(
    args: argparse.Namespace,
    *,
    python: str,
) -> dict[str, list[str]]:
    output = args.output.resolve()
    source = args.source.resolve()
    train_manifest = output / "train-cache" / "manifest.json"
    training = output / "train-seed0"
    calibration = output / "calibration.json"
    evaluation = output / "seed0-evaluation.json"
    return {
        "TRAIN_CACHE": [
            python,
            "-m",
            "scripts.cache_gcqf_evidence",
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--data",
            str(args.data.resolve()),
            "--split",
            "train",
            "--image-path",
            str(args.train_images.resolve()),
            "--dataset-signature",
            EXPECTED_TRAIN10_SIGNATURE,
            "--output",
            str(output / "train-cache"),
            "--expected-records",
            "647",
            "--workers",
            "8",
            "--records-per-shard",
            "8",
        ],
        "TRAIN_SEED0": [
            python,
            "-m",
            "scripts.train_gcqf_g0",
            "--train-cache",
            str(train_manifest),
            "--output",
            str(training),
            "--seed",
            "0",
            "--source-commit",
            str(args.source_commit).lower(),
        ],
        "CALIBRATE": [
            python,
            "-m",
            "scripts.calibrate_sr_peg_g0",
            "--cache",
            str(train_manifest),
            "--module",
            str(training / "best-module.pt"),
            "--data",
            str(args.data.resolve()),
            "--output",
            str(calibration),
        ],
        "EVALUATE": [
            python,
            "-m",
            "scripts.evaluate_gcqf_g0",
            "--cache",
            str(args.val_cache.resolve()),
            "--module",
            str(training / "best-module.pt"),
            "--calibration",
            str(calibration),
            "--data",
            str(args.data.resolve()),
            "--output",
            str(evaluation),
            "--anchor-reference",
            str(args.anchor_reference.resolve()),
        ],
    }


def _stage_artifacts(output: Path) -> dict[str, Path]:
    return {
        "TRAIN_CACHE": output / "train-cache" / "manifest.json",
        "TRAIN_SEED0": output / "train-seed0" / "manifest.json",
        "CALIBRATE": output / "calibration.json",
        "EVALUATE": output / "seed0-evaluation.json",
    }


def _verify_stage(stage: str, artifact: Path) -> None:
    if not artifact.is_file():
        raise RuntimeError(f"{stage} did not produce {artifact}")
    if stage == "TRAIN_CACHE":
        cache = VerifiedEvidenceCache(artifact)
        if (
            cache.manifest["schema_version"] != SRPEG_CACHE_SCHEMA_VERSION
            or cache.manifest["record_count"] != 647
            or cache.manifest["dataset_signature"]
            != EXPECTED_TRAIN10_SIGNATURE
            or cache.manifest["baseline_sha256"]
            != EXPECTED_BASELINE_SHA256
        ):
            raise RuntimeError("TRAIN_CACHE authority drift")
        return
    value = json.loads(artifact.read_text(encoding="utf-8"))
    if stage == "TRAIN_SEED0":
        required = (
            value.get("schema_version") == "gcte-gcqf-training/v2"
            and value.get("seed") == 0
            and len(value.get("train_image_ids", [])) == 518
            and len(value.get("calibration_image_ids", [])) == 129
            and (artifact.parent / "best-module.pt").is_file()
        )
    elif stage == "CALIBRATE":
        required = (
            value.get("schema_version") == "gcte-sr-peg-calibration/v1"
            and value.get("image_count") == 129
            and len(value.get("candidates", [])) == 27
            and isinstance(value.get("selected_thresholds"), dict)
        )
    elif stage == "EVALUATE":
        required = (
            value.get("schema_version") == "gcte-gcqf-five-state/v2"
            and set(value.get("metrics", {}))
            == {
                "Global",
                "Raw-Union",
                "Fixed-SADED",
                "Residual-Off",
                "Full-GCQF",
            }
        )
    else:
        raise ValueError(f"unknown stage: {stage}")
    if not required:
        raise RuntimeError(f"{stage} artifact contract drift")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    commit = _validate_source_commit(source, args.source_commit)
    paths = (
        args.checkpoint,
        args.data,
        args.train_images,
        args.val_cache,
        args.anchor_reference,
    )
    for path in paths:
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    if _sha256_file(args.checkpoint) != EXPECTED_BASELINE_SHA256:
        raise ValueError("baseline checkpoint authority mismatch")
    validate_runtime_environment(_observed_runtime())

    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(str(args.data.resolve()), autodownload=False)
    subset = subset_signature(
        args.train_images.resolve(),
        root=Path(dataset["path"]).resolve(),
    )
    if subset != {
        "count": 647,
        "sha256": EXPECTED_TRAIN10_SIGNATURE,
    }:
        raise ValueError("train10 semantic signature mismatch")
    val_cache = VerifiedEvidenceCache(
        args.val_cache,
        expected_baseline_sha256=EXPECTED_BASELINE_SHA256,
        expected_dataset_signature=EXPECTED_VAL_SIGNATURE,
    )
    if (
        val_cache.manifest["schema_version"] != CACHE_SCHEMA_VERSION
        or val_cache.manifest["record_count"] != 548
    ):
        raise ValueError("sealed val cache authority mismatch")
    output_parent = args.output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(
            f"insufficient free disk: {free_bytes} < {MIN_FREE_BYTES}"
        )
    return {
        "source_commit": commit,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "train10": subset,
        "val_signature": EXPECTED_VAL_SIGNATURE,
        "runtime": _observed_runtime(),
        "free_bytes": free_bytes,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finalize_pipeline(output: Path, evaluation: Path) -> None:
    value = json.loads(evaluation.read_text(encoding="utf-8"))
    if value.get("per_seed_gate", {}).get("passed") is not True:
        raise RuntimeError("seed0 evaluation failed the frozen hard gate")
    _write_json(
        output / "PIPELINE_COMPLETE",
        {
            "evaluation": evaluation.resolve().as_posix(),
            "evaluation_sha256": _sha256_file(evaluation),
            "seed": 0,
        },
    )


def run(args: argparse.Namespace) -> Path:
    if args.seed != 0:
        raise ValueError("only seed0 is permitted")
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    if (output / "PIPELINE_COMPLETE").exists():
        raise FileExistsError(output / "PIPELINE_COMPLETE")
    if (output / "PIPELINE_FAILED").exists():
        raise RuntimeError("existing failed pipeline requires a new output path")

    try:
        preflight = _preflight(args)
        _write_json(output / "preflight.json", preflight)
        commands = build_stage_commands(args, python=sys.executable)
        artifacts = _stage_artifacts(output)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(args.source.resolve())
        environment["PYTHONHASHSEED"] = "0"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for stage in STAGES:
            artifact = artifacts[stage]
            marker = output / f"{stage}_COMPLETE.json"
            if marker.exists():
                marker_value = json.loads(marker.read_text(encoding="utf-8"))
                _verify_stage(stage, artifact)
                if marker_value.get("artifact_sha256") != _sha256_file(artifact):
                    raise RuntimeError(f"{stage} completion checksum drift")
                continue
            if artifact.exists() or (
                stage in {"TRAIN_CACHE", "TRAIN_SEED0"}
                and artifact.parent.exists()
            ):
                raise RuntimeError(f"{stage} has incomplete existing output")
            _write_json(
                output / "PIPELINE_STATUS",
                {"stage": stage, "seed": 0},
            )
            log_path = logs / f"{stage.lower()}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    commands[stage],
                    cwd=args.source.resolve(),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{stage} failed with exit code {completed.returncode}"
                )
            _verify_stage(stage, artifact)
            _write_json(
                marker,
                {
                    "stage": stage,
                    "artifact": artifact.resolve().as_posix(),
                    "artifact_sha256": _sha256_file(artifact),
                },
            )
        finalize_pipeline(output, artifacts["EVALUATE"])
        _write_json(
            output / "PIPELINE_STATUS",
            {"stage": "COMPLETE", "seed": 0},
        )
        return output / "PIPELINE_COMPLETE"
    except Exception as error:
        _write_json(
            output / "PIPELINE_FAILED",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

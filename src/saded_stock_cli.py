"""Fail-closed CLI contract for the fresh SADED stock baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import torch
import yaml

from src.tascv_cli import build_settings as build_tascv_settings
from src.tascv_protocol import (
    APPROVED_TASCV_PARENT,
    EXPECTED_COMMON_FINGERPRINTS,
    EXPECTED_DATASET_FILE_COUNT,
    EXPECTED_DATASET_ROOT,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ENVIRONMENT,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_UPSTREAM_SOURCE_SHA256,
    FROZEN_TRAINING_CONTRACT,
    REPO_SOURCE_FILES,
    current_environment,
    current_upstream_source_hashes,
    require_clean_repo,
    sha256_file,
    source_bundle_sha256,
    validate_initial_state_artifact,
)
from src.tascv_stage import TASCVStage


PROTOCOL_SCHEMA = "saded-fresh100-stock/v1"
SOURCE_FILES = tuple(
    sorted(
        set(REPO_SOURCE_FILES)
        | {
            "scripts/train_rtdetr_saded_stock.py",
            "src/saded_single_model_evidence.py",
            "src/saded_stock_cli.py",
        }
    )
)
MIN_FREE_BYTES = 3 * 1024**3
EXPECTED_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen fresh seed-0 SADED stock baseline."
    )
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _reject_forbidden(value: str | Path) -> None:
    raw = str(value).replace("\\", "/").lower()
    resolved = Path(value).resolve().as_posix().lower()
    if "test-dev" in raw or "test_dev" in raw:
        raise ValueError(f"test-dev is forbidden: {value}")
    if "test-dev" in resolved or "test_dev" in resolved:
        raise ValueError(f"test-dev is forbidden: {value}")


def _reject_forbidden_record(value: object) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _reject_forbidden_record(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_record(nested)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if "test-dev" in normalized or "test_dev" in normalized:
            raise ValueError(f"test-dev is forbidden: {value}")


def current_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_closure(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    return {
        relative: sha256_file(root / relative)
        for relative in SOURCE_FILES
    }


def build_settings(args: Namespace) -> dict:
    forwarded = Namespace(
        stage=TASCVStage.FORMAL_100,
        data=args.data,
        project=args.project,
        name=args.name,
        device=args.device,
        seed=args.seed,
    )
    return build_tascv_settings(forwarded)


def _validate_train_only_yaml(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"path", "train", "val", "names"}
    ):
        raise ValueError("fresh stock data YAML must be a mapping")
    expected_train = (
        Path(EXPECTED_DATASET_ROOT).resolve() / "images" / "train"
    ).as_posix()
    if (
        Path(payload.get("path", "")).resolve().as_posix()
        != Path(EXPECTED_DATASET_ROOT).resolve().as_posix()
        or Path(payload.get("train", "")).resolve().as_posix()
        != expected_train
        or Path(payload.get("val", "")).resolve().as_posix()
        != expected_train
    ):
        raise ValueError("fresh stock data must bind only the full train split")
    if payload.get("names") != EXPECTED_NAMES:
        raise ValueError("fresh stock data class mapping drift")
    _reject_forbidden_record(payload)


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def validate_protocol_inputs(
    args: Namespace,
    *,
    repo_root: Path | None = None,
    require_fresh_target: bool = True,
) -> dict:
    for value in (
        args.protocol_manifest,
        args.initial_state,
        args.data,
        args.project,
        args.name,
    ):
        _reject_forbidden(value)
    if str(args.device) != "0":
        raise ValueError("fresh stock training requires single GPU device 0")
    if args.seed != 0:
        raise ValueError("fresh stock training requires seed 0")

    manifest_path = Path(args.protocol_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_forbidden_record(manifest)
    if manifest.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("fresh stock protocol schema drift")
    if {
        "stage": manifest.get("stage"),
        "arm": manifest.get("arm"),
        "fresh_start": manifest.get("fresh_start"),
        "predecessor_required": manifest.get("predecessor_required"),
        "checkpoint_reuse": manifest.get("checkpoint_reuse"),
    } != {
        "stage": "FORMAL_100",
        "arm": "stock_control",
        "fresh_start": True,
        "predecessor_required": False,
        "checkpoint_reuse": "forbidden",
    }:
        raise ValueError("fresh stock scientific boundary drift")
    if manifest.get("environment") != EXPECTED_ENVIRONMENT:
        raise ValueError("fresh stock environment authority drift")
    if current_environment() != EXPECTED_ENVIRONMENT:
        raise ValueError("fresh stock live environment drift")
    if manifest.get("training_contract") != FROZEN_TRAINING_CONTRACT:
        raise ValueError("fresh stock training contract drift")
    if manifest.get("dataset") != {
        "root": EXPECTED_DATASET_ROOT,
        "sha256": EXPECTED_DATASET_SHA256,
        "file_count": EXPECTED_DATASET_FILE_COUNT,
        "train_images": 6471,
        "val_images": 548,
        "classes": 10,
    }:
        raise ValueError("fresh stock dataset authority drift")

    resolved_repo = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    require_clean_repo(resolved_repo)
    source = manifest.get("runtime_source", {})
    live_sources = source_closure(resolved_repo)
    live_commit = current_commit(resolved_repo)
    if (
        source.get("commit") != live_commit
        or source.get("repo_files") != live_sources
        or source.get("repo_bundle_sha256")
        != source_bundle_sha256(live_sources)
        or source.get("upstream") != EXPECTED_UPSTREAM_SOURCE_SHA256
        or source.get("upstream_bundle_sha256")
        != source_bundle_sha256(EXPECTED_UPSTREAM_SOURCE_SHA256)
        or current_upstream_source_hashes()
        != EXPECTED_UPSTREAM_SOURCE_SHA256
    ):
        raise ValueError("fresh stock source closure drift")
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            APPROVED_TASCV_PARENT["commit"],
            "HEAD",
        ],
        cwd=resolved_repo,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("fresh stock source is not descended from authority")

    initial_path = Path(args.initial_state).resolve()
    initial = manifest.get("initial_state", {})
    actual_initial_sha = sha256_file(initial_path)
    if (
        Path(initial.get("path", "")).resolve() != initial_path
        or initial.get("sha256") != actual_initial_sha
        or actual_initial_sha != EXPECTED_INITIAL_STATE_SHA256[0]
        or initial.get("common_fingerprint")
        != EXPECTED_COMMON_FINGERPRINTS[0]
    ):
        raise ValueError("fresh stock initial-state binding drift")
    artifact = torch.load(initial_path, map_location="cpu", weights_only=False)
    validate_initial_state_artifact(artifact, seed=0)

    data_path = Path(args.data).resolve()
    data = manifest.get("data", {})
    if (
        Path(data.get("path", "")).resolve() != data_path
        or data.get("sha256") != sha256_file(data_path)
    ):
        raise ValueError("fresh stock data binding drift")
    _validate_train_only_yaml(data_path)

    project = Path(args.project).resolve()
    target = project / args.name
    endpoint = manifest.get("endpoint", {})
    if (
        Path(endpoint.get("project", "")).resolve() != project
        or endpoint.get("name") != args.name
        or Path(endpoint.get("target_dir", "")).resolve() != target
    ):
        raise ValueError("fresh stock endpoint drift")
    if require_fresh_target and target.exists():
        raise ValueError("fresh stock output target already exists")
    if not require_fresh_target and not target.is_dir():
        raise ValueError("completed fresh stock output target is missing")
    if target.as_posix().startswith("/mnt/uav/"):
        raise ValueError("fresh stock refuses output under /mnt/uav")
    if (
        require_fresh_target
        and shutil.disk_usage(_existing_parent(project)).free
        < MIN_FREE_BYTES
    ):
        raise ValueError("fresh stock requires at least 3 GiB free")
    return manifest


__all__ = [
    "EXPECTED_ENVIRONMENT",
    "EXPECTED_UPSTREAM_SOURCE_SHA256",
    "PROTOCOL_SCHEMA",
    "SOURCE_FILES",
    "build_parser",
    "build_settings",
    "current_commit",
    "current_environment",
    "current_upstream_source_hashes",
    "sha256_file",
    "source_bundle_sha256",
    "source_closure",
    "validate_protocol_inputs",
]

"""Authority helpers for fresh-stock single-endpoint SADED evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.saded import (
    FRAGMENT_IOS,
    LARGE_EFFECTIVE_SIZE,
    MATCH_IOU,
    MAX_DET,
    ROUTER_K,
    TINY_EFFECTIVE_SIZE,
)
from src.saded_single_model_adjudicator import FORMAL_THRESHOLDS
from src.sbr_artifacts import sha256_file
from src.sbr_g0 import FrozenSBRProtocol
from src.tascv_protocol import (
    REPO_SOURCE_FILES,
    require_clean_repo,
    source_bundle_sha256,
)


SCHEMA_VERSION = "saded-fresh-stock-evaluation-protocol/v1"
EXPECTED_IMAGE_LIST_SHA256 = (
    "87C1B9FE8CD39CAF7F46494E7FE55DC4315573B64EE83A5B71778DBF55933B3A"
)
EXPECTED_DATASET_SIGNATURE = (
    "A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A"
)
EXPECTED_DATASET_YAML = (
    "/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml"
)
EXPECTED_DATASET_YAML_SHA256 = (
    "7EB91FCEF62A687A26A8EF76E9075B9793B52BC8BB110E4235FACF3E2B958324"
)
EXPECTED_IMAGE_ROOT = "/mnt/uav/datasets/VisDrone/images/val"
EXPECTED_IMAGE_COUNT = 548
POSTPROCESS_SOURCE_FILES = tuple(
    sorted(
        set(REPO_SOURCE_FILES)
        | {
            "docs/superpowers/specs/"
            "2026-07-25-saded-fresh100-postprocess-design.md",
            "scripts/adjudicate_saded_stock_fresh.py",
            "scripts/cache_saded_stock_endpoint.py",
            "scripts/evaluate_saded_stock_single.py",
            "scripts/prepare_saded_stock_evaluation_protocol.py",
            "scripts/route_saded_stock_single.py",
            "src/saded_single_model_adjudicator.py",
            "src/saded_single_model_evidence.py",
            "src/saded_stock_evaluation_protocol.py",
            "src/saded_stock_postprocess.py",
            "tests/test_saded_stock_evaluation_protocol.py",
            "tests/test_saded_stock_postprocess.py",
        }
    )
)


def reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            reject_forbidden(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            reject_forbidden(nested)
    elif isinstance(value, (str, Path)):
        normalized = str(value).replace("\\", "/").lower()
        if "test-dev" in normalized or "test_dev" in normalized:
            raise ValueError(f"test-dev is forbidden: {value}")


def frozen_route_contract() -> dict[str, Any]:
    return {
        "tiny_effective_size": TINY_EFFECTIVE_SIZE,
        "large_effective_size": LARGE_EFFECTIVE_SIZE,
        "match_iou_strictly_greater_than": MATCH_IOU,
        "fragment_ios": FRAGMENT_IOS,
        "router_k": ROUTER_K,
        "max_det": MAX_DET,
        "views": ["full", "TL", "TR", "BL", "BR"],
        "sbr_protocol": dict(FrozenSBRProtocol().__dict__),
        "formal_thresholds": dict(FORMAL_THRESHOLDS),
    }


def canonical_image_list_bytes(image_list: list[str]) -> bytes:
    return (
        json.dumps(
            image_list,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def build_image_authority(
    image_root: Path | str,
    image_list: list[str],
) -> dict[str, Any]:
    root = Path(image_root).resolve()
    if (
        not image_list
        or len(set(image_list)) != len(image_list)
        or any(not isinstance(item, str) or not item for item in image_list)
    ):
        raise ValueError("image authority requires nonempty unique IDs")
    records: list[dict[str, Any]] = []
    for image_id in image_list:
        candidate = Path(image_id)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("image authority ID is not canonical")
        path = (root / candidate).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"image authority file is missing: {image_id}")
        records.append(
            {
                "image_id": image_id,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "saded-dev-val-image-authority/v1",
        "root": root.as_posix(),
        "image_count": len(records),
        "image_list_sha256": hashlib.sha256(
            canonical_image_list_bytes(image_list)
        ).hexdigest().upper(),
        "images": records,
    }


def postprocess_source_closure(
    repo_root: Path | str,
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    return {
        relative: sha256_file(root / relative)
        for relative in POSTPROCESS_SOURCE_FILES
    }


def current_commit(repo_root: Path | str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root).resolve(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def postprocess_source_state(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    require_clean_repo(root)
    files = postprocess_source_closure(root)
    return {
        "commit": current_commit(root),
        "files": files,
        "bundle_sha256": source_bundle_sha256(files),
    }


__all__ = [
    "EXPECTED_DATASET_SIGNATURE",
    "EXPECTED_DATASET_YAML",
    "EXPECTED_DATASET_YAML_SHA256",
    "EXPECTED_IMAGE_COUNT",
    "EXPECTED_IMAGE_LIST_SHA256",
    "EXPECTED_IMAGE_ROOT",
    "POSTPROCESS_SOURCE_FILES",
    "SCHEMA_VERSION",
    "build_image_authority",
    "canonical_image_list_bytes",
    "current_commit",
    "frozen_route_contract",
    "postprocess_source_closure",
    "postprocess_source_state",
    "reject_forbidden",
]

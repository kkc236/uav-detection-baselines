"""Authority helpers for fresh-stock single-endpoint SADED evaluation."""

from __future__ import annotations

from argparse import Namespace
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
            "scripts/route_saded.py",
            "scripts/route_saded_stock_single.py",
            "scripts/train_rtdetr_saded_stock.py",
            "src/saded_single_model_adjudicator.py",
            "src/saded_single_model_evidence.py",
            "src/saded_stock_cli.py",
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digests_equal(left: object, right: object) -> bool:
    return str(left).lower() == str(right).lower()


def _bound_file(record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} binding schema drift")
    path = Path(str(record["path"])).resolve()
    reject_forbidden(path)
    if (
        not path.is_file()
        or not digests_equal(sha256_file(path), record["sha256"])
    ):
        raise ValueError(f"{label} binding drift")
    return path


def verify_named_checksums(
    checksum_path: Path | str,
    *,
    root: Path | str,
    expected_names: set[str],
) -> dict[str, str]:
    path = Path(checksum_path).resolve()
    artifact_root = Path(root).resolve()
    observed: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        observed[relative] = digest.lower()
    if set(observed) != expected_names or any(
        sha256_file(artifact_root / relative) != digest
        for relative, digest in observed.items()
    ):
        raise ValueError("fresh evaluation protocol checksum drift")
    return observed


def verify_image_authority(
    authority: dict[str, Any],
    image_list: list[str],
    *,
    verify_bytes: bool,
) -> None:
    if (
        authority.get("schema_version")
        != "saded-dev-val-image-authority/v1"
        or authority.get("root")
        != Path(EXPECTED_IMAGE_ROOT).resolve().as_posix()
        or authority.get("image_count") != EXPECTED_IMAGE_COUNT
        or authority.get("image_list_sha256")
        != EXPECTED_IMAGE_LIST_SHA256
        or len(authority.get("images", [])) != EXPECTED_IMAGE_COUNT
        or [row.get("image_id") for row in authority["images"]]
        != image_list
    ):
        raise ValueError("dev-val image authority drift")
    if verify_bytes:
        root = Path(EXPECTED_IMAGE_ROOT).resolve()
        for row in authority["images"]:
            path = (root / str(row["image_id"])).resolve()
            if (
                root not in path.parents
                or not path.is_file()
                or path.stat().st_size != row.get("size")
                or sha256_file(path) != row.get("sha256")
            ):
                raise ValueError(
                    f"dev-val image bytes drift: {row.get('image_id')}"
                )


def validate_completed_stock_endpoint(
    protocol: dict[str, Any],
    *,
    training_repo: Path,
) -> dict[str, Any]:
    from scripts.train_rtdetr_saded_stock import validate_runtime_summary
    from src.saded_stock_cli import (
        build_parser as build_stock_parser,
        validate_protocol_inputs,
    )

    training = protocol.get("training", {})
    stock_protocol_path = _bound_file(
        training.get("protocol"),
        label="stock training protocol",
    )
    summary_path = _bound_file(
        training.get("summary"),
        label="stock training summary",
    )
    checkpoint_path = _bound_file(
        training.get("checkpoint"),
        label="stock checkpoint",
    )
    stock_protocol = _read_json(stock_protocol_path)
    endpoint = stock_protocol.get("endpoint", {})
    initial = stock_protocol.get("initial_state", {})
    data = stock_protocol.get("data", {})
    args = build_stock_parser().parse_args(
        [
            "--protocol-manifest",
            str(stock_protocol_path),
            "--initial-state",
            str(initial.get("path", "")),
            "--data",
            str(data.get("path", "")),
            "--project",
            str(endpoint.get("project", "")),
            "--name",
            str(endpoint.get("name", "")),
            "--device",
            "0",
            "--seed",
            "0",
        ]
    )
    validate_protocol_inputs(
        args,
        repo_root=training_repo,
        require_fresh_target=False,
    )
    expected_summary = (
        Path(endpoint["target_dir"]).resolve()
        / "saded_stock_training_summary.json"
    )
    expected_checkpoint = (
        Path(endpoint["target_dir"]).resolve() / "weights" / "last.pt"
    )
    if (
        summary_path != expected_summary
        or checkpoint_path != expected_checkpoint
    ):
        raise ValueError("completed stock endpoint path drift")
    summary = _read_json(summary_path)
    failures = validate_runtime_summary(summary)
    if failures:
        raise ValueError(
            "completed stock endpoint runtime drift: " + "; ".join(failures)
        )
    exact = {
        "schema_version": "saded-stock-training-summary/v1",
        "stage": "FORMAL_100",
        "arm": "stock_control",
        "seed": 0,
        "protocol_manifest": stock_protocol_path.as_posix(),
        "protocol_manifest_sha256": sha256_file(stock_protocol_path),
        "protocol_source_commit": stock_protocol["runtime_source"]["commit"],
        "source_repo_bundle_sha256": stock_protocol["runtime_source"][
            "repo_bundle_sha256"
        ],
        "initial_state": initial["path"],
        "initial_state_sha256": initial["sha256"],
        "initial_state_common_fingerprint": initial[
            "common_fingerprint"
        ],
        "data": data["path"],
        "data_sha256": data["sha256"],
    }
    for key, value in exact.items():
        actual = summary.get(key)
        matches = (
            digests_equal(actual, value)
            if key.endswith("sha256")
            else actual == value
        )
        if not matches:
            raise ValueError(f"completed stock summary drift: {key}")
    if (
        summary.get("checkpoint", {}).get("path")
        != checkpoint_path.as_posix()
        or not digests_equal(
            summary["checkpoint"].get("sha256"),
            sha256_file(checkpoint_path),
        )
    ):
        raise ValueError("completed stock checkpoint summary drift")
    return summary


def validate_evaluation_protocol(
    manifest_path: Path | str,
    *,
    repo_root: Path | str,
    verify_images: bool,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    reject_forbidden(path)
    protocol = _read_json(path)
    reject_forbidden(protocol)
    if (
        not isinstance(protocol, dict)
        or protocol.get("schema_version") != SCHEMA_VERSION
        or path.name != "protocol_manifest.json"
        or path.parent.name != protocol.get("run_id")
    ):
        raise ValueError("fresh evaluation protocol identity drift")
    if protocol.get("source") != postprocess_source_state(repo_root):
        raise ValueError("fresh evaluation source closure drift")
    if protocol.get("route_contract") != frozen_route_contract():
        raise ValueError("fresh evaluation route contract drift")
    if protocol.get("dataset") != {
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
    }:
        raise ValueError("fresh evaluation dataset authority drift")
    if not digests_equal(
        sha256_file(Path(EXPECTED_DATASET_YAML)),
        EXPECTED_DATASET_YAML_SHA256,
    ):
        raise ValueError("fresh evaluation dataset YAML drift")

    artifacts = protocol.get("protocol_artifacts", {})
    image_list_path = _bound_file(
        artifacts.get("image_list"),
        label="evaluation image list",
    )
    authority_path = _bound_file(
        artifacts.get("image_authority"),
        label="evaluation image authority",
    )
    _bound_file(
        artifacts.get("endpoint_anchor"),
        label="stock endpoint anchor",
    )
    image_list = _read_json(image_list_path)
    if (
        not isinstance(image_list, list)
        or len(image_list) != EXPECTED_IMAGE_COUNT
        or hashlib.sha256(canonical_image_list_bytes(image_list))
        .hexdigest()
        .upper()
        != EXPECTED_IMAGE_LIST_SHA256
    ):
        raise ValueError("fresh evaluation image list drift")
    authority = _read_json(authority_path)
    verify_image_authority(
        authority,
        image_list,
        verify_bytes=verify_images,
    )

    checksum_path = path.parent / "checksums.sha256"
    anchor_path = (
        path.parent.parent / f"{protocol['run_id']}_protocol_anchor.json"
    )
    expected_names = {
        "endpoint_anchor.json",
        "image_authority.json",
        "image_list.json",
        "protocol_manifest.json",
    }
    verify_named_checksums(
        checksum_path,
        root=path.parent,
        expected_names=expected_names,
    )
    anchor = _read_json(anchor_path)
    if (
        anchor.get("schema_version")
        != "saded-fresh-stock-evaluation-protocol-anchor/v1"
        or anchor.get("manifest_sha256") != sha256_file(path)
        or anchor.get("checksums_sha256") != sha256_file(checksum_path)
    ):
        raise ValueError("fresh evaluation protocol anchor drift")
    training_repo = Path(
        protocol.get("training", {}).get("source_repo", "")
    ).resolve()
    validate_completed_stock_endpoint(
        protocol,
        training_repo=training_repo,
    )
    return protocol


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
    "digests_equal",
    "frozen_route_contract",
    "postprocess_source_closure",
    "postprocess_source_state",
    "reject_forbidden",
    "validate_completed_stock_endpoint",
    "validate_evaluation_protocol",
    "verify_image_authority",
    "verify_named_checksums",
]

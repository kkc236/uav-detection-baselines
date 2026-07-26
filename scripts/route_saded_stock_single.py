#!/usr/bin/env python3
"""Route one fresh-stock cache into A and SADED-SM without GT."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_saded_stock_endpoint import (  # noqa: E402
    CACHE_ARTIFACTS,
    CACHE_SCHEMA,
)
from scripts.route_saded import _aggregate_capacity  # noqa: E402
from src.saded_stock_evaluation_protocol import (  # noqa: E402
    postprocess_source_state,
    reject_forbidden,
    validate_evaluation_protocol,
)
from src.saded_stock_postprocess import route_single_cache  # noqa: E402
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl_gz,
    sha256_file,
    write_checksums,
)


ROUTE_SCHEMA = "saded-fresh-stock-single-route/v1"
ROUTE_ARTIFACTS = {
    "route_manifest.json",
    "predictions.jsonl.gz",
    "capacity.json",
    "route_invariants.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal fresh-stock single-cache SADED route."
    )
    parser.add_argument(
        "--evaluation-protocol",
        required=True,
        type=Path,
    )
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [json.loads(line) for line in handle]


def _verify_checksums(
    root: Path,
    artifacts: set[str],
) -> dict[str, str]:
    path = root / "checksums.sha256"
    lines = path.read_text(encoding="ascii").splitlines()
    observed: dict[str, str] = {}
    for line in lines:
        digest, relative = line.split("  ", 1)
        observed[relative] = digest.lower()
    if set(observed) != artifacts or any(
        sha256_file(root / relative) != digest
        for relative, digest in observed.items()
    ):
        raise ValueError("fresh cache checksum closure drift")
    return observed


def _verify_cache(
    protocol: dict[str, Any],
    protocol_path: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    cache = Path(protocol["outputs"]["cache"]).resolve()
    anchor_path = cache.parent / "cache_anchor.json"
    expected = CACHE_ARTIFACTS | {"checksums.sha256"}
    if (
        not cache.is_dir()
        or {item.name for item in cache.iterdir()} != expected
        or not anchor_path.is_file()
    ):
        raise ValueError("fresh cache artifact set drift")
    checksums = _verify_checksums(cache, CACHE_ARTIFACTS)
    manifest = _read_json(cache / "cache_manifest.json")
    invariants = _read_json(cache / "cache_invariants.json")
    anchor = _read_json(anchor_path)
    if (
        manifest.get("schema_version") != CACHE_SCHEMA
        or manifest.get("evaluation_protocol", {}).get("sha256")
        != sha256_file(protocol_path)
        or manifest.get("source") != protocol["source"]
        or manifest.get("checkpoint")
        != protocol["training"]["checkpoint"]
        or invariants.get("passed") is not True
        or anchor.get("schema_version")
        != "saded-fresh-stock-cache-anchor/v1"
        or anchor.get("cache_checksums_sha256")
        != sha256_file(cache / "checksums.sha256")
        or anchor.get("cache_manifest_sha256")
        != checksums["cache_manifest.json"]
        or anchor.get("predictions_sha256")
        != checksums["predictions.jsonl.gz"]
    ):
        raise ValueError("fresh cache nested closure drift")
    rows = _read_jsonl_gz(cache / "predictions.jsonl.gz")
    if (
        len(rows) != 548
        or [row.get("image_id") for row in rows]
        != _read_json(
            Path(
                protocol["protocol_artifacts"]["image_list"]["path"]
            )
        )
    ):
        raise ValueError("fresh cache image identity drift")
    return cache, rows, anchor


def route(args: argparse.Namespace) -> Path:
    reject_forbidden(vars(args))
    if "src.sbr_metrics" in sys.modules:
        raise ValueError("GT evaluator imported before fresh routing")
    protocol_path = args.evaluation_protocol.resolve()
    protocol = validate_evaluation_protocol(
        protocol_path,
        repo_root=REPO_ROOT,
        verify_images=False,
    )
    output = Path(protocol["outputs"]["route"]).resolve()
    anchor_path = output.parent / "route_anchor.json"
    if output.exists() or anchor_path.exists():
        raise FileExistsError("fresh route target exists")
    source_before = postprocess_source_state(REPO_ROOT)
    cache, cache_rows, cache_anchor = _verify_cache(
        protocol,
        protocol_path,
    )
    cache_snapshot = {
        item.name: sha256_file(item)
        for item in cache.iterdir()
        if item.is_file()
    }
    cache_snapshot["cache_anchor.json"] = sha256_file(
        cache.parent / "cache_anchor.json"
    )
    rows, invariants = route_single_cache(cache_rows)
    capacity = _aggregate_capacity(rows)
    source_after = postprocess_source_state(REPO_ROOT)
    cache_snapshot_after = {
        item.name: sha256_file(item)
        for item in cache.iterdir()
        if item.is_file()
    }
    cache_snapshot_after["cache_anchor.json"] = sha256_file(
        cache.parent / "cache_anchor.json"
    )
    invariants.update(
        {
            "image_count_exact": len(rows) == 548,
            "source_unchanged": source_before == source_after,
            "cache_snapshot_unchanged": (
                cache_snapshot == cache_snapshot_after
            ),
            "gt_module_absent": "src.sbr_metrics" not in sys.modules,
        }
    )
    invariants["passed"] = all(
        value is True
        for key, value in invariants.items()
        if key != "image_count"
    )
    if not invariants["passed"]:
        raise ValueError("fresh single-cache route invariants failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".route-staging-", dir=output.parent)
    )
    try:
        predictions_path = atomic_write_jsonl_gz(
            staging / "predictions.jsonl.gz",
            rows,
        )
        capacity_path = atomic_write_json(
            staging / "capacity.json",
            capacity,
        )
        invariants_path = atomic_write_json(
            staging / "route_invariants.json",
            invariants,
        )
        manifest_path = atomic_write_json(
            staging / "route_manifest.json",
            {
                "schema_version": ROUTE_SCHEMA,
                "evaluation_protocol_sha256": sha256_file(protocol_path),
                "source": source_after,
                "cache": {
                    "path": cache.as_posix(),
                    "anchor_sha256": sha256_file(
                        cache.parent / "cache_anchor.json"
                    ),
                    "checksums_sha256": sha256_file(
                        cache / "checksums.sha256"
                    ),
                    "manifest_sha256": sha256_file(
                        cache / "cache_manifest.json"
                    ),
                    "predictions_sha256": sha256_file(
                        cache / "predictions.jsonl.gz"
                    ),
                },
                "checkpoint": protocol["training"]["checkpoint"],
                "route_contract": protocol["route_contract"],
                "arms": ["A", "route_control"],
                "image_count": len(rows),
                "predictions_sha256": sha256_file(predictions_path),
                "capacity_sha256": sha256_file(capacity_path),
                "invariants_sha256": sha256_file(invariants_path),
                "required_artifacts": sorted(
                    ROUTE_ARTIFACTS | {"checksums.sha256"}
                ),
            },
        )
        checksums = write_checksums(
            staging / "checksums.sha256",
            [
                manifest_path,
                predictions_path,
                capacity_path,
                invariants_path,
            ],
            root=staging,
        )
        staging.rename(output)
        atomic_write_json(
            anchor_path,
            {
                "schema_version": (
                    "saded-fresh-stock-single-route-anchor/v1"
                ),
                "route_checksums_sha256": sha256_file(
                    output / checksums.name
                ),
                "route_manifest_sha256": sha256_file(
                    output / manifest_path.name
                ),
                "predictions_sha256": sha256_file(
                    output / predictions_path.name
                ),
                "cache_anchor_sha256": sha256_file(
                    cache.parent / "cache_anchor.json"
                ),
                "evaluation_protocol_sha256": sha256_file(protocol_path),
            },
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if anchor_path.exists():
            anchor_path.unlink()
        raise
    return output


def main() -> None:
    args = build_parser().parse_args()
    print(route(args))


if __name__ == "__main__":
    main()

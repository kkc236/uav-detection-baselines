#!/usr/bin/env python3
"""Independently adjudicate the sealed fresh-stock five-gate result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_saded_stock_single import (  # noqa: E402
    EVALUATION_ARTIFACTS,
    EVALUATION_SCHEMA,
    _snapshot,
    _verify_route,
)
from src.saded_single_model_adjudicator import (  # noqa: E402
    adjudicate_single_model,
)
from src.saded_stock_evaluation_protocol import (  # noqa: E402
    postprocess_source_state,
    reject_forbidden,
    validate_evaluation_protocol,
    verify_named_checksums,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    write_checksums,
)
from src.sbr_ppaf import metric_deltas  # noqa: E402


ADJUDICATION_SCHEMA = "saded-fresh-stock-single-adjudication-manifest/v1"
ADJUDICATION_ARTIFACTS = {
    "manifest.json",
    "bindings.json",
    "adjudication.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjudicate the fresh-stock single-route five gates."
    )
    parser.add_argument(
        "--evaluation-protocol",
        required=True,
        type=Path,
    )
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decide(
    *,
    arm_a: dict[str, Any],
    route_control: dict[str, Any],
    invariants_passed: bool,
) -> dict[str, Any]:
    """Recompute the five frozen gates from absolute sealed metrics."""

    return adjudicate_single_model(
        arm_a=arm_a,
        route_control=route_control,
        invariants_passed=invariants_passed,
    )


def exit_code_for_decision(decision: str) -> int:
    return {
        "SADED_SINGLE_SEED_GO": 0,
        "SADED_SINGLE_SEED_STOP": 1,
        "INVALID": 2,
    }.get(decision, 2)


def _verify_evaluation(
    protocol: dict[str, Any],
    protocol_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    evaluation = Path(protocol["outputs"]["evaluation"]).resolve()
    anchor_path = evaluation.parent / "evaluation_anchor.json"
    claim_path = Path(protocol["outputs"]["evaluation_claim"]).resolve()
    expected = EVALUATION_ARTIFACTS | {"checksums.sha256"}
    if (
        not evaluation.is_dir()
        or {item.name for item in evaluation.iterdir()} != expected
        or not anchor_path.is_file()
        or not claim_path.is_file()
    ):
        raise ValueError("fresh evaluation artifact set drift")
    checksums = verify_named_checksums(
        evaluation / "checksums.sha256",
        root=evaluation,
        expected_names=EVALUATION_ARTIFACTS,
    )
    manifest = _read_json(evaluation / "evaluation_manifest.json")
    invariants = _read_json(evaluation / "evaluation_invariants.json")
    metrics = _read_json(evaluation / "metrics.json")
    deltas = _read_json(evaluation / "deltas.json")
    anchor = _read_json(anchor_path)
    claim = _read_json(claim_path)
    route_path = Path(protocol["outputs"]["route"]).resolve()
    route_anchor_path = route_path.parent / "route_anchor.json"
    if (
        manifest.get("schema_version") != EVALUATION_SCHEMA
        or manifest.get("evaluation_protocol_sha256")
        != sha256_file(protocol_path)
        or manifest.get("source") != protocol["source"]
        or manifest.get("route", {}).get("path")
        != route_path.as_posix()
        or manifest.get("route", {}).get("anchor_sha256")
        != sha256_file(route_anchor_path)
        or manifest.get("claim", {}).get("path")
        != claim_path.as_posix()
        or manifest.get("claim", {}).get("sha256")
        != sha256_file(claim_path)
        or manifest.get("claim", {}).get("retry_permitted") is not False
        or manifest.get("dataset") != protocol["dataset"]
        or manifest.get("checkpoint")
        != protocol["training"]["checkpoint"]
        or manifest.get("arms") != ["A", "route_control"]
        or manifest.get("image_count") != 548
        or manifest.get("artifacts", {}).get("metrics_sha256")
        != checksums["metrics.json"]
        or manifest.get("artifacts", {}).get("deltas_sha256")
        != checksums["deltas.json"]
        or manifest.get("artifacts", {}).get("invariants_sha256")
        != checksums["evaluation_invariants.json"]
        or invariants.get("passed") is not True
        or set(metrics) != {"A", "route_control"}
        or deltas
        != metric_deltas(metrics["route_control"], metrics["A"])
        or claim.get("schema_version")
        != "saded-fresh-stock-evaluation-claim/v1"
        or claim.get("state") != "CONSUMED"
        or claim.get("retry_permitted") is not False
        or claim.get("evaluation_protocol_sha256")
        != sha256_file(protocol_path)
        or claim.get("route_anchor_sha256")
        != sha256_file(route_anchor_path)
        or anchor.get("schema_version")
        != "saded-fresh-stock-single-evaluation-anchor/v1"
        or anchor.get("evaluation_checksums_sha256")
        != sha256_file(evaluation / "checksums.sha256")
        or anchor.get("evaluation_manifest_sha256")
        != checksums["evaluation_manifest.json"]
        or anchor.get("metrics_sha256") != checksums["metrics.json"]
        or anchor.get("route_anchor_sha256")
        != sha256_file(route_anchor_path)
        or anchor.get("claim_sha256") != sha256_file(claim_path)
        or anchor.get("evaluation_protocol_sha256")
        != sha256_file(protocol_path)
    ):
        raise ValueError("fresh evaluation nested closure drift")
    paths = [
        claim_path,
        anchor_path,
        *(evaluation / name for name in expected),
    ]
    return metrics, deltas, manifest, _snapshot(paths)


def adjudicate(args: argparse.Namespace) -> Path:
    reject_forbidden(vars(args))
    protocol_path = args.evaluation_protocol.resolve()
    protocol = validate_evaluation_protocol(
        protocol_path,
        repo_root=REPO_ROOT,
        verify_images=False,
    )
    output = Path(protocol["outputs"]["adjudication"]).resolve()
    anchor_path = output.parent / "adjudication_anchor.json"
    if output.exists() or anchor_path.exists():
        raise FileExistsError("fresh adjudication target exists")
    source_before = postprocess_source_state(REPO_ROOT)
    _, _, route_snapshot = _verify_route(protocol, protocol_path)
    metrics, recorded_deltas, evaluation_manifest, evaluation_snapshot = (
        _verify_evaluation(protocol, protocol_path)
    )
    evidence_invariants = {
        "protocol_valid": True,
        "route_snapshot_matches_evaluation": (
            evaluation_manifest.get("route", {}).get("snapshot")
            == route_snapshot
        ),
        "evaluation_closure_valid": True,
        "metric_arms_exact": set(metrics) == {"A", "route_control"},
        "recorded_deltas_exact": (
            recorded_deltas
            == metric_deltas(metrics["route_control"], metrics["A"])
        ),
        "image_count_exact": (
            evaluation_manifest.get("image_count") == 548
        ),
    }
    evidence_invariants["passed"] = all(evidence_invariants.values())
    decision = decide(
        arm_a=metrics.get("A", {}),
        route_control=metrics.get("route_control", {}),
        invariants_passed=evidence_invariants["passed"],
    )

    _, _, route_snapshot_after = _verify_route(protocol, protocol_path)
    _, _, _, evaluation_snapshot_after = _verify_evaluation(
        protocol,
        protocol_path,
    )
    source_after = postprocess_source_state(REPO_ROOT)
    evidence_invariants.update(
        {
            "source_unchanged": source_after == source_before,
            "route_snapshot_unchanged": (
                route_snapshot_after == route_snapshot
            ),
            "evaluation_snapshot_unchanged": (
                evaluation_snapshot_after == evaluation_snapshot
            ),
        }
    )
    evidence_invariants["passed"] = all(
        value is True
        for key, value in evidence_invariants.items()
        if key != "passed"
    )
    if not evidence_invariants["passed"]:
        decision = decide(
            arm_a={},
            route_control={},
            invariants_passed=False,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".adjudication-staging-",
            dir=output.parent,
        )
    )
    try:
        manifest_path = atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": ADJUDICATION_SCHEMA,
                "evaluation_protocol_sha256": sha256_file(protocol_path),
                "source": source_after,
                "checkpoint": protocol["training"]["checkpoint"],
                "image_count": 548,
                "method": "SADED-SM",
                "seed": 0,
                "detector_epochs": 100,
                "training_free_router": True,
                "decision": decision["decision"],
                "required_artifacts": sorted(
                    ADJUDICATION_ARTIFACTS | {"checksums.sha256"}
                ),
            },
        )
        bindings_path = atomic_write_json(
            staging / "bindings.json",
            {
                "protocol": {
                    "path": protocol_path.as_posix(),
                    "sha256": sha256_file(protocol_path),
                },
                "route_snapshot": route_snapshot,
                "evaluation_snapshot": evaluation_snapshot,
                "evidence_invariants": evidence_invariants,
            },
        )
        adjudication_path = atomic_write_json(
            staging / "adjudication.json",
            {
                **decision,
                "absolute_metrics": {
                    "A": metrics["A"],
                    "route_control": metrics["route_control"],
                },
            },
        )
        checksums_path = write_checksums(
            staging / "checksums.sha256",
            [
                manifest_path,
                bindings_path,
                adjudication_path,
            ],
            root=staging,
        )
        staging.rename(output)
        atomic_write_json(
            anchor_path,
            {
                "schema_version": (
                    "saded-fresh-stock-single-adjudication-anchor/v1"
                ),
                "checksums_sha256": sha256_file(
                    output / checksums_path.name
                ),
                "manifest_sha256": sha256_file(
                    output / manifest_path.name
                ),
                "adjudication_sha256": sha256_file(
                    output / adjudication_path.name
                ),
                "evaluation_anchor_sha256": sha256_file(
                    Path(protocol["outputs"]["evaluation"]).resolve().parent
                    / "evaluation_anchor.json"
                ),
                "decision": decision["decision"],
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = adjudicate(args)
        result = _read_json(output / "adjudication.json")
    except Exception as error:
        print(f"SADED_FRESH_INVALID: {error}", file=sys.stderr)
        return 2
    print(result["decision"])
    return exit_code_for_decision(result["decision"])


if __name__ == "__main__":
    raise SystemExit(main())

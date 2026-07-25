#!/usr/bin/env python3
"""Independently close the formal seed-0 single-model SADED result."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.saded_single_model_adjudicator import (  # noqa: E402
    adjudicate_single_model,
)
from src.saded_single_model_evidence import (  # noqa: E402
    load_bound_json,
    sha256_file,
    source_state,
    validate_checkpoint_metadata,
    validate_binding_hashes,
    verify_checksum_closure,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    write_checksums,
)
from src.tascv_protocol import reject_forbidden_path  # noqa: E402


EXPECTED_BINDINGS = {
    "tascv_protocol": (
        "13d0e3ef66bfa2d35bb6037640888f7ac97993f2e43c090c6ec261a9701c25e3"
    ),
    "tascv_gate": (
        "e039a9111e22badee2884fd25f25dd9b38805da29f678ffb5476abeed7652e40"
    ),
    "tascv_adjudication_anchor": (
        "8341638a812483d7f9bfdaa713b3615f7fbf0a5da9128606f9ccb2b7abe247dd"
    ),
    "r0_input_manifest": (
        "aa85a80d2f43bc0a72d6a083657aa2fe539746bb79f8cabbef71516dc014cbff"
    ),
    "r0_route_anchor": (
        "e3c3a391496774412c60c921bf2db11cdbc2de908a562e5ad173123f36fb077c"
    ),
    "r0_route_checksums": (
        "6a5a4430dc53d4b196364ea5022ef88fcb3b5d165053db808db54689f7bf74fe"
    ),
    "r0_predictions": (
        "4c8e4998f0cbdbbc5963fecbf05ac4dc26d56db6b95d71a076fd129a66aa740e"
    ),
    "r0_evaluation_checksums": (
        "7a9598773b7c4b32ffe0d1658f785d4131146438015cbd3a32a2c946cb1efc69"
    ),
    "r0_metrics": (
        "1708f636d60d16090b69e691a2d4d28ba16af202ae9821ed00bf97c31f45905e"
    ),
    "checkpoint": (
        "54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b"
    ),
}
ROUTE_ARTIFACTS = {
    "route_manifest.json",
    "predictions.jsonl.gz",
    "capacity.json",
    "route_invariants.json",
}
EVALUATION_ARTIFACTS = {
    "evaluation_manifest.json",
    "metrics.json",
    "deltas.json",
    "capacity.json",
    "evaluation_invariants.json",
    "r0_gate.json",
}
OUTPUT_ARTIFACTS = {
    "manifest.json",
    "bindings.json",
    "adjudication.json",
}
PRIMARY_KEYS = tuple(
    (
        "AP-tiny-SBR",
        "mAP50-95",
        "tiny_recall",
        "AP75",
        "AP-large-SBR",
    )
)
SOURCE_FILES = (
    "docs/superpowers/specs/2026-07-25-saded-single-model-formal-design.md",
    "scripts/adjudicate_saded_single_model.py",
    "src/saded_single_model_adjudicator.py",
    "src/saded_single_model_evidence.py",
    "tests/test_saded_single_model_adjudicator.py",
    "tests/test_saded_single_model_cli.py",
    "tests/test_saded_single_model_evidence.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjudicate sealed single-model SADED seed-0 evidence"
    )
    parser.add_argument("--tascv-protocol", required=True, type=Path)
    parser.add_argument("--tascv-gate", required=True, type=Path)
    parser.add_argument(
        "--tascv-adjudication-anchor",
        required=True,
        type=Path,
    )
    parser.add_argument("--r0-input-manifest", required=True, type=Path)
    parser.add_argument("--r0-route-root", required=True, type=Path)
    parser.add_argument("--r0-evaluation", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _primary(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in PRIMARY_KEYS}


def adjudicate(args: argparse.Namespace) -> Path:
    safe_paths = {
        name: reject_forbidden_path(
            getattr(args, name),
            context="SADED single-model formal adjudication",
        )
        for name in (
            "tascv_protocol",
            "tascv_gate",
            "tascv_adjudication_anchor",
            "r0_input_manifest",
            "r0_route_root",
            "r0_evaluation",
            "checkpoint",
            "output",
        )
    }
    output = safe_paths["output"]
    anchor = output.parent / f"{output.name}_anchor.json"
    if output.exists() or anchor.exists():
        raise FileExistsError("formal output or external anchor already exists")

    route_root = safe_paths["r0_route_root"]
    route_dir = route_root / "route"
    evaluation_dir = safe_paths["r0_evaluation"]
    paths = {
        "tascv_protocol": safe_paths["tascv_protocol"],
        "tascv_gate": safe_paths["tascv_gate"],
        "tascv_adjudication_anchor": safe_paths[
            "tascv_adjudication_anchor"
        ],
        "r0_input_manifest": safe_paths["r0_input_manifest"],
        "r0_route_anchor": route_root / "route_anchor.json",
        "r0_route_checksums": route_dir / "checksums.sha256",
        "r0_predictions": route_dir / "predictions.jsonl.gz",
        "r0_evaluation_checksums": (
            evaluation_dir / "checksums.sha256"
        ),
        "r0_metrics": evaluation_dir / "metrics.json",
        "checkpoint": safe_paths["checkpoint"],
    }
    source_before = source_state(REPO_ROOT, SOURCE_FILES)
    bindings = validate_binding_hashes(paths, EXPECTED_BINDINGS)
    route_closure = verify_checksum_closure(
        route_dir,
        expected_artifacts=ROUTE_ARTIFACTS,
    )
    evaluation_closure = verify_checksum_closure(
        evaluation_dir,
        expected_artifacts=EVALUATION_ARTIFACTS,
    )

    stopped_gate = load_bound_json(
        paths["tascv_gate"],
        bindings["tascv_gate"],
    )
    stopped_anchor = load_bound_json(
        paths["tascv_adjudication_anchor"],
        bindings["tascv_adjudication_anchor"],
    )
    route_anchor = load_bound_json(
        paths["r0_route_anchor"],
        bindings["r0_route_anchor"],
    )
    route_manifest = load_bound_json(
        route_dir / "route_manifest.json",
        route_closure["artifacts"]["route_manifest.json"],
    )
    route_invariants = load_bound_json(
        route_dir / "route_invariants.json",
        route_closure["artifacts"]["route_invariants.json"],
    )
    evaluation_manifest = load_bound_json(
        evaluation_dir / "evaluation_manifest.json",
        evaluation_closure["artifacts"]["evaluation_manifest.json"],
    )
    evaluation_invariants = load_bound_json(
        evaluation_dir / "evaluation_invariants.json",
        evaluation_closure["artifacts"][
            "evaluation_invariants.json"
        ],
    )
    r0_gate = load_bound_json(
        evaluation_dir / "r0_gate.json",
        evaluation_closure["artifacts"]["r0_gate.json"],
    )
    metrics = load_bound_json(
        paths["r0_metrics"],
        bindings["r0_metrics"],
    )
    recorded_deltas = load_bound_json(
        evaluation_dir / "deltas.json",
        evaluation_closure["artifacts"]["deltas.json"],
    )
    stopped_checksums = paths["tascv_gate"].parent / "checksums.sha256"
    if not stopped_checksums.is_file():
        raise ValueError("T-ASCV adjudication checksum closure is missing")
    expected_tascv_checksum_line = (
        f"{bindings['tascv_gate']}  gate.json"
    )
    tascv_checksum_lines = stopped_checksums.read_text(
        encoding="ascii"
    ).splitlines()
    tascv_anchor_semantics = (
        stopped_anchor.get("schema_version")
        == "saded-adjudication-anchor/v1"
        and stopped_anchor.get("stage") == "SCREEN_SEED0"
        and stopped_anchor.get("decision") == "TASCV_STOP"
        and stopped_anchor.get("gate_sha256")
        == bindings["tascv_gate"]
        and stopped_anchor.get("checksums_sha256")
        == sha256_file(stopped_checksums)
        and tascv_checksum_lines == [expected_tascv_checksum_line]
    )
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required to authenticate checkpoint metadata"
        ) from exc
    checkpoint_payload = paths["checkpoint"].read_bytes()
    if (
        hashlib.sha256(checkpoint_payload).hexdigest()
        != bindings["checkpoint"]
    ):
        raise ValueError("checkpoint changed before metadata load")
    checkpoint_metadata = validate_checkpoint_metadata(
        torch.load(
            io.BytesIO(checkpoint_payload),
            map_location="cpu",
            weights_only=False,
        )
    )

    evidence_invariants = {
        "tascv_is_terminal_stop": (
            stopped_gate.get("decision") == "TASCV_STOP"
        ),
        "tascv_anchor_semantics": tascv_anchor_semantics,
        "route_anchor_binds_input": (
            route_anchor.get("input_manifest_sha256")
            == bindings["r0_input_manifest"]
        ),
        "route_anchor_binds_closure": (
            route_anchor.get("route_checksums_sha256")
            == bindings["r0_route_checksums"]
        ),
        "route_anchor_binds_predictions": (
            route_anchor.get("predictions_sha256")
            == bindings["r0_predictions"]
        ),
        "route_manifest_schema": (
            route_manifest.get("schema_version")
            == "sbr-saded-route/v1"
        ),
        "route_manifest_checkpoint": (
            route_manifest.get("input_file_sha256", {}).get(
                "checkpoint"
            )
            == bindings["checkpoint"]
        ),
        "route_manifest_input": (
            route_manifest.get("input_manifest_sha256")
            == bindings["r0_input_manifest"]
        ),
        "route_manifest_predictions": (
            route_manifest.get("predictions_sha256")
            == bindings["r0_predictions"]
        ),
        "route_source_exact": (
            route_manifest.get("route_source", {}).get("commit")
            == "ada48a1f09e468138e70eaa4b20cd426de6157da"
        ),
        "original_source_exact": (
            route_manifest.get("original_source", {}).get("commit")
            == "51ee6c446ffd967c12481894a9ac1cf00cad2105"
        ),
        "route_arms_exact": (
            route_manifest.get("arms") == ["A", "route_control"]
        ),
        "route_image_count": route_manifest.get("image_count") == 548,
        "route_invariants_passed": (
            route_invariants.get("passed") is True
        ),
        "evaluation_schema": (
            evaluation_manifest.get("schema_version")
            == "sbr-saded-r0-evaluation/v1"
        ),
        "evaluation_binds_route": (
            evaluation_manifest.get("route_checksums_sha256")
            == bindings["r0_route_checksums"]
            and evaluation_manifest.get("route_anchor_sha256")
            == bindings["r0_route_anchor"]
        ),
        "evaluation_decision_r0_go": (
            evaluation_manifest.get("decision") == "R0_GO"
            and r0_gate.get("decision") == "R0_GO"
        ),
        "evaluation_image_count": (
            evaluation_manifest.get("image_count") == 548
        ),
        "evaluation_invariants_passed": (
            evaluation_invariants.get("passed") is True
        ),
        "metric_arms_exact": set(metrics) == {"A", "route_control"},
        "route_closure_verified": route_closure.get("passed") is True,
        "evaluation_closure_verified": (
            evaluation_closure.get("passed") is True
        ),
        "checkpoint_epoch_100_verified": (
            checkpoint_metadata.get("passed") is True
        ),
    }
    evidence_invariants["passed"] = all(
        evidence_invariants.values()
    )
    decision = adjudicate_single_model(
        arm_a=metrics.get("A", {}),
        route_control=metrics.get("route_control", {}),
        invariants_passed=evidence_invariants["passed"],
    )
    if decision["decision"] != "INVALID":
        exact_recorded_delta = all(
            float(recorded_deltas[key]) == decision["deltas"][key]
            for key in PRIMARY_KEYS
        )
        evidence_invariants["recorded_delta_exact"] = (
            exact_recorded_delta
        )
        evidence_invariants["passed"] = (
            evidence_invariants["passed"] and exact_recorded_delta
        )
        if not exact_recorded_delta:
            decision = adjudicate_single_model(
                arm_a={},
                route_control={},
                invariants_passed=False,
            )
    source_after = source_state(REPO_ROOT, SOURCE_FILES)
    bindings_after = validate_binding_hashes(
        paths,
        EXPECTED_BINDINGS,
    )
    route_closure_after = verify_checksum_closure(
        route_dir,
        expected_artifacts=ROUTE_ARTIFACTS,
    )
    evaluation_closure_after = verify_checksum_closure(
        evaluation_dir,
        expected_artifacts=EVALUATION_ARTIFACTS,
    )
    inputs_unchanged = (
        bindings_after == bindings
        and route_closure_after == route_closure
        and evaluation_closure_after == evaluation_closure
        and stopped_anchor.get("checksums_sha256")
        == sha256_file(stopped_checksums)
    )
    source_unchanged = source_after == source_before
    evidence_invariants["successor_source_unchanged"] = (
        source_unchanged
    )
    evidence_invariants["passed"] = (
        evidence_invariants["passed"] and source_unchanged
    )
    evidence_invariants["input_snapshot_unchanged"] = (
        inputs_unchanged
    )
    evidence_invariants["passed"] = (
        evidence_invariants["passed"] and inputs_unchanged
    )
    if not source_unchanged or not inputs_unchanged:
        decision = adjudicate_single_model(
            arm_a={},
            route_control={},
            invariants_passed=False,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        manifest_path = atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": (
                    "saded-single-model-formal-manifest/v1"
                ),
                "method": "SADED route-control",
                "seed": 0,
                "detector_epochs": 100,
                "training_free_router": True,
                "image_count": 548,
                "parent_tascv_decision": "TASCV_STOP",
                "successor_source": source_after,
                "checkpoint_metadata": checkpoint_metadata,
                "decision": decision["decision"],
                "required_artifacts": sorted(
                    OUTPUT_ARTIFACTS | {"checksums.sha256"}
                ),
            },
        )
        bindings_path = atomic_write_json(
            staging / "bindings.json",
            {
                "expected": dict(EXPECTED_BINDINGS),
                "actual": bindings,
                "route_closure": route_closure,
                "evaluation_closure": evaluation_closure,
                "evidence_invariants": evidence_invariants,
                "successor_source": source_after,
                "checkpoint_metadata": checkpoint_metadata,
            },
        )
        adjudication_path = atomic_write_json(
            staging / "adjudication.json",
            {
                **decision,
                "absolute_metrics": {
                    "A": _primary(metrics["A"]),
                    "route_control": _primary(
                        metrics["route_control"]
                    ),
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
            anchor,
            {
                "schema_version": (
                    "saded-single-model-formal-anchor/v1"
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
                "decision": decision["decision"],
            },
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if anchor.exists():
            anchor.unlink()
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = adjudicate(args)
        result = _read_json(output / "adjudication.json")
    except Exception as exc:
        print(
            f"SADED_SINGLE_MODEL_INVALID: {exc}",
            file=sys.stderr,
        )
        return 2
    print(result["decision"])
    return 0 if result["decision"] == "SADED_SINGLE_SEED_GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())

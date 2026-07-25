"""Standalone verification and adjudication for sealed SADED stages."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from scripts.evaluate_saded_stage import (
    EVALUATION_FILES,
    _jsonable,
    _metric_row,
    _three_way_deltas,
)
from scripts.route_saded_pair import (
    _parse_checksums,
    _read_json,
    _snapshot,
)
from src.saded_stage import ROUTE_ARMS, screen_seed0_gate
from src.saded_stage_protocol import stage_source_state
from src.sbr_artifacts import sha256_file
from src.tascv_protocol import (
    FROZEN_FORMAL_THRESHOLDS,
    FROZEN_SCREEN_GATE,
    reject_forbidden_path,
)


def _strict_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_strict_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _strict_equal(a, b) for a, b in zip(left, right)
            )
        )
    if (
        isinstance(left, Real)
        and not isinstance(left, bool)
        and isinstance(right, Real)
        and not isinstance(right, bool)
    ):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and float(left) == float(right)
        )
    return type(left) is type(right) and left == right


def verify_evaluation_closure(
    evaluation_root: Path | str,
    *,
    evaluation_anchor_sha256: str,
    recompute_metrics: bool = True,
) -> dict[str, Any]:
    root = reject_forbidden_path(
        evaluation_root,
        context="SADED adjudicator evaluation root",
    )
    if (
        not root.is_dir()
        or {path.name for path in root.iterdir()}
        != {"evaluation", "evaluation_anchor.json"}
    ):
        raise ValueError("SADED evaluation root closure drift")
    anchor_path = root / "evaluation_anchor.json"
    expected_anchor = str(evaluation_anchor_sha256).lower()
    if (
        len(expected_anchor) != 64
        or sha256_file(anchor_path) != expected_anchor
    ):
        raise ValueError("SADED evaluation external anchor drift")
    evaluation_dir = root / "evaluation"
    expected_files = set(EVALUATION_FILES) | {"checksums.sha256"}
    if (
        not evaluation_dir.is_dir()
        or {path.name for path in evaluation_dir.iterdir()}
        != expected_files
    ):
        raise ValueError("SADED evaluation artifact closure drift")
    paths = [
        anchor_path,
        *(evaluation_dir / name for name in expected_files),
    ]
    before_snapshot = _snapshot(paths)
    checksums_path = evaluation_dir / "checksums.sha256"
    checksums = _parse_checksums(checksums_path)
    if set(checksums) != set(EVALUATION_FILES):
        raise ValueError("SADED evaluation checksum closure drift")
    for name, digest in checksums.items():
        if sha256_file(evaluation_dir / name) != digest:
            raise ValueError(f"SADED evaluation checksum drift: {name}")
    anchor = _read_json(anchor_path)
    if (
        set(anchor)
        != {
            "schema_version",
            "evaluation_manifest_sha256",
            "evaluation_checksums_sha256",
            "route_anchor_sha256",
            "training_protocol_sha256",
            "training_source_commit",
            "evaluation_source_commit",
        }
        or anchor["schema_version"]
        != "saded-stage-evaluation-anchor/v1"
        or anchor["evaluation_manifest_sha256"]
        != sha256_file(
            evaluation_dir / "evaluation_manifest.json"
        )
        or anchor["evaluation_checksums_sha256"]
        != sha256_file(checksums_path)
    ):
        raise ValueError("SADED evaluation anchor binding drift")
    manifest = _read_json(
        evaluation_dir / "evaluation_manifest.json"
    )
    source = stage_source_state(
        Path(__file__).resolve().parents[1]
    )
    if (
        manifest.get("schema_version")
        != "saded-stage-evaluation/v1"
        or manifest.get("required_artifacts")
        != list(EVALUATION_FILES) + ["checksums.sha256"]
        or manifest.get("evaluation_source") != source
        or manifest.get("identity")
        not in (
            {
                "stage": "SCREEN_10",
                "seed": 0,
            },
            {
                "stage": "SCREEN_10",
                "seed": 1,
            },
            {
                "stage": "SCREEN_10",
                "seed": 2,
            },
            {
                "stage": "FORMAL_100",
                "seed": 0,
            },
            {
                "stage": "FORMAL_100",
                "seed": 1,
            },
            {
                "stage": "FORMAL_100",
                "seed": 2,
            },
        )
        or anchor["evaluation_source_commit"] != source["commit"]
        or anchor["training_protocol_sha256"]
        != manifest["training_protocol"]["sha256"]
        or anchor["training_source_commit"]
        != manifest["training_protocol"]["source_commit"]
    ):
        raise ValueError("SADED evaluation manifest identity drift")
    artifact_bindings = manifest.get("artifacts")
    expected_artifacts = {
        "metrics_sha256": sha256_file(
            evaluation_dir / "metrics.json"
        ),
        "deltas_sha256": sha256_file(
            evaluation_dir / "deltas.json"
        ),
        "capacity_sha256": sha256_file(
            evaluation_dir / "capacity.json"
        ),
        "invariants_sha256": sha256_file(
            evaluation_dir / "evaluation_invariants.json"
        ),
    }
    if artifact_bindings != expected_artifacts:
        raise ValueError("SADED evaluation artifact binding drift")
    from scripts.evaluate_saded_stage import _verify_route

    route_binding = manifest["route"]
    route_manifest, route_rows, route_capacity, route_snapshot = (
        _verify_route(
            Path(route_binding["root"]),
            expected_anchor_sha256=route_binding["anchor_sha256"],
            evaluation_source=source,
        )
    )
    if (
        route_binding["manifest_sha256"]
        != sha256_file(
            Path(route_binding["root"])
            / "route/route_manifest.json"
        )
        or route_binding["snapshot"] != route_snapshot
        or anchor["route_anchor_sha256"]
        != route_binding["anchor_sha256"]
    ):
        raise ValueError("SADED evaluation route binding drift")
    metrics = _read_json(evaluation_dir / "metrics.json")
    deltas = _read_json(evaluation_dir / "deltas.json")
    invariants = _read_json(
        evaluation_dir / "evaluation_invariants.json"
    )
    capacity = _read_json(evaluation_dir / "capacity.json")
    if (
        set(metrics) != set(ROUTE_ARMS)
        or not _strict_equal(deltas, _three_way_deltas(metrics))
        or invariants.get("passed") is not True
        or not _strict_equal(capacity, route_capacity)
    ):
        raise ValueError("SADED evaluation semantic closure drift")
    if recompute_metrics:
        # This is a second, standalone GT-aware pass over the sealed route.
        from src.sbr_artifacts import load_dataset
        from src.sbr_metrics import evaluate_dataset

        protocol_path = Path(
            route_manifest["training_protocol"]["path"]
        )
        protocol = json.loads(
            protocol_path.read_text(encoding="utf-8")
        )
        dataset = load_dataset(
            Path(protocol["dataset"]["full_yaml"]),
            split="val",
            root_override=Path(protocol["dataset"]["root"]),
        )
        image_list = json.loads(
            Path(route_manifest["dataset"]["image_list"]).read_text(
                encoding="utf-8"
            )
        )
        sealed_dataset = manifest["dataset"]
        if (
            dataset["image_list"] != image_list
            or dataset["image_count"] != 548
            or sealed_dataset
            != {
                "root": dataset["root"].as_posix(),
                "yaml_path": dataset["yaml_path"].as_posix(),
                "yaml_hash": dataset["yaml_hash"],
                "dataset_signature": dataset[
                    "dataset_signature"
                ],
                "image_count": dataset["image_count"],
            }
        ):
            raise ValueError("SADED adjudicator dataset closure drift")
        image_by_id = {
            image["relative_path"]: image
            for image in dataset["images"]
        }
        rows_by_arm: dict[str, list[dict[str, Any]]] = {
            arm: [] for arm in ROUTE_ARMS
        }
        for route_row in route_rows:
            image = image_by_id[route_row["image_id"]]
            if (
                int(route_row["width"]) != int(image["width"])
                or int(route_row["height"]) != int(image["height"])
            ):
                raise ValueError(
                    "SADED adjudicator route dimension drift"
                )
            for arm in ROUTE_ARMS:
                rows_by_arm[arm].append(
                    _metric_row(image, route_row["arms"][arm])
                )
        replayed_metrics = {
            arm: _jsonable(evaluate_dataset(rows))
            for arm, rows in rows_by_arm.items()
        }
        if not _strict_equal(replayed_metrics, metrics):
            raise ValueError("SADED adjudicator metric replay drift")
    if _snapshot(paths) != before_snapshot:
        raise ValueError("SADED evaluation changed during adjudication")
    return {
        "root": root,
        "anchor_sha256": expected_anchor,
        "manifest": manifest,
        "metrics": metrics,
        "deltas": deltas,
        "invariants": invariants,
        "capacity": capacity,
        "source": source,
    }


def adjudicate_screen_seed0(
    evaluation_root: Path | str,
    *,
    evaluation_anchor_sha256: str,
) -> dict[str, Any]:
    verified = verify_evaluation_closure(
        evaluation_root,
        evaluation_anchor_sha256=evaluation_anchor_sha256,
        recompute_metrics=True,
    )
    manifest = verified["manifest"]
    if manifest["identity"] != {"stage": "SCREEN_10", "seed": 0}:
        raise ValueError("not a screen seed0 evaluation")
    metrics = verified["metrics"]
    decision = screen_seed0_gate(
        route_control=metrics["route_control"],
        route_treatment=metrics["route_treatment"],
        invariants_passed=verified["invariants"]["passed"],
    )
    decision.update(
        {
            "protocol_manifest_sha256": manifest[
                "training_protocol"
            ]["sha256"],
            "protocol_source_commit": manifest[
                "training_protocol"
            ]["source_commit"],
            "evaluation_source": verified["source"],
            "evaluation_binding": {
                "root": verified["root"].as_posix(),
                "anchor_sha256": verified["anchor_sha256"],
                "manifest_sha256": sha256_file(
                    verified["root"]
                    / "evaluation/evaluation_manifest.json"
                ),
                "checksums_sha256": sha256_file(
                    verified["root"]
                    / "evaluation/checksums.sha256"
                ),
                "metrics_sha256": sha256_file(
                    verified["root"] / "evaluation/metrics.json"
                ),
                "deltas_sha256": sha256_file(
                    verified["root"] / "evaluation/deltas.json"
                ),
            },
        }
    )
    return decision


def _mean_deltas(
    values: list[Mapping[str, Any]],
) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("three seed delta mean requires three values")
    keys = {
        "mAP50-95",
        "AP-tiny-SBR",
        "tiny_recall",
        "AP75",
        "AP-large-SBR",
    }
    if any(set(value) != keys for value in values):
        raise ValueError("three seed delta schema drift")
    return {
        key: sum(float(value[key]) for value in values) / 3.0
        for key in sorted(keys)
    }


def _evaluation_binding(verified: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(verified["root"])
    return {
        "root": root.as_posix(),
        "anchor_sha256": verified["anchor_sha256"],
        "manifest_sha256": sha256_file(
            root / "evaluation/evaluation_manifest.json"
        ),
        "checksums_sha256": sha256_file(
            root / "evaluation/checksums.sha256"
        ),
        "metrics_sha256": sha256_file(
            root / "evaluation/metrics.json"
        ),
        "deltas_sha256": sha256_file(
            root / "evaluation/deltas.json"
        ),
    }


def _verify_three_evaluations(
    evaluations: Mapping[int, tuple[Path | str, str]],
    *,
    stage: str,
    recompute_metrics: bool = True,
) -> dict[int, dict[str, Any]]:
    if set(evaluations) != {0, 1, 2}:
        raise ValueError("SADED three-seed evaluation set drift")
    verified = {
        seed: verify_evaluation_closure(
            root,
            evaluation_anchor_sha256=anchor,
            recompute_metrics=recompute_metrics,
        )
        for seed, (root, anchor) in evaluations.items()
    }
    protocol_bindings = {
        (
            item["manifest"]["training_protocol"]["sha256"],
            item["manifest"]["training_protocol"]["source_commit"],
            item["source"]["commit"],
        )
        for item in verified.values()
    }
    if (
        len(protocol_bindings) != 1
        or any(
            item["manifest"]["identity"]
            != {"stage": stage, "seed": seed}
            for seed, item in verified.items()
        )
    ):
        raise ValueError("SADED three-seed source/identity drift")
    return verified


def _attribution_three_seed_gate(
    verified: Mapping[int, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    deltas = [
        verified[seed]["deltas"][
            "route_treatment_vs_route_control"
        ]
        for seed in (0, 1, 2)
    ]
    mean = _mean_deltas(deltas)
    gate = FROZEN_SCREEN_GATE["three_seed"]
    positive_mAP = sum(
        float(item["mAP50-95"]) > 0.0 for item in deltas
    )
    nonnegative_tiny = sum(
        float(item["AP-tiny-SBR"]) >= 0.0 for item in deltas
    )
    nonnegative_recall = sum(
        float(item["tiny_recall"]) >= 0.0 for item in deltas
    )
    failures: list[str] = []
    if positive_mAP < gate["mAP_positive_wins_minimum"]:
        failures.append("mAP_positive_wins<2")
    if mean["mAP50-95"] <= 0.0:
        failures.append("mean_mAP50-95<=0")
    if (
        nonnegative_tiny
        < gate["AP-tiny-SBR_nonnegative_wins_minimum"]
    ):
        failures.append("AP-tiny-SBR_nonnegative_wins<2")
    if (
        nonnegative_recall
        < gate["tiny_recall_nonnegative_wins_minimum"]
    ):
        failures.append("tiny_recall_nonnegative_wins<2")
    for key, threshold in gate["mean_guards"].items():
        if mean[key] < threshold - 1e-12:
            failures.append(f"mean_{key}<{threshold}")
    return failures, {
        "per_seed": {
            str(seed): deltas[seed] for seed in (0, 1, 2)
        },
        "mean": mean,
        "counts": {
            "mAP_positive": positive_mAP,
            "AP-tiny-SBR_nonnegative": nonnegative_tiny,
            "tiny_recall_nonnegative": nonnegative_recall,
        },
        "thresholds": gate,
    }


def adjudicate_screen_three_seed(
    evaluations: Mapping[int, tuple[Path | str, str]],
) -> dict[str, Any]:
    verified = _verify_three_evaluations(
        evaluations,
        stage="SCREEN_10",
    )
    failures, attribution = _attribution_three_seed_gate(verified)
    first = verified[0]
    return {
        "schema_version":
            "tascv-screen-three-seed-adjudication/v1",
        "decision": (
            "TASCV_SCREEN_GO" if not failures else "TASCV_STOP"
        ),
        "failures": failures,
        "protocol_manifest_sha256": first["manifest"][
            "training_protocol"
        ]["sha256"],
        "protocol_source_commit": first["manifest"][
            "training_protocol"
        ]["source_commit"],
        "evaluation_source": first["source"],
        "evaluation_bindings": {
            str(seed): _evaluation_binding(verified[seed])
            for seed in (0, 1, 2)
        },
        "attribution": attribution,
    }


def _formal_primary_failures(
    deltas: Mapping[str, Any],
) -> list[str]:
    thresholds = FROZEN_FORMAL_THRESHOLDS[
        "primary_route_treatment_minus_arm_a"
    ]
    return [
        f"{key}<{threshold}"
        for key, threshold in thresholds.items()
        if float(deltas[key]) < threshold - 1e-12
    ]


def adjudicate_formal_seed0(
    evaluation_root: Path | str,
    *,
    evaluation_anchor_sha256: str,
) -> dict[str, Any]:
    verified = verify_evaluation_closure(
        evaluation_root,
        evaluation_anchor_sha256=evaluation_anchor_sha256,
        recompute_metrics=True,
    )
    if verified["manifest"]["identity"] != {
        "stage": "FORMAL_100",
        "seed": 0,
    }:
        raise ValueError("not a formal seed0 evaluation")
    primary = verified["deltas"]["route_treatment_vs_A"]
    attribution = verified["deltas"][
        "route_treatment_vs_route_control"
    ]
    failures = _formal_primary_failures(primary)
    if float(attribution["mAP50-95"]) <= 0.0:
        failures.append("attribution_mAP50-95<=0")
    manifest = verified["manifest"]
    return {
        "schema_version": "tascv-formal-seed0-adjudication/v1",
        "decision": (
            "TASCV_FORMAL_SEED0_GO"
            if not failures
            else "TASCV_STOP"
        ),
        "failures": failures,
        "protocol_manifest_sha256": manifest[
            "training_protocol"
        ]["sha256"],
        "protocol_source_commit": manifest[
            "training_protocol"
        ]["source_commit"],
        "evaluation_source": verified["source"],
        "evaluation_binding": _evaluation_binding(verified),
        "primary": {
            "deltas": primary,
            "thresholds": FROZEN_FORMAL_THRESHOLDS[
                "primary_route_treatment_minus_arm_a"
            ],
        },
        "attribution": {
            "deltas": attribution,
            "mAP50-95_strictly_greater_than": 0.0,
        },
    }


def adjudicate_formal_three_seed(
    evaluations: Mapping[int, tuple[Path | str, str]],
    *,
    recompute_metrics: bool = True,
) -> dict[str, Any]:
    verified = _verify_three_evaluations(
        evaluations,
        stage="FORMAL_100",
        recompute_metrics=recompute_metrics,
    )
    primary_per_seed = [
        verified[seed]["deltas"]["route_treatment_vs_A"]
        for seed in (0, 1, 2)
    ]
    primary_mean = _mean_deltas(primary_per_seed)
    failures = _formal_primary_failures(primary_mean)
    attribution_failures, attribution = (
        _attribution_three_seed_gate(verified)
    )
    failures.extend(
        f"attribution:{failure}"
        for failure in attribution_failures
    )
    first = verified[0]
    return {
        "schema_version":
            "tascv-formal-three-seed-adjudication/v1",
        "decision": (
            "TASCV_FORMAL_GO" if not failures else "TASCV_STOP"
        ),
        "failures": failures,
        "protocol_manifest_sha256": first["manifest"][
            "training_protocol"
        ]["sha256"],
        "protocol_source_commit": first["manifest"][
            "training_protocol"
        ]["source_commit"],
        "evaluation_source": first["source"],
        "evaluation_bindings": {
            str(seed): _evaluation_binding(verified[seed])
            for seed in (0, 1, 2)
        },
        "primary": {
            "per_seed": {
                str(seed): primary_per_seed[seed]
                for seed in (0, 1, 2)
            },
            "mean": primary_mean,
            "thresholds": FROZEN_FORMAL_THRESHOLDS[
                "primary_route_treatment_minus_arm_a"
            ],
        },
        "attribution": attribution,
    }


def replay_screen_seed0_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    if (
        gate.get("schema_version")
        != "tascv-screen-seed0-adjudication/v1"
        or gate.get("decision") != "TASCV_SCREEN_SEED0_GO"
    ):
        raise ValueError("not a T-ASCV screen seed0 GO gate")
    binding = gate.get("evaluation_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("T-ASCV screen evaluation binding missing")
    root = reject_forbidden_path(
        binding.get("root", ""),
        context="T-ASCV screen replay evaluation",
    )
    if (
        sha256_file(root / "evaluation/evaluation_manifest.json")
        != binding.get("manifest_sha256")
        or sha256_file(root / "evaluation/checksums.sha256")
        != binding.get("checksums_sha256")
        or sha256_file(root / "evaluation/metrics.json")
        != binding.get("metrics_sha256")
        or sha256_file(root / "evaluation/deltas.json")
        != binding.get("deltas_sha256")
    ):
        raise ValueError("T-ASCV screen evaluation binding drift")
    replayed = adjudicate_screen_seed0(
        root,
        evaluation_anchor_sha256=str(binding["anchor_sha256"]),
    )
    if not _strict_equal(dict(gate), replayed):
        raise ValueError("T-ASCV screen seed0 gate replay drift")
    return dict(gate)


def _evaluation_arguments_from_gate(
    gate: Mapping[str, Any],
) -> dict[int, tuple[Path, str]]:
    bindings = gate.get("evaluation_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "0",
        "1",
        "2",
    }:
        raise ValueError("T-ASCV three-seed evaluation bindings drift")
    return {
        seed: (
            reject_forbidden_path(
                bindings[str(seed)].get("root", ""),
                context="T-ASCV three-seed replay evaluation",
            ),
            str(bindings[str(seed)].get("anchor_sha256", "")),
        )
        for seed in (0, 1, 2)
    }


def replay_screen_three_seed_gate(
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        gate.get("schema_version")
        != "tascv-screen-three-seed-adjudication/v1"
        or gate.get("decision") != "TASCV_SCREEN_GO"
    ):
        raise ValueError("not a T-ASCV three-seed screen GO gate")
    replayed = adjudicate_screen_three_seed(
        _evaluation_arguments_from_gate(gate)
    )
    if not _strict_equal(dict(gate), replayed):
        raise ValueError("T-ASCV three-seed screen gate replay drift")
    return dict(gate)


def replay_formal_seed0_gate(
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        gate.get("schema_version")
        != "tascv-formal-seed0-adjudication/v1"
        or gate.get("decision") != "TASCV_FORMAL_SEED0_GO"
    ):
        raise ValueError("not a T-ASCV formal seed0 GO gate")
    binding = gate.get("evaluation_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("T-ASCV formal seed0 evaluation binding missing")
    replayed = adjudicate_formal_seed0(
        reject_forbidden_path(
            binding.get("root", ""),
            context="T-ASCV formal seed0 replay evaluation",
        ),
        evaluation_anchor_sha256=str(
            binding.get("anchor_sha256", "")
        ),
    )
    if not _strict_equal(dict(gate), replayed):
        raise ValueError("T-ASCV formal seed0 gate replay drift")
    return dict(gate)


def replay_formal_three_seed_gate(
    gate: Mapping[str, Any],
    *,
    recompute_metrics: bool = True,
) -> dict[str, Any]:
    if (
        gate.get("schema_version")
        != "tascv-formal-three-seed-adjudication/v1"
        or gate.get("decision") != "TASCV_FORMAL_GO"
    ):
        raise ValueError("not a T-ASCV formal three-seed GO gate")
    replayed = adjudicate_formal_three_seed(
        _evaluation_arguments_from_gate(gate),
        recompute_metrics=recompute_metrics,
    )
    if not _strict_equal(dict(gate), replayed):
        raise ValueError("T-ASCV formal three-seed gate replay drift")
    return dict(gate)


def adjudicate_confirmation_result(
    result_root: Path | str,
    *,
    result_anchor_sha256: str,
) -> dict[str, Any]:
    from src.saded_confirmation import (
        adjudicate_confirmation_metrics,
        verify_confirmation_predictions,
    )

    root = Path(result_root).resolve()
    expected_files = {
        "result_manifest.json",
        "metrics.json",
        "gate.json",
        "evaluation_invariants.json",
        "checksums.sha256",
        "result_anchor.json",
    }
    if (
        not root.is_dir()
        or {path.name for path in root.iterdir()} != expected_files
    ):
        raise ValueError("confirmation result closure drift")
    anchor_path = root / "result_anchor.json"
    expected_anchor = str(result_anchor_sha256).lower()
    if (
        len(expected_anchor) != 64
        or sha256_file(anchor_path) != expected_anchor
    ):
        raise ValueError("confirmation result external anchor drift")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    manifest_path = root / "result_manifest.json"
    checksums_path = root / "checksums.sha256"
    if (
        anchor.get("schema_version")
        != "saded-confirmation-result-anchor/v1"
        or anchor.get("result_manifest_sha256")
        != sha256_file(manifest_path)
        or anchor.get("result_checksums_sha256")
        != sha256_file(checksums_path)
        or anchor.get("terminal") is not True
    ):
        raise ValueError("confirmation result anchor drift")
    checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest.lower()
    if set(checksums) != {
        "result_manifest.json",
        "metrics.json",
        "gate.json",
        "evaluation_invariants.json",
    } or any(
        sha256_file(root / name) != digest
        for name, digest in checksums.items()
    ):
        raise ValueError("confirmation result checksum drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(
        (root / "metrics.json").read_text(encoding="utf-8")
    )
    sealed_gate = json.loads(
        (root / "gate.json").read_text(encoding="utf-8")
    )
    invariants = json.loads(
        (root / "evaluation_invariants.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        manifest.get("schema_version")
        != "saded-confirmation-result/v1"
        or manifest.get("terminal") is not True
        or invariants.get("passed") is not True
        or manifest.get("artifacts")
        != {
            "metrics_sha256": sha256_file(root / "metrics.json"),
            "gate_sha256": sha256_file(root / "gate.json"),
            "invariants_sha256": sha256_file(
                root / "evaluation_invariants.json"
            ),
        }
    ):
        raise ValueError("confirmation result manifest drift")
    prediction = verify_confirmation_predictions(
        Path(manifest["prediction_root"]),
        anchor_sha256=manifest["prediction_anchor_sha256"],
    )
    if (
        prediction["snapshot"] != manifest["prediction_snapshot"]
        or anchor["prediction_anchor_sha256"]
        != prediction["anchor_sha256"]
    ):
        raise ValueError("confirmation result prediction drift")
    claim = manifest.get("claim", {})
    claim_path = Path(claim.get("path", "")).resolve()
    if (
        not claim_path.is_file()
        or sha256_file(claim_path) != claim.get("sha256")
        or anchor.get("claim_sha256") != claim.get("sha256")
        or claim.get("retry_permitted") is not False
    ):
        raise ValueError("confirmation one-shot claim drift")
    replayed_gate = adjudicate_confirmation_metrics(metrics)
    if (
        not _strict_equal(replayed_gate, sealed_gate)
        or manifest.get("decision") != replayed_gate["decision"]
        or anchor.get("decision") != replayed_gate["decision"]
    ):
        raise ValueError("confirmation terminal gate replay drift")
    return {
        **replayed_gate,
        "result_binding": {
            "root": root.as_posix(),
            "anchor_sha256": expected_anchor,
            "manifest_sha256": sha256_file(manifest_path),
            "checksums_sha256": sha256_file(checksums_path),
            "metrics_sha256": sha256_file(root / "metrics.json"),
            "claim_sha256": claim["sha256"],
        },
    }


def replay_confirmation_gate(
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if gate.get("decision") not in {
        "TASCV_CONFIRMATION_GO",
        "TASCV_STOP",
    }:
        raise ValueError("not a terminal confirmation gate")
    binding = gate.get("result_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("confirmation result binding missing")
    replayed = adjudicate_confirmation_result(
        binding["root"],
        result_anchor_sha256=str(binding["anchor_sha256"]),
    )
    if not _strict_equal(dict(gate), replayed):
        raise ValueError("confirmation gate replay drift")
    return dict(gate)


__all__ = [
    "adjudicate_formal_seed0",
    "adjudicate_formal_three_seed",
    "adjudicate_confirmation_result",
    "adjudicate_screen_seed0",
    "adjudicate_screen_three_seed",
    "replay_formal_seed0_gate",
    "replay_formal_three_seed_gate",
    "replay_confirmation_gate",
    "replay_screen_seed0_gate",
    "replay_screen_three_seed_gate",
    "verify_evaluation_closure",
]

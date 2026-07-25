"""Frozen one-shot confirmation helpers for the paper-ready SADED result."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from src.saded_adjudicator import (
    _attribution_three_seed_gate,
    _formal_primary_failures,
    _mean_deltas,
    replay_formal_three_seed_gate,
)
from src.saded_stage import PREDICTION_KEYS, ROUTE_ARMS
from src.saded_stage_protocol import stage_source_state
from src.sbr_artifacts import sha256_file
from src.sbr_ppaf import metric_deltas
from src.tascv_protocol import FROZEN_CONFIRMATION_CONTRACT


def adjudicate_confirmation_metrics(
    metrics_by_seed: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(metrics_by_seed) != {"0", "1", "2"}:
        return {
            "schema_version": "saded-confirmation-adjudication/v1",
            "decision": "INVALID",
            "failures": ["seed_set_drift"],
        }
    primary_per_seed: list[dict[str, float]] = []
    attribution_per_seed: list[dict[str, float]] = []
    try:
        for seed in ("0", "1", "2"):
            metrics = metrics_by_seed[seed]
            if set(metrics) != set(ROUTE_ARMS):
                raise ValueError("arm set drift")
            primary_per_seed.append(
                metric_deltas(metrics["route_treatment"], metrics["A"])
            )
            attribution_per_seed.append(
                metric_deltas(
                    metrics["route_treatment"],
                    metrics["route_control"],
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "schema_version": "saded-confirmation-adjudication/v1",
            "decision": "INVALID",
            "failures": [f"metric_schema:{error}"],
        }
    primary_mean = _mean_deltas(primary_per_seed)
    failures = [
        f"primary:{failure}"
        for failure in _formal_primary_failures(primary_mean)
    ]
    fake_verified = {
        seed: {
            "deltas": {
                "route_treatment_vs_route_control":
                    attribution_per_seed[seed]
            }
        }
        for seed in (0, 1, 2)
    }
    attribution_failures, attribution = (
        _attribution_three_seed_gate(fake_verified)
    )
    failures.extend(
        f"attribution:{failure}"
        for failure in attribution_failures
    )
    return {
        "schema_version": "saded-confirmation-adjudication/v1",
        "decision": (
            "TASCV_CONFIRMATION_GO"
            if not failures
            else "TASCV_STOP"
        ),
        "failures": failures,
        "primary": {
            "per_seed": {
                str(seed): primary_per_seed[seed]
                for seed in (0, 1, 2)
            },
            "mean": primary_mean,
        },
        "attribution": attribution,
    }


def verify_confirmation_authority(
    root: Path | str,
    *,
    anchor_sha256: str,
) -> dict[str, Any]:
    authority_root = Path(root).resolve()
    if (
        not authority_root.is_dir()
        or {path.name for path in authority_root.iterdir()}
        != {"authority", "authority_anchor.json"}
    ):
        raise ValueError("confirmation authority root closure drift")
    anchor_path = authority_root / "authority_anchor.json"
    expected_anchor = str(anchor_sha256).lower()
    if (
        len(expected_anchor) != 64
        or sha256_file(anchor_path) != expected_anchor
    ):
        raise ValueError("confirmation authority external anchor drift")
    directory = authority_root / "authority"
    if (
        not directory.is_dir()
        or {path.name for path in directory.iterdir()}
        != {
            "authority_manifest.json",
            "image_list.json",
            "checksums.sha256",
        }
    ):
        raise ValueError("confirmation authority artifact closure drift")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    manifest_path = directory / "authority_manifest.json"
    image_list_path = directory / "image_list.json"
    if (
        anchor.get("schema_version")
        != "saded-confirmation-authority-anchor/v1"
        or anchor.get("authority_manifest_sha256")
        != sha256_file(manifest_path)
        or anchor.get("authority_checksums_sha256")
        != sha256_file(directory / "checksums.sha256")
        or anchor.get("image_list_sha256")
        != sha256_file(image_list_path)
    ):
        raise ValueError("confirmation authority anchor binding drift")
    checksums = {}
    for line in (directory / "checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest.lower()
    if checksums != {
        "authority_manifest.json": sha256_file(manifest_path),
        "image_list.json": sha256_file(image_list_path),
    }:
        raise ValueError("confirmation authority checksum drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_list = json.loads(image_list_path.read_text(encoding="utf-8"))
    image_root = Path(manifest["image_root"]).resolve()
    if (
        manifest.get("schema_version")
        != "saded-confirmation-authority/v1"
        or manifest.get("image_list_sha256")
        != sha256_file(image_list_path)
        or manifest.get("image_count") != len(image_list)
        or not image_list
        or len(set(image_list)) != len(image_list)
        or not image_root.is_dir()
    ):
        raise ValueError("confirmation authority manifest drift")
    for relative in image_list:
        path = (image_root / relative).resolve()
        if image_root not in path.parents or not path.is_file():
            raise ValueError("confirmation image list path drift")
    formal_gate = Path(manifest["formal_gate"]["path"]).resolve()
    if (
        not formal_gate.is_file()
        or sha256_file(formal_gate)
        != str(manifest["formal_gate"]["sha256"]).lower()
    ):
        raise ValueError("confirmation formal gate binding drift")
    return {
        "root": authority_root,
        "anchor_sha256": expected_anchor,
        "manifest": manifest,
        "image_list": image_list,
        "image_list_path": image_list_path,
        "image_root": image_root,
    }


def verify_confirmation_predictions(
    root: Path | str,
    *,
    anchor_sha256: str,
) -> dict[str, Any]:
    prediction_root = Path(root).resolve()
    prediction_names = set(
        FROZEN_CONFIRMATION_CONTRACT["prediction_files"]
    )
    expected_names = prediction_names | {
        "prediction_manifest.json",
        "checksums.sha256",
        "prediction_anchor.json",
    }
    if (
        not prediction_root.is_dir()
        or {path.name for path in prediction_root.iterdir()}
        != expected_names
    ):
        raise ValueError("confirmation prediction closure drift")
    anchor_path = prediction_root / "prediction_anchor.json"
    expected_anchor = str(anchor_sha256).lower()
    if (
        len(expected_anchor) != 64
        or sha256_file(anchor_path) != expected_anchor
    ):
        raise ValueError("confirmation prediction external anchor drift")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    manifest_path = prediction_root / "prediction_manifest.json"
    checksums_path = prediction_root / "checksums.sha256"
    if (
        anchor.get("schema_version")
        != "saded-confirmation-prediction-anchor/v1"
        or anchor.get("prediction_manifest_sha256")
        != sha256_file(manifest_path)
        or anchor.get("prediction_checksums_sha256")
        != sha256_file(checksums_path)
        or anchor.get("prediction_file_count") != 9
        or anchor.get("exact_nine_sealed") is not True
    ):
        raise ValueError("confirmation prediction anchor drift")
    checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in checksums:
            raise ValueError("confirmation checksum schema drift")
        checksums[parts[1]] = parts[0].lower()
    if set(checksums) != prediction_names | {
        "prediction_manifest.json"
    }:
        raise ValueError("confirmation checksum target drift")
    for name, digest in checksums.items():
        if sha256_file(prediction_root / name) != digest:
            raise ValueError("confirmation prediction checksum drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = stage_source_state(Path(__file__).resolve().parents[1])
    if (
        manifest.get("schema_version")
        != "saded-confirmation-predictions/v1"
        or manifest.get("source") != source
        or manifest.get("exact_nine_sealed") is not True
        or manifest.get("annotation_inputs_opened") is not False
        or set(manifest.get("prediction_files", {}))
        != prediction_names
        or manifest.get("prediction_files")
        != {
            name: sha256_file(prediction_root / name)
            for name in prediction_names
        }
    ):
        raise ValueError("confirmation prediction manifest drift")
    protocol_path = Path(manifest["protocol"]["path"]).resolve()
    formal_gate_path = Path(manifest["formal_gate"]["path"]).resolve()
    if (
        not protocol_path.is_file()
        or sha256_file(protocol_path)
        != str(manifest["protocol"]["sha256"]).lower()
        or not formal_gate_path.is_file()
        or sha256_file(formal_gate_path)
        != str(manifest["formal_gate"]["sha256"]).lower()
        or anchor["formal_gate_sha256"]
        != manifest["formal_gate"]["sha256"]
    ):
        raise ValueError("confirmation prediction authority drift")
    gate = json.loads(formal_gate_path.read_text(encoding="utf-8"))
    replay_formal_three_seed_gate(gate, recompute_metrics=False)
    image_list = manifest.get("image_list")
    if (
        not isinstance(image_list, list)
        or not image_list
        or len(image_list) != manifest.get("image_count")
        or len(set(image_list)) != len(image_list)
    ):
        raise ValueError("confirmation prediction image identity drift")
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in prediction_names:
        value = json.loads(
            (prediction_root / name).read_text(encoding="utf-8")
        )
        if (
            not isinstance(value, list)
            or len(value) != len(image_list)
            or [row.get("image_id") for row in value] != image_list
        ):
            raise ValueError("confirmation prediction row identity drift")
        for row in value:
            if (
                set(row)
                != {"image_id", "width", "height", "predictions"}
                or int(row["width"]) <= 0
                or int(row["height"]) <= 0
                or len(row["predictions"]) > 300
                or any(
                    set(prediction) != PREDICTION_KEYS
                    for prediction in row["predictions"]
                )
            ):
                raise ValueError(
                    "confirmation prediction row schema drift"
                )
        rows[name] = value
    snapshot = {
        name: sha256_file(prediction_root / name)
        for name in sorted(expected_names)
    }
    return {
        "root": prediction_root,
        "anchor_sha256": expected_anchor,
        "anchor": anchor,
        "manifest": manifest,
        "rows": rows,
        "snapshot": snapshot,
        "source": source,
    }


__all__ = [
    "adjudicate_confirmation_metrics",
    "verify_confirmation_authority",
    "verify_confirmation_predictions",
]

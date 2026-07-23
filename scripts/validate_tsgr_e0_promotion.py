from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ebc_qp_causal_audit import compare_tsgr_audit_runs  # noqa: E402
from src.ebc_qp_config import EBCQPConfig  # noqa: E402
from src.ebc_qp_protocol import (  # noqa: E402
    E1_CONTROLLED_AMP_GROWTH_INTERVAL,
    E1_CONTROLLED_AMP_SCALE,
    E1_EXPECTED_OPTIMIZER_ATTEMPTS,
    dataset_signature,
    state_fingerprint,
    subset_signature,
)


EXPECTED_TRAINING_SOURCES = frozenset(
    {
        "scripts/audit_ebc_qp_aux_causality.py",
        "src/ebc_qp_causal_audit.py",
        "src/rtdetr_ebc_qp.py",
        "src/ebc_qp_decoder.py",
        "src/ebc_qp_config.py",
    }
)
EXPECTED_ULTRALYTICS_SOURCES = frozenset({"head.py", "tasks.py", "rtdetr-l.yaml"})
EXPECTED_E1_TRAINING = {
    "epochs": 10,
    "fraction": 1.0,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "device": "0",
    "amp": True,
    "controlled_amp_scale": E1_CONTROLLED_AMP_SCALE,
    "controlled_amp_growth_interval": E1_CONTROLLED_AMP_GROWTH_INTERVAL,
    "expected_optimizer_attempts": E1_EXPECTED_OPTIMIZER_ATTEMPTS,
    "save_period": -1,
    "retained_zero_based_epoch_checkpoints": [7, 8, 9],
    "deterministic": True,
    "nbs": 64,
    "nms": False,
    "max_det": 300,
    "optimizer": "MuSGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
}
EXPECTED_TSGR_CONFIG = EBCQPConfig(
    lambda_p2=0.1,
    lambda_quality=0.0,
    lambda_ebc=0.0,
    learnable_fusion_gamma=False,
    query_injection_enabled=False,
    quality_gated_p2=False,
    p2_c2_grad_scale=0.1,
    contribution_separated_aux_gradients=True,
).as_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the immutable AMP128 E0 promotion sidecar.")
    parser.add_argument("--a0", required=True, type=Path)
    parser.add_argument("--h0", required=True, type=Path)
    parser.add_argument("--h1", required=True, type=Path)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--frozen-experiment", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--initial-state", required=True, type=Path)
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--a0-repeat", required=True, type=Path)
    parser.add_argument("--production-preflight-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest().upper()


def validate_authoritative_artifacts(
    protocol_path: Path,
    frozen_experiment_path: Path,
    data_path: Path,
    initial_state_path: Path,
    subset_path: Path,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    protocol_path = Path(protocol_path).resolve()
    frozen_experiment_path = Path(frozen_experiment_path).resolve()
    data_path = Path(data_path).resolve()
    initial_state_path = Path(initial_state_path).resolve()
    subset_path = Path(subset_path).resolve()
    if set(source_paths) != EXPECTED_ULTRALYTICS_SOURCES:
        errors.append("Ultralytics source keyset mismatch")
    required = (protocol_path, frozen_experiment_path, data_path, initial_state_path, subset_path, *source_paths.values())
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        return {"evidence_valid": False, "errors": errors + [f"missing artifact: {path}" for path in missing]}

    manifest = json.loads(protocol_path.read_text(encoding="utf-8"))
    signature = manifest.get("signature")
    unsigned = dict(manifest)
    unsigned.pop("signature", None)
    if manifest.get("format_version") != 1:
        errors.append("protocol format_version mismatch")
    if manifest.get("seed") != 0:
        errors.append("protocol seed must be 0 for E0 promotion")
    if not signature or signature != _json_sha256(unsigned):
        errors.append("protocol canonical signature mismatch")

    frozen = json.loads(frozen_experiment_path.read_text(encoding="utf-8"))
    experiment_signature = frozen.get("experiment_signature")
    if frozen.get("format_version") != 1:
        errors.append("frozen experiment format_version mismatch")
    if not experiment_signature or experiment_signature != _json_sha256(frozen.get("payload")):
        errors.append("frozen experiment payload signature mismatch")
    if experiment_signature != manifest.get("experiment_signature"):
        errors.append("protocol/frozen experiment signature mismatch")

    data_sha = _file_sha256(data_path)
    initial_sha = _file_sha256(initial_state_path)
    if data_sha != manifest.get("data", {}).get("sha256"):
        errors.append("data SHA mismatch")
    if initial_sha != manifest.get("initial_state", {}).get("sha256"):
        errors.append("initial-state SHA mismatch")
    if str(data_path) != manifest.get("data", {}).get("path"):
        errors.append("protocol data path mismatch")
    if str(initial_state_path) != manifest.get("initial_state", {}).get("path"):
        errors.append("protocol initial-state path mismatch")
    if str(subset_path) != manifest.get("subset", {}).get("path"):
        errors.append("protocol subset path mismatch")

    source_sha = {name: _file_sha256(Path(path)) for name, path in sorted(source_paths.items())}
    if source_sha != manifest.get("source_sha256"):
        errors.append("Ultralytics source SHA mismatch")

    dataset_record: dict[str, Any] | None = None
    subset_record: dict[str, Any] | None = None
    category_mapping_sha: str | None = None
    try:
        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        dataset_root = Path(data["path"]).resolve()
        if Path(data["train"]).resolve() != subset_path:
            errors.append("data YAML train path does not select the frozen subset")
        dataset_record = dataset_signature(dataset_root)
        if dataset_record != manifest.get("dataset"):
            errors.append("dataset semantic signature mismatch")
        selected = [Path(line) for line in subset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        subset_record = {
            "count": len(selected),
            "fraction": manifest.get("subset", {}).get("fraction"),
            "sha256": subset_signature(selected, root=dataset_root),
        }
        expected_subset = {key: manifest.get("subset", {}).get(key) for key in ("count", "fraction", "sha256")}
        if subset_record != expected_subset:
            errors.append("subset semantic signature mismatch")
        category_mapping_sha = _json_sha256(data["names"])
        if category_mapping_sha != manifest.get("category_mapping_sha256"):
            errors.append("category mapping signature mismatch")
    except Exception as exc:
        errors.append(f"dataset semantic validation failed: {type(exc).__name__}: {exc}")

    payload = frozen.get("payload") or {}
    if payload.get("e1_training") != EXPECTED_E1_TRAINING:
        errors.append("frozen E1 training contract mismatch")
    if payload.get("tsgr_config") != EXPECTED_TSGR_CONFIG:
        errors.append("frozen TSGR config mismatch")
    expected_subset_payload = {
        key: manifest.get("subset", {}).get(key) for key in ("count", "fraction", "sha256")
    }
    authoritative_pairs = (
        ("dataset", manifest.get("dataset")),
        ("category_mapping_sha256", manifest.get("category_mapping_sha256")),
        ("subset", expected_subset_payload),
        ("data_sha256", manifest.get("data", {}).get("sha256")),
        ("source_sha256", manifest.get("source_sha256")),
        ("git_commit", manifest.get("git_commit")),
        ("environment", manifest.get("environment")),
    )
    for field, expected in authoritative_pairs:
        if payload.get(field) != expected:
            errors.append(f"frozen experiment/manifest mismatch: {field}")

    common_fingerprint: str | None = None
    innovation_fingerprint: str | None = None
    try:
        initial = torch.load(initial_state_path, map_location="cpu", weights_only=False)
        metadata = initial.get("metadata") or {}
        metadata_expected = {
            "seed": manifest.get("seed"),
            "dataset": manifest.get("dataset"),
            "category_mapping_sha256": manifest.get("category_mapping_sha256"),
            "subset": expected_subset_payload,
            "source_sha256": manifest.get("source_sha256"),
            "git_commit": manifest.get("git_commit"),
            "environment": manifest.get("environment"),
            "experiment_signature": manifest.get("experiment_signature"),
        }
        for field, expected in metadata_expected.items():
            if metadata.get(field) != expected:
                errors.append(f"initial-state metadata mismatch: {field}")
        fingerprints = initial.get("fingerprints") or {}
        common_fingerprint = state_fingerprint(initial.get("common_state") or {})
        innovation_fingerprint = state_fingerprint(initial.get("innovation_state") or {})
        if common_fingerprint != fingerprints.get("common"):
            errors.append("initial-state common fingerprint mismatch")
        if innovation_fingerprint != fingerprints.get("innovation"):
            errors.append("initial-state innovation fingerprint mismatch")
    except Exception as exc:
        errors.append(f"initial-state metadata validation failed: {type(exc).__name__}: {exc}")

    return {
        "evidence_valid": not errors,
        "errors": errors,
        "paths": {
            "protocol": str(protocol_path),
            "frozen_experiment": str(frozen_experiment_path),
            "data": str(data_path),
            "initial_state": str(initial_state_path),
            "subset": str(subset_path),
            "ultralytics_sources": {name: str(Path(path).resolve()) for name, path in sorted(source_paths.items())},
        },
        "protocol_sha256": _file_sha256(protocol_path),
        "protocol_signature": signature,
        "experiment_sha256": _file_sha256(frozen_experiment_path),
        "experiment_signature": experiment_signature,
        "data_sha256": data_sha,
        "initial_state_sha256": initial_sha,
        "common_initial_fingerprint": common_fingerprint,
        "innovation_initial_fingerprint": innovation_fingerprint,
        "subset_file_sha256": _file_sha256(subset_path),
        "dataset": dataset_record,
        "subset": subset_record,
        "category_mapping_sha256": category_mapping_sha,
        "source_sha256": source_sha,
        "training_commit": manifest.get("git_commit"),
        "environment": manifest.get("environment"),
    }


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{commit}:{relative_path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest().upper()


def validate_trace_bundle(
    trace_paths: Mapping[str, Path],
    *,
    protocol_path: Path,
    data_path: Path,
    initial_state_path: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    traces = {arm: json.loads(Path(path).read_text(encoding="utf-8")) for arm, path in trace_paths.items()}
    trace_sha = {arm: _file_sha256(path) for arm, path in trace_paths.items()}
    manifest = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    training_commit = str(manifest.get("git_commit"))
    protocol_sha = _file_sha256(protocol_path)
    data_sha = _file_sha256(data_path)
    initial_sha = _file_sha256(initial_state_path)
    for arm, trace in traces.items():
        evidence = trace.get("evidence") or {}
        if evidence.get("git_commit") != training_commit:
            errors.append(f"{arm} training commit mismatch")
        if evidence.get("protocol_manifest_sha256") != protocol_sha:
            errors.append(f"{arm} protocol SHA mismatch")
        if evidence.get("protocol_signature") != manifest.get("signature"):
            errors.append(f"{arm} protocol signature mismatch")
        if evidence.get("data_sha256") != data_sha:
            errors.append(f"{arm} data SHA mismatch")
        if trace.get("initial_state_sha256") != initial_sha:
            errors.append(f"{arm} initial-state SHA mismatch")
        if trace.get("initial_state_path") != str(Path(initial_state_path).resolve()):
            errors.append(f"{arm} initial-state path mismatch")
        if trace.get("common_initial_fingerprint") != authority.get("common_initial_fingerprint"):
            errors.append(f"{arm} common initial-state fingerprint mismatch")
        sources = evidence.get("sources") or {}
        if set(sources) != EXPECTED_TRAINING_SOURCES:
            errors.append(f"{arm} training source keyset mismatch")
        for relative_path, expected in sources.items():
            try:
                if _git_blob_sha256(training_commit, relative_path) != expected:
                    errors.append(f"{arm} training source mismatch: {relative_path}")
            except subprocess.CalledProcessError:
                errors.append(f"{arm} training source missing: {relative_path}")
        runtime_expected = {
            "python": manifest.get("environment", {}).get("python"),
            "torch": manifest.get("environment", {}).get("torch"),
            "cuda_runtime": manifest.get("environment", {}).get("cuda"),
            "ultralytics": manifest.get("environment", {}).get("ultralytics"),
            "gpu": manifest.get("environment", {}).get("gpu"),
        }
        for field, expected in runtime_expected.items():
            if evidence.get(field) != expected:
                errors.append(f"{arm} runtime/authority mismatch: {field}")
        authority_fields = {
            "experiment_signature": manifest.get("experiment_signature"),
            "dataset": manifest.get("dataset"),
            "category_mapping_sha256": manifest.get("category_mapping_sha256"),
            "subset": manifest.get("subset"),
            "source_sha256": manifest.get("source_sha256"),
        }
        for field, expected in authority_fields.items():
            if field in evidence and evidence.get(field) != expected:
                errors.append(f"{arm} embedded authority mismatch: {field}")

    comparison = compare_tsgr_audit_runs(traces["a0"], traces["h0"], traces["h1"])
    if not comparison.get("passed"):
        errors.extend(f"comparator: {error}" for error in comparison.get("errors", []))
    repeatability = None
    try:
        from src.ebc_qp_causal_audit import compare_a0_repeats

        repeatability = compare_a0_repeats(traces["a0"], traces["a0-repeat"])
    except (KeyError, ValueError) as exc:
        errors.append(f"A0 repeatability evidence invalid: {exc}")

    repeat_fields = (
        "stock_grad_preclip_sha256",
        "stock_delta_sha256",
        "stock_bn_sha256",
        "stock_ema_parameters",
        "optimizer_state_parameters",
        "stock_state_parameters",
        "optimizer_groups",
    )
    divergence_profiles = {}
    for label, other in (("a0_repeat", traces.get("a0-repeat")), ("a0_h0", traces.get("h0"))):
        if other is None:
            continue
        divergence_profiles[label] = {
            field: {
                "count": sum(
                    left.get(field) != right.get(field)
                    for left, right in zip(traces["a0"].get("steps", []), other.get("steps", []))
                ),
                "first": next(
                    (
                        index
                        for index, (left, right) in enumerate(
                            zip(traces["a0"].get("steps", []), other.get("steps", [])), start=1
                        )
                        if left.get(field) != right.get(field)
                    ),
                    None,
                ),
            }
            for field in repeat_fields
        }
    repeatability_floor_match = (
        divergence_profiles.get("a0_repeat") == divergence_profiles.get("a0_h0")
        if len(divergence_profiles) == 2
        else False
    )
    numeric_fields = (
        "stock_grad_preclip_parameters",
        "stock_delta_parameters",
        "stock_bn_parameters",
        "stock_ema_parameters",
        "optimizer_state_parameters",
        "stock_state_parameters",
    )
    numeric_profiles = {}
    for label, other in (("a0_repeat", traces.get("a0-repeat")), ("a0_h0", traces.get("h0"))):
        if other is None:
            continue
        numeric_profiles[label] = {}
        for field in numeric_fields:
            distances = [
                _relative_signature_distance(left.get(field) or {}, right.get(field) or {})
                for left, right in zip(traces["a0"].get("steps", []), other.get("steps", []))
            ]
            ordered = sorted(distances)
            numeric_profiles[label][field] = {
                "min": min(distances) if distances else None,
                "median": statistics.median(distances) if distances else None,
                "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None,
                "max": max(distances) if distances else None,
                "first_nonzero": next((index for index, value in enumerate(distances, start=1) if value > 0), None),
            }
    return {
        "mechanism_gate_passed": bool(authority.get("evidence_valid")) and not errors,
        "errors": errors,
        "trace_sha256": trace_sha,
        "embedded_authority_fields_complete": {
            arm: all(
                field in (trace.get("evidence") or {})
                for field in ("experiment_signature", "dataset", "category_mapping_sha256", "subset", "source_sha256")
            )
            for arm, trace in traces.items()
        },
        "comparison": comparison,
        "repeatability": repeatability,
        "repeatability_divergence_profiles": divergence_profiles,
        "repeatability_numeric_distance_profiles": numeric_profiles,
        "repeatability_floor_profile_match": repeatability_floor_match,
        "repeatability_classification": (
            "NO_EXCESS_H0_COUPLING_ABOVE_REPEATABILITY_FLOOR"
            if repeatability and repeatability.get("pairing_valid") and repeatability_floor_match
            else "REPEATABILITY_ATTRIBUTION_UNRESOLVED"
        ),
    }


def _relative_signature_distance(
    left: Mapping[str, Mapping[str, object]], right: Mapping[str, Mapping[str, object]]
) -> float:
    values = []
    for name in set(left).union(right):
        left_record = left.get(name) or {}
        right_record = right.get(name) or {}
        for field in ("l2", "sum", "max_abs"):
            values.append((float(left_record.get(field, 0.0)), float(right_record.get(field, 0.0))))
    numerator = math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in values))
    denominator = max(
        math.sqrt(sum(left_value**2 for left_value, _ in values)),
        math.sqrt(sum(right_value**2 for _, right_value in values)),
        1e-12,
    )
    return numerator / denominator


def validate_production_preflight(manifest_path: Path, *, expected_commit: str) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        return {"verified": False, "errors": [f"missing production preflight manifest: {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = manifest.get("signature")
    unsigned = dict(manifest)
    unsigned.pop("signature", None)
    if signature != _json_sha256(unsigned):
        errors.append("production preflight manifest signature mismatch")
    if (manifest.get("format_version"), manifest.get("stage"), manifest.get("arm"), manifest.get("seed")) != (
        1,
        "e1",
        "tsgr-p2",
        0,
    ):
        errors.append("production preflight identity mismatch")
    if manifest.get("git_commit_start") != expected_commit or manifest.get("git_commit_end") != expected_commit:
        errors.append("production preflight Git commit mismatch")
    if manifest.get("ebc_config") != EXPECTED_TSGR_CONFIG:
        errors.append("production preflight TSGR config mismatch")
    controlled = manifest.get("controlled_amp") or {}
    expected_controlled = {
        "init_scale": E1_CONTROLLED_AMP_SCALE,
        "growth_interval": E1_CONTROLLED_AMP_GROWTH_INTERVAL,
        "expected_optimizer_attempts": E1_EXPECTED_OPTIMIZER_ATTEMPTS,
        "optimizer_attempts": E1_EXPECTED_OPTIMIZER_ATTEMPTS,
        "skipped_attempts": 0,
    }
    if controlled != expected_controlled:
        errors.append("production preflight controlled AMP contract mismatch")

    artifact = (manifest.get("artifacts") or {}).get("optimizer_evidence") or {}
    evidence_path = Path(artifact.get("path") or "").resolve()
    records: list[dict[str, Any]] = []
    if not evidence_path.is_file():
        errors.append("production preflight optimizer evidence is missing")
    else:
        if artifact.get("bytes") != evidence_path.stat().st_size or artifact.get("sha256") != _file_sha256(evidence_path):
            errors.append("production preflight optimizer evidence artifact mismatch")
        try:
            records = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as exc:
            errors.append(f"production preflight optimizer evidence is invalid: {exc}")
    if len(records) != E1_EXPECTED_OPTIMIZER_ATTEMPTS:
        errors.append("production preflight optimizer attempt count mismatch")
    ratios: list[float] = []
    for index, record in enumerate(records, start=1):
        if record.get("optimizer_attempt") != index:
            errors.append(f"production preflight optimizer attempt {index} is out of order")
            break
        ratio = record.get("update_monitor_ratio")
        if not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)) or float(ratio) < 0.0:
            errors.append(f"production preflight optimizer attempt {index} has invalid monitor ratio")
            break
        ratios.append(float(ratio))
        if (
            record.get("amp_step_skipped")
            or record.get("nonfinite_fields")
            or record.get("runtime_violation")
            or record.get("update_monitor_abort")
            or record.get("p2_entry_count") != 0
            or record.get("ordinary_query_count") != 300
            or float(record.get("amp_scale_before", -1)) != E1_CONTROLLED_AMP_SCALE
            or float(record.get("amp_scale_after", -1)) != E1_CONTROLLED_AMP_SCALE
        ):
            errors.append(f"production preflight optimizer attempt {index} violated the runtime contract")
            break
    return {
        "verified": not errors,
        "errors": errors,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_signature": signature,
        "optimizer_evidence_path": str(evidence_path),
        "optimizer_evidence_sha256": _file_sha256(evidence_path) if evidence_path.is_file() else None,
        "optimizer_attempts": len(records),
        "update_monitor_ratio_min": min(ratios) if ratios else None,
        "update_monitor_ratio_median": sorted(ratios)[len(ratios) // 2] if ratios else None,
        "update_monitor_ratio_max": max(ratios) if ratios else None,
        "update_monitor_abort_count": sum(bool(record.get("update_monitor_abort")) for record in records),
    }


def _installed_ultralytics_sources() -> dict[str, Path]:
    import ultralytics
    import ultralytics.nn.modules.head as ultralytics_head
    import ultralytics.nn.tasks as ultralytics_tasks

    package_root = Path(ultralytics.__file__).parent
    return {
        "head.py": Path(ultralytics_head.__file__),
        "tasks.py": Path(ultralytics_tasks.__file__),
        "rtdetr-l.yaml": package_root / "cfg" / "models" / "rt-detr" / "rtdetr-l.yaml",
    }


def _atomic_write_locked(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite promotion evidence: {path}")
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if checksum_path.exists():
        raise FileExistsError(f"refusing to overwrite promotion checksum: {checksum_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    os.chmod(path, 0o444)
    checksum = _file_sha256(path)
    checksum_tmp = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    checksum_tmp.write_text(f"{checksum}  {path.name}\n", encoding="ascii")
    checksum_tmp.replace(checksum_path)
    os.chmod(checksum_path, 0o444)
    return checksum


def _validator_identity() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise SystemExit("promotion validator requires a tracked-clean worktree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    records = {}
    for path in (Path(__file__).resolve(), ROOT / "src" / "ebc_qp_causal_audit.py"):
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True
        )
        if tracked.returncode != 0:
            raise SystemExit(f"promotion validator source is not tracked: {relative}")
        worktree_sha = _file_sha256(path)
        commit_sha = _git_blob_sha256(commit, relative)
        if worktree_sha != commit_sha:
            raise SystemExit(f"promotion validator source differs from HEAD: {relative}")
        records[relative] = worktree_sha
    return {"commit": commit, "sources": records}


def main() -> None:
    args = build_parser().parse_args()
    validator_identity = _validator_identity()
    authority = validate_authoritative_artifacts(
        args.protocol_manifest,
        args.frozen_experiment,
        args.data,
        args.initial_state,
        args.subset,
        _installed_ultralytics_sources(),
    )
    traces = validate_trace_bundle(
        {"a0": args.a0, "a0-repeat": args.a0_repeat, "h0": args.h0, "h1": args.h1},
        protocol_path=args.protocol_manifest,
        data_path=args.data,
        initial_state_path=args.initial_state,
        authority=authority,
    )
    mechanism_gate_passed = bool(traces["mechanism_gate_passed"])
    production = (
        validate_production_preflight(
            args.production_preflight_manifest,
            expected_commit=validator_identity["commit"],
        )
        if args.production_preflight_manifest is not None
        else {"verified": False, "errors": ["production preflight is pending"]}
    )
    promotion_ready = mechanism_gate_passed and bool(production["verified"])
    classification = (
        "TSGR_E0B_PASS"
        if promotion_ready
        else "MECHANISM_PASS_MONITOR_PENDING"
        if mechanism_gate_passed and args.production_preflight_manifest is None
        else "INVALID_PRODUCTION_PREFLIGHT"
        if mechanism_gate_passed
        else "INVALID_E0_EVIDENCE"
    )
    payload = {
        "format_version": 1,
        "classification": classification,
        "evidence_valid": bool(authority["evidence_valid"]),
        "mechanism_gate_passed": mechanism_gate_passed,
        "promotion_ready": promotion_ready,
        "production_update_monitor_verified": bool(production["verified"]),
        "production_preflight": production,
        "authority": authority,
        "traces": traces,
        "training_evidence_commit": authority.get("training_commit"),
        "promotion_validator_commit": validator_identity["commit"],
        "promotion_validator_sources": validator_identity["sources"],
        "promotion_validator_sha256": _file_sha256(Path(__file__).resolve()),
        "promotion_comparator_sha256": _file_sha256(ROOT / "src" / "ebc_qp_causal_audit.py"),
    }
    payload["signature"] = _json_sha256(payload)
    sidecar_sha256 = _atomic_write_locked(args.output, payload)
    print(json.dumps({
        **{key: payload[key] for key in ("classification", "evidence_valid", "mechanism_gate_passed", "promotion_ready", "training_evidence_commit", "promotion_validator_commit")},
        "sidecar_sha256": sidecar_sha256,
    }))
    if promotion_ready:
        return
    if not mechanism_gate_passed or args.production_preflight_manifest is not None:
        raise SystemExit(1)
    raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tascv_cli import (
    current_environment,
    current_upstream_source_hashes,
)
from src.tascv_protocol import (
    APPROVED_TASCV_PARENT,
    CONTROL_SLOTS,
    EXPECTED_CATEGORY_MAPPING_SHA256,
    EXPECTED_COMMON_FINGERPRINTS,
    EXPECTED_DATASET_FILE_COUNT,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ENVIRONMENT,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_SUBSET_COUNT,
    EXPECTED_SUBSET_FILE_SHA256,
    EXPECTED_SUBSET_SHA256,
    EXPECTED_UPSTREAM_SOURCE_SHA256,
    FROZEN_CROP_CONTRACT,
    FROZEN_FORMAL_THRESHOLDS,
    FROZEN_MECHANISM_GATE,
    FROZEN_SCREEN_GATE,
    FROZEN_STAGE_CONTRACT,
    FROZEN_STATE_MACHINE,
    FROZEN_TASCV_CONTRACT,
    FROZEN_TRAINING_CONTRACT,
    PROTOCOL_VERSION,
    R0_EVALUATION_ANCHOR_SHA256,
    R0_EVALUATION_CHECKSUM_ROOT,
    R0_EVALUATION_MANIFEST_SHA256,
    R0_ROUTE_ANCHOR_SHA256,
    repo_source_hashes,
    resolve_control_allowlist,
    require_clean_repo,
    sha256_file,
    source_bundle_sha256,
    subset_signature,
    validate_initial_state_artifact,
    validate_parent_attestation,
    validate_r0_authority,
    validate_r0_closure as validate_frozen_r0_closure,
)
from src.tascv_stage import (
    TASCVStage,
    allowed_observed_tensor_batch_sizes,
    allowed_seeds,
    stage_policy,
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def _reject_forbidden(path: Path) -> None:
    raw = str(path).replace("\\", "/").lower()
    resolved = path.resolve().as_posix().lower()
    if any(
        token in raw or token in resolved
        for token in ("test-dev", "test_dev")
    ):
        raise ValueError(f"test-dev is forbidden: {path}")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_approved_parent(repo_root: Path) -> None:
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            APPROVED_TASCV_PARENT["commit"],
            "HEAD",
        ],
        cwd=repo_root,
        check=True,
    )
    tree = _git(
        repo_root,
        "rev-parse",
        f"{APPROVED_TASCV_PARENT['commit']}^{{tree}}",
    )
    if tree != APPROVED_TASCV_PARENT["tree"]:
        raise ValueError("approved T-ASCV parent tree drift")
    hashes = {
        relative: sha256_file(repo_root / relative)
        for relative in APPROVED_TASCV_PARENT["files"]
    }
    if hashes != APPROVED_TASCV_PARENT["files"]:
        raise ValueError("approved T-ASCV parent runtime files drift")
    if (
        source_bundle_sha256(hashes)
        != APPROVED_TASCV_PARENT["bundle_sha256"]
    ):
        raise ValueError("approved T-ASCV parent bundle drift")


def _validate_stage_contract() -> None:
    for stage in TASCVStage:
        policy = stage_policy(stage)
        record = FROZEN_STAGE_CONTRACT[stage.value]
        actual = {
            "seeds": sorted(allowed_seeds(stage)),
            "arms": (
                ["tascv"]
                if stage is TASCVStage.TINY_MECHANISM_500
                else ["control", "tascv"]
            ),
            "epochs": policy.epochs,
            "uses_hashed_subset": policy.uses_hashed_subset,
            "max_train_batches": policy.max_train_batches,
            "expected_successful_batches": (
                policy.expected_successful_batches
            ),
            "expected_optimizer_attempts": (
                policy.expected_optimizer_attempts
            ),
            "allowed_observed_tensor_batch_sizes": sorted(
                allowed_observed_tensor_batch_sizes(stage)
            ),
        }
        if actual != record:
            raise ValueError(f"T-ASCV stage contract drift: {stage.value}")


def _checksum_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if len(digest) != 64 or name in names:
            raise ValueError(f"invalid checksum closure: {path}")
        names.add(name)
    return names


def _validate_r0_closure(
    *,
    route_anchor: Path,
    route_checksums: Path,
    evaluation_anchor: Path,
    evaluation_manifest: Path,
    evaluation_checksums: Path,
) -> dict:
    authority = validate_r0_authority(
        route_anchor=route_anchor,
        evaluation_anchor=evaluation_anchor,
    )
    route_record = json.loads(
        route_anchor.read_text(encoding="utf-8")
    )
    if (
        sha256_file(route_checksums)
        != route_record["route_checksums_sha256"].upper()
        or _checksum_names(route_checksums)
        != {
            "capacity.json",
            "predictions.jsonl.gz",
            "route_invariants.json",
            "route_manifest.json",
        }
    ):
        raise ValueError("T-ASCV R0 route closure drift")
    evaluation_record = json.loads(
        evaluation_anchor.read_text(encoding="utf-8")
    )
    if (
        sha256_file(evaluation_checksums)
        != R0_EVALUATION_CHECKSUM_ROOT
        or _checksum_names(evaluation_checksums)
        != {
            "capacity.json",
            "deltas.json",
            "evaluation_invariants.json",
            "evaluation_manifest.json",
            "metrics.json",
            "r0_gate.json",
        }
        or sha256_file(evaluation_manifest)
        != R0_EVALUATION_MANIFEST_SHA256
    ):
        raise ValueError("T-ASCV R0 evaluation closure drift")
    manifest = json.loads(
        evaluation_manifest.read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version")
        != "sbr-saded-r0-evaluation/v1"
        or manifest.get("decision") != "R0_GO"
        or manifest.get("route_snapshot_verified") is not True
        or manifest.get("route_anchor_sha256", "").upper()
        != R0_ROUTE_ANCHOR_SHA256
        or manifest.get("source", {}).get("commit")
        != authority["source_commit"]
        or manifest.get("source", {}).get("clean_tracked") is not True
        or manifest.get("source", {}).get("untracked") is not False
    ):
        raise ValueError("T-ASCV R0 evaluation manifest drift")
    if (
        evaluation_record.get("evaluation_checksums_sha256", "").upper()
        != sha256_file(evaluation_checksums)
    ):
        raise ValueError("T-ASCV R0 external anchor drift")
    record = {
        **authority,
        "route_checksums": route_checksums.resolve().as_posix(),
        "route_checksums_sha256": sha256_file(route_checksums),
        "evaluation_manifest": evaluation_manifest.resolve().as_posix(),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest),
        "evaluation_checksums": evaluation_checksums.resolve().as_posix(),
        "evaluation_checksums_sha256": sha256_file(
            evaluation_checksums
        ),
    }
    return validate_frozen_r0_closure(record)


def _load_baseline_provenance(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "ascv-loc-matched/v2":
        raise ValueError("baseline provenance schema drift")
    if (
        manifest.get("dataset", {}).get("sha256")
        != EXPECTED_DATASET_SHA256
        or manifest.get("dataset", {}).get("file_count")
        != EXPECTED_DATASET_FILE_COUNT
        or manifest.get("category_mapping_sha256")
        != EXPECTED_CATEGORY_MAPPING_SHA256
        or manifest.get("subset", {}).get("count")
        != EXPECTED_SUBSET_COUNT
        or manifest.get("subset", {}).get("semantic_sha256")
        != EXPECTED_SUBSET_SHA256
        or manifest.get("subset", {}).get("file_sha256")
        != EXPECTED_SUBSET_FILE_SHA256
    ):
        raise ValueError("baseline dataset provenance drift")
    root = Path(manifest["dataset"]["root"])
    subset = Path(manifest["subset"]["path"])
    if (
        sha256_file(subset) != EXPECTED_SUBSET_FILE_SHA256
        or subset_signature(subset, root=root)
        != {
            "count": EXPECTED_SUBSET_COUNT,
            "sha256": EXPECTED_SUBSET_SHA256,
        }
    ):
        raise ValueError("baseline subset provenance drift")
    for seed in range(3):
        validate_parent_attestation(manifest, seed)
        initial = manifest["initial_states"][str(seed)]
        initial_path = Path(initial["path"])
        if (
            sha256_file(initial_path)
            != EXPECTED_INITIAL_STATE_SHA256[seed]
            or initial["common_fingerprint"]
            != EXPECTED_COMMON_FINGERPRINTS[seed]
        ):
            raise ValueError("baseline common-state provenance drift")
        artifact = torch.load(
            initial_path,
            map_location="cpu",
            weights_only=False,
        )
        validate_initial_state_artifact(artifact, seed=seed)
    return manifest


def _control_requirements(
    *,
    baseline: dict,
    run_root: Path,
    runtime_commit: str,
    repo_bundle: str,
    upstream_bundle: str,
) -> dict:
    slots: dict[str, dict] = {}
    for slot in CONTROL_SLOTS:
        _prefix, stage, seed_text = slot.split(":")
        seed = int(seed_text)
        project = run_root / "controls" / stage / f"seed{seed}"
        canary_positions = {
            "PREFLIGHT_1": [[0, 1]],
            "SCREEN_10": [[0, 1], [0, 2], [1, 82]],
            "FORMAL_100": [[0, 1], [0, 2], [1, 810]],
        }[stage]
        slots[slot] = {
            "slot_id": slot,
            "provenance": {
                "stage": stage,
                "seed": seed,
                "model": "Ultralytics RT-DETR-L stock",
                "runtime_source_commit": runtime_commit,
                "repo_bundle_sha256": repo_bundle,
                "upstream_bundle_sha256": upstream_bundle,
                "approved_tascv_parent": APPROVED_TASCV_PARENT,
                "r0_evaluation_anchor_sha256": (
                    R0_EVALUATION_ANCHOR_SHA256
                ),
                "initial_state_sha256": baseline["initial_states"][
                    seed_text
                ]["sha256"],
                "common_fingerprint": baseline["initial_states"][
                    seed_text
                ]["common_fingerprint"],
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "subset_sha256": EXPECTED_SUBSET_SHA256,
                "subset_binding": {
                    "count": EXPECTED_SUBSET_COUNT,
                    "semantic_sha256": EXPECTED_SUBSET_SHA256,
                    "file_sha256": EXPECTED_SUBSET_FILE_SHA256,
                },
                "data_yaml_sha256": (
                    baseline["train_only_yaml"]["sha256"]
                    if FROZEN_STAGE_CONTRACT[stage][
                        "uses_hashed_subset"
                    ]
                    else baseline["full_train_only_yaml"]["sha256"]
                ),
                "training_contract": FROZEN_TRAINING_CONTRACT,
                "stage_contract": FROZEN_STAGE_CONTRACT[stage],
                "batch_canary_contract": {
                    "digest_schema": FROZEN_TRAINING_CONTRACT[
                        "batch_digest_schema"
                    ],
                    "required_epoch_global_batch_positions": (
                        canary_positions
                    ),
                },
                "endpoint_contract": {
                    "checkpoint_name": "last.pt",
                    "training_summary_name": (
                        "tascv_training_summary.json"
                    ),
                    "raw_predictions_binding_required": True,
                    "evaluator_binding_required": True,
                },
            },
            "fresh_target": {
                "project": project.resolve().as_posix(),
                "name": "control",
                "target_dir": (project / "control").resolve().as_posix(),
                "summary": (
                    project / "control/tascv_training_summary.json"
                ).resolve().as_posix(),
                "checkpoint": (
                    project / "control/weights/last.pt"
                ).resolve().as_posix(),
            },
        }
    return {
        "schema_version": "saded-control-requirements/v1",
        "slots": slots,
    }


def prepare_requirements(
    *,
    baseline_protocol: Path,
    run_root: Path,
    output: Path,
    repo_root: Path,
) -> dict:
    for path in (baseline_protocol, run_root, output, repo_root):
        _reject_forbidden(path)
    require_clean_repo(repo_root)
    _validate_approved_parent(repo_root)
    baseline = _load_baseline_provenance(baseline_protocol)
    if current_environment() != EXPECTED_ENVIRONMENT:
        raise ValueError("runtime environment drift")
    upstream = current_upstream_source_hashes()
    if upstream != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("Ultralytics source drift")
    requirements = _control_requirements(
        baseline=baseline,
        run_root=run_root,
        runtime_commit=_git(repo_root, "rev-parse", "HEAD"),
        repo_bundle=source_bundle_sha256(
            repo_source_hashes(repo_root)
        ),
        upstream_bundle=source_bundle_sha256(upstream),
    )
    _atomic_json(output, requirements)
    return requirements


def finalize_protocol(
    *,
    baseline_protocol: Path,
    full_dataset_yaml: Path,
    route_anchor: Path,
    route_checksums: Path,
    evaluation_anchor: Path,
    evaluation_manifest: Path,
    evaluation_checksums: Path,
    control_allowlist: Path,
    run_root: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict:
    paths = (
        baseline_protocol,
        full_dataset_yaml,
        route_anchor,
        route_checksums,
        evaluation_anchor,
        evaluation_manifest,
        evaluation_checksums,
        control_allowlist,
        run_root,
        output_dir,
        repo_root,
    )
    for path in paths:
        _reject_forbidden(path)
    require_clean_repo(repo_root)
    _validate_approved_parent(repo_root)
    _validate_stage_contract()
    if current_environment() != EXPECTED_ENVIRONMENT:
        raise ValueError("runtime environment drift")
    upstream = current_upstream_source_hashes()
    if upstream != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("Ultralytics source drift")
    baseline = _load_baseline_provenance(baseline_protocol)
    r0 = _validate_r0_closure(
        route_anchor=route_anchor,
        route_checksums=route_checksums,
        evaluation_anchor=evaluation_anchor,
        evaluation_manifest=evaluation_manifest,
        evaluation_checksums=evaluation_checksums,
    )
    allowlist = json.loads(
        control_allowlist.read_text(encoding="utf-8")
    )
    if (
        allowlist.get("schema_version")
        != "saded-control-allowlist/v1"
        or set(allowlist.get("slots", {})) != set(CONTROL_SLOTS)
        or any(
            record.get("resolution") not in {"RUN_FRESH", "BOUND"}
            for record in allowlist["slots"].values()
        )
        or allowlist["slots"]["B:PREFLIGHT_1:0"].get("resolution")
        != "RUN_FRESH"
    ):
        raise ValueError("control allowlist is not finalizable")
    requirements_record = allowlist.get("requirements", {})
    requirements_path = Path(requirements_record["path"])
    if (
        sha256_file(requirements_path)
        != requirements_record.get("sha256")
    ):
        raise ValueError("control requirements binding drift")
    expected_requirements = _control_requirements(
        baseline=baseline,
        run_root=run_root,
        runtime_commit=_git(repo_root, "rev-parse", "HEAD"),
        repo_bundle=source_bundle_sha256(
            repo_source_hashes(repo_root)
        ),
        upstream_bundle=source_bundle_sha256(upstream),
    )
    if json.loads(
        requirements_path.read_text(encoding="utf-8")
    ) != expected_requirements:
        raise ValueError("control requirements content drift")
    bound_candidates = [
        record["candidate"]
        for record in allowlist["slots"].values()
        if record.get("resolution") == "BOUND"
    ]
    replayed_allowlist = resolve_control_allowlist(
        expected_requirements,
        bound_candidates,
    )
    if replayed_allowlist["slots"] != allowlist["slots"]:
        raise ValueError("control allowlist replay drift")

    full_data = yaml.safe_load(
        full_dataset_yaml.read_text(encoding="utf-8")
    )
    old_subset_yaml = yaml.safe_load(
        Path(baseline["train_only_yaml"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if full_data.get("names") != old_subset_yaml.get("names"):
        raise ValueError("class mapping drift")
    dataset_root = Path(baseline["dataset"]["root"]).resolve()
    subset_path = Path(baseline["subset"]["path"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_yaml = output_dir / "tascv_subset_train_only.yaml"
    full_yaml = output_dir / "tascv_full_train_only.yaml"
    _atomic_write(
        subset_yaml,
        yaml.safe_dump(
            {
                "path": dataset_root.as_posix(),
                "train": subset_path.as_posix(),
                "val": subset_path.as_posix(),
                "names": full_data["names"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
    )
    full_train = dataset_root / "images/train"
    if not full_train.is_dir():
        raise FileNotFoundError(full_train)
    _atomic_write(
        full_yaml,
        yaml.safe_dump(
            {
                "path": dataset_root.as_posix(),
                "train": full_train.as_posix(),
                "val": full_train.as_posix(),
                "names": full_data["names"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
    )
    if (
        sha256_file(subset_yaml)
        != baseline["train_only_yaml"]["sha256"]
        or sha256_file(full_yaml)
        != baseline["full_train_only_yaml"]["sha256"]
    ):
        raise ValueError("T-ASCV train-only YAML content drift")
    sources = repo_source_hashes(repo_root)
    runtime_commit = _git(repo_root, "rev-parse", "HEAD")
    treatment_endpoints = {}
    for stage, contract in FROZEN_STAGE_CONTRACT.items():
        for seed in contract["seeds"]:
            project = (
                run_root / "treatments" / stage / f"seed{seed}"
            )
            treatment_endpoints[f"T:{stage}:{seed}"] = {
                "project": project.resolve().as_posix(),
                "name": "tascv",
                "target_dir": (project / "tascv").resolve().as_posix(),
            }
    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_id": f"final-tascv-{runtime_commit[:8]}",
        "runtime_source": {
            "commit": runtime_commit,
            "repo_files": sources,
            "repo_bundle_sha256": source_bundle_sha256(sources),
            "upstream": upstream,
            "upstream_bundle_sha256": source_bundle_sha256(upstream),
        },
        "approved_tascv_parent": APPROVED_TASCV_PARENT,
        "r0_authority": r0,
        "environment": EXPECTED_ENVIRONMENT,
        "dataset": baseline["dataset"],
        "category_mapping_sha256": EXPECTED_CATEGORY_MAPPING_SHA256,
        "subset": baseline["subset"],
        "initial_states": baseline["initial_states"],
        "parent_lineage": baseline["parent_lineage"],
        "train_only_yaml": {
            "path": subset_yaml.resolve().as_posix(),
            "sha256": sha256_file(subset_yaml),
        },
        "full_train_only_yaml": {
            "path": full_yaml.resolve().as_posix(),
            "sha256": sha256_file(full_yaml),
        },
        "stage_contract": FROZEN_STAGE_CONTRACT,
        "training_contract": FROZEN_TRAINING_CONTRACT,
        "scientific_contract": {
            "state_machine": list(FROZEN_STATE_MACHINE),
            "crop": FROZEN_CROP_CONTRACT,
            "tascv": FROZEN_TASCV_CONTRACT,
            "mechanism_gate": FROZEN_MECHANISM_GATE,
            "screen_gate": FROZEN_SCREEN_GATE,
            "formal_thresholds": FROZEN_FORMAL_THRESHOLDS,
        },
        "control_allowlist": {
            "path": control_allowlist.resolve().as_posix(),
            "sha256": sha256_file(control_allowlist),
            "slots": allowlist["slots"],
        },
        "treatment_endpoints": treatment_endpoints,
        "reuse_policy": {
            "old_ascv_preflight_authorizes": False,
            "old_ascv_m500_endpoint_reusable": False,
            "invalid_or_partial_reusable": False,
            "stock_control_provenance_only": True,
        },
        "forbidden_data": ["test-dev", "test_dev"],
    }
    _atomic_json(output_dir / "protocol_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen T-ASCV protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    requirements = subparsers.add_parser("requirements")
    requirements.add_argument(
        "--baseline-protocol",
        type=Path,
        required=True,
    )
    requirements.add_argument("--run-root", type=Path, required=True)
    requirements.add_argument("--output", type=Path, required=True)
    requirements.add_argument("--repo-root", type=Path, default=ROOT)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument(
        "--baseline-protocol",
        type=Path,
        required=True,
    )
    finalize.add_argument(
        "--full-dataset-yaml",
        type=Path,
        required=True,
    )
    finalize.add_argument("--route-anchor", type=Path, required=True)
    finalize.add_argument("--route-checksums", type=Path, required=True)
    finalize.add_argument(
        "--evaluation-anchor",
        type=Path,
        required=True,
    )
    finalize.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
    )
    finalize.add_argument(
        "--evaluation-checksums",
        type=Path,
        required=True,
    )
    finalize.add_argument(
        "--control-allowlist",
        type=Path,
        required=True,
    )
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "requirements":
        value = prepare_requirements(
            baseline_protocol=args.baseline_protocol.resolve(),
            run_root=args.run_root.resolve(),
            output=args.output.resolve(),
            repo_root=args.repo_root.resolve(),
        )
    else:
        value = finalize_protocol(
            baseline_protocol=args.baseline_protocol.resolve(),
            full_dataset_yaml=args.full_dataset_yaml.resolve(),
            route_anchor=args.route_anchor.resolve(),
            route_checksums=args.route_checksums.resolve(),
            evaluation_anchor=args.evaluation_anchor.resolve(),
            evaluation_manifest=args.evaluation_manifest.resolve(),
            evaluation_checksums=args.evaluation_checksums.resolve(),
            control_allowlist=args.control_allowlist.resolve(),
            run_root=args.run_root.resolve(),
            output_dir=args.output.resolve(),
            repo_root=args.repo_root.resolve(),
        )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

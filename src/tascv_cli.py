"""Fail-closed CLI contract for frozen T-ASCV training stages."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import yaml

from src.tascv_protocol import (
    APPROVED_TASCV_PARENT,
    FROZEN_STAGE_CONTRACT,
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
    FROZEN_STATE_MACHINE,
    FROZEN_TASCV_CONTRACT,
    FROZEN_TRAINING_CONTRACT,
    PROTOCOL_VERSION,
    repo_source_hashes,
    require_clean_repo,
    sha256_file,
    source_bundle_sha256,
    subset_signature,
    validate_initial_state_artifact,
    validate_parent_attestation,
    validate_r0_closure,
    validate_runtime_manifest,
)
from src.tascv_stage import TASCVStage, allowed_seeds, stage_policy
from src.tascv_adjudicator import (
    replay_mechanism_gate,
    replay_preflight_gate,
    validate_paired_control_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen matched T-ASCV training endpoint."
    )
    parser.add_argument(
        "--stage",
        type=TASCVStage,
        choices=list(TASCVStage),
        required=True,
    )
    parser.add_argument(
        "--arm",
        choices=("control", "tascv"),
        required=True,
    )
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--predecessor-evidence", type=Path)
    return parser


def _reject_forbidden(value: str | Path) -> None:
    raw = str(value).replace("\\", "/").lower()
    resolved = Path(value).resolve().as_posix().lower()
    if any(
        token in raw or token in resolved
        for token in ("test-dev", "test_dev")
    ):
        raise ValueError(f"test-dev is forbidden in T-ASCV training: {value}")


def _reject_forbidden_manifest_paths(record: object) -> None:
    path_fields = {
        "path",
        "root",
        "full_yaml",
        "route_anchor",
        "route_checksums",
        "evaluation_anchor",
        "evaluation_manifest",
        "evaluation_checksums",
        "gate",
        "parent_protocol",
        "project",
        "target_dir",
        "summary",
        "checkpoint",
    }
    if isinstance(record, dict):
        for key, value in record.items():
            if isinstance(value, str) and (
                key in path_fields
                or key.endswith("_path")
                or key.endswith("_yaml")
            ):
                _reject_forbidden(value)
            else:
                _reject_forbidden_manifest_paths(value)
    elif isinstance(record, list):
        for value in record:
            _reject_forbidden_manifest_paths(value)


def current_environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def current_upstream_source_hashes() -> dict[str, str]:
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        raise RuntimeError("Ultralytics is not installed")
    root = Path(spec.origin).resolve().parent
    paths = {
        "head.py": root / "nn/modules/head.py",
        "tasks.py": root / "nn/tasks.py",
        "rtdetr-l.yaml": root / "cfg/models/rt-detr/rtdetr-l.yaml",
        "data/augment.py": root / "data/augment.py",
        "data/build.py": root / "data/build.py",
        "data/dataset.py": root / "data/dataset.py",
        "engine/trainer.py": root / "engine/trainer.py",
        "models/rtdetr/model.py": root / "models/rtdetr/model.py",
        "models/rtdetr/train.py": root / "models/rtdetr/train.py",
        "models/utils/loss.py": root / "models/utils/loss.py",
        "models/utils/ops.py": root / "models/utils/ops.py",
        "nn/modules/block.py": root / "nn/modules/block.py",
        "nn/modules/conv.py": root / "nn/modules/conv.py",
        "nn/modules/transformer.py": root / "nn/modules/transformer.py",
        "optim/muon.py": root / "optim/muon.py",
        "utils/loss.py": root / "utils/loss.py",
        "utils/torch_utils.py": root / "utils/torch_utils.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _scientific_contract() -> dict:
    return {
        "state_machine": list(FROZEN_STATE_MACHINE),
        "crop": FROZEN_CROP_CONTRACT,
        "tascv": FROZEN_TASCV_CONTRACT,
        "mechanism_gate": FROZEN_MECHANISM_GATE,
        "screen_gate": FROZEN_SCREEN_GATE,
        "formal_thresholds": FROZEN_FORMAL_THRESHOLDS,
    }


def _validate_train_only_yaml(
    data_path: Path,
    manifest: dict,
    *,
    uses_subset: bool,
) -> None:
    record = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or "test" in record:
        raise ValueError("T-ASCV data YAML is not train-only")
    if record.get("train") != record.get("val"):
        raise ValueError("T-ASCV train-only YAML aliases differ")
    if uses_subset:
        expected = Path(manifest["subset"]["path"]).resolve()
        if Path(record["train"]).resolve() != expected:
            raise ValueError("T-ASCV subset YAML path drift")
    else:
        root = Path(manifest["dataset"]["root"]).resolve()
        if (
            Path(record.get("path", "")).resolve() != root
            or Path(record["train"]).resolve() != root / "images/train"
        ):
            raise ValueError("T-ASCV full train-only YAML path drift")


def _validate_predecessor(
    args: argparse.Namespace,
    manifest: dict,
    manifest_sha: str,
) -> None:
    if args.stage is TASCVStage.PREFLIGHT_1:
        if args.predecessor_evidence is not None:
            raise ValueError("PREFLIGHT_1 forbids predecessor evidence")
        return
    if args.predecessor_evidence is None:
        raise ValueError(f"{args.stage.value} requires predecessor evidence")
    _reject_forbidden(args.predecessor_evidence)
    predecessor = json.loads(
        args.predecessor_evidence.resolve().read_text(encoding="utf-8")
    )
    expected = {
        TASCVStage.TINY_MECHANISM_500: "TASCV_PREFLIGHT_GO",
    }.get(args.stage)
    expected_schema = {
        TASCVStage.TINY_MECHANISM_500: (
            "tascv-preflight-adjudication/v1"
        ),
    }.get(args.stage)
    if args.stage is TASCVStage.SCREEN_10:
        expected = (
            "TASCV_MECHANISM_GO"
            if args.seed == 0
            else "TASCV_SCREEN_SEED0_GO"
        )
        expected_schema = (
            "tascv-mechanism-adjudication/v1"
            if args.seed == 0
            else "tascv-screen-seed0-adjudication/v1"
        )
    if args.stage is TASCVStage.FORMAL_100:
        expected = (
            "TASCV_SCREEN_GO"
            if args.seed == 0
            else "TASCV_FORMAL_SEED0_GO"
        )
        expected_schema = (
            "tascv-screen-three-seed-adjudication/v1"
            if args.seed == 0
            else "tascv-formal-seed0-adjudication/v1"
        )
    if (
        predecessor.get("schema_version") != expected_schema
        or predecessor.get("decision") != expected
    ):
        raise ValueError("T-ASCV predecessor decision does not authorize stage")
    if args.stage is TASCVStage.TINY_MECHANISM_500:
        replay_preflight_gate(predecessor)
    elif (
        args.stage is TASCVStage.SCREEN_10
        and args.seed == 0
    ):
        replay_mechanism_gate(predecessor)
    elif args.stage is TASCVStage.SCREEN_10:
        raise ValueError(
            "T-ASCV screen seeds1/2 remain closed until the new "
            "seed0 attribution predecessor replay is implemented"
        )
    elif args.stage is TASCVStage.FORMAL_100:
        raise ValueError(
            "T-ASCV formal launch remains closed until the new "
            "SADED screen predecessor replay is implemented"
        )
    if (
        predecessor.get("protocol_manifest_sha256") != manifest_sha
        or predecessor.get("protocol_source_commit")
        != manifest.get("runtime_source", {}).get("commit")
    ):
        raise ValueError("T-ASCV predecessor protocol binding drift")


def validate_protocol_inputs(args: argparse.Namespace) -> dict:
    for value in (
        args.initial_state,
        args.data,
        args.protocol_manifest,
        args.project,
    ):
        _reject_forbidden(value)
    if str(args.device) != "0":
        raise ValueError("T-ASCV requires the single GPU device 0")
    if args.seed not in allowed_seeds(args.stage):
        raise ValueError("T-ASCV stage/seed mismatch")

    manifest_path = args.protocol_manifest.resolve()
    manifest, manifest_sha = validate_runtime_manifest(manifest_path)
    _reject_forbidden_manifest_paths(manifest)
    if manifest.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError("unexpected T-ASCV protocol schema")
    if manifest.get("environment") != EXPECTED_ENVIRONMENT:
        raise ValueError("T-ASCV protocol environment drift")
    if manifest.get("approved_tascv_parent") != APPROVED_TASCV_PARENT:
        raise ValueError("T-ASCV approved-parent binding drift")
    if manifest.get("stage_contract") != FROZEN_STAGE_CONTRACT:
        raise ValueError("T-ASCV stage contract drift")
    if manifest.get("training_contract") != FROZEN_TRAINING_CONTRACT:
        raise ValueError("T-ASCV training contract drift")
    if manifest.get("scientific_contract") != _scientific_contract():
        raise ValueError("T-ASCV scientific contract drift")
    r0 = manifest.get("r0_authority", {})
    validate_r0_closure(r0)
    _validate_predecessor(args, manifest, manifest_sha)

    repo_root = Path(__file__).resolve().parents[1]
    require_clean_repo(repo_root)
    actual_sources = repo_source_hashes(repo_root)
    source = manifest.get("runtime_source", {})
    if (
        actual_sources != source.get("repo_files")
        or source_bundle_sha256(actual_sources)
        != source.get("repo_bundle_sha256")
    ):
        raise ValueError("T-ASCV source closure drift")
    if (
        source_bundle_sha256(source.get("upstream", {}))
        != source.get("upstream_bundle_sha256")
    ):
        raise ValueError("T-ASCV upstream bundle drift")
    if current_environment() != EXPECTED_ENVIRONMENT:
        raise ValueError("T-ASCV runtime environment drift")
    upstream = current_upstream_source_hashes()
    if (
        upstream != EXPECTED_UPSTREAM_SOURCE_SHA256
        or upstream != source.get("upstream")
    ):
        raise ValueError("T-ASCV upstream source drift")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != source.get("commit"):
        raise ValueError("T-ASCV source commit drift")
    for relative, expected_sha in APPROVED_TASCV_PARENT["files"].items():
        if sha256_file(repo_root / relative) != expected_sha:
            raise ValueError("T-ASCV approved-parent file drift")

    allowlist_record = manifest.get("control_allowlist", {})
    allowlist_path = Path(allowlist_record.get("path", "")).resolve()
    if (
        not allowlist_path.is_file()
        or sha256_file(allowlist_path)
        != allowlist_record.get("sha256")
    ):
        raise ValueError("T-ASCV control allowlist binding drift")
    allowlist = json.loads(
        allowlist_path.read_text(encoding="utf-8")
    )
    if allowlist.get("slots") != allowlist_record.get("slots"):
        raise ValueError("T-ASCV inline control allowlist drift")
    for slot_id, slot_record in allowlist["slots"].items():
        if slot_record.get("resolution") == "BOUND":
            checkpoint = slot_record.get("candidate", {}).get(
                "checkpoint",
                {},
            )
            checkpoint_path = Path(
                checkpoint.get("path", "")
            ).resolve()
            _reject_forbidden(checkpoint_path)
            if (
                checkpoint.get("kind") != "last.pt"
                or not checkpoint_path.is_file()
                or sha256_file(checkpoint_path)
                != checkpoint.get("sha256", "").upper()
            ):
                raise ValueError(
                    f"T-ASCV bound control drift: {slot_id}"
                )

    stage_record = FROZEN_STAGE_CONTRACT[args.stage.value]
    if (
        args.seed not in stage_record["seeds"]
        or args.arm not in stage_record["arms"]
    ):
        raise ValueError("T-ASCV stage/seed/arm mismatch")
    if (
        args.stage is TASCVStage.TINY_MECHANISM_500
        and args.arm != "tascv"
    ):
        raise ValueError("T-ASCV mechanism stage is treatment-only")
    if args.arm == "control":
        slot = f"B:{args.stage.value}:{args.seed}"
        control = manifest["control_allowlist"]["slots"].get(slot)
        if not control or control.get("resolution") != "RUN_FRESH":
            raise ValueError("T-ASCV control endpoint is not RUN_FRESH")
        expected_endpoint = control["fresh_target"]
    else:
        slot = f"T:{args.stage.value}:{args.seed}"
        expected_endpoint = manifest["treatment_endpoints"][slot]
    project = Path(args.project).resolve()
    if (
        project.as_posix() != expected_endpoint["project"]
        or args.name != expected_endpoint["name"]
    ):
        raise ValueError("T-ASCV output endpoint drift")
    target = project / args.name
    if target.exists():
        raise ValueError("T-ASCV fixed output target already exists")
    if target.as_posix().startswith("/mnt/uav/"):
        raise ValueError("T-ASCV refuses to write under read-only /mnt/uav")
    existing = project
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if shutil.disk_usage(existing).free < 2 * 1024**3:
        raise ValueError("T-ASCV requires at least 2 GiB free")

    policy = stage_policy(args.stage)
    key = (
        "train_only_yaml"
        if policy.uses_hashed_subset
        else "full_train_only_yaml"
    )
    data_record = manifest[key]
    data_path = args.data.resolve()
    if (
        data_path != Path(data_record["path"]).resolve()
        or sha256_file(data_path) != data_record["sha256"]
    ):
        raise ValueError("T-ASCV data binding drift")
    _validate_train_only_yaml(
        data_path,
        manifest,
        uses_subset=policy.uses_hashed_subset,
    )
    if policy.uses_hashed_subset:
        subset = manifest["subset"]
        subset_path = Path(subset["path"]).resolve()
        if (
            sha256_file(subset_path) != EXPECTED_SUBSET_FILE_SHA256
            or subset_signature(
                subset_path,
                root=Path(manifest["dataset"]["root"]),
            )
            != {
                "count": EXPECTED_SUBSET_COUNT,
                "sha256": EXPECTED_SUBSET_SHA256,
            }
        ):
            raise ValueError("T-ASCV subset binding drift")

    validate_parent_attestation(manifest, args.seed)
    expected = manifest["initial_states"][str(args.seed)]
    initial_path = args.initial_state.resolve()
    actual_initial = sha256_file(initial_path)
    if (
        initial_path != Path(expected["path"]).resolve()
        or actual_initial != expected["sha256"]
        or actual_initial != EXPECTED_INITIAL_STATE_SHA256[args.seed]
    ):
        raise ValueError("T-ASCV initial-state binding drift")
    artifact = torch.load(
        initial_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_initial_state_artifact(artifact, seed=args.seed)
    paired_slot_id = f"B:{args.stage.value}:{args.seed}"
    if (
        args.arm == "tascv"
        and paired_slot_id
        in manifest["control_allowlist"]["slots"]
    ):
        slot_record = manifest["control_allowlist"]["slots"][
            paired_slot_id
        ]
        if slot_record.get("resolution") == "RUN_FRESH":
            control_summary_path = Path(
                slot_record["fresh_target"]["summary"]
            ).resolve()
            if not control_summary_path.is_file():
                raise ValueError(
                    "T-ASCV treatment requires completed paired control"
                )
            control_summary = json.loads(
                control_summary_path.read_text(encoding="utf-8")
            )
            validate_paired_control_summary(
                control_summary,
                stage=args.stage.value,
                seed=args.seed,
            )
            expected_control = {
                "protocol_manifest_sha256": manifest_sha,
                "protocol_source_commit": source["commit"],
                "source_repo_bundle_sha256": source[
                    "repo_bundle_sha256"
                ],
                "source_upstream_bundle_sha256": source[
                    "upstream_bundle_sha256"
                ],
                "approved_tascv_parent": APPROVED_TASCV_PARENT,
                "r0_evaluation_anchor_sha256": r0[
                    "evaluation_anchor_sha256"
                ],
                "initial_state_sha256": actual_initial,
                "initial_state_common_fingerprint": expected[
                    "common_fingerprint"
                ],
                "data_sha256": sha256_file(data_path),
                "control_slot": slot_record,
            }
            for key, value in expected_control.items():
                if control_summary.get(key) != value:
                    raise ValueError(
                        f"T-ASCV paired control binding drift: {key}"
                    )
    return manifest


def build_settings(args: argparse.Namespace) -> dict:
    policy = stage_policy(args.stage)
    return {
        "model": "rtdetr-l.yaml",
        "data": str(Path(args.data).resolve()),
        "epochs": policy.epochs,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": str(args.device),
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "exist_ok": False,
        "pretrained": False,
        "resume": False,
        "cache": False,
        "amp": True,
        "compile": False,
        "deterministic": True,
        "seed": args.seed,
        "fraction": 1.0,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": -1,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "plots": False,
        "val": False,
    }


__all__ = [
    "build_parser",
    "build_settings",
    "current_environment",
    "current_upstream_source_hashes",
    "validate_protocol_inputs",
]

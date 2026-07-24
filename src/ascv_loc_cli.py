from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import torch
import yaml

from src.ascv_loc_protocol import (
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
    repo_source_hashes,
    require_clean_repo,
    sha256_file,
    source_bundle_sha256,
    subset_signature,
    validate_initial_state_artifact,
    validate_parent_attestation,
)
from src.ascv_loc_stage import ASCVStage, allowed_seeds, stage_policy
from src.ascv_loc_adjudicator import replay_preflight_gate


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen ASCV-Loc training stage.")
    parser.add_argument("--stage", type=ASCVStage, choices=list(ASCVStage), required=True)
    parser.add_argument("--arm", choices=("control", "ascv"), required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Protocol-generated train-only YAML.")
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predecessor-evidence", type=Path)
    return parser


def _reject_forbidden(value: str | Path) -> None:
    normalized = str(value).replace("\\", "/").lower()
    if "test-dev" in normalized or "test_dev" in normalized:
        raise ValueError(f"test-dev is forbidden in ASCV-Loc training: {value}")


def current_environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def current_upstream_source_hashes() -> dict[str, str]:
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        raise RuntimeError("Ultralytics is not installed")
    root = Path(spec.origin).resolve().parent
    paths = {
        "head.py": root / "nn" / "modules" / "head.py",
        "tasks.py": root / "nn" / "tasks.py",
        "rtdetr-l.yaml": root / "cfg" / "models" / "rt-detr" / "rtdetr-l.yaml",
        "data/augment.py": root / "data" / "augment.py",
        "data/build.py": root / "data" / "build.py",
        "data/dataset.py": root / "data" / "dataset.py",
        "engine/trainer.py": root / "engine" / "trainer.py",
        "models/rtdetr/model.py": root / "models" / "rtdetr" / "model.py",
        "models/rtdetr/train.py": root / "models" / "rtdetr" / "train.py",
        "models/utils/loss.py": root / "models" / "utils" / "loss.py",
        "models/utils/ops.py": root / "models" / "utils" / "ops.py",
        "nn/modules/block.py": root / "nn" / "modules" / "block.py",
        "nn/modules/conv.py": root / "nn" / "modules" / "conv.py",
        "nn/modules/transformer.py": root / "nn" / "modules" / "transformer.py",
        "optim/muon.py": root / "optim" / "muon.py",
        "utils/loss.py": root / "utils" / "loss.py",
        "utils/torch_utils.py": root / "utils" / "torch_utils.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _validate_train_only_yaml(data_path: Path, manifest: dict, *, uses_subset: bool) -> None:
    record = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or "test" in record:
        raise ValueError("stage data YAML is not train-only")
    train = record.get("train")
    val = record.get("val")
    if train != val:
        raise ValueError("train-only YAML train/val aliases differ")
    if uses_subset:
        expected = Path(manifest["subset"]["path"]).resolve()
        if Path(train).resolve() != expected:
            raise ValueError("train-only YAML does not resolve to the sealed subset")
    else:
        expected = Path(manifest["dataset"]["root"]).resolve() / "images" / "train"
        if Path(record.get("path", "")).resolve() != Path(manifest["dataset"]["root"]).resolve():
            raise ValueError("full train-only YAML dataset root drift")
        if Path(train).resolve() != expected:
            raise ValueError("full train-only YAML does not resolve to sealed images/train")


def validate_protocol_inputs(args: argparse.Namespace) -> dict:
    for value in (args.initial_state, args.data, args.protocol_manifest, args.project):
        _reject_forbidden(value)
    if str(args.device) != "0":
        raise ValueError("ASCV-Loc requires the single GPU device 0")
    manifest_path = args.protocol_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "ascv-loc-matched/v2":
        raise ValueError("unexpected ASCV-Loc protocol schema")
    if manifest.get("environment", {}).get("ultralytics") != "8.4.90":
        raise ValueError("protocol is not frozen to Ultralytics 8.4.90")
    expected_scientific_contract = {
        "state_machine": list(FROZEN_STATE_MACHINE),
        "crop": FROZEN_CROP_CONTRACT,
        "mechanism_gate": FROZEN_MECHANISM_GATE,
        "screen_gate": FROZEN_SCREEN_GATE,
        "formal_thresholds": FROZEN_FORMAL_THRESHOLDS,
    }
    if manifest.get("scientific_contract") != expected_scientific_contract:
        raise ValueError("protocol scientific contract drift")
    predecessor_decisions = {
        ASCVStage.MECHANISM_500: "PREFLIGHT_GO",
        ASCVStage.SCREEN_10: "GO",
        ASCVStage.SEED0_100: "SCREEN_GO",
        ASCVStage.SEED1_100: "FORMAL_SEED0_GO",
        ASCVStage.SEED2_100: "FORMAL_SEED0_GO",
    }
    predecessor_path = args.predecessor_evidence
    if args.stage is ASCVStage.PREFLIGHT_1:
        if predecessor_path is not None:
            raise ValueError("PREFLIGHT_1 must not declare predecessor evidence")
    else:
        if predecessor_path is None:
            raise ValueError(f"{args.stage.value} requires predecessor evidence")
        _reject_forbidden(predecessor_path)
        predecessor = json.loads(predecessor_path.resolve().read_text(encoding="utf-8"))
        if predecessor.get("decision") != predecessor_decisions[args.stage]:
            raise ValueError("predecessor decision does not authorize this stage")
        if args.stage is ASCVStage.MECHANISM_500:
            predecessor = replay_preflight_gate(predecessor)
            protocol_binding = predecessor["protocol"]
            if protocol_binding.get("manifest_sha256") != sha256_file(manifest_path):
                raise ValueError("predecessor protocol manifest checksum mismatch")
            if protocol_binding.get("source_commit") != manifest.get("source_commit"):
                raise ValueError("predecessor source commit mismatch")
        else:
            if predecessor.get("protocol_manifest_sha256") != sha256_file(manifest_path):
                raise ValueError("predecessor protocol manifest checksum mismatch")
            if predecessor.get("protocol_source_commit") != manifest.get("source_commit"):
                raise ValueError("predecessor source commit mismatch")
    repo_root = Path(__file__).resolve().parents[1]
    require_clean_repo(repo_root)
    actual_sources = repo_source_hashes(repo_root)
    source_record = manifest.get("source", {})
    if actual_sources != source_record.get("repo_files"):
        raise ValueError("source file checksum does not match the frozen protocol")
    if source_bundle_sha256(actual_sources) != source_record.get("repo_bundle_sha256"):
        raise ValueError("source bundle checksum does not match the frozen protocol")
    if current_environment() != EXPECTED_ENVIRONMENT:
        raise ValueError("runtime environment does not match the frozen protocol")
    upstream_hashes = current_upstream_source_hashes()
    if (
        upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256
        or upstream_hashes != source_record.get("upstream")
    ):
        raise ValueError("Ultralytics source checksum does not match the frozen protocol")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if manifest.get("source_commit") != current_commit:
        raise ValueError(
            f"source commit does not match the frozen protocol: {current_commit}"
        )

    policy = stage_policy(args.stage)
    data_record = manifest["train_only_yaml"] if policy.uses_hashed_subset else manifest["full_train_only_yaml"]
    data_path = args.data.resolve()
    if data_path.as_posix() != Path(data_record["path"]).resolve().as_posix():
        raise ValueError("stage data path does not match the frozen protocol")
    if sha256_file(data_path).upper() != data_record["sha256"].upper():
        raise ValueError("stage data checksum does not match the frozen protocol")
    _validate_train_only_yaml(data_path, manifest, uses_subset=policy.uses_hashed_subset)

    if int(args.seed) not in allowed_seeds(args.stage):
        raise ValueError(f"stage/seed mismatch: {args.stage.value} cannot use seed {args.seed}")
    if args.stage is ASCVStage.MECHANISM_500 and args.arm != "ascv":
        raise ValueError("MECHANISM_500 is ASCV-only")
    if policy.uses_hashed_subset:
        subset = manifest["subset"]
        subset_path = Path(subset["path"]).resolve()
        if sha256_file(subset_path) != subset["file_sha256"] or sha256_file(subset_path) != EXPECTED_SUBSET_FILE_SHA256:
            raise ValueError("subset file checksum does not match the frozen protocol")
        semantic = subset_signature(subset_path, root=Path(manifest["dataset"]["root"]))
        expected_semantic = {"count": EXPECTED_SUBSET_COUNT, "sha256": EXPECTED_SUBSET_SHA256}
        if semantic != expected_semantic or semantic != {
            "count": int(subset["count"]),
            "sha256": subset["semantic_sha256"],
        }:
            raise ValueError("subset semantic signature does not match the frozen protocol")
    validate_parent_attestation(manifest, int(args.seed))
    expected = manifest["initial_states"][str(args.seed)]
    initial_path = args.initial_state.resolve()
    if initial_path.as_posix() != Path(expected["path"]).resolve().as_posix():
        raise ValueError("initial-state path does not match the frozen protocol")
    if sha256_file(initial_path).upper() != expected["sha256"].upper():
        raise ValueError("initial-state checksum does not match the frozen protocol")
    if sha256_file(initial_path).upper() != EXPECTED_INITIAL_STATE_SHA256[int(args.seed)]:
        raise ValueError("initial-state checksum is not allowlisted")
    artifact = torch.load(initial_path, map_location="cpu", weights_only=False)
    validate_initial_state_artifact(artifact, seed=int(args.seed))
    return manifest


def build_settings(args: argparse.Namespace) -> dict:
    policy = stage_policy(args.stage)
    return {
        "model": "rtdetr-l.yaml",
        "data": str(args.data.resolve()),
        "epochs": policy.epochs,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": args.device,
        "project": str(args.project.resolve()),
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
        "save_period": 1,
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

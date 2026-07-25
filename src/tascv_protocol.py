"""Frozen authority and source closure for the independent T-ASCV route."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import subprocess

import torch
import yaml

from src.ascv_loc_protocol import (
    EXPECTED_CATEGORY_MAPPING_SHA256,
    EXPECTED_COMMON_FINGERPRINTS,
    EXPECTED_DATASET_FILE_COUNT,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ENVIRONMENT,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_PARENT_ATTESTATION_SHA256,
    EXPECTED_PARENT_SOURCE_SHA256,
    EXPECTED_SUBSET_COUNT,
    EXPECTED_SUBSET_FILE_SHA256,
    EXPECTED_SUBSET_SHA256,
    EXPECTED_UPSTREAM_SOURCE_SHA256,
    require_clean_repo,
    sha256_file,
    source_bundle_sha256,
    state_fingerprint,
    subset_signature,
    training_batch_sha256,
    validate_initial_state_artifact,
    validate_parent_attestation,
)


PROTOCOL_VERSION = "tascv-saded-protocol/v1"
EXPECTED_DATASET_ROOT = "/mnt/uav/datasets/VisDrone"
EXPECTED_SUBSET_PATH = (
    "/mnt/uav/protocols/ebc-qp-d2-musgd-seed0/"
    "d2-train-10pct.txt"
)
EXPECTED_INITIAL_STATE_PATHS = {
    0: (
        "/mnt/uav/protocols/ebc-qp-d2-musgd-seed0/"
        "initial-state-seed0.pt"
    ),
    1: "/mnt/uav/protocols/tsgr-p2-e1/initial-state-seed1.pt",
    2: "/mnt/uav/protocols/tsgr-p2-e1/initial-state-seed2.pt",
}
APPROVED_TASCV_PARENT = {
    "commit": "c8fc52db0744177481c8e742b16871df76dd175a",
    "tree": "7aa8eab098da70b8546e010f0b3277427c0a9191",
    "bundle_sha256": (
        "9831E5204DEB83847E913731B6AA2CB5CB211D3AAE5FF866F3B89184CCFFE875"
    ),
    "files": {
        "src/tascv.py": (
            "FE3BE8202EFA17BB9499DBCAB914D370750281A11E4372215CBBE607CA99DD80"
        ),
        "src/tascv_stage.py": (
            "5A8D2A7A7B8A0F981F9D8AD55712FA4C6D216B64E9CF0E0D0FC1A90AE99AFB78"
        ),
        "src/tascv_diagnostics.py": (
            "C8687ECD9C116224F0F995CDEB40C6A25AEB31CDA8F626C4B289BA01EBA22766"
        ),
        "src/rtdetr_tascv.py": (
            "84D695F51F1FA8FB60ABE575B2A241E13BE0088C64CEF39A945B9E0F2F0472DB"
        ),
    },
}
R0_SOURCE_COMMIT = "ada48a1f09e468138e70eaa4b20cd426de6157da"
R0_ROUTE_ANCHOR_SHA256 = (
    "E3C3A391496774412C60C921BF2DB11CDBC2DE908A562E5AD173123F36FB077C"
)
R0_EVALUATION_ANCHOR_SHA256 = (
    "CBCD803318F59372B3BB0FEFFC234F829E9CD7653399FECF58E2E48C46926CC8"
)
R0_EVALUATION_CHECKSUM_ROOT = (
    "7A9598773B7C4B32FFE0D1658F785D4131146438015CBD3A32A2C946CB1EFC69"
)
R0_EVALUATION_MANIFEST_SHA256 = (
    "793A6A541100A81A3133D3EBC9D8937446B266D23CE6AE61B3A315AD94E3DFF6"
)

FROZEN_STATE_MACHINE = (
    "R0_GO",
    "PREFLIGHT_1_SEED0_PAIRED",
    "TINY_MECHANISM_500_SEED1",
    "SCREEN_10_PAIRED_SEED0",
    "SCREEN_SEED0_ATTRIBUTION_GO",
    "SCREEN_10_PAIRED_SEEDS_1_2",
    "SCREEN_THREE_SEED_ATTRIBUTION_GO",
    "FORMAL_100_PAIRED_SEED0",
    "FORMAL_SEED0_PRIMARY_AND_ATTRIBUTION_GO",
    "FORMAL_100_PAIRED_SEEDS_1_2",
    "ONE_SEALED_TEST_DEV_ADJUDICATION",
)
FROZEN_CROP_CONTRACT = {
    "protocol": "ascv-loc/crop-v2",
    "input_hw": [640, 640],
    "crop_hw": [384, 384],
    "containment_tolerance_px": 1e-6,
    "identity": (
        "resolved_im_file.relative_to("
        "resolved_dataset_root).as_posix()"
    ),
}
FROZEN_TASCV_CONTRACT = {
    "target_effective_size_px_max": 16.0,
    "teacher": "mapped_local_prediction_detached",
    "student": "full_view_prediction",
    "loss": "fp32_mean_l1_plus_one_minus_aligned_giou",
    "lambda": 0.1,
    "warmup_epochs": 3,
    "non_tiny_auxiliary_contribution": 0,
}
FROZEN_MECHANISM_GATE = {
    "successful_batches": 500,
    "optimizer_attempts": 106,
    "scientific_tail_window": [401, 500],
    "minimum_tiny_pairs": 100,
    "minimum_tiny_batches": 80,
    "mean_advantage_strictly_positive": True,
    "win_rate_strictly_greater_than": 0.5,
    "auxiliary_non_tiny_pairs": 0,
}
FROZEN_SCREEN_GATE = {
    "seeds": [0, 1, 2],
    "attribution": "route_treatment_minus_route_control",
    "seed0": {
        "mAP50-95": 0.0,
        "AP-tiny-SBR": 0.0,
        "tiny_recall": 0.0,
        "AP75": -0.002,
        "AP-large-SBR": -0.005,
        "mAP_strictly_greater": True,
    },
    "three_seed": {
        "mAP_positive_wins_minimum": 2,
        "mAP_mean_strictly_positive": True,
        "AP-tiny-SBR_nonnegative_wins_minimum": 2,
        "tiny_recall_nonnegative_wins_minimum": 2,
        "mean_guards": {
            "AP-tiny-SBR": 0.0,
            "tiny_recall": 0.0,
            "AP75": -0.002,
            "AP-large-SBR": -0.005,
        },
    },
}
FROZEN_FORMAL_THRESHOLDS = {
    "primary_route_treatment_minus_arm_a": {
        "AP-tiny-SBR": 0.010,
        "mAP50-95": 0.003,
        "tiny_recall": 0.020,
        "AP75": -0.002,
        "AP-large-SBR": -0.005,
    },
    "attribution_route_treatment_minus_route_control": {
        "mAP50-95": 0.0,
        "strictly_greater": True,
    },
}
FROZEN_CONFIRMATION_CONTRACT = {
    "state": "SEALED_UNOPENED",
    "seeds": [0, 1, 2],
    "systems": ["A", "route_control", "route_treatment"],
    "prediction_files": [
        f"seed{seed}_{system}.json"
        for seed in range(3)
        for system in ("A", "route_control", "route_treatment")
    ],
    "image_root_derivation": {
        "base": "dataset.root",
        "relative_parts": ["images", "test", "dev"],
        "join_rule": "parts[0]/(parts[1]+'-'+parts[2])",
    },
    "claim_file": "confirmation_open_claim.json",
    "claim_creation": "O_CREAT|O_EXCL",
    "one_shot": True,
}
FROZEN_STAGE_CONTRACT = {
    "PREFLIGHT_1": {
        "seeds": [0],
        "arms": ["control", "tascv"],
        "epochs": 100,
        "uses_hashed_subset": True,
        "max_train_batches": 1,
        "expected_successful_batches": 1,
        "expected_optimizer_attempts": 1,
        "allowed_observed_tensor_batch_sizes": [8],
    },
    "TINY_MECHANISM_500": {
        "seeds": [1],
        "arms": ["tascv"],
        "epochs": 100,
        "uses_hashed_subset": True,
        "max_train_batches": 500,
        "expected_successful_batches": 500,
        "expected_optimizer_attempts": 106,
        "allowed_observed_tensor_batch_sizes": [7, 8],
    },
    "SCREEN_10": {
        "seeds": [0, 1, 2],
        "arms": ["control", "tascv"],
        "epochs": 10,
        "uses_hashed_subset": True,
        "max_train_batches": None,
        "expected_successful_batches": 810,
        "expected_optimizer_attempts": 145,
        "allowed_observed_tensor_batch_sizes": [7, 8],
    },
    "FORMAL_100": {
        "seeds": [0, 1, 2],
        "arms": ["control", "tascv"],
        "epochs": 100,
        "uses_hashed_subset": False,
        "max_train_batches": None,
        "expected_successful_batches": 80_900,
        "expected_optimizer_attempts": 10_556,
        "allowed_observed_tensor_batch_sizes": [7, 8],
    },
}
FROZEN_TRAINING_CONTRACT = {
    "model": "rtdetr-l.yaml",
    "pretrained": False,
    "resume": False,
    "cache": False,
    "compile": False,
    "val": False,
    "plots": False,
    "save": True,
    "save_period": -1,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "device": "0",
    "deterministic": True,
    "fraction": 1.0,
    "optimizer": "MuSGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "nbs": 64,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": False,
    "amp": True,
    "amp_scale": 128.0,
    "batch_digest_schema": (
        "ascv-loc/training-batch-v1-legacy-encoding-only"
    ),
    "query_count": 300,
    "max_det": 300,
    "nms": False,
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
}
FROZEN_OPTIMIZER_OBSERVATION = {
    "class": "MuSGD",
    "requested_lr0": 0.01,
    "requested_momentum": 0.937,
    "groups": [
        {
            "lr": lr,
            "momentum": 0.937,
            "param_group": group,
            "parameter_count": count,
            "use_muon": use_muon,
            "weight_decay": decay,
        }
        for group, use_muon, decay, lr, count in (
            ("weight", False, 0.0005, 0.03, 0),
            ("weight", False, 0.0005, 0.01, 0),
            ("bn", False, 0.0, 0.03, 0),
            ("bn", False, 0.0, 0.01, 143),
            ("bias", False, 0.0, 0.03, 0),
            ("bias", False, 0.0, 0.01, 226),
            ("muon", True, 0.0005, 0.03, 0),
            ("muon", True, 0.0005, 0.01, 206),
        )
    ],
}
CONTROL_SLOTS = tuple(
    [f"B:PREFLIGHT_1:0"]
    + [f"B:SCREEN_10:{seed}" for seed in range(3)]
    + [f"B:FORMAL_100:{seed}" for seed in range(3)]
)

REPO_SOURCE_FILES = (
    "scripts/adjudicate_saded_stage.py",
    "scripts/adjudicate_tascv.py",
    "scripts/cache_saded_endpoint.py",
    "scripts/evaluate_saded_confirmation_once.py",
    "scripts/evaluate_saded_stage.py",
    "scripts/prepare_tascv_protocol.py",
    "scripts/resolve_saded_controls.py",
    "scripts/route_saded_pair.py",
    "scripts/seal_saded_confirmation_predictions.py",
    "scripts/train_rtdetr_tascv.py",
    "src/ascv_loc.py",
    "src/ascv_loc_protocol.py",
    "src/rtdetr_tascv.py",
    "src/saded.py",
    "src/saded_adjudicator.py",
    "src/saded_confirmation.py",
    "src/saded_stage.py",
    "src/saded_stage_protocol.py",
    "src/sbr_artifacts.py",
    "src/sbr_fusion.py",
    "src/sbr_g0.py",
    "src/sbr_geometry.py",
    "src/sbr_metrics.py",
    "src/sbr_ppaf.py",
    "src/sbr_v2_audit.py",
    "src/tascv.py",
    "src/tascv_adjudicator.py",
    "src/tascv_cli.py",
    "src/tascv_diagnostics.py",
    "src/tascv_protocol.py",
    "src/tascv_stage.py",
)


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
    relative_paths = {
        "head.py": "nn/modules/head.py",
        "tasks.py": "nn/tasks.py",
        "rtdetr-l.yaml": "cfg/models/rt-detr/rtdetr-l.yaml",
        "data/augment.py": "data/augment.py",
        "data/build.py": "data/build.py",
        "data/dataset.py": "data/dataset.py",
        "engine/trainer.py": "engine/trainer.py",
        "models/rtdetr/model.py": "models/rtdetr/model.py",
        "models/rtdetr/train.py": "models/rtdetr/train.py",
        "models/utils/loss.py": "models/utils/loss.py",
        "models/utils/ops.py": "models/utils/ops.py",
        "nn/modules/block.py": "nn/modules/block.py",
        "nn/modules/conv.py": "nn/modules/conv.py",
        "nn/modules/transformer.py": "nn/modules/transformer.py",
        "optim/muon.py": "optim/muon.py",
        "utils/loss.py": "utils/loss.py",
        "utils/torch_utils.py": "utils/torch_utils.py",
    }
    return {
        name: sha256_file(root / relative)
        for name, relative in relative_paths.items()
    }


def _read_json(path: Path) -> dict:
    raw = str(path).replace("\\", "/").lower()
    resolved = path.resolve().as_posix().lower()
    if any(
        token in raw or token in resolved
        for token in ("test-dev", "test_dev")
    ):
        raise ValueError(f"test-dev is forbidden in T-ASCV protocol: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"T-ASCV R0 artifact is not an object: {path}")
    return record


def reject_forbidden_path(path: str | Path, *, context: str) -> Path:
    candidate = Path(path)
    raw = str(path).replace("\\", "/").lower()
    resolved = candidate.resolve()
    normalized = resolved.as_posix().lower()
    if any(
        token in raw or token in normalized
        for token in ("test-dev", "test_dev")
    ):
        raise ValueError(f"test-dev is forbidden in {context}: {path}")
    return resolved


def validate_r0_authority(
    *,
    route_anchor: Path,
    evaluation_anchor: Path,
) -> dict:
    for path in (route_anchor, evaluation_anchor):
        raw = str(path).replace("\\", "/").lower()
        resolved = path.resolve().as_posix().lower()
        if any(
            token in raw or token in resolved
            for token in ("test-dev", "test_dev")
        ):
            raise ValueError(
                f"test-dev is forbidden in T-ASCV R0 authority: {path}"
            )
    actual = {
        "route": sha256_file(route_anchor),
        "evaluation": sha256_file(evaluation_anchor),
    }
    expected = {
        "route": R0_ROUTE_ANCHOR_SHA256,
        "evaluation": R0_EVALUATION_ANCHOR_SHA256,
    }
    if actual != expected:
        raise ValueError(
            f"T-ASCV R0 artifact checksum mismatch: {actual}"
        )
    route = _read_json(route_anchor)
    evaluation = _read_json(evaluation_anchor)
    if route.get("schema_version") != "sbr-saded-route-anchor/v1":
        raise ValueError("T-ASCV R0 route anchor schema mismatch")
    if (
        evaluation.get("schema_version") != "sbr-saded-r0-anchor/v1"
        or evaluation.get("decision") != "R0_GO"
        or evaluation.get("route_anchor_sha256", "").upper()
        != R0_ROUTE_ANCHOR_SHA256
        or evaluation.get("evaluation_checksums_sha256", "").upper()
        != R0_EVALUATION_CHECKSUM_ROOT
        or evaluation.get("evaluation_manifest_sha256", "").upper()
        != R0_EVALUATION_MANIFEST_SHA256
    ):
        raise ValueError("T-ASCV R0 evaluation anchor is not authoritative")
    return {
        "decision": "R0_GO",
        "source_commit": R0_SOURCE_COMMIT,
        "route_anchor": route_anchor.resolve().as_posix(),
        "route_anchor_sha256": R0_ROUTE_ANCHOR_SHA256,
        "evaluation_anchor": evaluation_anchor.resolve().as_posix(),
        "evaluation_anchor_sha256": R0_EVALUATION_ANCHOR_SHA256,
    }


def _checksum_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if len(digest) != 64 or name in names:
            raise ValueError(f"invalid T-ASCV checksum closure: {path}")
        names.add(name)
    return names


def validate_r0_closure(authority: dict) -> dict:
    path_fields = (
        "route_anchor",
        "route_checksums",
        "evaluation_anchor",
        "evaluation_manifest",
        "evaluation_checksums",
    )
    paths = {
        name: reject_forbidden_path(
            authority[name],
            context="T-ASCV R0 closure",
        )
        for name in path_fields
    }
    base = validate_r0_authority(
        route_anchor=paths["route_anchor"],
        evaluation_anchor=paths["evaluation_anchor"],
    )
    route = _read_json(paths["route_anchor"])
    if (
        sha256_file(paths["route_checksums"])
        != route["route_checksums_sha256"].upper()
        or _checksum_names(paths["route_checksums"])
        != {
            "capacity.json",
            "predictions.jsonl.gz",
            "route_invariants.json",
            "route_manifest.json",
        }
    ):
        raise ValueError("T-ASCV R0 route closure drift")
    if (
        sha256_file(paths["evaluation_checksums"])
        != R0_EVALUATION_CHECKSUM_ROOT
        or _checksum_names(paths["evaluation_checksums"])
        != {
            "capacity.json",
            "deltas.json",
            "evaluation_invariants.json",
            "evaluation_manifest.json",
            "metrics.json",
            "r0_gate.json",
        }
        or sha256_file(paths["evaluation_manifest"])
        != R0_EVALUATION_MANIFEST_SHA256
    ):
        raise ValueError("T-ASCV R0 evaluation closure drift")
    manifest = _read_json(paths["evaluation_manifest"])
    if (
        manifest.get("schema_version")
        != "sbr-saded-r0-evaluation/v1"
        or manifest.get("decision") != "R0_GO"
        or manifest.get("route_snapshot_verified") is not True
        or manifest.get("route_anchor_sha256", "").upper()
        != R0_ROUTE_ANCHOR_SHA256
        or manifest.get("source", {}).get("commit")
        != R0_SOURCE_COMMIT
        or manifest.get("source", {}).get("clean_tracked") is not True
        or manifest.get("source", {}).get("untracked") is not False
    ):
        raise ValueError("T-ASCV R0 evaluation manifest drift")
    expected = {
        **base,
        "route_checksums": paths["route_checksums"].as_posix(),
        "route_checksums_sha256": sha256_file(
            paths["route_checksums"]
        ),
        "evaluation_manifest": paths["evaluation_manifest"].as_posix(),
        "evaluation_manifest_sha256": sha256_file(
            paths["evaluation_manifest"]
        ),
        "evaluation_checksums": paths[
            "evaluation_checksums"
        ].as_posix(),
        "evaluation_checksums_sha256": sha256_file(
            paths["evaluation_checksums"]
        ),
    }
    if authority != expected:
        raise ValueError("T-ASCV R0 authority manifest binding drift")
    return expected


def repo_source_hashes(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REPO_SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
    return hashes


def frozen_scientific_contract() -> dict:
    return {
        "state_machine": list(FROZEN_STATE_MACHINE),
        "crop": FROZEN_CROP_CONTRACT,
        "tascv": FROZEN_TASCV_CONTRACT,
        "mechanism_gate": FROZEN_MECHANISM_GATE,
        "screen_gate": FROZEN_SCREEN_GATE,
        "formal_thresholds": FROZEN_FORMAL_THRESHOLDS,
        "confirmation": FROZEN_CONFIRMATION_CONTRACT,
    }


def _reject_forbidden_manifest_values(value: object) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _reject_forbidden_manifest_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_forbidden_manifest_values(nested)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if "test-dev" in normalized or "test_dev" in normalized:
            raise ValueError("test-dev is forbidden in T-ASCV manifest")


def _rehash_binding(binding: dict, *, context: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
    }:
        raise ValueError(f"invalid {context} binding")
    path = reject_forbidden_path(binding["path"], context=context)
    if (
        not path.is_file()
        or not isinstance(binding["sha256"], str)
        or len(binding["sha256"]) != 64
        or sha256_file(path) != binding["sha256"].upper()
    ):
        raise ValueError(f"{context} checksum drift")
    return path


def validate_bound_control_candidate(
    candidate: dict,
    *,
    slot: str,
    requirement: dict,
) -> None:
    if (
        not isinstance(requirement, dict)
        or requirement.get("slot_id") != slot
        or candidate.get("provenance") != requirement.get("provenance")
    ):
        raise ValueError(f"bound control provenance drift for {slot}")
    provenance = requirement["provenance"]
    checkpoint = candidate.get("checkpoint", {})
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "kind",
        "path",
        "sha256",
    }:
        raise ValueError(f"invalid bound checkpoint schema for {slot}")
    checkpoint_path = reject_forbidden_path(
        checkpoint["path"],
        context=f"bound control checkpoint {slot}",
    )
    if (
        checkpoint.get("kind") != "last.pt"
        or checkpoint_path.name != "last.pt"
        or not checkpoint_path.is_file()
        or sha256_file(checkpoint_path)
        != str(checkpoint.get("sha256", "")).upper()
    ):
        raise ValueError(f"invalid bound last.pt checkpoint for {slot}")
    for artifact_name in (
        "training_summary",
        "raw_predictions",
        "evaluator",
    ):
        _rehash_binding(
            candidate.get(artifact_name, {}),
            context=f"bound control {artifact_name} {slot}",
        )
    summary_path = _rehash_binding(
        candidate["training_summary"],
        context=f"bound control training summary {slot}",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _prefix, stage, seed_text = slot.split(":")
    seed = int(seed_text)
    stage_contract = provenance["stage_contract"]
    exact = {
        "schema_version": "tascv-training-summary/v1",
        "stage": stage,
        "arm": "control",
        "seed": seed,
        "protocol_source_commit": provenance[
            "runtime_source_commit"
        ],
        "source_repo_bundle_sha256": provenance[
            "repo_bundle_sha256"
        ],
        "source_upstream_bundle_sha256": provenance[
            "upstream_bundle_sha256"
        ],
        "approved_tascv_parent": provenance[
            "approved_tascv_parent"
        ],
        "r0_evaluation_anchor_sha256": provenance[
            "r0_evaluation_anchor_sha256"
        ],
        "initial_state_sha256": provenance[
            "initial_state_sha256"
        ],
        "initial_state_common_fingerprint": provenance[
            "common_fingerprint"
        ],
        "data_sha256": provenance["data_yaml_sha256"],
        "subset_binding": provenance["subset_binding"],
        "batch": 8,
        "observed_tensor_batch_sizes": stage_contract[
            "allowed_observed_tensor_batch_sizes"
        ],
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": stage_contract[
            "expected_successful_batches"
        ],
        "optimizer_attempts": stage_contract[
            "expected_optimizer_attempts"
        ],
        "expected_successful_batches": stage_contract[
            "expected_successful_batches"
        ],
        "expected_optimizer_attempts": stage_contract[
            "expected_optimizer_attempts"
        ],
        "workers": 8,
        "loader": {
            "trainer_batch_size": 8,
            "per_rank_batch_size": 8,
            "loader_batch_size": 8,
            "loader_num_workers": 8,
        },
        "optimizer": FROZEN_OPTIMIZER_OBSERVATION,
        "local_forward_calls": 0,
        "local_forward_call_histogram": {"1": 0, "2": 0},
        "local_bn_preserved_batches": 0,
        "internal_validation_bypass_count": 1,
        "test_loader_is_none": True,
        "auxiliary_non_tiny_pair_count": 0,
        "checkpoint": checkpoint,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"bound control training summary drift for {slot}: {key}"
            )
    canaries = summary.get("batch_canaries")
    expected_positions = [
        tuple(value)
        for value in provenance["batch_canary_contract"][
            "required_epoch_global_batch_positions"
        ]
    ]
    if (
        not isinstance(canaries, list)
        or [
            (record.get("epoch"), record.get("batch"))
            for record in canaries
        ]
        != expected_positions
        or any(
            not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            for record in canaries
        )
    ):
        raise ValueError(
            f"bound control batch canary drift for {slot}"
        )
    manifest_path = reject_forbidden_path(
        summary.get("protocol_manifest", ""),
        context=f"bound control source manifest {slot}",
    )
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path)
        != str(summary.get("protocol_manifest_sha256", "")).upper()
    ):
        raise ValueError(
            f"bound control source manifest drift for {slot}"
        )


def _validate_train_only_yaml(
    path: Path,
    *,
    dataset_root: Path,
    subset_path: Path,
    uses_subset: bool,
) -> dict:
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "train", "val", "names"}
        or "test" in record
        or Path(record["path"]).resolve() != dataset_root
        or record["train"] != record["val"]
    ):
        raise ValueError("T-ASCV train-only YAML contract drift")
    expected_train = (
        subset_path
        if uses_subset
        else dataset_root / "images/train"
    )
    if Path(record["train"]).resolve() != expected_train:
        raise ValueError("T-ASCV train-only YAML source drift")
    return record


def validate_runtime_manifest(
    manifest_path: str | Path,
    *,
    repo_root: Path | None = None,
) -> tuple[dict, str]:
    """Rebuild the complete execution authority from sealed live inputs."""

    path = reject_forbidden_path(
        manifest_path,
        context="T-ASCV runtime manifest",
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("T-ASCV manifest must be an object")
    _reject_forbidden_manifest_values(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"forbidden_data", "scientific_contract"}
        }
    )
    source = manifest.get("runtime_source", {})
    commit = source.get("commit")
    if (
        manifest.get("schema_version") != PROTOCOL_VERSION
        or not isinstance(commit, str)
        or manifest.get("protocol_id") != f"final-tascv-{commit[:8]}"
        or path.name != "protocol_manifest.json"
        or path.parent.name != manifest.get("protocol_id")
    ):
        raise ValueError("T-ASCV manifest identity drift")
    if manifest.get("environment") != EXPECTED_ENVIRONMENT:
        raise ValueError("T-ASCV manifest environment drift")
    if current_environment() != EXPECTED_ENVIRONMENT:
        raise ValueError("T-ASCV live environment drift")
    if manifest.get("approved_tascv_parent") != APPROVED_TASCV_PARENT:
        raise ValueError("T-ASCV approved parent drift")
    if manifest.get("stage_contract") != FROZEN_STAGE_CONTRACT:
        raise ValueError("T-ASCV stage contract drift")
    if manifest.get("training_contract") != FROZEN_TRAINING_CONTRACT:
        raise ValueError("T-ASCV training contract drift")
    if manifest.get("scientific_contract") != frozen_scientific_contract():
        raise ValueError("T-ASCV scientific contract drift")
    if manifest.get("forbidden_data") != ["test-dev", "test_dev"]:
        raise ValueError("T-ASCV forbidden-data contract drift")
    if manifest.get("reuse_policy") != {
        "old_ascv_preflight_authorizes": False,
        "old_ascv_m500_endpoint_reusable": False,
        "invalid_or_partial_reusable": False,
        "stock_control_provenance_only": True,
    }:
        raise ValueError("T-ASCV reuse policy drift")
    expected_dataset = {
        "authority": "sealed-parent-attestation-only-before-val",
        "classes": 10,
        "file_count": EXPECTED_DATASET_FILE_COUNT,
        "full_yaml": (
            "/mnt/uav/protocols/tsgr-p2-e1/"
            "source-VisDrone-full.yaml"
        ),
        "full_yaml_sha256": (
            "7EB91FCEF62A687A26A8EF76E9075B9793B52BC8BB110E4235FACF3E2B958324"
        ),
        "root": EXPECTED_DATASET_ROOT,
        "sha256": EXPECTED_DATASET_SHA256,
        "train_images": 6471,
        "val_images": 548,
    }
    if (
        manifest.get("dataset") != expected_dataset
        or manifest.get("category_mapping_sha256")
        != EXPECTED_CATEGORY_MAPPING_SHA256
    ):
        raise ValueError("T-ASCV dataset authority drift")
    full_source_yaml = reject_forbidden_path(
        expected_dataset["full_yaml"],
        context="T-ASCV source dataset YAML",
    )
    if (
        not full_source_yaml.is_file()
        or sha256_file(full_source_yaml)
        != expected_dataset["full_yaml_sha256"]
    ):
        raise ValueError("T-ASCV source dataset YAML drift")
    expected_subset = {
        "count": EXPECTED_SUBSET_COUNT,
        "file_sha256": EXPECTED_SUBSET_FILE_SHA256,
        "path": EXPECTED_SUBSET_PATH,
        "selection": "reused_sealed_D2",
        "semantic_sha256": EXPECTED_SUBSET_SHA256,
    }
    if manifest.get("subset") != expected_subset:
        raise ValueError("T-ASCV subset manifest drift")
    dataset_root = Path(EXPECTED_DATASET_ROOT).resolve()
    subset_path = Path(EXPECTED_SUBSET_PATH).resolve()
    if (
        not dataset_root.is_dir()
        or sha256_file(subset_path) != EXPECTED_SUBSET_FILE_SHA256
        or subset_signature(subset_path, root=dataset_root)
        != {
            "count": EXPECTED_SUBSET_COUNT,
            "sha256": EXPECTED_SUBSET_SHA256,
        }
    ):
        raise ValueError("T-ASCV live subset authority drift")
    expected_initial_states = {
        str(seed): {
            "common_fingerprint": EXPECTED_COMMON_FINGERPRINTS[seed],
            "path": EXPECTED_INITIAL_STATE_PATHS[seed],
            "sha256": EXPECTED_INITIAL_STATE_SHA256[seed],
        }
        for seed in range(3)
    }
    if manifest.get("initial_states") != expected_initial_states:
        raise ValueError("T-ASCV initial-state manifest drift")
    expected_parent_lineage = {
        "0": {
            "parent_protocol": (
                "/mnt/uav/protocols/ebc-qp-d2-musgd-seed0/"
                "protocol-seed0.json"
            ),
            "parent_protocol_sha256": EXPECTED_PARENT_ATTESTATION_SHA256[0],
        },
        "1": {
            "parent_protocol": (
                "/mnt/uav/protocols/tsgr-p2-e1/protocol-seed1.json"
            ),
            "parent_protocol_sha256": EXPECTED_PARENT_ATTESTATION_SHA256[1],
        },
        "2": {
            "parent_protocol": (
                "/mnt/uav/protocols/tsgr-p2-e1/protocol-seed2.json"
            ),
            "parent_protocol_sha256": EXPECTED_PARENT_ATTESTATION_SHA256[2],
        },
    }
    if manifest.get("parent_lineage") != expected_parent_lineage:
        raise ValueError("T-ASCV parent-lineage manifest drift")
    for seed in range(3):
        validate_parent_attestation(manifest, seed)
        initial_path = Path(EXPECTED_INITIAL_STATE_PATHS[seed]).resolve()
        if sha256_file(initial_path) != EXPECTED_INITIAL_STATE_SHA256[seed]:
            raise ValueError("T-ASCV initial-state checksum drift")
        artifact = torch.load(
            initial_path,
            map_location="cpu",
            weights_only=False,
        )
        validate_initial_state_artifact(artifact, seed=seed)
    repo = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    require_clean_repo(repo)
    live_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    live_repo_files = repo_source_hashes(repo)
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            APPROVED_TASCV_PARENT["commit"],
            "HEAD",
        ],
        cwd=repo,
        check=False,
    )
    if (
        ancestor.returncode != 0
        or live_commit != commit
        or source.get("repo_files") != live_repo_files
        or source.get("repo_bundle_sha256")
        != source_bundle_sha256(live_repo_files)
        or source.get("upstream") != EXPECTED_UPSTREAM_SOURCE_SHA256
        or source.get("upstream_bundle_sha256")
        != source_bundle_sha256(EXPECTED_UPSTREAM_SOURCE_SHA256)
        or current_upstream_source_hashes()
        != EXPECTED_UPSTREAM_SOURCE_SHA256
    ):
        raise ValueError("T-ASCV live source closure drift")
    for relative, expected_sha in APPROVED_TASCV_PARENT["files"].items():
        if sha256_file(repo / relative) != expected_sha:
            raise ValueError("T-ASCV approved-parent file drift")
    if (
        source_bundle_sha256(APPROVED_TASCV_PARENT["files"])
        != APPROVED_TASCV_PARENT["bundle_sha256"]
    ):
        raise ValueError("T-ASCV approved-parent bundle drift")
    validate_r0_closure(manifest.get("r0_authority", {}))
    allowlist_record = manifest.get("control_allowlist", {})
    allowlist_path = _rehash_binding(
        {
            "path": allowlist_record.get("path"),
            "sha256": allowlist_record.get("sha256"),
        },
        context="T-ASCV control allowlist",
    )
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    if (
        allowlist.get("schema_version")
        != "saded-control-allowlist/v1"
        or set(allowlist.get("slots", {})) != set(CONTROL_SLOTS)
        or allowlist.get("slots") != allowlist_record.get("slots")
        or allowlist_path.parent != path.parent
        or allowlist_path.name != "control_allowlist.json"
    ):
        raise ValueError("T-ASCV control allowlist content drift")
    requirements_path = _rehash_binding(
        allowlist.get("requirements", {}),
        context="T-ASCV control requirements",
    )
    requirements = json.loads(
        requirements_path.read_text(encoding="utf-8")
    )
    if (
        requirements.get("schema_version")
        != "saded-control-requirements/v1"
        or set(requirements.get("slots", {})) != set(CONTROL_SLOTS)
        or requirements_path.parent != path.parent
        or requirements_path.name != "control_requirements.json"
    ):
        raise ValueError("T-ASCV control requirements drift")
    for slot, record in allowlist["slots"].items():
        if record.get("resolution") == "BOUND":
            validate_bound_control_candidate(
                record.get("candidate", {}),
                slot=slot,
                requirement=requirements["slots"][slot],
            )
        elif record.get("resolution") == "RUN_FRESH":
            target = record.get("fresh_target", {})
            if (
                set(target)
                != {
                    "project",
                    "name",
                    "target_dir",
                    "summary",
                    "checkpoint",
                }
                or target.get("name") != "control"
                or Path(target["target_dir"]).resolve()
                != Path(target["project"]).resolve() / "control"
                or Path(target["summary"]).resolve()
                != Path(target["target_dir"]).resolve()
                / "tascv_training_summary.json"
                or Path(target["checkpoint"]).resolve()
                != Path(target["target_dir"]).resolve()
                / "weights/last.pt"
            ):
                raise ValueError(f"T-ASCV fresh control endpoint drift: {slot}")
        else:
            raise ValueError(f"T-ASCV control resolution drift: {slot}")
    expected_treatment_keys = {
        f"T:{stage}:{seed}"
        for stage, contract in FROZEN_STAGE_CONTRACT.items()
        for seed in contract["seeds"]
    }
    treatment_endpoints = manifest.get("treatment_endpoints", {})
    if set(treatment_endpoints) != expected_treatment_keys:
        raise ValueError("T-ASCV treatment endpoint set drift")
    for slot, endpoint in treatment_endpoints.items():
        if (
            set(endpoint) != {"project", "name", "target_dir"}
            or endpoint.get("name") != "tascv"
            or Path(endpoint["target_dir"]).resolve()
            != Path(endpoint["project"]).resolve() / "tascv"
        ):
            raise ValueError(f"T-ASCV treatment endpoint drift: {slot}")
    subset_yaml_path = _rehash_binding(
        manifest.get("train_only_yaml", {}),
        context="T-ASCV subset train-only YAML",
    )
    full_yaml_path = _rehash_binding(
        manifest.get("full_train_only_yaml", {}),
        context="T-ASCV full train-only YAML",
    )
    if (
        subset_yaml_path != path.parent / "tascv_subset_train_only.yaml"
        or full_yaml_path != path.parent / "tascv_full_train_only.yaml"
    ):
        raise ValueError("T-ASCV train-only YAML endpoint drift")
    subset_yaml = _validate_train_only_yaml(
        subset_yaml_path,
        dataset_root=dataset_root,
        subset_path=subset_path,
        uses_subset=True,
    )
    full_yaml = _validate_train_only_yaml(
        full_yaml_path,
        dataset_root=dataset_root,
        subset_path=subset_path,
        uses_subset=False,
    )
    if subset_yaml["names"] != full_yaml["names"]:
        raise ValueError("T-ASCV class mapping YAML drift")
    return manifest, sha256_file(path)


_FORBIDDEN_CONTROL_KEYS = {
    "ap",
    "map",
    "metric",
    "metrics",
    "result",
    "results",
    "delta",
    "deltas",
    "gate",
    "decision",
    "fitness",
    "val_annotation",
    "val_annotations",
}
_FORBIDDEN_CONTROL_PATH_TOKENS = (
    "test-dev",
    "test_dev",
    "/metrics/",
    "/results/",
    "/deltas/",
    "/gates/",
    "val_annotation",
)


def _reject_control_performance_data(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if (
                normalized in _FORBIDDEN_CONTROL_KEYS
                or "metric" in normalized
            ):
                raise ValueError(
                    f"forbidden control-candidate field: {key}"
                )
            _reject_control_performance_data(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_control_performance_data(nested)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if any(
            token in normalized
            for token in _FORBIDDEN_CONTROL_PATH_TOKENS
        ):
            raise ValueError(
                f"forbidden control-candidate path/value: {value}"
            )


def resolve_control_allowlist(
    requirements: dict,
    candidates: list[dict],
) -> dict:
    if (
        requirements.get("schema_version")
        != "saded-control-requirements/v1"
        or set(requirements.get("slots", {})) != set(CONTROL_SLOTS)
    ):
        raise ValueError("invalid control requirements")
    matches: dict[str, list[dict]] = {
        slot: [] for slot in CONTROL_SLOTS
    }
    for candidate in candidates:
        _reject_control_performance_data(candidate)
        if (
            candidate.get("schema_version")
            != "saded-stock-control-candidate/v1"
            or set(candidate)
            != {
                "schema_version",
                "slot_id",
                "provenance",
                "training_summary",
                "checkpoint",
                "raw_predictions",
                "evaluator",
            }
        ):
            raise ValueError("invalid control candidate schema")
        slot = candidate.get("slot_id")
        if slot not in matches:
            raise ValueError("control candidate targets an unknown slot")
        if slot == "B:PREFLIGHT_1:0":
            raise ValueError(
                "preflight control is forced RUN_FRESH"
            )
        requirement = requirements["slots"][slot]
        provenance = candidate.get("provenance")
        if (
            not isinstance(provenance, dict)
            or set(provenance) != set(requirement["provenance"])
        ):
            raise ValueError("invalid control candidate provenance schema")
        checkpoint = candidate.get("checkpoint")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "kind",
            "path",
            "sha256",
        }:
            raise ValueError("invalid control candidate checkpoint schema")
        for artifact_name in (
            "training_summary",
            "raw_predictions",
            "evaluator",
        ):
            artifact = candidate.get(artifact_name)
            if not isinstance(artifact, dict) or set(artifact) != {
                "path",
                "sha256",
            }:
                raise ValueError(
                    f"invalid control candidate {artifact_name} schema"
                )
        if candidate.get("provenance") == requirement["provenance"]:
            matches[slot].append(candidate)
    resolutions: dict[str, dict] = {}
    for slot in CONTROL_SLOTS:
        found = matches[slot]
        if len(found) > 1:
            raise ValueError(
                f"multiple provenance-only control matches for {slot}"
            )
        if found:
            candidate = found[0]
            checkpoint = candidate.get("checkpoint", {})
            checkpoint_path = reject_forbidden_path(
                checkpoint.get("path", ""),
                context=f"bound control checkpoint {slot}",
            )
            if (
                checkpoint.get("kind") != "last.pt"
                or not isinstance(checkpoint.get("path"), str)
                or not isinstance(checkpoint.get("sha256"), str)
                or len(checkpoint["sha256"]) != 64
                or checkpoint_path.name != "last.pt"
                or not checkpoint_path.is_file()
                or sha256_file(checkpoint_path)
                != checkpoint["sha256"].upper()
            ):
                raise ValueError(
                    f"invalid bound last.pt checkpoint for {slot}"
                )
            for artifact_name in (
                "training_summary",
                "raw_predictions",
                "evaluator",
            ):
                artifact = candidate[artifact_name]
                artifact_path = reject_forbidden_path(
                    artifact["path"],
                    context=f"bound control {artifact_name} {slot}",
                )
                if (
                    not isinstance(artifact["sha256"], str)
                    or len(artifact["sha256"]) != 64
                    or not artifact_path.is_file()
                    or sha256_file(artifact_path)
                    != artifact["sha256"].upper()
                ):
                    raise ValueError(
                        f"invalid bound artifact {artifact_name} for {slot}"
                    )
            validate_bound_control_candidate(
                candidate,
                slot=slot,
                requirement=requirements["slots"][slot],
            )
            resolutions[slot] = {
                "resolution": "BOUND",
                "candidate": candidate,
            }
        else:
            resolutions[slot] = {
                "resolution": "RUN_FRESH",
                "fresh_target": requirements["slots"][slot][
                    "fresh_target"
                ],
            }
    return {
        "schema_version": "saded-control-allowlist/v1",
        "slots": resolutions,
    }


__all__ = [
    name
    for name in globals()
    if name.startswith("EXPECTED_")
    or name.startswith("FROZEN_")
    or name.startswith("R0_")
] + [
    "PROTOCOL_VERSION",
    "APPROVED_TASCV_PARENT",
    "CONTROL_SLOTS",
    "REPO_SOURCE_FILES",
    "current_environment",
    "current_upstream_source_hashes",
    "frozen_scientific_contract",
    "resolve_control_allowlist",
    "repo_source_hashes",
    "reject_forbidden_path",
    "require_clean_repo",
    "sha256_file",
    "source_bundle_sha256",
    "state_fingerprint",
    "subset_signature",
    "training_batch_sha256",
    "validate_initial_state_artifact",
    "validate_parent_attestation",
    "validate_bound_control_candidate",
    "validate_r0_authority",
    "validate_r0_closure",
    "validate_runtime_manifest",
]

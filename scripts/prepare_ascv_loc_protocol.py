from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path

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
    state_fingerprint,
    subset_signature,
    validate_initial_state_artifact,
)

PROTOCOL_VERSION = "ascv-loc-matched/v2"
REQUIRED_PARENT_SEEDS = frozenset({0, 1, 2})


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _reject_forbidden(path: Path) -> None:
    normalized = path.resolve().as_posix().lower()
    if "test-dev" in normalized or "test_dev" in normalized:
        raise ValueError(f"test-dev path is forbidden during ASCV-Loc development: {path}")


def _source_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _load_parent_protocol(path: Path) -> tuple[int, dict]:
    path = path.resolve()
    _reject_forbidden(path)
    parent_sha = sha256_file(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    seed = int(record["seed"])
    if parent_sha != EXPECTED_PARENT_ATTESTATION_SHA256.get(seed):
        raise ValueError(f"parent attestation checksum mismatch in {path}")
    if record.get("dataset") != {
        "file_count": EXPECTED_DATASET_FILE_COUNT,
        "sha256": EXPECTED_DATASET_SHA256,
    }:
        raise ValueError(f"dataset signature mismatch in {path}")
    if record.get("category_mapping_sha256") != EXPECTED_CATEGORY_MAPPING_SHA256:
        raise ValueError(f"category mapping mismatch in {path}")
    subset = record.get("subset", {})
    if int(subset.get("count", -1)) != EXPECTED_SUBSET_COUNT:
        raise ValueError(f"subset count mismatch in {path}")
    if subset.get("sha256") != EXPECTED_SUBSET_SHA256:
        raise ValueError(f"subset signature mismatch in {path}")
    subset_path = Path(subset["path"]).resolve()
    if sha256_file(subset_path) != EXPECTED_SUBSET_FILE_SHA256:
        raise ValueError(f"subset file checksum mismatch in {path}")
    parent_data_path = Path(record["data"]["path"]).resolve()
    parent_data = yaml.safe_load(parent_data_path.read_text(encoding="utf-8"))
    dataset_root = Path(parent_data["path"]).resolve()
    semantic = subset_signature(subset_path, root=dataset_root)
    if semantic != {"count": EXPECTED_SUBSET_COUNT, "sha256": EXPECTED_SUBSET_SHA256}:
        raise ValueError(f"subset semantic signature mismatch in {path}")
    if record.get("environment") != EXPECTED_ENVIRONMENT:
        raise ValueError(f"environment mismatch in {path}")
    if record.get("source_sha256") != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError(f"upstream source signature mismatch in {path}")
    initial = record.get("initial_state", {})
    initial_path = Path(initial["path"]).resolve()
    initial_sha = sha256_file(initial_path)
    if initial_sha != initial["sha256"] or initial_sha != EXPECTED_INITIAL_STATE_SHA256[seed]:
        raise ValueError(f"initial-state checksum mismatch in {path}")
    artifact = torch.load(initial_path, map_location="cpu", weights_only=False)
    validate_initial_state_artifact(artifact, seed=seed)
    if sha256_file(parent_data_path) != record["data"]["sha256"]:
        raise ValueError(f"parent data YAML checksum mismatch in {path}")
    return seed, record


def prepare_protocol(
    *,
    parent_protocols: list[Path],
    full_dataset_yaml: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict:
    if not parent_protocols:
        raise ValueError("at least one matched parent protocol is required")
    full_dataset_yaml = full_dataset_yaml.resolve()
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()
    for path in (full_dataset_yaml, output_dir, repo_root):
        _reject_forbidden(path)
    require_clean_repo(repo_root)
    if current_environment() != EXPECTED_ENVIRONMENT:
        raise ValueError("runtime environment does not match the authoritative baseline")
    upstream_hashes = current_upstream_source_hashes()
    if upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("Ultralytics source files do not match the authoritative baseline")

    parents: dict[int, tuple[Path, dict]] = {}
    for path in parent_protocols:
        seed, record = _load_parent_protocol(path)
        if seed in parents:
            raise ValueError(f"duplicate parent protocol for seed {seed}")
        parents[seed] = (path.resolve(), record)
    if set(parents) != set(REQUIRED_PARENT_SEEDS):
        raise ValueError("authoritative ASCV protocol requires parent attestations for seeds 0, 1, and 2")

    seed0 = parents[0][1]
    subset_path = Path(seed0["subset"]["path"]).resolve()
    parent_data = yaml.safe_load(Path(seed0["data"]["path"]).read_text(encoding="utf-8"))
    full_data = yaml.safe_load(full_dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(parent_data.get("names"), (dict, list)):
        raise ValueError("parent data YAML has no class mapping")
    if parent_data["names"] != full_data.get("names"):
        raise ValueError("parent/full class mappings differ")
    source_hashes = repo_source_hashes(repo_root)

    subset_yaml = output_dir / "matched_subset_train_only.yaml"
    subset_config = {
        "path": parent_data.get("path"),
        "train": subset_path.as_posix(),
        "val": subset_path.as_posix(),
        "names": parent_data["names"],
    }
    _atomic_write(subset_yaml, yaml.safe_dump(subset_config, sort_keys=False, allow_unicode=True))

    full_yaml = output_dir / "matched_full_train_only.yaml"
    full_train = full_data["train"]
    full_config = {
        "path": full_data.get("path"),
        "train": full_train,
        "val": full_train,
        "names": full_data["names"],
    }
    _atomic_write(full_yaml, yaml.safe_dump(full_config, sort_keys=False, allow_unicode=True))

    initial_states = {}
    lineage = {}
    for seed, (path, record) in sorted(parents.items()):
        initial_states[str(seed)] = {
            "path": str(Path(record["initial_state"]["path"]).resolve()),
            "sha256": record["initial_state"]["sha256"],
        }
        lineage[str(seed)] = {
            "parent_protocol": path.as_posix(),
            "parent_protocol_sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "source_commit": _source_commit(repo_root),
        "environment": EXPECTED_ENVIRONMENT,
        "dataset": {
            "sha256": EXPECTED_DATASET_SHA256,
            "file_count": EXPECTED_DATASET_FILE_COUNT,
            "train_images": 6471,
            "val_images": 548,
            "classes": 10,
            "root": str(Path(parent_data["path"]).resolve()),
            "authority": "sealed-parent-attestation-only-before-val",
            "full_yaml": full_dataset_yaml.as_posix(),
            "full_yaml_sha256": sha256_file(full_dataset_yaml),
        },
        "category_mapping_sha256": EXPECTED_CATEGORY_MAPPING_SHA256,
        "subset": {
            "count": EXPECTED_SUBSET_COUNT,
            "path": subset_path.as_posix(),
            "semantic_sha256": EXPECTED_SUBSET_SHA256,
            "file_sha256": EXPECTED_SUBSET_FILE_SHA256,
            "selection": "reused_sealed_D2",
        },
        "initial_states": initial_states,
        "parent_lineage": lineage,
        "source": {
            "repo_files": source_hashes,
            "repo_bundle_sha256": source_bundle_sha256(source_hashes),
            "upstream": upstream_hashes,
        },
        "train_only_yaml": {
            "path": subset_yaml.as_posix(),
            "sha256": sha256_file(subset_yaml),
        },
        "full_train_only_yaml": {
            "path": full_yaml.as_posix(),
            "sha256": sha256_file(full_yaml),
        },
        "training_contract": {
            "pretrained": False,
            "imgsz": 640,
            "batch": 8,
            "workers": 8,
            "amp": True,
            "amp_scale": 128.0,
            "optimizer": "MuSGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "nbs": 64,
            "query_count": 300,
            "max_det": 300,
            "nms": False,
        },
        "scientific_contract": {
            "state_machine": list(FROZEN_STATE_MACHINE),
            "crop": FROZEN_CROP_CONTRACT,
            "mechanism_gate": FROZEN_MECHANISM_GATE,
            "screen_gate": FROZEN_SCREEN_GATE,
            "formal_thresholds": FROZEN_FORMAL_THRESHOLDS,
        },
        "forbidden_data": ["test-dev", "test_dev"],
    }
    manifest_path = output_dir / "protocol_manifest.json"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind ASCV-Loc to the exact matched-baseline protocol.")
    parser.add_argument("--parent-protocol", type=Path, action="append", required=True)
    parser.add_argument("--full-dataset-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_protocol(
        parent_protocols=args.parent_protocol,
        full_dataset_yaml=args.full_dataset_yaml,
        output_dir=args.output,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

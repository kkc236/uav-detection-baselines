from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

import scripts.prepare_ascv_loc_protocol as protocol_module
import src.ascv_loc_protocol as authority_module
from scripts.prepare_ascv_loc_protocol import (
    prepare_protocol,
    require_clean_repo,
    sha256_file,
    subset_signature,
)


def _parent_protocol(tmp_path: Path, monkeypatch, *, seed: int = 0) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    for directory in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset / directory).mkdir(parents=True, exist_ok=True)
    for index in range(10):
        (dataset / "images/train" / f"{index:06d}.jpg").write_bytes(f"image-{index}".encode())
        (dataset / "labels/train" / f"{index:06d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (dataset / "images/val" / "000000.jpg").write_bytes(b"val-image")
    (dataset / "labels/val" / "000000.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    subset = tmp_path / f"subset-{seed}.txt"
    subset.write_text(
        "".join(f"{(dataset / 'images/train' / f'{index:06d}.jpg').as_posix()}\n" for index in range(10))
    )
    subset_sha = sha256_file(subset)
    subset_semantic_sha = subset_signature(subset, root=dataset)["sha256"]
    dataset_record = {
        "file_count": protocol_module.EXPECTED_DATASET_FILE_COUNT,
        "sha256": protocol_module.EXPECTED_DATASET_SHA256,
    }
    monkeypatch.setattr(protocol_module, "EXPECTED_SUBSET_SHA256", subset_semantic_sha)
    monkeypatch.setattr(protocol_module, "EXPECTED_SUBSET_FILE_SHA256", subset_sha)
    monkeypatch.setattr(protocol_module, "EXPECTED_SUBSET_COUNT", 10)
    monkeypatch.setattr(protocol_module, "EXPECTED_CATEGORY_MAPPING_SHA256", "CATEGORY")
    monkeypatch.setattr(
        protocol_module,
        "EXPECTED_UPSTREAM_SOURCE_SHA256",
        {"head.py": "HEAD", "rtdetr-l.yaml": "YAML", "tasks.py": "TASKS"},
    )
    monkeypatch.setattr(protocol_module, "REQUIRED_PARENT_SEEDS", frozenset({seed}))
    monkeypatch.setattr(protocol_module, "current_environment", lambda: protocol_module.EXPECTED_ENVIRONMENT)
    monkeypatch.setattr(
        protocol_module,
        "current_upstream_source_hashes",
        lambda: {"head.py": "HEAD", "rtdetr-l.yaml": "YAML", "tasks.py": "TASKS"},
    )
    monkeypatch.setattr(protocol_module, "require_clean_repo", lambda _root: None)

    data = tmp_path / f"parent-{seed}.yaml"
    data.write_text(
        yaml.safe_dump({"path": str(dataset), "train": str(subset), "val": "images/val", "names": {0: "car"}})
    )
    initial = tmp_path / f"initial-{seed}.pt"
    metadata = {
        "seed": seed,
        "dataset": dataset_record,
        "category_mapping_sha256": "CATEGORY",
        "subset": {
            "count": 10,
            "fraction": 0.1,
            "sha256": subset_semantic_sha,
        },
        "source_sha256": {"head.py": "HEAD", "rtdetr-l.yaml": "YAML", "tasks.py": "TASKS"},
        "environment": protocol_module.EXPECTED_ENVIRONMENT,
        "control_parameters": 32_826_626,
        "innovation_seed": seed + 10_000,
    }
    torch.save(
        {
            "format_version": 1,
            "metadata": metadata,
            "common_state": {},
            "innovation_state": {},
            "fingerprints": {"common": protocol_module.state_fingerprint({}), "innovation": protocol_module.state_fingerprint({})},
        },
        initial,
    )
    parent = {
        "seed": seed,
        "dataset": dataset_record,
        "category_mapping_sha256": "CATEGORY",
        "subset": {"count": 10, "path": str(subset), "sha256": subset_semantic_sha},
        "environment": protocol_module.EXPECTED_ENVIRONMENT,
        "initial_state": {"path": str(initial), "sha256": sha256_file(initial)},
        "data": {"path": str(data), "sha256": sha256_file(data)},
        "source_sha256": {"head.py": "HEAD", "rtdetr-l.yaml": "YAML", "tasks.py": "TASKS"},
    }
    parent_path = tmp_path / f"parent-{seed}.json"
    parent_path.write_text(json.dumps(parent))
    monkeypatch.setattr(
        protocol_module,
        "EXPECTED_PARENT_ATTESTATION_SHA256",
        {seed: sha256_file(parent_path)},
    )
    monkeypatch.setattr(
        protocol_module,
        "EXPECTED_INITIAL_STATE_SHA256",
        {seed: sha256_file(initial)},
    )
    monkeypatch.setattr(
        protocol_module,
        "EXPECTED_COMMON_FINGERPRINTS",
        {seed: protocol_module.state_fingerprint({})},
    )
    monkeypatch.setattr(authority_module, "EXPECTED_CATEGORY_MAPPING_SHA256", "CATEGORY")
    monkeypatch.setattr(authority_module, "EXPECTED_DATASET_FILE_COUNT", dataset_record["file_count"])
    monkeypatch.setattr(authority_module, "EXPECTED_DATASET_SHA256", dataset_record["sha256"])
    monkeypatch.setattr(authority_module, "EXPECTED_SUBSET_COUNT", 10)
    monkeypatch.setattr(authority_module, "EXPECTED_SUBSET_SHA256", subset_semantic_sha)
    monkeypatch.setattr(
        authority_module,
        "EXPECTED_UPSTREAM_SOURCE_SHA256",
        {"head.py": "HEAD", "rtdetr-l.yaml": "YAML", "tasks.py": "TASKS"},
    )
    monkeypatch.setattr(authority_module, "EXPECTED_ENVIRONMENT", protocol_module.EXPECTED_ENVIRONMENT)
    monkeypatch.setattr(authority_module, "EXPECTED_COMMON_FINGERPRINTS", {seed: authority_module.state_fingerprint({})})
    return parent_path, data


def test_protocol_reuses_exact_parent_subset_and_initial_state(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _parent_protocol(tmp_path, monkeypatch)
    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(yaml.safe_dump({"path": "/dataset", "train": "images/train", "val": "images/val", "names": {0: "car"}}))

    manifest = prepare_protocol(
        parent_protocols=[parent],
        full_dataset_yaml=full_yaml,
        output_dir=tmp_path / "output",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert manifest["schema_version"] == "ascv-loc-matched/v2"
    assert manifest["subset"]["count"] == 10
    assert manifest["subset"]["semantic_sha256"] == protocol_module.EXPECTED_SUBSET_SHA256
    assert manifest["subset"]["file_sha256"] == protocol_module.EXPECTED_SUBSET_FILE_SHA256
    assert manifest["training_contract"]["batch"] == 8
    assert manifest["training_contract"]["optimizer"] == "MuSGD"
    assert manifest["training_contract"]["amp_scale"] == 128.0
    assert manifest["initial_states"]["0"]["sha256"]
    assert manifest["dataset"]["file_count"] == protocol_module.EXPECTED_DATASET_FILE_COUNT
    assert manifest["category_mapping_sha256"] == "CATEGORY"
    assert manifest["source"]["repo_files"]
    assert manifest["source"]["upstream"] == {"head.py": "HEAD", "rtdetr-l.yaml": "YAML", "tasks.py": "TASKS"}
    assert manifest["scientific_contract"]["mechanism_gate"]["scientific_tail_window"] == [401, 500]
    assert manifest["scientific_contract"]["screen_gate"]["seeds"] == [0, 1, 2]
    assert manifest["scientific_contract"]["formal_thresholds"]["AP-large-SBR"] == -0.005


def test_train_only_yamls_never_resolve_real_val_or_test_dev(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _parent_protocol(tmp_path, monkeypatch)
    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(yaml.safe_dump({"path": "/dataset", "train": "images/train", "val": "images/val", "names": {0: "car"}}))
    output = tmp_path / "output"

    prepare_protocol(
        parent_protocols=[parent],
        full_dataset_yaml=full_yaml,
        output_dir=output,
        repo_root=Path(__file__).resolve().parents[1],
    )
    subset_data = yaml.safe_load((output / "matched_subset_train_only.yaml").read_text())
    full_data = yaml.safe_load((output / "matched_full_train_only.yaml").read_text())

    assert subset_data["train"] == subset_data["val"]
    assert full_data["train"] == full_data["val"]
    assert "test" not in subset_data and "test" not in full_data


def test_changed_parent_subset_fails_closed(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _parent_protocol(tmp_path, monkeypatch)
    record = json.loads(parent.read_text())
    Path(record["subset"]["path"]).write_text("changed\n")
    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(yaml.safe_dump({"path": "/dataset", "train": "images/train", "names": {0: "car"}}))

    with pytest.raises(ValueError, match="subset file checksum mismatch"):
        prepare_protocol(
            parent_protocols=[parent],
            full_dataset_yaml=full_yaml,
            output_dir=tmp_path / "output",
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_changed_subset_semantics_fail_even_if_manifest_file_hash_is_rewritten(
    tmp_path: Path, monkeypatch
) -> None:
    parent, _ = _parent_protocol(tmp_path, monkeypatch)
    record = json.loads(parent.read_text())
    subset_path = Path(record["subset"]["path"])
    lines = subset_path.read_text().splitlines()
    subset_path.write_text("\n".join(reversed(lines)) + "\n")
    monkeypatch.setattr(protocol_module, "EXPECTED_SUBSET_FILE_SHA256", sha256_file(subset_path))

    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path / "dataset"), "train": "images/train", "names": {0: "car"}})
    )
    with pytest.raises(ValueError, match="subset semantic signature mismatch"):
        prepare_protocol(
            parent_protocols=[parent],
            full_dataset_yaml=full_yaml,
            output_dir=tmp_path / "output",
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_parent_attestation_drift_fails_without_rehashing_val_data(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _parent_protocol(tmp_path, monkeypatch)
    record = json.loads(parent.read_text())
    record["dataset"]["sha256"] = "0" * 64
    parent.write_text(json.dumps(record))
    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path / "dataset"), "train": "images/train", "names": {0: "car"}})
    )

    with pytest.raises(ValueError, match="parent attestation checksum mismatch"):
        prepare_protocol(
            parent_protocols=[parent],
            full_dataset_yaml=full_yaml,
            output_dir=tmp_path / "output",
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_protocol_preparation_never_opens_val_or_test_dev_files(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _parent_protocol(tmp_path, monkeypatch)
    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path / "dataset"), "train": "images/train", "names": {0: "car"}})
    )
    opened: list[str] = []
    original_open = Path.open

    def audited_open(path: Path, *args, **kwargs):
        opened.append(path.resolve().as_posix().lower())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    prepare_protocol(
        parent_protocols=[parent],
        full_dataset_yaml=full_yaml,
        output_dir=tmp_path / "output",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert not any("/images/val/" in path or "/labels/val/" in path or "test-dev" in path for path in opened)


def test_initial_state_must_carry_authoritative_scratch_provenance(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _parent_protocol(tmp_path, monkeypatch)
    record = json.loads(parent.read_text())
    initial_path = Path(record["initial_state"]["path"])
    artifact = torch.load(initial_path, map_location="cpu", weights_only=False)
    artifact["metadata"]["control_parameters"] = 1
    torch.save(artifact, initial_path)
    record["initial_state"]["sha256"] = sha256_file(initial_path)
    parent.write_text(json.dumps(record))
    monkeypatch.setattr(protocol_module, "EXPECTED_INITIAL_STATE_SHA256", {0: sha256_file(initial_path)})
    monkeypatch.setattr(
        protocol_module,
        "EXPECTED_PARENT_ATTESTATION_SHA256",
        {0: sha256_file(parent)},
    )
    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path / "dataset"), "train": "images/train", "names": {0: "car"}})
    )

    with pytest.raises(ValueError, match="scratch provenance"):
        prepare_protocol(
            parent_protocols=[parent],
            full_dataset_yaml=full_yaml,
            output_dir=tmp_path / "output",
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_clean_repo_gate_rejects_tracked_and_untracked_source_drift(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    source = repo / "source.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=ASCV Test", "-c", "user.email=ascv@example.invalid", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    require_clean_repo(repo)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="source tree is not clean"):
        require_clean_repo(repo)
    source.write_text("VALUE = 1\n")
    (repo / "shadow.py").write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="source tree is not clean"):
        require_clean_repo(repo)

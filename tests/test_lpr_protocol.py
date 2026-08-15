from __future__ import annotations

from pathlib import Path

import torch

from src.lpr_protocol import (
    CATEGORY_NAMES,
    EXPECTED_COMMON_FINGERPRINTS,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ENVIRONMENT,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SUBSET_SHA256,
    build_initial_state,
    category_mapping_sha256,
    dataset_signature,
    environment_violations,
    file_sha256,
    load_initial_state,
    select_hashed_subset,
    source_violations,
    state_fingerprint,
    subset_signature,
    validate_initial_state_authority,
)


def test_frozen_authority_constants_match_strict_contract() -> None:
    assert EXPECTED_DATASET_SHA256 == "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
    assert EXPECTED_SUBSET_SHA256 == "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
    assert EXPECTED_ENVIRONMENT == {
        "gpu": "NVIDIA GeForce RTX 4090",
        "driver": "550.142",
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda": "12.1",
        "ultralytics": "8.4.90",
    }
    assert len(CATEGORY_NAMES) == 10
    assert EXPECTED_COMMON_FINGERPRINTS == {
        0: "0B968046FDC89BE5A31581C81F7335A9742BC422503428113637B1CC829F0FA0",
        1: "A73D3A57F5DCF3F62FA4B30329C32204E3A74BC57AA4FFEE577873D14F0A3D65",
        2: "1CCA2D745106F949268B3978722A415439623376D01C1D188B1450C6230AF1B2",
    }
    assert EXPECTED_SOURCE_SHA256 == {
        "head.py": "5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F",
        "tasks.py": "B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92",
        "rtdetr-l.yaml": "85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83",
    }


def test_installed_ultralytics_sources_match_frozen_authority() -> None:
    assert source_violations() == {}


def test_hash_selected_subset_is_order_independent_and_signed(tmp_path) -> None:
    root = tmp_path / "VisDrone"
    paths = [root / "images" / "train" / f"image-{index:03}.jpg" for index in range(20)]
    selected = select_hashed_subset(reversed(paths), root=root, fraction=0.10)

    assert len(selected) == 2
    assert selected == select_hashed_subset(paths, root=root, fraction=0.10)
    assert len(subset_signature(selected, root=root)) == 64


def test_dataset_signature_hashes_relative_names_and_contents(tmp_path) -> None:
    root = tmp_path / "VisDrone"
    for relative, content in {
        "images/train/a.jpg": b"train-image",
        "labels/train/a.txt": b"0 0.5 0.5 0.1 0.1\n",
        "images/val/b.jpg": b"val-image",
        "labels/val/b.txt": b"1 0.4 0.4 0.2 0.2\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    first = dataset_signature(root)
    (root / "labels" / "val" / "b.txt").write_bytes(b"changed")
    second = dataset_signature(root)

    assert first["file_count"] == 4
    assert len(first["sha256"]) == 64
    assert first != second


def test_category_mapping_hash_is_stable_for_mapping_or_sequence() -> None:
    mapping = {index: name for index, name in enumerate(CATEGORY_NAMES)}

    assert category_mapping_sha256(mapping) == category_mapping_sha256(list(CATEGORY_NAMES))
    assert category_mapping_sha256(mapping) == "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"


def test_environment_gate_reports_each_protocol_drift() -> None:
    actual = dict(EXPECTED_ENVIRONMENT)
    actual["driver"] = "595.84"
    actual["python"] = "3.12.3"

    violations = environment_violations(actual)

    assert violations == {
        "driver": {"expected": "550.142", "actual": "595.84"},
        "python": {"expected": "3.10.12", "actual": "3.12.3"},
    }


def test_initial_state_preserves_common_state_and_detects_corruption() -> None:
    common = {"model.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    lpr = {
        **common,
        "model.decoder.lpr_refiners.0.alpha": torch.tensor(0.0),
    }
    artifact = build_initial_state(common, lpr, metadata={"seed": 0})

    assert artifact["fingerprints"]["common"] == state_fingerprint(common)
    assert set(artifact["lpr_state"]) == {"model.decoder.lpr_refiners.0.alpha"}

    class _StateTarget:
        def __init__(self, state):
            self.state = state
            self.loaded = None

        def state_dict(self):
            return self.state

        def load_state_dict(self, state, strict=True):
            assert strict
            self.loaded = state

    control = _StateTarget(dict(common))
    method = _StateTarget(dict(lpr))
    load_initial_state(control, artifact, variant="control")
    load_initial_state(method, artifact, variant="lpr")
    assert set(control.loaded) == set(common)
    assert set(method.loaded) == set(lpr)

    artifact["common_state"]["model.weight"][0, 0] += 1
    try:
        load_initial_state(control, artifact, variant="control")
    except ValueError as error:
        assert "fingerprint" in str(error)
    else:
        raise AssertionError("corrupted initial state was accepted")


def test_initial_state_authority_requires_linux_common_fingerprint_and_file_hash(tmp_path, monkeypatch) -> None:
    common = {"model.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    method = {**common, "model.decoder.lpr_refiners.0.alpha": torch.tensor(0.0)}
    artifact = build_initial_state(common, method, metadata={"seed": 0})
    path = tmp_path / "initial-state-seed0.pt"
    torch.save(artifact, path)
    monkeypatch.setitem(EXPECTED_COMMON_FINGERPRINTS, 0, artifact["fingerprints"]["common"])
    record = {
        "path": str(path),
        "sha256": file_sha256(path),
        "fingerprints": artifact["fingerprints"],
    }

    validate_initial_state_authority(path, seed=0, manifest_record=record)

    record["sha256"] = "BAD"
    try:
        validate_initial_state_authority(path, seed=0, manifest_record=record)
    except ValueError as error:
        assert "file SHA" in str(error)
    else:
        raise AssertionError("changed initial-state file hash was accepted")

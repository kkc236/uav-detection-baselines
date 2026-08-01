from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping

import pytest
import torch
from torch import nn

from src.iber_protocol import (
    DESIGN_VERSION,
    EXECUTION_ENVIRONMENT,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PRIVATE_OPTIMIZER,
    PRIVATE_SEED,
    PROBES,
    PROBE_EPOCHS,
    PROTOCOL_PAYLOAD,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    SCREEN_CONTRACT,
    SCREEN_EPOCHS,
    SCREEN_TRAIN_COUNT,
    SCREEN_VAL_COUNT,
    execution_environment,
    file_sha256,
    module_state_sha256,
    validate_screen_contract,
    write_immutable_report,
)


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _assert_recursively_immutable(value: object) -> None:
    assert not isinstance(value, (dict, list))
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_recursively_immutable(item)
    elif isinstance(value, tuple):
        for item in value:
            _assert_recursively_immutable(item)


def _screen_contract() -> dict[str, object]:
    return {
        "seed": 0,
        "epochs": 30,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "amp_scale": 128.0,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "max_det": 300,
        "nms": False,
    }


def test_iber_protocol_is_independent_and_frozen() -> None:
    assert DESIGN_VERSION == "iber-be-v1.0"
    assert PROBES == frozenset(("b0", "b1", "b2", "b3"))
    assert PROBE_EPOCHS == 12
    assert SCREEN_EPOCHS == 30
    assert SCREEN_TRAIN_COUNT == 647
    assert SCREEN_VAL_COUNT == 548
    assert PRIVATE_SEED == 10_000
    assert PRIVATE_OPTIMIZER == {
        "name": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "clip": 10.0,
    }
    assert EXPECTED_BASELINE_SHA256 == (
        "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
    )
    assert EXPECTED_DATASET_SHA256 == (
        "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
    )
    assert EXPECTED_SUBSET_SHA256 == (
        "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
    )


def test_execution_environment_and_runtime_amendment_are_iber_owned() -> None:
    assert execution_environment() == {
        "gpu": "NVIDIA GeForce RTX 4090",
        "reported_memory_mib": 49140,
        "driver": "570.133.07",
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda": "12.1",
        "ultralytics": "8.4.90",
    }
    assert execution_environment() is not EXECUTION_ENVIRONMENT
    assert RUNTIME_AMENDMENT == {
        "amendment_id": "iber-be-v1.0-runtime-driver-2026-08-01",
        "approved_on": "2026-08-01",
        "baseline_driver": "550.142",
        "execution_driver": "570.133.07",
        "allowed_differences": ("driver", "reported_memory_mib"),
        "comparison": "same-checkpoint-stock-vs-refined",
    }
    serialized = json.dumps(
        _json_compatible(RUNTIME_AMENDMENT),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert RUNTIME_AMENDMENT_SHA256 == hashlib.sha256(serialized).hexdigest().upper()
    assert RUNTIME_AMENDMENT_SHA256 == (
        "3B1DE94AB38955CAC309A5E3685B46801ECBC79CDEB907E23CCF36789A37C6BF"
    )


def test_protocol_sha256_covers_canonical_iber_payload() -> None:
    serialized = json.dumps(
        _json_compatible(PROTOCOL_PAYLOAD),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert PROTOCOL_SHA256 == hashlib.sha256(serialized).hexdigest().upper()
    assert PROTOCOL_SHA256 == (
        "D273BAAA90C734FB497AA11536F1AAD2142F532BCB347CC155F915B23988E097"
    )
    assert len(PROTOCOL_SHA256) == 64
    assert PROTOCOL_PAYLOAD["design_version"] == DESIGN_VERSION
    assert "itber" not in serialized.decode("utf-8").lower()


@pytest.mark.parametrize(
    ("authority", "key", "replacement"),
    [
        (SCREEN_CONTRACT, "epochs", 29),
        (PRIVATE_OPTIMIZER, "lr", 2e-3),
        (EXECUTION_ENVIRONMENT, "driver", "changed"),
        (RUNTIME_AMENDMENT, "comparison", "changed"),
        (PROTOCOL_PAYLOAD, "design_version", "changed"),
    ],
)
def test_exported_authority_rejects_top_level_mutation(
    authority: Mapping[str, object], key: str, replacement: object
) -> None:
    original = authority[key]
    protocol_sha256 = PROTOCOL_SHA256
    amendment_sha256 = RUNTIME_AMENDMENT_SHA256
    try:
        with pytest.raises(TypeError):
            authority[key] = replacement  # type: ignore[index]
    finally:
        if isinstance(authority, dict):
            authority[key] = original
    assert PROTOCOL_SHA256 == protocol_sha256
    assert RUNTIME_AMENDMENT_SHA256 == amendment_sha256


def test_exported_authority_rejects_nested_mapping_mutation() -> None:
    expected_hashes = PROTOCOL_PAYLOAD["expected_sha256"]
    assert isinstance(expected_hashes, Mapping)
    original = expected_hashes["baseline"]
    protocol_sha256 = PROTOCOL_SHA256
    amendment_sha256 = RUNTIME_AMENDMENT_SHA256
    try:
        with pytest.raises(TypeError):
            expected_hashes["baseline"] = "0" * 64  # type: ignore[index]
    finally:
        if isinstance(expected_hashes, dict):
            expected_hashes["baseline"] = original
    assert PROTOCOL_SHA256 == protocol_sha256
    assert RUNTIME_AMENDMENT_SHA256 == amendment_sha256


def test_exported_authority_rejects_nested_sequence_mutation() -> None:
    allowed_differences = RUNTIME_AMENDMENT["allowed_differences"]
    assert isinstance(allowed_differences, (list, tuple))
    original = tuple(allowed_differences)
    protocol_sha256 = PROTOCOL_SHA256
    amendment_sha256 = RUNTIME_AMENDMENT_SHA256
    try:
        with pytest.raises(TypeError):
            allowed_differences[0] = "cuda"  # type: ignore[index]
    finally:
        if isinstance(allowed_differences, list):
            allowed_differences[:] = original
    assert PROTOCOL_SHA256 == protocol_sha256
    assert RUNTIME_AMENDMENT_SHA256 == amendment_sha256


def test_exported_authority_is_recursively_immutable_without_hash_drift() -> None:
    protocol_sha256 = PROTOCOL_SHA256
    amendment_sha256 = RUNTIME_AMENDMENT_SHA256

    for authority in (
        SCREEN_CONTRACT,
        PRIVATE_OPTIMIZER,
        EXECUTION_ENVIRONMENT,
        RUNTIME_AMENDMENT,
        PROTOCOL_PAYLOAD,
    ):
        _assert_recursively_immutable(authority)

    assert PROTOCOL_SHA256 == protocol_sha256
    assert RUNTIME_AMENDMENT_SHA256 == amendment_sha256


def test_screen_contract_accepts_only_the_frozen_values() -> None:
    report = validate_screen_contract(_screen_contract())

    assert report["status"] == "passed_with_runtime_amendment"
    assert report["contract"] == _screen_contract()
    assert report["violations"] == {}
    assert report["protocol_sha256"] == PROTOCOL_SHA256
    assert report["runtime_amendment_sha256"] == RUNTIME_AMENDMENT_SHA256


def test_screen_contract_rejects_changed_epoch() -> None:
    contract = _screen_contract()
    contract["epochs"] = 29

    report = validate_screen_contract(contract)

    assert report["status"] == "engineering_invalid"
    assert report["violations"]["epochs"] == {"expected": 30, "actual": 29}


def test_screen_contract_rejects_missing_key() -> None:
    contract = _screen_contract()
    del contract["workers"]

    report = validate_screen_contract(contract)

    assert report["status"] == "engineering_invalid"
    assert report["violations"]["workers"] == {"expected": 8, "actual": None}


def test_screen_contract_rejects_non_mapping_input() -> None:
    report = validate_screen_contract([("seed", 0)])  # type: ignore[arg-type]

    assert report["status"] == "engineering_invalid"
    assert report["contract"] is None
    assert report["violations"]["contract"] == {
        "expected": "mapping",
        "actual": "list",
    }


def test_screen_contract_rejects_bool_for_integer() -> None:
    contract = _screen_contract()
    contract["seed"] = False

    report = validate_screen_contract(contract)

    assert report["status"] == "engineering_invalid"
    assert report["violations"]["seed"] == {"expected": 0, "actual": False}


def test_screen_contract_rejects_non_string_extra_key() -> None:
    contract = _screen_contract()
    contract[7] = "extra"  # type: ignore[index]

    report = validate_screen_contract(contract)

    assert report["status"] == "engineering_invalid"
    assert report["violations"][7] == {"expected": None, "actual": "extra"}


@pytest.mark.parametrize("identity", ["itber-v1.1", "I-TBER", "itber-probe"])
def test_screen_contract_rejects_itber_identities(identity: str) -> None:
    contract = _screen_contract()
    contract["design_version"] = identity

    report = validate_screen_contract(contract)

    assert report["status"] == "engineering_invalid"
    assert report["violations"]["design_version"]["actual"] == identity


def test_module_state_sha256_changes_with_module_state() -> None:
    module = nn.Linear(2, 2)
    first = module_state_sha256(module)
    module.weight.data.add_(1.0)
    second = module_state_sha256(module)

    assert len(first) == 64
    assert first != second
    assert second == module_state_sha256(module)


@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor.*:UserWarning")
def test_module_state_sha256_rejects_quantized_persistent_buffer() -> None:
    module = nn.Module()
    module.register_buffer(
        "quantized_buffer",
        torch.quantize_per_tensor(
            torch.tensor([1.0, 2.0]), scale=0.1, zero_point=0, dtype=torch.qint8
        ),
        persistent=True,
    )

    with pytest.raises(TypeError, match="state entry 'quantized_buffer'.*quantized"):
        module_state_sha256(module)


def test_module_state_sha256_rejects_sparse_persistent_buffer() -> None:
    module = nn.Module()
    module.register_buffer(
        "sparse_buffer",
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([1.0, 2.0]),
            size=(2, 2),
            check_invariants=False,
        ),
        persistent=True,
    )

    with pytest.raises(TypeError, match="state entry 'sparse_buffer'.*non-strided"):
        module_state_sha256(module)


def test_file_sha256_streams_an_uppercase_digest(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"independent-iber-protocol\n")

    expected = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    assert file_sha256(path) == expected


def test_immutable_report_is_canonical_and_exclusive(tmp_path) -> None:
    path = tmp_path / "immutable" / "authority.json"
    payload = {"status": "passed_with_runtime_amendment", "design_version": DESIGN_VERSION}

    result = write_immutable_report(path, payload)

    assert result == path
    assert path.read_text(encoding="utf-8") == (
        '{"design_version":"iber-be-v1.0","status":"passed_with_runtime_amendment"}\n'
    )
    with pytest.raises(FileExistsError):
        write_immutable_report(path, {"status": "changed"})


def test_immutable_report_rejects_direct_overwrite(tmp_path) -> None:
    path = tmp_path / "authority.json"
    write_immutable_report(path, {"design_version": DESIGN_VERSION})
    original = path.read_bytes()

    try:
        if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
            assert path.stat().st_mode & stat.S_IWUSR == 0
        else:
            with pytest.raises(PermissionError):
                path.write_text('{"changed":true}\n', encoding="utf-8")
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)
        if path.read_bytes() != original:
            path.write_bytes(original)


def test_immutable_report_rejects_symlink_destination(tmp_path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"target":true}\n', encoding="utf-8")
    destination = tmp_path / "authority.json"
    try:
        destination.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|reparse"):
        write_immutable_report(destination, {"design_version": DESIGN_VERSION})

    assert target.read_text(encoding="utf-8") == '{"target":true}\n'


def test_immutable_report_rejects_symlink_parent(tmp_path) -> None:
    target_parent = tmp_path / "target"
    target_parent.mkdir()
    linked_parent = tmp_path / "linked"
    try:
        linked_parent.symlink_to(target_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|reparse"):
        write_immutable_report(
            linked_parent / "authority.json", {"design_version": DESIGN_VERSION}
        )


def test_immutable_report_requires_json_suffix(tmp_path) -> None:
    with pytest.raises(ValueError, match="end in .json"):
        write_immutable_report(tmp_path / "authority.txt", {"status": "passed"})

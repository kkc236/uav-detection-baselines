from __future__ import annotations

import hashlib
import json

import pytest
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
    SCREEN_EPOCHS,
    SCREEN_TRAIN_COUNT,
    SCREEN_VAL_COUNT,
    execution_environment,
    file_sha256,
    module_state_sha256,
    validate_screen_contract,
    write_immutable_report,
)


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
        "allowed_differences": ["driver", "reported_memory_mib"],
        "comparison": "same-checkpoint-stock-vs-refined",
    }
    serialized = json.dumps(
        RUNTIME_AMENDMENT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert RUNTIME_AMENDMENT_SHA256 == hashlib.sha256(serialized).hexdigest().upper()


def test_protocol_sha256_covers_canonical_iber_payload() -> None:
    serialized = json.dumps(
        PROTOCOL_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert PROTOCOL_SHA256 == hashlib.sha256(serialized).hexdigest().upper()
    assert len(PROTOCOL_SHA256) == 64
    assert PROTOCOL_PAYLOAD["design_version"] == DESIGN_VERSION
    assert "itber" not in serialized.decode("utf-8").lower()


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


def test_immutable_report_requires_json_suffix(tmp_path) -> None:
    with pytest.raises(ValueError, match="end in .json"):
        write_immutable_report(tmp_path / "authority.txt", {"status": "passed"})

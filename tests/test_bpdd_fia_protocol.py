from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.bpdd_protocol import BPDD_PROTOCOL
from src.fdr_protocol import FDR_PROTOCOL, FDR_PROTOCOL_SHA256
from src.bpdd_fia_protocol import (
    BPDD_FIA_PROTOCOL,
    BPDD_FIA_PROTOCOL_SHA256,
    FDR_INITIAL_STATE_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_resume_authority,
    write_create_only_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_bpdd_fia_protocol.py"
EXPECTED_FDR_SOURCE_COMMIT = "d97e1eb7f98414752a1c1f38287697db3f2a0679"
EXPECTED_INITIAL_STATE_SHA256 = (
    "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"
)
EXPECTED_DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)


def _source() -> dict[str, str]:
    return {"git_commit": "a" * 40, "tree_sha256": "B" * 64}


def _load_prepare_module():
    assert PREPARE_SCRIPT.is_file(), "BPDD+FIA protocol preparer is missing"
    spec = importlib.util.spec_from_file_location(
        "prepare_bpdd_fia_protocol", PREPARE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_binds_frozen_fdr_and_complete_baseline_settings() -> None:
    assert BPDD_FIA_PROTOCOL["fdr_authority"] == {
        "protocol_sha256": FDR_PROTOCOL_SHA256,
        "source_commit": EXPECTED_FDR_SOURCE_COMMIT,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
    }
    assert BPDD_FIA_PROTOCOL["environment"] == FDR_PROTOCOL["environment"]
    assert BPDD_FIA_PROTOCOL["dataset"] == FDR_PROTOCOL["dataset"]
    assert BPDD_FIA_PROTOCOL["dataset"]["sha256"] == EXPECTED_DATASET_SHA256
    assert BPDD_FIA_PROTOCOL["training"] == FDR_PROTOCOL["training"]
    assert BPDD_FIA_PROTOCOL["augmentation"] == FDR_PROTOCOL["augmentation"]
    assert BPDD_FIA_PROTOCOL["bpdd"] == BPDD_PROTOCOL["bpdd"]
    assert BPDD_FIA_PROTOCOL_SHA256 == public_state_sha256(BPDD_FIA_PROTOCOL)


def test_protocol_exposes_only_one_formal_combined_variant() -> None:
    assert BPDD_FIA_PROTOCOL["variants"] == {
        "fdr_bpdd_fia": {
            "model_yaml": "configs/rtdetr-l-fdr-bpdd-fia.yaml",
            "fdr_enabled": True,
            "bpdd_enabled": True,
            "fia_enabled": True,
        }
    }
    assert BPDD_FIA_PROTOCOL["stages"] == {
        "formal": {"schedule_epochs": 100, "fresh_start": True}
    }
    assert BPDD_FIA_PROTOCOL["seed"] == 0
    assert BPDD_FIA_PROTOCOL["fia"] == {
        "feature_level": "P3",
        "channels": 256,
        "refinement_blocks": 2,
        "depthwise_kernel": 3,
        "outer_residual_gate": True,
        "outer_residual_gate_init": 0.0,
        "private_seed": 20000,
    }


def test_run_identity_binds_source_protocol_initial_state_and_formal_arm() -> None:
    source = _source()
    identity = build_run_identity(
        source, stage="formal", variant="fdr_bpdd_fia", seed=0
    )

    assert identity == {
        "source_sha256": public_state_sha256(source),
        "protocol_sha256": BPDD_FIA_PROTOCOL_SHA256,
        "fdr_protocol_sha256": FDR_PROTOCOL_SHA256,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "run_id": identity["run_id"],
        "stage": "formal",
        "variant": "fdr_bpdd_fia",
        "seed": 0,
    }
    assert identity["run_id"].startswith("fdr_bpdd_fia-formal-seed0-")


@pytest.mark.parametrize(
    ("stage", "variant", "seed", "message"),
    [
        ("screen", "fdr_bpdd_fia", 0, "stage"),
        ("formal", "fdr_bpdd", 0, "variant"),
        ("formal", "fdr_bpdd_fia", 1, "seed0"),
    ],
)
def test_run_identity_rejects_non_authorized_runs(
    stage: str, variant: str, seed: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_run_identity(_source(), stage=stage, variant=variant, seed=seed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_sha256", "A" * 64),
        ("protocol_sha256", "B" * 64),
        ("fdr_protocol_sha256", "C" * 64),
        ("initial_state_sha256", "D" * 64),
        ("run_id", "foreign-run"),
        ("stage", "screen"),
        ("variant", "fdr_bpdd"),
        ("seed", 1),
    ],
)
def test_resume_rejects_every_authority_drift(
    field: str, replacement: object
) -> None:
    expected = build_run_identity(
        _source(), stage="formal", variant="fdr_bpdd_fia", seed=0
    )
    checkpoint = deepcopy(expected)
    checkpoint[field] = replacement

    with pytest.raises(ValueError, match=field):
        validate_resume_authority(checkpoint, expected)


def test_prepare_manifest_has_one_identity_canonical_hash_and_create_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_prepare_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"test-state")
    output = tmp_path / "bpdd-fia-protocol.json"
    monkeypatch.setattr(module, "_file_sha256", lambda _path: EXPECTED_INITIAL_STATE_SHA256)
    fingerprints = {
        "public": "1" * 64,
        "fdr_public": "2" * 64,
        "replaced_control": "3" * 64,
        "private": "4" * 64,
        "control": "5" * 64,
        "fdr": "6" * 64,
    }
    monkeypatch.setattr(
        module, "_validate_initial_state", lambda _path: {"fingerprints": fingerprints}
    )

    manifest = module.prepare_manifest(
        source_commit="a" * 40,
        source_tree_sha256="B" * 64,
        initial_state=state,
        output=output,
    )

    assert manifest["source"] == _source()
    assert manifest["source_sha256"] == public_state_sha256(_source())
    assert manifest["protocol"] == BPDD_FIA_PROTOCOL
    assert manifest["protocol_sha256"] == BPDD_FIA_PROTOCOL_SHA256
    assert manifest["initial_state"] == {
        "path": str(state.resolve()),
        "sha256": EXPECTED_INITIAL_STATE_SHA256,
        "fingerprints": fingerprints,
    }
    assert set(manifest["run_identities"]) == {"fdr_bpdd_fia_formal"}
    assert manifest["manifest_sha256"] == public_state_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert output.read_bytes() == canonical_json_bytes(manifest) + b"\n"

    with pytest.raises(FileExistsError):
        module.prepare_manifest(
            source_commit="a" * 40,
            source_tree_sha256="B" * 64,
            initial_state=state,
            output=output,
        )


@pytest.mark.parametrize(
    ("commit", "tree", "message"),
    [
        ("x" * 40, "B" * 64, "source_commit"),
        ("a" * 39, "B" * 64, "source_commit"),
        ("a" * 40, "z" * 64, "source_tree_sha256"),
        ("a" * 40, "B" * 63, "source_tree_sha256"),
    ],
)
def test_prepare_manifest_rejects_malformed_source_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commit: str,
    tree: str,
    message: str,
) -> None:
    module = _load_prepare_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"test-state")
    monkeypatch.setattr(module, "_file_sha256", lambda _path: EXPECTED_INITIAL_STATE_SHA256)
    monkeypatch.setattr(
        module, "_validate_initial_state", lambda _path: {"fingerprints": {}}
    )

    with pytest.raises(ValueError, match=message):
        module.prepare_manifest(
            source_commit=commit,
            source_tree_sha256=tree,
            initial_state=state,
            output=tmp_path / "manifest.json",
        )


def test_prepare_cli_exposes_only_authority_inputs() -> None:
    module = _load_prepare_module()
    actions = {action.dest for action in module.build_parser()._actions}
    assert actions == {
        "help",
        "source_commit",
        "source_tree_sha256",
        "initial_state",
        "output",
    }

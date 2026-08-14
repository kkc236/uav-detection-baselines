from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import src.pr_ira_protocol as pr_ira_protocol_module
from src.bpdd_protocol import BPDD_PROTOCOL
from src.fdr_protocol import FDR_PROTOCOL, FDR_PROTOCOL_SHA256
from src.pr_ira_protocol import (
    FDR_INITIAL_STATE_SHA256,
    FDR_SOURCE_COMMIT,
    PR_IRA_PROTOCOL,
    PR_IRA_PROTOCOL_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_resume_authority,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_pr_ira_protocol.py"
EXPECTED_FDR_SOURCE_COMMIT = "d97e1eb7f98414752a1c1f38287697db3f2a0679"
EXPECTED_INITIAL_STATE_SHA256 = (
    "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"
)
EXPECTED_DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)
EXPECTED_SUBSET_SHA256 = (
    "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
)
SCREEN_VARIANTS = (
    "fdr_bpdd",
    "fdr_bpdd_pr_ira",
    "fdr",
    "fdr_pr_ira",
)


def _source() -> dict[str, str]:
    return {"git_commit": "a" * 40, "tree_sha256": "B" * 64}


def _load_prepare_module():
    assert PREPARE_SCRIPT.is_file(), "PR-IRA protocol preparer is missing"
    spec = importlib.util.spec_from_file_location(
        "prepare_pr_ira_protocol", PREPARE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_freezes_inherited_authority_and_runtime() -> None:
    assert FDR_SOURCE_COMMIT == EXPECTED_FDR_SOURCE_COMMIT
    assert FDR_INITIAL_STATE_SHA256 == EXPECTED_INITIAL_STATE_SHA256
    assert PR_IRA_PROTOCOL["fdr_authority"] == {
        "protocol_sha256": FDR_PROTOCOL_SHA256,
        "source_commit": EXPECTED_FDR_SOURCE_COMMIT,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
    }
    assert PR_IRA_PROTOCOL["environment"] == FDR_PROTOCOL["environment"]
    assert PR_IRA_PROTOCOL["environment"]["ultralytics"] == "8.4.90"
    assert PR_IRA_PROTOCOL["dataset"] == FDR_PROTOCOL["dataset"]
    assert PR_IRA_PROTOCOL["dataset"]["sha256"] == EXPECTED_DATASET_SHA256
    assert PR_IRA_PROTOCOL["dataset"]["screen_train_images"] == 647
    assert PR_IRA_PROTOCOL["dataset"]["screen_sha256"] == EXPECTED_SUBSET_SHA256
    assert PR_IRA_PROTOCOL["augmentation"] == FDR_PROTOCOL["augmentation"]
    assert PR_IRA_PROTOCOL["bpdd"] == BPDD_PROTOCOL["bpdd"] == {
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1e-6,
    }
    assert PR_IRA_PROTOCOL_SHA256 == public_state_sha256(PR_IRA_PROTOCOL)


def test_protocol_freezes_seed_optimizer_and_pr_ira_schedule() -> None:
    expected_training = deepcopy(FDR_PROTOCOL["training"])
    expected_training.update(
        {
            "screen_schedule_epochs": 30,
            "screen_cutoff_epoch": 30,
        }
    )
    assert PR_IRA_PROTOCOL["training"] == expected_training
    assert PR_IRA_PROTOCOL["training"]["seeds"] == [0]
    assert PR_IRA_PROTOCOL["training"]["optimizer"] == "MuSGD"
    assert PR_IRA_PROTOCOL["training"]["momentum"] == 0.937
    assert PR_IRA_PROTOCOL["training"]["weight_decay"] == 0.0005
    assert PR_IRA_PROTOCOL["pr_ira"] == {
        "feature_level": "P3",
        "channels": 256,
        "alpha_max": 0.20,
        "epsilon": 1e-6,
        "private_lr_multiplier": 0.1,
        "private_seed_namespace": 20_000,
        "private_seed_formula": "20000 + experiment_seed",
        "schedule": {
            "screen30": {
                "epochs": 30,
                "identity": [1, 3],
                "linear_open": [4, 9],
                "fully_open": [10, 30],
                "private_update": [4, 30],
                "private_frozen": [],
            },
            "formal100": {
                "epochs": 100,
                "identity": [1, 10],
                "linear_open": [11, 30],
                "fully_open": [31, 100],
                "private_update": [11, 60],
                "private_frozen": [61, 100],
            },
        },
    }
    assert PR_IRA_PROTOCOL["seed"] == 0


@pytest.mark.parametrize(
    ("epoch", "epochs", "expected"),
    [
        (3, 30, False),
        (4, 30, True),
        (30, 30, True),
        (10, 100, False),
        (11, 100, True),
        (60, 100, True),
        (61, 100, False),
        (100, 100, False),
    ],
)
def test_private_update_window_boundaries(
    epoch: int, epochs: int, expected: bool
) -> None:
    assert (
        pr_ira_protocol_module.pr_ira_private_update_enabled(epoch, epochs)
        is expected
    )


@pytest.mark.parametrize(
    ("epoch", "epochs"),
    [
        (0, 30),
        (31, 30),
        (101, 100),
        (1, 50),
    ],
)
def test_private_update_window_rejects_invalid_values(
    epoch: int, epochs: int
) -> None:
    with pytest.raises(ValueError):
        pr_ira_protocol_module.pr_ira_private_update_enabled(epoch, epochs)


@pytest.mark.parametrize(
    ("epoch", "epochs"),
    [
        (True, 30),
        (1.0, 30),
        ("1", 30),
        (1, True),
        (1, 30.0),
        (1, "30"),
    ],
)
def test_private_update_window_rejects_non_integer_types(
    epoch: object, epochs: object
) -> None:
    with pytest.raises(TypeError):
        pr_ira_protocol_module.pr_ira_private_update_enabled(epoch, epochs)  # type: ignore[arg-type]


def test_private_update_helper_is_exported() -> None:
    assert "pr_ira_private_update_enabled" in pr_ira_protocol_module.__all__


def test_protocol_freezes_variants_stages_and_every_gate_threshold() -> None:
    assert PR_IRA_PROTOCOL["variants"] == {
        "fdr_bpdd": {"fdr_enabled": True, "bpdd_enabled": True, "pr_ira_enabled": False},
        "fdr_bpdd_pr_ira": {"fdr_enabled": True, "bpdd_enabled": True, "pr_ira_enabled": True},
        "fdr": {"fdr_enabled": True, "bpdd_enabled": False, "pr_ira_enabled": False},
        "fdr_pr_ira": {"fdr_enabled": True, "bpdd_enabled": False, "pr_ira_enabled": True},
    }
    assert PR_IRA_PROTOCOL["stages"] == {
        "screen": {
            "schedule_epochs": 30,
            "fresh_start": True,
            "eligible_variants": list(SCREEN_VARIANTS),
        },
        "formal": {
            "schedule_epochs": 100,
            "fresh_start": True,
            "eligible_variants": ["fdr_bpdd_pr_ira"],
        },
    }
    assert PR_IRA_PROTOCOL["gate_thresholds"] == {
        "final_map50_95_delta_gt": 0.0,
        "tail3_map50_95_delta_gt": 0.0,
        "final_ap75_delta_gt": 0.0,
        "tail3_ap75_delta_gt": 0.0,
        "final_ap50_delta_gte": -0.0005,
        "final_precision_delta_gte": -0.0020,
        "tiny_or_small_map_delta_gt": 0.0,
        "finite_gradients_required": True,
        "max_abs_effective_amplitude": 0.20,
        "sustained_amplitude_saturation_forbidden": True,
        "residual_rms_tolerance": 1e-5,
        "firewall_rtol": 1e-5,
        "firewall_atol": 1e-7,
        "max_parameter_increase_ratio": 0.10,
    }
    assert PR_IRA_PROTOCOL["independence_gate_reuses_compatibility"] is True


@pytest.mark.parametrize("variant", SCREEN_VARIANTS)
def test_screen_run_identities_bind_every_authority(variant: str) -> None:
    identity = build_run_identity(_source(), stage="screen", variant=variant, seed=0)

    assert identity["source_sha256"] == public_state_sha256(_source())
    assert identity["protocol_sha256"] == PR_IRA_PROTOCOL_SHA256
    assert identity["fdr_protocol_sha256"] == FDR_PROTOCOL_SHA256
    assert identity["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert identity["dataset_sha256"] == EXPECTED_DATASET_SHA256
    assert identity["screen_subset_sha256"] == EXPECTED_SUBSET_SHA256
    assert identity["stage"] == "screen"
    assert identity["variant"] == variant
    assert identity["seed"] == 0
    assert identity["run_id"].startswith(f"{variant}-screen30-seed0-")


def test_only_the_combined_pr_ira_arm_has_a_formal100_identity() -> None:
    identity = build_run_identity(
        _source(), stage="formal", variant="fdr_bpdd_pr_ira", seed=0
    )
    assert identity["run_id"].startswith("fdr_bpdd_pr_ira-formal100-seed0-")
    assert identity["stage"] == "formal"

    for variant in ("fdr_bpdd", "fdr", "fdr_pr_ira"):
        with pytest.raises(ValueError, match="not eligible"):
            build_run_identity(_source(), stage="formal", variant=variant, seed=0)


@pytest.mark.parametrize(
    ("stage", "variant", "seed", "message"),
    [
        ("probe", "fdr_bpdd", 0, "stage"),
        ("screen", "stock", 0, "variant"),
        ("screen", "fdr_bpdd", 1, "seed0"),
    ],
)
def test_run_identity_rejects_non_authorized_inputs(
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
        ("dataset_sha256", "E" * 64),
        ("screen_subset_sha256", "F" * 64),
        ("run_id", "foreign-run"),
        ("stage", "formal"),
        ("variant", "fdr"),
        ("seed", 1),
    ],
)
def test_resume_rejects_source_protocol_state_dataset_variant_or_seed_drift(
    field: str, replacement: object
) -> None:
    expected = build_run_identity(
        _source(), stage="screen", variant="fdr_bpdd_pr_ira", seed=0
    )
    checkpoint = deepcopy(expected)
    checkpoint[field] = replacement

    with pytest.raises(ValueError, match=field):
        validate_resume_authority(checkpoint, expected)


def test_prepare_manifest_creates_four_screens_and_one_formal_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_prepare_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"test-state")
    output = tmp_path / "pr-ira-protocol.json"
    monkeypatch.setattr(module, "_file_sha256", lambda _path: EXPECTED_INITIAL_STATE_SHA256)
    fingerprints = {"public": "1" * 64, "private": "2" * 64}
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
    assert manifest["protocol"] == PR_IRA_PROTOCOL
    assert manifest["protocol_sha256"] == PR_IRA_PROTOCOL_SHA256
    assert manifest["initial_state"] == {
        "path": str(state.resolve()),
        "sha256": EXPECTED_INITIAL_STATE_SHA256,
        "fingerprints": fingerprints,
    }
    assert set(manifest["run_identities"]) == {
        "fdr_bpdd_screen",
        "fdr_bpdd_pr_ira_screen",
        "fdr_screen",
        "fdr_pr_ira_screen",
        "fdr_bpdd_pr_ira_formal",
    }
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
    monkeypatch.setattr(module, "_validate_initial_state", lambda _path: {"fingerprints": {}})

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
    for forbidden in (
        "alpha_max",
        "private_lr_multiplier",
        "private_seed",
        "bpdd_weight",
        "seed",
    ):
        assert forbidden not in actions

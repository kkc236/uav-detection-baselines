from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from src.rtdetr_fdr import FDRRTDETRDetectionModel
from src.rtdetr_scads import (
    SCADSFDRRTDETRDetectionModel,
    SCADSTrainer,
)
from src.scads_protocol import (
    SCADS_PROTOCOL_SHA256,
    build_run_identity,
    build_scads_initial_state,
    load_scads_initial_state,
    validate_scads_initial_state,
)


def _paired_models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        fdr = FDRRTDETRDetectionModel(
            "configs/rtdetr-l-fdr.yaml",
            nc=10,
            verbose=False,
            private_seed=10_000,
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        scads = SCADSFDRRTDETRDetectionModel(
            "configs/rtdetr-l-fdr-scads.yaml",
            nc=10,
            verbose=False,
            private_seed=10_000,
            support_private_seed=20_000,
        )
    return fdr, scads


def _artifact():
    fdr, scads = _paired_models()
    artifact = build_scads_initial_state(
        fdr.state_dict(),
        scads.state_dict(),
        metadata={"seed": 0},
    )
    return fdr, scads, artifact


def test_scads_initial_state_partitions_only_approved_private_tensors() -> None:
    fdr, scads, artifact = _artifact()
    validate_scads_initial_state(artifact)
    assert set(artifact["common_state"]) == set(fdr.state_dict())
    assert set(artifact["common_state"]) | set(
        artifact["scads_private_state"]
    ) == set(scads.state_dict())
    assert all(
        name.startswith("model.28.decoder.support_router.")
        or name.startswith("model.28.decoder.adaptive_integral.")
        for name in artifact["scads_private_state"]
    )


def test_scads_initial_state_loads_both_variants_strictly() -> None:
    fdr, scads, artifact = _artifact()
    with torch.no_grad():
        next(fdr.parameters()).add_(1.0)
        next(scads.parameters()).sub_(1.0)
    load_scads_initial_state(fdr, artifact, variant="fdr")
    load_scads_initial_state(scads, artifact, variant="scads")
    for name, expected in artifact["common_state"].items():
        torch.testing.assert_close(fdr.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(scads.state_dict()[name], expected, rtol=0, atol=0)


def test_scads_initial_state_rejects_private_tensor_tampering() -> None:
    _fdr, _scads, artifact = _artifact()
    changed = deepcopy(artifact)
    first = next(iter(changed["scads_private_state"]))
    changed["scads_private_state"][first].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        validate_scads_initial_state(changed)


def test_scads_run_identity_binds_variant_stage_and_protocol() -> None:
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    identity = build_run_identity(source, stage="screen", variant="scads", seed=0)
    assert identity["variant"] == "scads"
    assert identity["stage"] == "screen"
    assert identity["protocol_sha256"] == SCADS_PROTOCOL_SHA256
    assert identity["run_id"].startswith("scads-screen-seed0-")


def test_scads_gradient_groups_cover_every_trainable_parameter_once() -> None:
    _fdr, model, _artifact_value = _artifact()
    trainer = object.__new__(SCADSTrainer)
    trainer.model = model
    groups = trainer.gradient_parameter_groups()
    assert set(groups) == {
        "gradient_norm",
        "fdr_gradient_norm",
        "scads_gradient_norm",
    }
    identifiers = [id(parameter) for values in groups.values() for parameter in values]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == set(expected)

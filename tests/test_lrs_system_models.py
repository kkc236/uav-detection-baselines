from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from src.bpdd_loss import BPDDDetectionLoss
from src.fdr_loss import FDRDetectionLoss
from src.fdr_protocol import build_fdr_initial_state
from src.fia import FIA
from src import rtdetr_lrs_system as lrs_system
from src.rtdetr_fdr import FDRTrainer
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel
from src.rtdetr_fdr_bpdd import FDRBPDDTrainer
from src.rtdetr_lrs_system import (
    ARM_CONFIGS,
    FIA_MODEL_INDEX,
    FIA_STATE_PREFIX,
    MODEL_TYPES,
    ROOT,
    TRAINER_TYPES,
    LRSFDRBPDDFIADetectionModel,
    LRSFDRBPDDFIATrainer,
    LRSFDRBPDDDetectionModel,
    LRSFDRBPDDTrainer,
    LRSFDRFIADetectionModel,
    LRSFDRFIATrainer,
    initialize_fia_graph,
    load_fia_initial_state,
    remap_fia_shared_key,
)


TEST_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def arm_models() -> dict[str, FDRBPDDDetectionModel]:
    constructors = {
        "g": LRSFDRBPDDDetectionModel,
        "h": LRSFDRFIADetectionModel,
        "i": LRSFDRBPDDFIADetectionModel,
    }
    models: dict[str, FDRBPDDDetectionModel] = {}
    for arm, constructor in constructors.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9100)
            models[arm] = constructor(nc=10, verbose=False)
    return models


def _is_fdr_private(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            ".dec_bbox_head.",
            ".decoder.pre_bbox_head.",
            ".decoder.distribution_feedback.",
        )
    )


@pytest.fixture(scope="module")
def non_fia_artifact(
    arm_models: dict[str, FDRBPDDDetectionModel],
) -> dict[str, Any]:
    source = arm_models["g"].state_dict()
    public = {
        name: value for name, value in source.items() if not _is_fdr_private(name)
    }
    return build_fdr_initial_state(
        public,
        source,
        private_prefixes=(
            "model.28.dec_bbox_head.",
            "model.28.decoder.pre_bbox_head.",
            "model.28.decoder.distribution_feedback.",
        ),
        metadata={"source": "rebuilt-lrs-non-fia"},
    )


def test_bpdd_criterion_preserves_lrs_alpha() -> None:
    with (TEST_ROOT / "configs" / "rtdetr-l-lrs-fdr.yaml").open(
        encoding="utf-8"
    ) as stream:
        payload = yaml.safe_load(stream)
    payload["bpdd_loss"] = {
        "enabled": True,
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1e-6,
        "matched_layer": "final",
        "include_dn": False,
    }

    model = FDRBPDDDetectionModel(payload, nc=10, verbose=False)
    criterion = model.init_criterion()

    assert criterion.reliability_shrinkage_alpha == 0.25
    assert criterion.supervise_dn_fdr is False


def test_public_arm_contract_uses_only_the_three_new_yamls() -> None:
    assert ROOT == TEST_ROOT
    assert ARM_CONFIGS == {
        "g": TEST_ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd.yaml",
        "h": TEST_ROOT / "configs" / "rtdetr-l-lrs-fdr-fia.yaml",
        "i": TEST_ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd-fia.yaml",
    }
    assert FIA_MODEL_INDEX == 22
    assert FIA_STATE_PREFIX == "model.22."
    assert MODEL_TYPES == {
        "g": LRSFDRBPDDDetectionModel,
        "h": LRSFDRFIADetectionModel,
        "i": LRSFDRBPDDFIADetectionModel,
    }
    assert TRAINER_TYPES == {
        "g": LRSFDRBPDDTrainer,
        "h": LRSFDRFIATrainer,
        "i": LRSFDRBPDDFIATrainer,
    }


def test_arm_models_isolate_bpdd_and_fia(
    arm_models: dict[str, FDRBPDDDetectionModel],
) -> None:
    criteria = {arm: model.init_criterion() for arm, model in arm_models.items()}

    assert isinstance(criteria["g"], BPDDDetectionLoss)
    assert type(criteria["h"]) is FDRDetectionLoss
    assert isinstance(criteria["i"], BPDDDetectionLoss)
    assert all(
        criterion.reliability_shrinkage_alpha == 0.25
        for criterion in criteria.values()
    )
    assert arm_models["g"].bpdd_options.enabled is True
    assert not hasattr(arm_models["h"], "bpdd_options")
    assert arm_models["i"].bpdd_options.enabled is True
    assert not any(isinstance(module, FIA) for module in arm_models["g"].model)
    assert sum(isinstance(module, FIA) for module in arm_models["h"].model) == 1
    assert sum(isinstance(module, FIA) for module in arm_models["i"].model) == 1


@pytest.mark.parametrize("arm", ["h", "i"])
def test_fia_graph_is_p3_only_and_identity_safe(
    arm: str,
    arm_models: dict[str, FDRBPDDDetectionModel],
) -> None:
    model = arm_models[arm]

    assert len(model.model) == 30
    assert model.model[FIA_MODEL_INDEX] is model.fia
    assert model.fia.f == 21
    assert model.model[23].f == 21
    assert model.model[-1].f == [22, 25, 28]
    assert model.fia.residual_scale.item() == 0.0


def test_initialize_fia_graph_rejects_a_graph_without_fia(
    arm_models: dict[str, FDRBPDDDetectionModel],
) -> None:
    with pytest.raises((TypeError, ValueError), match="30 modules"):
        initialize_fia_graph(arm_models["g"], private_seed=20_000)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("model.0.weight", "model.0.weight"),
        ("model.21.block.bias", "model.21.block.bias"),
        ("model.22.weight", "model.23.weight"),
        ("model.28.decoder.weight", "model.29.decoder.weight"),
        ("model.999.weight", "model.1000.weight"),
        ("module.model.22.weight", "module.model.22.weight"),
        ("not-a-model-key", "not-a-model-key"),
    ],
)
def test_remap_fia_shared_key_shifts_only_model_indices_at_or_after_fia(
    source: str,
    expected: str,
) -> None:
    assert remap_fia_shared_key(source) == expected


@pytest.mark.parametrize("arm", ["h", "i"])
def test_fia_initial_state_loads_every_shared_tensor_exactly(
    arm: str,
    arm_models: dict[str, FDRBPDDDetectionModel],
    non_fia_artifact: dict[str, Any],
) -> None:
    model = arm_models[arm]
    source = {
        **non_fia_artifact["fdr_public_state"],
        **non_fia_artifact["private_state"],
    }

    report = load_fia_initial_state(model, non_fia_artifact)

    loaded = model.state_dict()
    for name, expected in source.items():
        torch.testing.assert_close(
            loaded[remap_fia_shared_key(name)], expected, rtol=0, atol=0
        )
    assert report["shared_tensor_count"] == len(source)
    assert report["shared_mismatch_count"] == 0
    assert report["missing_keys"] == report["fia_private_keys"]
    assert report["fia_private_keys"]
    assert all(
        name.startswith(FIA_STATE_PREFIX) for name in report["fia_private_keys"]
    )


@pytest.mark.parametrize(
    "trainer_type",
    [LRSFDRFIATrainer, LRSFDRBPDDFIATrainer],
)
def test_fia_gradient_groups_are_disjoint_exhaustive_and_nonempty(
    trainer_type: type[LRSFDRFIATrainer],
    arm_models: dict[str, FDRBPDDDetectionModel],
) -> None:
    arm = "h" if trainer_type is LRSFDRFIATrainer else "i"
    trainer = trainer_type.__new__(trainer_type)
    trainer.model = arm_models[arm]

    groups = trainer.gradient_parameter_groups()

    assert set(groups) == {
        "gradient_norm",
        "fdr_gradient_norm",
        "fia_gradient_norm",
    }
    identifiers = {
        name: {id(parameter) for parameter in parameters}
        for name, parameters in groups.items()
    }
    assert all(identifiers.values())
    assert identifiers["gradient_norm"].isdisjoint(
        identifiers["fdr_gradient_norm"]
    )
    assert identifiers["gradient_norm"].isdisjoint(
        identifiers["fia_gradient_norm"]
    )
    assert identifiers["fdr_gradient_norm"].isdisjoint(
        identifiers["fia_gradient_norm"]
    )
    assert set().union(*identifiers.values()) == {
        id(parameter)
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    }
    assert identifiers["fia_gradient_norm"] == {
        id(parameter) for parameter in trainer.model.fia.parameters()
    }
    fdr_parameters = {
        id(parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and _is_fdr_private(name)
    }
    assert identifiers["fdr_gradient_norm"] == fdr_parameters


@pytest.mark.parametrize(
    ("arm", "trainer_type", "model_name", "expected_seeds"),
    [
        ("g", LRSFDRBPDDTrainer, "LRSFDRBPDDDetectionModel", (10_037, None)),
        ("h", LRSFDRFIATrainer, "LRSFDRFIADetectionModel", (10_037, 20_037)),
        (
            "i",
            LRSFDRBPDDFIATrainer,
            "LRSFDRBPDDFIADetectionModel",
            (10_037, 20_037),
        ),
    ],
)
def test_trainer_dispatches_exact_config_and_private_seeds_without_gpu(
    arm: str,
    trainer_type: type[FDRTrainer],
    model_name: str,
    expected_seeds: tuple[int, int | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _ModelDouble:
        def __init__(self, cfg: Path, **kwargs: Any) -> None:
            captured["cfg"] = cfg
            captured.update(kwargs)

    monkeypatch.setattr(lrs_system, model_name, _ModelDouble)
    trainer = trainer_type.__new__(trainer_type)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 37
    trainer.initial_state_path = None

    model = trainer.get_model(cfg="ignored.yaml", weights=None, verbose=True)

    assert isinstance(model, _ModelDouble)
    assert captured["cfg"] == ARM_CONFIGS[arm]
    assert captured["nc"] == 10
    assert captured["ch"] == 3
    assert captured["verbose"] is True
    assert captured["private_seed"] == expected_seeds[0]
    if expected_seeds[1] is None:
        assert "fia_private_seed" not in captured
    else:
        assert captured["fia_private_seed"] == expected_seeds[1]


@pytest.mark.parametrize("trainer_type", [LRSFDRFIATrainer, LRSFDRBPDDFIATrainer])
def test_fia_trainers_reject_checkpoint_weights(
    trainer_type: type[FDRTrainer],
) -> None:
    trainer = trainer_type.__new__(trainer_type)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None

    with pytest.raises(ValueError, match="fresh-only"):
        trainer.get_model(weights="checkpoint.pt", verbose=False)


def test_public_types_inherit_the_required_existing_integrations() -> None:
    assert issubclass(LRSFDRBPDDDetectionModel, FDRBPDDDetectionModel)
    assert issubclass(LRSFDRBPDDFIADetectionModel, FDRBPDDDetectionModel)
    assert issubclass(LRSFDRBPDDTrainer, FDRBPDDTrainer)
    assert issubclass(LRSFDRBPDDFIATrainer, FDRBPDDTrainer)
    assert issubclass(LRSFDRFIATrainer, FDRTrainer)

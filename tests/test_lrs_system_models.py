from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from src.bpdd_loss import BPDDDetectionLoss
from src.fdr_loss import FDRDetectionLoss
from src.fdr_protocol import build_fdr_initial_state, validate_fdr_initial_state
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


def _small_non_fia_state() -> dict[str, torch.Tensor]:
    return {
        "model.0.weight": torch.tensor([1.0, 2.0]),
        "model.21.weight": torch.tensor([3.0, 4.0]),
        "model.22.weight": torch.tensor([5.0, 6.0]),
        "model.28.dec_bbox_head.0.weight": torch.tensor([7.0, 8.0]),
    }


def _build_small_non_fia_artifact(
    source: dict[str, torch.Tensor],
) -> dict[str, Any]:
    public = {
        name: value for name, value in source.items() if not _is_fdr_private(name)
    }
    return build_fdr_initial_state(
        public,
        source,
        private_prefixes=("model.28.dec_bbox_head.",),
        metadata={"source": "small-rebuilt-lrs-non-fia"},
    )


class _StateTarget:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = {name: value.clone() for name, value in state.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.state

    def load_state_dict(
        self,
        state: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> SimpleNamespace:
        missing = sorted(set(self.state) - set(state))
        unexpected = sorted(set(state) - set(self.state))
        if strict and (missing or unexpected):
            raise RuntimeError("strict state mismatch")
        for name in set(state) & set(self.state):
            self.state[name].copy_(state[name])
        return SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)


class _WriteMarkerOnLoad:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (self.marker.write_text, ("unsafe pickle executed",))


def _small_fia_target() -> _StateTarget:
    source = _small_non_fia_state()
    target = {
        remap_fia_shared_key(name): value.clone()
        for name, value in source.items()
    }
    target[f"{FIA_STATE_PREFIX}residual_scale"] = torch.zeros(())
    return _StateTarget(target)


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


@pytest.fixture(scope="class")
def controlled_fia_pair() -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8421)
        base = LRSFDRBPDDDetectionModel(nc=10, verbose=False)
        base_rng = torch.random.get_rng_state().clone()

        torch.manual_seed(8421)
        fia = LRSFDRFIADetectionModel(nc=10, verbose=False)
        fia_rng = torch.random.get_rng_state().clone()
    return {
        "base": base,
        "fia": fia,
        "base_rng": base_rng,
        "fia_rng": fia_rng,
    }


class TestFIAConstructionIsolation:
    def test_fia_construction_preserves_the_base_cpu_rng_trajectory(
        self,
        controlled_fia_pair: dict[str, Any],
    ) -> None:
        assert torch.equal(
            controlled_fia_pair["fia_rng"], controlled_fia_pair["base_rng"]
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(8422)
            before_private_init = torch.random.get_rng_state().clone()
            initialize_fia_graph(
                controlled_fia_pair["fia"], private_seed=20_000
            )
            after_private_init = torch.random.get_rng_state()
        assert torch.equal(after_private_init, before_private_init)

    def test_fia_shared_state_matches_base_before_artifact_loading(
        self,
        controlled_fia_pair: dict[str, Any],
    ) -> None:
        base_state = controlled_fia_pair["base"].state_dict()
        fia_state = controlled_fia_pair["fia"].state_dict()

        for name, expected in base_state.items():
            torch.testing.assert_close(
                fia_state[remap_fia_shared_key(name)], expected, rtol=0, atol=0
            )
        remaining = set(fia_state) - {
            remap_fia_shared_key(name) for name in base_state
        }
        assert remaining
        assert all(name.startswith(FIA_STATE_PREFIX) for name in remaining)


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
    representatives = {
        "pre_fia_public": next(
            name
            for name, value in source.items()
            if name.startswith("model.")
            and int(name.split(".", 2)[1]) < FIA_MODEL_INDEX
            and value.is_floating_point()
        ),
        "shifted_post_fia_shared": next(
            name
            for name, value in source.items()
            if name.startswith("model.")
            and FIA_MODEL_INDEX <= int(name.split(".", 2)[1]) < 28
            and not _is_fdr_private(name)
            and value.is_floating_point()
        ),
        "fdr_private": next(
            name
            for name, value in source.items()
            if _is_fdr_private(name) and value.is_floating_point()
        ),
    }
    target_state = model.state_dict()
    with torch.no_grad():
        for source_name in representatives.values():
            target_state[remap_fia_shared_key(source_name)].add_(1)
    for source_name in representatives.values():
        assert not torch.equal(
            target_state[remap_fia_shared_key(source_name)], source[source_name]
        )

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


def test_fia_artifact_loader_blocks_pickle_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "pickle-executed.txt"
    malicious = tmp_path / "malicious.pt"
    torch.save(_WriteMarkerOnLoad(marker), malicious)

    with pytest.raises(Exception):
        lrs_system._load_fia_artifact(_small_fia_target(), malicious)

    assert not marker.exists()


@pytest.mark.parametrize(
    ("malformation", "error"),
    [
        ("missing_source", "only model.22 FIA tensors may remain private"),
        ("unexpected_source", "shared keys are missing after FIA insertion"),
        ("shape", "shared tensor shape changed"),
        ("dtype", "shared tensor dtype changed"),
    ],
)
def test_fia_initial_state_rejects_every_target_incompatibility(
    malformation: str,
    error: str,
) -> None:
    source = _small_non_fia_state()
    target = _small_fia_target()
    if malformation == "missing_source":
        source.pop("model.21.weight")
    elif malformation == "unexpected_source":
        source["model.30.unexpected"] = torch.ones(1)
    elif malformation == "shape":
        source["model.0.weight"] = torch.ones(3)
    elif malformation == "dtype":
        source["model.21.weight"] = torch.ones(2, dtype=torch.int64)
    else:
        raise AssertionError(f"unhandled test malformation: {malformation}")
    artifact = _build_small_non_fia_artifact(source)
    validate_fdr_initial_state(artifact)

    with pytest.raises(ValueError, match=error):
        load_fia_initial_state(target, artifact)  # type: ignore[arg-type]


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
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    artifact_path = tmp_path / "valid-initial-state.pt"
    torch.save(
        _build_small_non_fia_artifact(_small_non_fia_state()), artifact_path
    )

    class _ModelDouble:
        def __init__(self, cfg: Path, **kwargs: Any) -> None:
            captured["cfg"] = cfg
            captured.update(kwargs)

    monkeypatch.setattr(lrs_system, model_name, _ModelDouble)
    if arm == "g":
        def _standard_loader_spy(
            model: Any,
            path: Path,
            *,
            variant: str,
        ) -> None:
            validate_fdr_initial_state(
                torch.load(path, map_location="cpu", weights_only=False)
            )
            captured["loader"] = "standard"
            captured["loader_model"] = model
            captured["loader_path"] = path
            captured["loader_variant"] = variant

        monkeypatch.setattr(lrs_system, "_load_initial_state", _standard_loader_spy)
    else:
        def _strict_loader_spy(model: Any, artifact: dict[str, Any]) -> None:
            validate_fdr_initial_state(artifact)
            captured["loader"] = "strict-remapped"
            captured["loader_model"] = model

        monkeypatch.setattr(lrs_system, "load_fia_initial_state", _strict_loader_spy)
    trainer = trainer_type.__new__(trainer_type)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 37
    trainer.initial_state_path = artifact_path

    model = trainer.get_model(cfg="ignored.yaml", weights=None, verbose=True)

    assert isinstance(model, _ModelDouble)
    assert captured["cfg"] == ARM_CONFIGS[arm]
    assert captured["nc"] == 10
    assert captured["ch"] == 3
    assert captured["verbose"] is True
    assert captured["private_seed"] == expected_seeds[0]
    assert captured["loader_model"] is model
    if arm == "g":
        assert captured["loader"] == "standard"
        assert captured["loader_path"] == artifact_path
        assert captured["loader_variant"] == "fdr"
    else:
        assert captured["loader"] == "strict-remapped"
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

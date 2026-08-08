from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from src.fdr_head import FDRRTDETRDecoder
from src.frequency_cm import FrequencyCM
from src.rtdetr_fdr import FDRRTDETRDetectionModel
from src.rtdetr_fdr_frequencycm import (
    FDR_FREQUENCYCM_MODEL_CFG,
    FDRFrequencyCMDetectionModel,
    FDRFrequencyCMTrainer,
    load_fdr_frequencycm_initial_state,
    remap_fdr_decoder_key,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "rtdetr-l-fdr.yaml"


@pytest.fixture(scope="module")
def model_pair() -> tuple[FDRRTDETRDetectionModel, FDRFrequencyCMDetectionModel]:
    torch.manual_seed(41)
    baseline = FDRRTDETRDetectionModel(nc=10, verbose=False)
    torch.manual_seed(41)
    method = FDRFrequencyCMDetectionModel(nc=10, verbose=False)
    return baseline, method


def test_original_fdr_yaml_remains_byte_unchanged() -> None:
    assert hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest().upper() == (
        "736FA7E6DD0A6DD6207AD4BE4E2E170985AD19BC06AC0E5BA69967245CFEC600"
    )


def test_frequencycm_is_one_standalone_yaml_layer(model_pair) -> None:
    _, method = model_pair

    assert len(method.model) == 30
    assert isinstance(method.model[28], FrequencyCM)
    assert isinstance(method.model[29], FDRRTDETRDecoder)
    assert method.yaml["head"][-2] == [-1, 1, "FrequencyCM", [256, 20_000]]
    assert method.yaml["head"][-1][0] == [21, 24, 28]
    assert sum(isinstance(module, FrequencyCM) for module in method.modules()) == 1


def test_decoder_key_alias_is_the_only_shared_structural_rename(model_pair) -> None:
    baseline, method = model_pair
    baseline_state = baseline.state_dict()
    method_state = method.state_dict()
    mapped = {remap_fdr_decoder_key(name): value for name, value in baseline_state.items()}

    private = sorted(set(method_state) - set(mapped))
    assert private
    assert all(name.startswith("model.28.") for name in private)
    assert set(mapped).issubset(method_state)
    for name, expected in mapped.items():
        torch.testing.assert_close(method_state[name], expected, rtol=0, atol=0)


def test_identity_frequencycm_preserves_initialized_fdr_prediction(model_pair) -> None:
    baseline, method = model_pair
    baseline.eval()
    method.eval()
    value = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        baseline_output = baseline(value)[0]
        method_output = method(value)[0]

    torch.testing.assert_close(method_output, baseline_output, rtol=0, atol=0)


def test_frequencycm_loader_accepts_only_declared_new_layer_keys(model_pair, monkeypatch) -> None:
    baseline, method = model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_frequencycm.validate_fdr_initial_state",
        lambda _artifact: None,
    )
    artifact = {
        "fdr_public_state": baseline.state_dict(),
        "private_state": {},
    }

    report = load_fdr_frequencycm_initial_state(method, artifact)

    assert report["shared_mismatch_count"] == 0
    assert report["shared_tensor_count"] == len(baseline.state_dict())
    assert report["private_keys"]
    assert all(name.startswith("model.28.") for name in report["private_keys"])


def test_frequencycm_trainer_partitions_disjoint_gradient_groups(model_pair) -> None:
    _, model = model_pair
    trainer = object.__new__(FDRFrequencyCMTrainer)
    trainer.model = model

    groups = trainer.gradient_parameter_groups()

    assert set(groups) == {
        "gradient_norm",
        "fdr_gradient_norm",
        "frequencycm_gradient_norm",
    }
    identifiers = [{id(parameter) for parameter in parameters} for parameters in groups.values()]
    assert all(identifiers)
    assert identifiers[0].isdisjoint(identifiers[1])
    assert identifiers[0].isdisjoint(identifiers[2])
    assert identifiers[1].isdisjoint(identifiers[2])
    assert set.union(*identifiers) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_frequencycm_config_path_is_dedicated() -> None:
    assert FDR_FREQUENCYCM_MODEL_CFG.name == "rtdetr-l-fdr-frequencycm.yaml"
    assert FDR_FREQUENCYCM_MODEL_CFG != BASE_CONFIG

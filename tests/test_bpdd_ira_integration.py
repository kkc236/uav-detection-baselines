from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.fdr_head import FDRRTDETRDecoder
from src.ira import IRA
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, FDRBPDDTrainer
from src.rtdetr_fdr_bpdd_ira import (
    BPDD_IRA_MODEL_CFG,
    FDRBPDDIRADetectionModel,
    FDRBPDDIRATrainer,
    load_fdr_bpdd_ira_initial_state,
    remap_bpdd_ira_shared_key,
)


BPDD_CFG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-bpdd.yaml"


@pytest.fixture(scope="module")
def model_pair() -> tuple[FDRBPDDDetectionModel, FDRBPDDIRADetectionModel]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9031)
        bpdd = FDRBPDDDetectionModel(BPDD_CFG, nc=10, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9031)
        combined = FDRBPDDIRADetectionModel(
            BPDD_IRA_MODEL_CFG,
            nc=10,
            verbose=False,
            ira_private_seed=20_000,
        )
    return bpdd, combined


def _artifact(state: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "fdr_public_state": dict(state),
        "private_state": {},
    }


def test_combined_model_has_the_exact_isolated_graph(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDIRADetectionModel],
) -> None:
    _bpdd, combined = model_pair

    assert isinstance(combined, FDRBPDDDetectionModel)
    assert len(combined.model) == 30
    assert isinstance(combined.model[22], IRA)
    assert combined.model[22].f == 21
    assert combined.model[23].f == 21
    assert combined.model[24].f == [23, 17]
    assert combined.model[25].f == -1
    assert combined.model[26].f == 25
    assert combined.model[27].f == [26, 12]
    assert combined.model[28].f == -1
    assert isinstance(combined.model[29], FDRRTDETRDecoder)
    assert combined.model[29].f == [22, 25, 28]
    assert sum(isinstance(module, IRA) for module in combined.modules()) == 1


def test_shared_state_loader_leaves_only_ira_private_keys(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDIRADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_ira.validate_fdr_initial_state",
        lambda _artifact: None,
    )

    report = load_fdr_bpdd_ira_initial_state(combined, _artifact(bpdd.state_dict()))

    assert report["shared_mismatch_count"] == 0
    assert report["shared_tensor_count"] == len(bpdd.state_dict())
    assert report["missing_keys"] == report["ira_private_keys"]
    assert report["ira_private_keys"]
    assert all(name.startswith("model.22.") for name in report["ira_private_keys"])
    combined_state = combined.state_dict()
    for name, expected in bpdd.state_dict().items():
        actual = combined_state[remap_bpdd_ira_shared_key(name)]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_gate_eval_output_is_bit_exact_to_bpdd_after_shared_load(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDIRADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_ira.validate_fdr_initial_state",
        lambda _artifact: None,
    )
    load_fdr_bpdd_ira_initial_state(combined, _artifact(bpdd.state_dict()))
    bpdd.eval()
    combined.eval()
    image = torch.zeros((1, 3, 128, 128))

    with torch.inference_mode():
        bpdd_output, bpdd_raw = bpdd(image)
        combined_output, combined_raw = combined(image)

    torch.testing.assert_close(combined_output, bpdd_output, rtol=0, atol=0)
    for actual, expected in zip(combined_raw[:-1], bpdd_raw[:-1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert combined_raw[-1] is bpdd_raw[-1] is None


def test_ira_private_initialization_uses_only_the_declared_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = FDRBPDDIRADetectionModel(
            nc=10, verbose=False, ira_private_seed=20_007
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        second = FDRBPDDIRADetectionModel(
            nc=10, verbose=False, ira_private_seed=20_007
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        third = FDRBPDDIRADetectionModel(
            nc=10, verbose=False, ira_private_seed=20_008
        )

    first_state = first.model[22].state_dict()
    second_state = second.model[22].state_dict()
    third_state = third.model[22].state_dict()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)
    assert any(
        not torch.equal(first_state[name], third_state[name])
        for name in first_state
        if name != "residual_scale"
    )


def test_trainer_subclasses_bpdd_and_partitions_all_parameters(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDIRADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    trainer = FDRBPDDIRATrainer.__new__(FDRBPDDIRATrainer)
    trainer.model = combined

    groups = trainer.gradient_parameter_groups()

    assert issubclass(FDRBPDDIRATrainer, FDRBPDDTrainer)
    assert set(groups) == {
        "gradient_norm",
        "fdr_gradient_norm",
        "ira_gradient_norm",
    }
    identifiers = [
        {id(parameter) for parameter in parameters} for parameters in groups.values()
    ]
    assert all(identifiers)
    assert all(
        identifiers[left].isdisjoint(identifiers[right])
        for left in range(len(identifiers))
        for right in range(left + 1, len(identifiers))
    )
    assert set.union(*identifiers) == {
        id(parameter) for parameter in combined.parameters() if parameter.requires_grad
    }
    assert identifiers[2] == {
        id(parameter) for parameter in combined.model[22].parameters()
    }


def test_zero_gate_first_step_has_rezero_gradient_semantics(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDIRADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    ira = combined.model[22]
    ira.zero_grad(set_to_none=True)
    feature = torch.randn(2, 256, 8, 8, requires_grad=True)

    ira(feature).square().mean().backward()

    assert ira.residual_scale.grad is not None
    assert torch.isfinite(ira.residual_scale.grad)
    assert torch.count_nonzero(ira.residual_scale.grad) > 0
    for name, parameter in ira.named_parameters():
        if name == "residual_scale":
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == 0


def test_trainer_uses_fdr_and_ira_private_seed_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, int] = {}

    class FakeModel:
        def __init__(self, *_args, private_seed: int, ira_private_seed: int, **_kwargs):
            seen["fdr"] = private_seed
            seen["ira"] = ira_private_seed

    monkeypatch.setattr("src.rtdetr_fdr_bpdd_ira.FDRBPDDIRADetectionModel", FakeModel)
    trainer = FDRBPDDIRATrainer.__new__(FDRBPDDIRATrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 17
    trainer.initial_state_path = None

    trainer.get_model(verbose=False)

    assert seen == {"fdr": 10_017, "ira": 20_017}

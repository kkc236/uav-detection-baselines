from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.fdr_head import FDRRTDETRDecoder
from src.fia import FIA
from src.rtdetr_fdr import FDRRTDETRDetectionModel
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, FDRBPDDTrainer
from src.rtdetr_fdr_bpdd_fia import (
    BPDD_FIA_MODEL_CFG,
    FDRBPDDFIADetectionModel,
    FDRBPDDFIATrainer,
    load_fdr_bpdd_fia_initial_state,
    remap_bpdd_fia_shared_key,
)


BPDD_CFG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-bpdd.yaml"


@pytest.fixture(scope="module")
def model_pair() -> tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9031)
        bpdd = FDRBPDDDetectionModel(BPDD_CFG, nc=10, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9031)
        combined = FDRBPDDFIADetectionModel(
            BPDD_FIA_MODEL_CFG,
            nc=10,
            verbose=False,
            fia_private_seed=20_000,
        )
    return bpdd, combined


def _artifact(state: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "fdr_public_state": dict(state),
        "private_state": {},
    }


def test_combined_model_has_the_exact_isolated_graph(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
) -> None:
    _bpdd, combined = model_pair

    assert isinstance(combined, FDRBPDDDetectionModel)
    assert len(combined.model) == 30
    assert isinstance(combined.model[22], FIA)
    assert combined.model[22].f == 21
    assert combined.model[23].f == 21
    assert combined.model[24].f == [23, 17]
    assert combined.model[25].f == -1
    assert combined.model[26].f == 25
    assert combined.model[27].f == [26, 12]
    assert combined.model[28].f == -1
    assert isinstance(combined.model[29], FDRRTDETRDecoder)
    assert combined.model[29].f == [22, 25, 28]
    assert sum(isinstance(module, FIA) for module in combined.modules()) == 1


def test_shared_state_loader_leaves_only_fia_private_keys(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_fia.validate_fdr_initial_state",
        lambda _artifact: None,
    )

    report = load_fdr_bpdd_fia_initial_state(combined, _artifact(bpdd.state_dict()))

    assert report["shared_mismatch_count"] == 0
    assert report["shared_tensor_count"] == len(bpdd.state_dict())
    assert report["missing_keys"] == report["fia_private_keys"]
    assert report["fia_private_keys"]
    assert all(name.startswith("model.22.") for name in report["fia_private_keys"])
    combined_state = combined.state_dict()
    for name, expected in bpdd.state_dict().items():
        actual = combined_state[remap_bpdd_fia_shared_key(name)]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_gate_eval_output_is_bit_exact_to_bpdd_after_shared_load(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_fia.validate_fdr_initial_state",
        lambda _artifact: None,
    )
    load_fdr_bpdd_fia_initial_state(combined, _artifact(bpdd.state_dict()))
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


def test_fia_private_initialization_uses_only_the_declared_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = FDRBPDDFIADetectionModel(
            nc=10, verbose=False, fia_private_seed=20_007
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        second = FDRBPDDFIADetectionModel(
            nc=10, verbose=False, fia_private_seed=20_007
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        third = FDRBPDDFIADetectionModel(
            nc=10, verbose=False, fia_private_seed=20_008
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
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    trainer = FDRBPDDFIATrainer.__new__(FDRBPDDFIATrainer)
    trainer.model = combined

    groups = trainer.gradient_parameter_groups()

    assert issubclass(FDRBPDDFIATrainer, FDRBPDDTrainer)
    assert set(groups) == {
        "gradient_norm",
        "fdr_gradient_norm",
        "fia_gradient_norm",
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
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    fia = combined.model[22]
    fia.zero_grad(set_to_none=True)
    feature = torch.randn(2, 256, 8, 8, requires_grad=True)

    fia(feature).square().mean().backward()

    assert fia.residual_scale.grad is not None
    assert torch.isfinite(fia.residual_scale.grad)
    assert torch.count_nonzero(fia.residual_scale.grad) > 0
    for name, parameter in fia.named_parameters():
        if name == "residual_scale":
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == 0


def test_trainer_uses_fdr_and_fia_private_seed_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, int] = {}

    class FakeModel:
        def __init__(self, *_args, private_seed: int, fia_private_seed: int, **_kwargs):
            seen["fdr"] = private_seed
            seen["fia"] = fia_private_seed

    monkeypatch.setattr("src.rtdetr_fdr_bpdd_fia.FDRBPDDFIADetectionModel", FakeModel)
    trainer = FDRBPDDFIATrainer.__new__(FDRBPDDFIATrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 17
    trainer.initial_state_path = None

    trainer.get_model(verbose=False)

    assert seen == {"fdr": 10_017, "fia": 20_017}


def _bare_trainer() -> FDRBPDDFIATrainer:
    trainer = FDRBPDDFIATrainer.__new__(FDRBPDDFIATrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None
    return trainer


def test_resume_rejects_plain_bpdd_checkpoint(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
) -> None:
    bpdd, _combined = model_pair

    with pytest.raises(ValueError, match=r"exact combined FDR\+BPDD\+FIA"):
        _bare_trainer().get_model(weights=bpdd, verbose=False)


def test_resume_rejects_plain_fdr_checkpoint() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9031)
        fdr = FDRRTDETRDetectionModel(nc=10, verbose=False)

    with pytest.raises(ValueError, match=r"exact combined FDR\+BPDD\+FIA"):
        _bare_trainer().get_model(weights=fdr, verbose=False)


def test_resume_rejects_incomplete_or_shifted_combined_state(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    incomplete = dict(combined.state_dict())
    incomplete.pop(next(iter(incomplete)))

    with pytest.raises(ValueError, match=r"exact combined FDR\+BPDD\+FIA"):
        _bare_trainer().get_model(
            weights={"state_dict": incomplete},
            verbose=False,
        )

    shifted = {
        name.replace("model.29.", "model.28.", 1): tensor
        for name, tensor in combined.state_dict().items()
    }
    with pytest.raises(ValueError, match=r"exact combined FDR\+BPDD\+FIA"):
        _bare_trainer().get_model(
            weights={"state_dict": shifted},
            verbose=False,
        )


def test_exact_combined_resume_round_trips_every_tensor(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    with torch.no_grad():
        combined.fia.residual_scale.fill_(0.25)

    resumed = _bare_trainer().get_model(weights=combined, verbose=False)

    assert isinstance(resumed, FDRBPDDFIADetectionModel)
    assert resumed.state_dict().keys() == combined.state_dict().keys()
    for name, expected in combined.state_dict().items():
        torch.testing.assert_close(
            resumed.state_dict()[name], expected, rtol=0, atol=0
        )


def test_exact_combined_state_mapping_round_trips_every_tensor(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDFIADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    with torch.no_grad():
        combined.fia.residual_scale.fill_(-0.125)
    state = {
        name: tensor.detach().clone()
        for name, tensor in combined.state_dict().items()
    }

    resumed = _bare_trainer().get_model(
        weights={"state_dict": state},
        verbose=False,
    )

    for name, expected in state.items():
        torch.testing.assert_close(
            resumed.state_dict()[name], expected, rtol=0, atol=0
        )


def test_combined_construction_matches_plain_bpdd_rng_and_shared_trajectory() -> None:
    torch.manual_seed(58_031)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(68_031)
    initial_cpu = torch.random.get_rng_state().clone()
    initial_cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )

    bpdd = FDRBPDDDetectionModel(BPDD_CFG, nc=10, verbose=False)
    expected_cpu = torch.random.get_rng_state().clone()
    expected_cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )

    torch.random.set_rng_state(initial_cpu)
    if initial_cuda:
        torch.cuda.set_rng_state_all(initial_cuda)
    combined = FDRBPDDFIADetectionModel(
        BPDD_FIA_MODEL_CFG,
        nc=10,
        verbose=False,
        fia_private_seed=20_000,
    )

    assert torch.equal(torch.random.get_rng_state(), expected_cpu)
    actual_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    assert len(actual_cuda) == len(expected_cuda)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_cuda, expected_cuda, strict=True)
    )
    combined_state = combined.state_dict()
    for name, expected in bpdd.state_dict().items():
        torch.testing.assert_close(
            combined_state[remap_bpdd_fia_shared_key(name)],
            expected,
            rtol=0,
            atol=0,
        )

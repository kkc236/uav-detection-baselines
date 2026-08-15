from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from ultralytics.nn import tasks as ultralytics_tasks

from src.fdr_head import FDRRTDETRDecoder
from src.pr_ira import PRIRA
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, FDRBPDDTrainer
from src.rtdetr_fdr_bpdd_ira import remap_bpdd_ira_shared_key
from src.rtdetr_fdr_bpdd_pr_ira import (
    BPDD_PR_IRA_MODEL_CFG,
    FDRBPDDPRIRADetectionModel,
    FDRBPDDPRIRATrainer,
    load_fdr_bpdd_pr_ira_initial_state,
    remap_bpdd_pr_ira_shared_key,
)


BPDD_CFG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-bpdd.yaml"
RUN_IDENTITY = {
    "source_sha256": "source",
    "protocol_sha256": "protocol",
    "fdr_protocol_sha256": "fdr-protocol",
    "initial_state_sha256": "initial-state",
    "dataset_sha256": "dataset",
    "screen_subset_sha256": "subset",
    "run_id": "fdr-bpdd-pr-ira-formal100-seed0",
    "stage": "formal",
    "variant": "fdr_bpdd_pr_ira",
    "seed": 0,
}


@pytest.fixture(scope="module")
def model_pair() -> tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19_031)
        bpdd = FDRBPDDDetectionModel(BPDD_CFG, nc=10, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19_031)
        combined = FDRBPDDPRIRADetectionModel(
            BPDD_PR_IRA_MODEL_CFG,
            nc=10,
            verbose=False,
            experiment_seed=0,
        )
    return bpdd, combined


@pytest.fixture(scope="module")
def training_model_pair() -> tuple[
    FDRBPDDDetectionModel,
    FDRBPDDPRIRADetectionModel,
]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29_031)
        bpdd = FDRBPDDDetectionModel(BPDD_CFG, nc=10, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29_031)
        combined = FDRBPDDPRIRADetectionModel(
            BPDD_PR_IRA_MODEL_CFG,
            nc=10,
            verbose=False,
            experiment_seed=0,
        )
    return bpdd, combined


def _artifact(state: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "fdr_public_state": dict(state),
        "private_state": {},
    }


def _yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def _private_state(*, ambient_seed: int, experiment_seed: int) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ambient_seed)
        model = FDRBPDDPRIRADetectionModel(
            nc=10,
            verbose=False,
            experiment_seed=experiment_seed,
        )
    return {
        name: tensor.detach().clone()
        for name, tensor in model.pr_ira.state_dict().items()
    }


def _training_batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.zeros(2, 3, 128, 128),
        "cls": torch.tensor([[1], [2]], dtype=torch.float32),
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.20, 0.20], [0.35, 0.40, 0.15, 0.10]],
            dtype=torch.float32,
        ),
        "batch_idx": torch.tensor([0, 1], dtype=torch.float32),
    }


def test_combined_model_has_exactly_one_declarative_p3_pr_ira(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
) -> None:
    _bpdd, combined = model_pair

    assert isinstance(combined, FDRBPDDDetectionModel)
    assert len(combined.model) == 30
    assert combined.pr_ira is combined.model[22]
    assert combined.model[22].f == 21
    assert combined.model[23].f == 21
    assert isinstance(combined.model[29], FDRRTDETRDecoder)
    assert combined.model[29].f == [22, 25, 28]
    assert sum(isinstance(module, PRIRA) for module in combined.modules()) == 1
    assert ultralytics_tasks.PRIRA is PRIRA


def test_yaml_layer_is_a_reversible_fdr_bpdd_ablation() -> None:
    mature = _yaml(BPDD_CFG)
    combined = _yaml(BPDD_PR_IRA_MODEL_CFG)
    head = combined["head"]
    assert isinstance(head, list)

    assert head[12] == [21, 1, "PRIRA", [256, 0.20]]
    ablated = deepcopy(combined)
    ablated_head = ablated["head"]
    assert isinstance(ablated_head, list)
    ablated_head.pop(12)
    ablated_head[12][0] = -1
    ablated_head[-1][0] = [21, 24, 27]

    assert ablated == mature


def test_shared_state_uses_the_mature_post_insertion_key_shift(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_pr_ira.validate_fdr_initial_state",
        lambda _artifact: None,
    )

    report = load_fdr_bpdd_pr_ira_initial_state(
        combined,
        _artifact(bpdd.state_dict()),
    )

    assert report["shared_mismatch_count"] == 0
    assert report["shared_tensor_count"] == len(bpdd.state_dict())
    assert report["missing_keys"] == report["pr_ira_private_keys"]
    assert report["pr_ira_private_keys"]
    assert all(
        name.startswith("model.22.")
        for name in report["pr_ira_private_keys"]
    )
    combined_state = combined.state_dict()
    for name, expected in bpdd.state_dict().items():
        shifted = remap_bpdd_pr_ira_shared_key(name)
        assert shifted == remap_bpdd_ira_shared_key(name)
        torch.testing.assert_close(
            combined_state[shifted],
            expected,
            rtol=0,
            atol=0,
        )


def test_identity_pr_ira_preserves_outputs_and_all_raw_decoder_tensors(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_pr_ira.validate_fdr_initial_state",
        lambda _artifact: None,
    )
    load_fdr_bpdd_pr_ira_initial_state(combined, _artifact(bpdd.state_dict()))
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


def test_private_initialization_is_ambient_independent_and_seed_sensitive() -> None:
    first = _private_state(ambient_seed=1, experiment_seed=7)
    second = _private_state(ambient_seed=999, experiment_seed=7)
    third = _private_state(ambient_seed=999, experiment_seed=8)

    assert first.keys() == second.keys() == third.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    post_zeroed = {
        "channel_gate.3.weight",
        "channel_gate.3.bias",
        "spatial_gate.weight",
        "spatial_gate.bias",
    }
    assert all(torch.count_nonzero(first[name]) == 0 for name in post_zeroed)
    assert any(
        not torch.equal(first[name], third[name])
        for name in first
        if name not in post_zeroed and name != "amplitude"
    )


def test_generic_private_initialization_keeps_both_gates_exactly_half() -> None:
    model = FDRBPDDPRIRADetectionModel(
        nc=10,
        verbose=False,
        experiment_seed=31,
    )
    module = model.pr_ira
    feature = torch.randn(1, module.channels, 5, 7)

    with torch.inference_mode():
        d_raw = module.local_blocks(feature) - feature
        magnitude = d_raw.abs()
        channel_gate = torch.sigmoid(module.channel_gate(magnitude))
        spatial_gate = torch.sigmoid(
            module.spatial_gate(
                torch.cat(
                    (
                        magnitude.mean(dim=1, keepdim=True),
                        magnitude.amax(dim=1, keepdim=True),
                    ),
                    dim=1,
                )
            )
        )

    torch.testing.assert_close(
        channel_gate,
        torch.full_like(channel_gate, 0.5),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        spatial_gate,
        torch.full_like(spatial_gate, 0.5),
        rtol=0,
        atol=0,
    )


def test_combined_construction_preserves_the_public_rng_trajectory() -> None:
    torch.manual_seed(58_031)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(68_031)
    initial_cpu = torch.random.get_rng_state().clone()
    initial_cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )

    FDRBPDDDetectionModel(BPDD_CFG, nc=10, verbose=False)
    expected_cpu = torch.random.get_rng_state().clone()
    expected_cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )

    torch.random.set_rng_state(initial_cpu)
    if initial_cuda:
        torch.cuda.set_rng_state_all(initial_cuda)
    combined = FDRBPDDPRIRADetectionModel(
        BPDD_PR_IRA_MODEL_CFG,
        nc=10,
        verbose=False,
        experiment_seed=0,
    )

    assert combined.pr_ira_private_seed == 20_000
    assert torch.equal(torch.random.get_rng_state(), expected_cpu)
    actual_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    assert len(actual_cuda) == len(expected_cuda)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_cuda, expected_cuda, strict=True)
    )


def test_trainer_derives_both_private_seed_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, int] = {}

    class FakeModel:
        def __init__(
            self,
            *_args: object,
            private_seed: int,
            experiment_seed: int,
            **_kwargs: object,
        ) -> None:
            seen["fdr"] = private_seed
            seen["experiment"] = experiment_seed

    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_pr_ira.FDRBPDDPRIRADetectionModel",
        FakeModel,
    )
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 17
    trainer.initial_state_path = None

    trainer.get_model(verbose=False)

    assert issubclass(FDRBPDDPRIRATrainer, FDRBPDDTrainer)
    assert seen == {"fdr": 10_017, "experiment": 17}


def _bare_trainer() -> FDRBPDDPRIRATrainer:
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None
    trainer.pr_ira_run_identity = dict(RUN_IDENTITY)
    return trainer


def test_resume_requires_and_round_trips_exact_combined_state(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
) -> None:
    bpdd, combined = model_pair
    with torch.no_grad():
        combined.pr_ira.amplitude.fill_(0.25)

    payload = {
        "model": None,
        "ema": combined,
        "train_args": {
            "pr_ira_run_identity": dict(RUN_IDENTITY),
        },
    }
    resumed = _bare_trainer().get_model(weights=payload, verbose=False)

    assert isinstance(resumed, FDRBPDDPRIRADetectionModel)
    assert resumed.state_dict().keys() == combined.state_dict().keys()
    for name, expected in combined.state_dict().items():
        torch.testing.assert_close(
            resumed.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )

    with pytest.raises(ValueError, match=r"exact combined FDR\+BPDD\+PR-IRA"):
        _bare_trainer().get_model(
            weights={
                "model": bpdd,
                "pr_ira_run_identity": dict(RUN_IDENTITY),
            },
            verbose=False,
        )

    incomplete = dict(combined.state_dict())
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match=r"exact combined FDR\+BPDD\+PR-IRA"):
        _bare_trainer().get_model(
            weights={
                "state_dict": incomplete,
                "pr_ira_run_identity": dict(RUN_IDENTITY),
            },
            verbose=False,
        )


def test_resume_rejects_missing_or_drifted_run_authority(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
) -> None:
    _bpdd, combined = model_pair

    trainer = _bare_trainer()
    trainer.resume = True
    combined.args = {}
    with pytest.raises(ValueError, match="run identity"):
        trainer.get_model(weights=combined, verbose=False)

    drifted = dict(RUN_IDENTITY)
    drifted["dataset_sha256"] = "foreign-dataset"
    with pytest.raises(ValueError, match="resume authority mismatch for dataset_sha256"):
        _bare_trainer().get_model(
            weights={
                "model": combined,
                "pr_ira_run_identity": drifted,
            },
            verbose=False,
        )


def test_real_resume_accepts_exact_module_handoff_only_with_module_authority(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
) -> None:
    bpdd, combined = model_pair
    combined.args = {"pr_ira_run_identity": dict(RUN_IDENTITY)}
    trainer = _bare_trainer()
    trainer.resume = True

    resumed = trainer.get_model(weights=combined, verbose=False)

    assert isinstance(resumed, FDRBPDDPRIRADetectionModel)
    assert trainer._pr_ira_full_resume_authority_validated is False

    trainer.resume = False
    with pytest.raises(ValueError, match="module weights require resume"):
        trainer.get_model(weights=combined, verbose=False)
    trainer.resume = True
    with pytest.raises(ValueError, match=r"exact combined FDR\+BPDD\+PR-IRA"):
        trainer.get_model(weights=bpdd, verbose=False)


def test_module_resume_rejects_missing_or_drifted_args_authority(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
) -> None:
    _bpdd, combined = model_pair
    trainer = _bare_trainer()
    trainer.resume = True
    original_args = getattr(combined, "args", None)
    try:
        combined.args = {}
        with pytest.raises(ValueError, match="run identity"):
            trainer.get_model(weights=combined, verbose=False)

        drifted = dict(RUN_IDENTITY)
        drifted["dataset_sha256"] = "foreign-dataset"
        combined.args = {"pr_ira_run_identity": drifted}
        with pytest.raises(
            ValueError,
            match="resume authority mismatch for dataset_sha256",
        ):
            trainer.get_model(weights=combined, verbose=False)
    finally:
        combined.args = original_args


def test_setup_model_validates_full_resume_payload_before_authorizing(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bpdd, combined = model_pair
    combined.args = {"pr_ira_run_identity": dict(RUN_IDENTITY)}
    drifted = dict(RUN_IDENTITY)
    drifted["dataset_sha256"] = "foreign-dataset"
    checkpoint = {
        "ema": combined,
        "train_args": {"pr_ira_run_identity": drifted},
    }
    calls: list[object] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "setup_model",
        lambda _self: calls.append("parent") or checkpoint,
    )
    trainer = _bare_trainer()
    trainer.resume = True
    trainer.epochs = 100
    trainer._pr_ira_full_resume_authority_validated = False

    with pytest.raises(
        ValueError,
        match="resume authority mismatch for dataset_sha256",
    ):
        trainer.setup_model()

    assert calls == ["parent"]
    assert trainer._pr_ira_full_resume_authority_validated is False


def test_setup_then_resume_requires_clean_state_and_calls_parents_in_order(
    model_pair: tuple[FDRBPDDDetectionModel, FDRBPDDPRIRADetectionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bpdd, combined = model_pair
    combined.args = {"pr_ira_run_identity": dict(RUN_IDENTITY)}
    checkpoint = {
        "epoch": 1,
        "ema": combined,
        "train_args": {"pr_ira_run_identity": dict(RUN_IDENTITY)},
    }
    calls: list[object] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "setup_model",
        lambda _self: calls.append("setup") or checkpoint,
    )
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "resume_training",
        lambda _self, value: calls.append(("resume", value)) or "resumed",
    )
    trainer = _bare_trainer()
    trainer.resume = True
    trainer.epochs = 100
    trainer.model = combined
    combined.clear_pr_ira_firewall_buffer()
    for parameter in combined.parameters():
        parameter.grad = None

    assert trainer.setup_model() is checkpoint
    assert trainer._pr_ira_full_resume_authority_validated is True
    assert trainer.resume_training(checkpoint) == "resumed"
    assert calls == ["setup", ("resume", checkpoint)]

    trainer._pr_ira_full_resume_authority_validated = False
    with pytest.raises(RuntimeError, match="full checkpoint authority"):
        trainer.resume_training(checkpoint)
    trainer._pr_ira_full_resume_authority_validated = True
    next(combined.parameters()).grad = torch.zeros_like(next(combined.parameters()))
    with pytest.raises(RuntimeError, match="gradients must be empty"):
        trainer.resume_training(checkpoint)
    for parameter in combined.parameters():
        parameter.grad = None


def test_combined_loss_exposes_exact_main_and_bpdd_components(
    training_model_pair: tuple[
        FDRBPDDDetectionModel,
        FDRBPDDPRIRADetectionModel,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = training_model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_pr_ira.validate_fdr_initial_state",
        lambda _artifact: None,
    )
    load_fdr_bpdd_pr_ira_initial_state(combined, _artifact(bpdd.state_dict()))
    combined.train()
    combined.pr_ira.set_training_progress(30, 30)
    with torch.no_grad():
        combined.pr_ira.amplitude.zero_()

    torch.manual_seed(44_021)
    total, _displayed = combined.loss(_training_batch())
    expected_main = sum(
        value
        for name, value in combined.last_fdr_losses.items()
        if name != "loss_bpdd"
    )

    assert combined.last_bpdd_loss is combined.last_fdr_losses["loss_bpdd"]
    torch.testing.assert_close(combined.last_main_loss, expected_main, rtol=0, atol=0)
    torch.testing.assert_close(
        total,
        combined.last_main_loss + combined.last_bpdd_loss,
        rtol=0,
        atol=0,
    )


def test_zero_amplitude_combined_loss_is_bit_exact_to_mature_bpdd(
    training_model_pair: tuple[
        FDRBPDDDetectionModel,
        FDRBPDDPRIRADetectionModel,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpdd, combined = training_model_pair
    monkeypatch.setattr(
        "src.rtdetr_fdr_bpdd_pr_ira.validate_fdr_initial_state",
        lambda _artifact: None,
    )
    load_fdr_bpdd_pr_ira_initial_state(combined, _artifact(bpdd.state_dict()))
    bpdd.train()
    combined.train()
    combined.pr_ira.set_training_progress(30, 30)
    with torch.no_grad():
        combined.pr_ira.amplitude.zero_()
    batch = _training_batch()

    torch.manual_seed(55_031)
    public_rng = torch.random.get_rng_state().clone()
    mature_total, mature_displayed = bpdd.loss(batch)
    mature_losses = {
        name: value.detach().clone()
        for name, value in bpdd.last_fdr_losses.items()
    }
    mature_statistics = {
        name: value.detach().clone()
        for name, value in bpdd.last_bpdd_statistics.items()
    }

    torch.random.set_rng_state(public_rng)
    combined_total, combined_displayed = combined.loss(batch)

    assert combined.last_fdr_losses.keys() == mature_losses.keys()
    torch.testing.assert_close(combined_total, mature_total, rtol=0, atol=0)
    torch.testing.assert_close(combined_displayed, mature_displayed, rtol=0, atol=0)
    for name, expected in mature_losses.items():
        torch.testing.assert_close(
            combined.last_fdr_losses[name],
            expected,
            rtol=0,
            atol=0,
        )
    assert combined.last_bpdd_statistics.keys() == mature_statistics.keys()
    for name, expected in mature_statistics.items():
        torch.testing.assert_close(
            combined.last_bpdd_statistics[name],
            expected,
            rtol=0,
            atol=0,
        )

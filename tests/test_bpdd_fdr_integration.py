from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.bpdd_loss import BPDDDetectionLoss, BPDDOptions
from src.rtdetr_fdr import FDRRTDETRDetectionModel, FDRTrainer
from src.rtdetr_fdr_bpdd import (
    BPDD_MODEL_CFG,
    FDRBPDDDetectionModel,
    FDRBPDDTrainer,
)


FDR_CFG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr.yaml"


def _models() -> tuple[FDRRTDETRDetectionModel, FDRBPDDDetectionModel]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8019)
        fdr = FDRRTDETRDetectionModel(FDR_CFG, nc=10, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8019)
        bpdd = FDRBPDDDetectionModel(BPDD_MODEL_CFG, nc=10, verbose=False)
    return fdr, bpdd


@pytest.fixture(scope="module")
def paired_models() -> tuple[FDRRTDETRDetectionModel, FDRBPDDDetectionModel]:
    return _models()


def test_bpdd_model_has_identical_state_and_parameter_contract(
    paired_models: tuple[FDRRTDETRDetectionModel, FDRBPDDDetectionModel],
) -> None:
    fdr, bpdd = paired_models
    fdr_state = fdr.state_dict()
    bpdd_state = bpdd.state_dict()

    assert fdr_state.keys() == bpdd_state.keys()
    for name in fdr_state:
        torch.testing.assert_close(bpdd_state[name], fdr_state[name], rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in fdr.parameters()) == sum(
        parameter.numel() for parameter in bpdd.parameters()
    )


def test_bpdd_criterion_uses_only_the_frozen_yaml_options(
    paired_models: tuple[FDRRTDETRDetectionModel, FDRBPDDDetectionModel],
) -> None:
    _fdr, bpdd = paired_models
    criterion = bpdd.init_criterion()

    assert isinstance(criterion, BPDDDetectionLoss)
    assert criterion.bpdd_options == BPDDOptions(
        enabled=True,
        weight=0.5,
        temperature=0.5,
        margin=0.02,
        eps=1e-6,
    )


def test_bpdd_eval_prediction_is_bit_exact_to_fdr(
    paired_models: tuple[FDRRTDETRDetectionModel, FDRBPDDDetectionModel],
) -> None:
    fdr, bpdd = paired_models
    bpdd.load_state_dict(fdr.state_dict(), strict=True)
    fdr.eval()
    bpdd.eval()
    # 128 is the smallest stride-compatible square with at least 300 encoder
    # positions (16x16 + 8x8 + 4x4 = 336) for the frozen Query count.
    image = torch.zeros((1, 3, 128, 128))

    with torch.inference_mode():
        fdr_output, fdr_raw = fdr(image)
        bpdd_output, bpdd_raw = bpdd(image)

    torch.testing.assert_close(bpdd_output, fdr_output, rtol=0, atol=0)
    for actual, expected in zip(bpdd_raw[:-1], fdr_raw[:-1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bpdd_raw[-1] is fdr_raw[-1] is None


def test_validation_loss_skips_bpdd_because_only_one_decoder_layer_is_emitted(
    paired_models: tuple[FDRRTDETRDetectionModel, FDRBPDDDetectionModel],
) -> None:
    _fdr, bpdd = paired_models
    bpdd.eval()
    image = torch.zeros((1, 3, 128, 128))
    batch = {
        "img": image,
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0.0]),
    }

    with torch.inference_mode():
        predictions = bpdd.predict(image)
        total, displayed = bpdd.loss(batch, predictions)

    assert torch.isfinite(total)
    assert torch.isfinite(displayed).all()
    assert "loss_bpdd" not in bpdd.last_fdr_losses
    assert bpdd.last_bpdd_statistics == {}


def test_bpdd_trainer_is_the_same_fixed_fdr_trainer_contract() -> None:
    assert issubclass(FDRBPDDTrainer, FDRTrainer)
    assert FDRBPDDTrainer.controlled_amp_scale == 128.0

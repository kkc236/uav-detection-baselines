from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer as UltralyticsRTDETRTrainer

from src.ebc_qp_config import EBCQPConfig
from src.ebc_qp_decoder import EBCQPDecoder
from src.rtdetr_ebc_qp import (
    EBCQPDetectionModel,
    EBCQPTrainer,
    PairedControlTrainer,
    build_ebc_qp_checkpoint_metadata,
    validate_ebc_qp_checkpoint_metadata,
)


CONFIG = Path(__file__).parents[1] / "configs" / "rtdetr-l-ebc-qp.yaml"


def test_control_overrides_stock_validator_to_use_the_shared_tiny_metric_code():
    assert PairedControlTrainer.get_validator is not UltralyticsRTDETRTrainer.get_validator


def test_model_adds_weighted_losses_but_keeps_stock_encoder_auxiliary_output():
    config = EBCQPConfig()
    model = EBCQPDetectionModel(CONFIG, ch=3, nc=3, verbose=False, ebc_config=config)
    model.train()
    model.set_ebc_progress(3)
    batch = {
        "img": torch.rand(1, 3, 160, 160),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "cls": torch.tensor([[1.0]]),
        "batch_idx": torch.tensor([0.0]),
    }

    total, items = model.loss(batch)
    state = model.ebc_head.last_state
    expected = state.stock_loss + config.lambda_p2 * state.p2_loss.detach() + config.lambda_ebc * state.ebc_loss.detach()

    torch.testing.assert_close(total.detach(), expected)
    assert items.shape == (5,)
    assert state.encoder_aux_source_is_stock


def test_epoch_callback_restores_activation_boundary_after_resume():
    model = EBCQPDetectionModel(CONFIG, ch=3, nc=3, verbose=False)
    ema_model = EBCQPDetectionModel(CONFIG, ch=3, nc=3, verbose=False)
    trainer = object.__new__(EBCQPTrainer)
    trainer.model = model
    trainer.ema = SimpleNamespace(ema=ema_model)
    trainer.epoch = 3

    trainer._set_ebc_progress()

    assert model.ebc_head.ebc_epoch == 3
    assert model.ebc_head.competition_active
    assert ema_model.ebc_head.ebc_epoch == 3
    assert ema_model.ebc_head.competition_active


def test_checkpoint_round_trip_preserves_p2_parameters_and_progress(tmp_path: Path):
    original = _small_head()
    original.set_progress(7)
    with torch.no_grad():
        original.p2_adapter[0].weight.fill_(0.125)
        original.p2_bbox_head.layers[-1].bias.fill_(0.25)
    checkpoint = {
        "model": original.state_dict(),
        "ebc_qp": build_ebc_qp_checkpoint_metadata(original.ebc_config, original.ebc_epoch),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    restored = _small_head()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_ebc_qp_checkpoint_metadata(loaded["ebc_qp"], restored.ebc_config)
    restored.load_state_dict(loaded["model"])
    restored.set_progress(loaded["ebc_qp"]["ebc_epoch"])

    assert restored.ebc_epoch == 7
    _assert_state_dict_equal(restored.p2_adapter, original.p2_adapter)
    _assert_state_dict_equal(restored.p2_bbox_head, original.p2_bbox_head)


def test_resume_rejects_changed_frozen_config():
    config = EBCQPConfig()
    metadata = build_ebc_qp_checkpoint_metadata(config, ebc_epoch=3)
    metadata["config"]["lambda_ebc"] = 0.0

    with pytest.raises(RuntimeError, match="config mismatch"):
        validate_ebc_qp_checkpoint_metadata(metadata, config)


def test_v1_checkpoint_metadata_without_quality_flag_remains_loadable():
    config = EBCQPConfig()
    metadata = build_ebc_qp_checkpoint_metadata(config, ebc_epoch=3)
    metadata["config"].pop("quality_weighted_ebc")

    validate_ebc_qp_checkpoint_metadata(metadata, config)


def _small_head() -> EBCQPDecoder:
    return EBCQPDecoder(
        nc=3,
        ch=(4, 8, 8, 8),
        hd=16,
        nq=8,
        ndp=2,
        nh=4,
        ndl=1,
        d_ffn=32,
        nd=0,
        ebc_config=EBCQPConfig(query_budget=8, p2_candidates=4),
    )


def _assert_state_dict_equal(first: torch.nn.Module, second: torch.nn.Module) -> None:
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])

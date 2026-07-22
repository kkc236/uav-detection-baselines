import json
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


def test_controlled_optimizer_evidence_persists_nonfinite_and_skip_before_failing(tmp_path: Path):
    trainer = object.__new__(PairedControlTrainer)
    trainer.controlled_amp_scale = 256.0
    trainer.save_dir = tmp_path
    trainer._initialize_optimizer_evidence()

    trainer._record_optimizer_evidence(
        {
            "amp_step_skipped": False,
            "amp_scale_before": 256.0,
            "amp_scale_after": 256.0,
            "pure_stock_preclip_norm": 3.0,
        }
    )
    with pytest.raises(RuntimeError, match="skipped optimizer attempt 2"):
        trainer._record_optimizer_evidence(
            {
                "amp_step_skipped": True,
                "amp_scale_before": 256.0,
                "amp_scale_after": 128.0,
                "pure_stock_preclip_norm": float("inf"),
            }
        )

    records = [json.loads(line) for line in trainer.optimizer_evidence_path.read_text().splitlines()]
    assert [record["optimizer_attempt"] for record in records] == [1, 2]
    assert records[1]["pure_stock_preclip_norm"] is None
    assert records[1]["nonfinite_fields"] == ["pure_stock_preclip_norm"]


def test_e1_retains_only_the_three_zero_based_tail_checkpoints(tmp_path: Path):
    trainer = object.__new__(PairedControlTrainer)
    trainer.controlled_amp_scale = 256.0
    trainer.args = SimpleNamespace(epochs=10, save_period=-1)
    trainer.wdir = tmp_path
    trainer.last = tmp_path / "last.pt"
    trainer.last.write_bytes(b"resumable-checkpoint")

    trainer.epoch = 6
    assert trainer._retain_e1_tail_checkpoint() is None
    trainer.epoch = 7
    retained = trainer._retain_e1_tail_checkpoint()

    assert retained == tmp_path / "epoch7.pt"
    assert retained.read_bytes() == trainer.last.read_bytes()
    assert not (tmp_path / "epoch7.pt.tmp").exists()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        trainer._retain_e1_tail_checkpoint()


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


def test_v1_checkpoint_metadata_without_later_optional_flags_remains_loadable():
    config = EBCQPConfig()
    metadata = build_ebc_qp_checkpoint_metadata(config, ebc_epoch=3)
    metadata["config"].pop("quality_weighted_ebc")
    metadata["config"].pop("query_injection_enabled")
    metadata["config"].pop("quality_gated_p2")
    metadata["config"].pop("lambda_quality")

    validate_ebc_qp_checkpoint_metadata(metadata, config)


def test_full_model_configures_gamma_after_yaml_construction():
    config = EBCQPConfig(learnable_fusion_gamma=True)

    model = EBCQPDetectionModel(CONFIG, ch=3, nc=3, verbose=False, ebc_config=config)

    assert isinstance(model.ebc_head.p2_fusion_gamma, torch.nn.Parameter)


def test_a1_single_batch_trains_p2_and_gamma_without_ebc_contribution():
    config = EBCQPConfig(lambda_ebc=0.0, learnable_fusion_gamma=True)
    model = EBCQPDetectionModel(CONFIG, ch=3, nc=3, verbose=False, ebc_config=config)
    model.train()
    model.set_ebc_progress(3)
    batch = {
        "img": torch.rand(1, 3, 160, 160),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "cls": torch.tensor([[1.0]]),
        "batch_idx": torch.tensor([0.0]),
    }

    total, _ = model.loss(batch)
    state = model.ebc_head.last_state
    expected = state.stock_loss + config.lambda_p2 * state.p2_loss.detach()
    torch.testing.assert_close(total.detach(), expected)
    total.backward()

    assert state.ordinary_query_count == config.query_budget
    assert _has_finite_nonzero_gradient(model.ebc_head.p2_adapter)
    assert _has_finite_nonzero_gradient(model.ebc_head.p2_bbox_head)
    gamma_gradient = model.ebc_head.p2_fusion_gamma.grad
    assert gamma_gradient is not None
    assert torch.isfinite(gamma_gradient)
    assert torch.count_nonzero(gamma_gradient) == 1


def test_tsgr_loss_buffers_only_private_and_exact_shallow_gradients():
    config = EBCQPConfig(
        lambda_p2=0.1,
        lambda_quality=0.0,
        lambda_ebc=0.0,
        query_injection_enabled=False,
        p2_c2_grad_scale=0.1,
        contribution_separated_aux_gradients=True,
    )
    model = EBCQPDetectionModel(CONFIG, ch=3, nc=3, verbose=False, ebc_config=config)
    model.train()
    model.set_isolated_auxiliary_gradient_scale(256.0)
    batch = {
        "img": torch.rand(1, 3, 160, 160),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "cls": torch.tensor([[1.0]]),
        "batch_idx": torch.tensor([0.0]),
    }

    total, _items = model.loss(batch)
    state = model.ebc_head.last_state
    torch.testing.assert_close(total.detach(), state.stock_loss, rtol=0, atol=0)
    total.backward()
    buffered, scale = model.pop_isolated_auxiliary_gradients()

    assert scale == 256.0
    assert any(name.startswith("model.0.") or name.startswith("model.1.") for name in buffered)
    assert any(".p2_adapter." in name or ".p2_bbox_head." in name for name in buffered)
    assert not any(name.startswith("model.2.") for name in buffered)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if ".p2_adapter." in name or ".p2_bbox_head." in name
    )

    model.clear_isolated_auxiliary_gradients()
    model.eval()
    with torch.no_grad():
        validation_total, _validation_items = model.loss(batch)
    validation_state = model.ebc_head.last_state
    torch.testing.assert_close(validation_total.detach(), validation_state.stock_loss, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="buffer is empty"):
        model.pop_isolated_auxiliary_gradients()


def test_qg_p2_single_batch_adds_only_weighted_quality_loss_and_quality_parameters():
    config = EBCQPConfig(
        lambda_ebc=0.0,
        learnable_fusion_gamma=True,
        quality_gated_p2=True,
    )
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
    expected = (
        state.stock_loss
        + config.lambda_p2 * state.p2_loss.detach()
        + config.lambda_quality * state.quality_loss.detach()
    )
    torch.testing.assert_close(total.detach(), expected)
    total.backward()

    assert items.shape == (6,)
    assert state.ordinary_query_count == config.query_budget
    assert _has_finite_nonzero_gradient(model.ebc_head.p2_quality_head)
    assert _has_finite_nonzero_gradient(model.ebc_head.p2_adapter)
    assert _has_finite_nonzero_gradient(model.ebc_head.p2_bbox_head)


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


def _has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients) and any(
        torch.count_nonzero(gradient) for gradient in gradients
    )

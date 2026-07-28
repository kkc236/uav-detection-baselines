from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-acr-eg.yaml"


def _integrated_checkpoint() -> dict:
    from src.rtdetr_acr_eg import ACREGDetectionModel

    return {
        "ema": ACREGDetectionModel(_config_path(), nc=10, verbose=False),
        "optimizer": {"state": {0: {"momentum_buffer": torch.ones(1)}}},
        "scaler": {
            "scale": 128.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2**31 - 1,
            "_growth_tracker": 1,
        },
        "epoch": 8,
        "updates": 1,
    }


def _remove_registered_state_entry(model: torch.nn.Module, key: str) -> None:
    module = model
    parts = key.split(".")
    for part in parts[:-1]:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    name = parts[-1]
    if name in module._parameters:
        del module._parameters[name]
    elif name in module._buffers:
        del module._buffers[name]
    else:
        raise AssertionError(f"state entry is not registered: {key}")


def test_query_logit_injection_only_changes_final_detection_queries():
    from src.rtdetr_acr_eg import inject_query_retention_logits

    decoder_scores = torch.zeros(2, 1, 5, 3, requires_grad=True)
    retention_logits = torch.tensor([[[2.0], [-1.0], [0.5]]], requires_grad=True)

    fused = inject_query_retention_logits(
        decoder_scores,
        retention_logits,
        num_queries=3,
        gain=0.2,
    )

    assert torch.equal(fused[0], decoder_scores[0])
    assert torch.equal(fused[1, :, :2], decoder_scores[1, :, :2])
    expected = retention_logits.tanh() * 0.2
    assert torch.allclose(fused[1, :, 2:], expected.expand_as(fused[1, :, 2:]))

    fused.sum().backward()
    assert retention_logits.grad is not None
    assert torch.count_nonzero(retention_logits.grad).item() == retention_logits.numel()


def test_acr_eg_module_is_registered_on_rtdetr_model_subclass():
    from src.rtdetr_acr_eg import ACREGDetectionModel

    assert issubclass(ACREGDetectionModel, torch.nn.Module)
    assert "loss" in ACREGDetectionModel.__dict__
    assert "predict" in ACREGDetectionModel.__dict__


def test_yaml_model_registers_acr_eg_parameters():
    from src.rtdetr_acr_eg import ACREGDetectionModel

    config = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-acr-eg.yaml"
    model = ACREGDetectionModel(config, nc=10, verbose=False)

    assert model.nc == 10
    assert any(name.startswith("acr_eg.") for name in model.state_dict())


def test_training_output_contract_accepts_rtdetr_denoising_metadata():
    from src.rtdetr_acr_eg import _require_raw_training_output

    output = _require_raw_training_output(
        (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            {"dn_num_split": [100, 300]},
        )
    )

    assert output[4]["dn_num_split"] == [100, 300]


def test_integrated_resume_requires_custom_ema_optimizer_scaler_epoch_and_updates():
    from src.rtdetr_acr_eg import validate_acr_eg_resume_checkpoint

    valid = _integrated_checkpoint()
    for key in ("ema", "optimizer", "scaler", "epoch", "updates"):
        checkpoint = dict(valid)
        checkpoint.pop(key)
        with pytest.raises(ValueError, match=f"ACR_EG_RESUME_MISSING_{key.upper()}"):
            validate_acr_eg_resume_checkpoint(checkpoint)


def test_integrated_resume_rejects_stock_rtdetr_checkpoint():
    from src.rtdetr_acr_eg import validate_acr_eg_resume_checkpoint

    checkpoint = _integrated_checkpoint()
    checkpoint["ema"] = torch.nn.Linear(1, 1)

    with pytest.raises(ValueError, match="ACR_EG_RESUME_MODEL_IDENTITY_MISMATCH"):
        validate_acr_eg_resume_checkpoint(checkpoint)


def test_integrated_resume_requires_all_acr_eg_state_keys():
    from src.rtdetr_acr_eg import validate_acr_eg_resume_checkpoint

    checkpoint = _integrated_checkpoint()
    state_keys = set(checkpoint["ema"].state_dict())
    acr_key = next(key for key in state_keys if key.startswith("acr_eg."))
    _remove_registered_state_entry(checkpoint["ema"], acr_key)

    with pytest.raises(ValueError, match="ACR_EG_RESUME_STATE_IDENTITY_MISMATCH"):
        validate_acr_eg_resume_checkpoint(
            checkpoint,
            expected_model_state_keys=state_keys,
        )


def test_resume_model_is_acr_eg_and_loads_integrated_weights():
    from src.rtdetr_acr_eg import ACREGDetectionModel, ACREGFormalTrainer

    source = ACREGDetectionModel(_config_path(), nc=10, verbose=False)
    trainer = object.__new__(ACREGFormalTrainer)
    trainer.model_yaml = _config_path()
    trainer.data = {"nc": 10, "channels": 3}

    loaded = trainer.get_model(cfg="rtdetr-l.yaml", weights=source, verbose=False)

    assert type(loaded) is ACREGDetectionModel
    source_state = source.state_dict()
    loaded_state = loaded.state_dict()
    assert set(loaded_state) == set(source_state)
    for key, value in source_state.items():
        assert torch.equal(loaded_state[key], value), key

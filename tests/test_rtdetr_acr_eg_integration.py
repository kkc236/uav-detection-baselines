from __future__ import annotations

from pathlib import Path

import torch


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

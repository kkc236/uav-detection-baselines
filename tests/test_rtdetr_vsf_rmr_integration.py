from __future__ import annotations

import torch

from src.rtdetr_vsf_rmr import LOSS_NAMES, VSFRMRDetectionModel, VSFRMRTrainer
from src.vsf_rmr import VSFRMR


def synthetic_predictions(class_count: int = 2):
    layers = 6
    queries = 4
    boxes = torch.tensor(
        [[0.40, 0.50, 0.10, 0.10], [0.52, 0.50, 0.10, 0.10], [0.70, 0.70, 0.08, 0.08], [0.9, 0.9, 0.05, 0.05]]
    )
    dec_boxes = boxes.view(1, 1, queries, 4).repeat(layers, 1, 1, 1).requires_grad_()
    dec_scores = torch.zeros((layers, 1, queries, class_count), requires_grad=True)
    enc_boxes = boxes.unsqueeze(0).clone().requires_grad_()
    enc_scores = torch.zeros((1, queries, class_count), requires_grad=True)
    return dec_boxes, dec_scores, enc_boxes, enc_scores, None


def synthetic_batch():
    return {
        "img": torch.zeros((1, 3, 160, 160)),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.40, 0.50, 0.10, 0.10], [0.52, 0.50, 0.10, 0.10]]),
        "batch_idx": torch.tensor([0.0, 0.0]),
    }


def test_custom_model_contains_only_vsf_rmr_innovation():
    model = VSFRMRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    names = [module.__class__.__name__ for module in model.modules()]

    assert names.count("VSFRMR") == 1
    assert "BTDSE" not in names
    assert "P3SamplingProbe" not in names
    assert model.loss_names == LOSS_NAMES


def test_eval_prediction_routes_decoder_features_and_keeps_stock_output():
    model = VSFRMRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    image = torch.rand(1, 3, 160, 160)
    calls: list[list[tuple[int, ...]]] = []

    def record_inputs(module, args):
        calls.append([tuple(value.shape) for value in args[0]])

    handle = model.vsf_rmr.register_forward_pre_hook(record_inputs)
    with torch.no_grad():
        output = model.predict(image)
    handle.remove()

    assert output is not None
    assert calls == [[(1, 256, 20, 20), (1, 256, 10, 10), (1, 256, 5, 5)]]
    assert model.vsf_rmr.peek_auxiliary_state() is None


def test_training_loss_adds_fp32_vsf_items_and_consumes_cache_once():
    model = VSFRMRDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).train()
    auxiliary_features = [
        torch.randn(1, 256, 8, 8),
        torch.randn(1, 256, 4, 4),
        torch.randn(1, 256, 2, 2),
    ]
    model.vsf_rmr(auxiliary_features)

    total, items = model.loss(synthetic_batch(), preds=synthetic_predictions())
    total.backward()

    assert total.dtype == torch.float32
    assert items.shape == (5,)
    assert torch.isfinite(total)
    assert torch.isfinite(items).all()
    assert model.last_vsf_result is not None
    assert model.last_vsf_result.local.dtype == torch.float32
    assert model.last_vsf_result.global_.dtype == torch.float32
    assert model.vsf_rmr.peek_auxiliary_state() is None


def test_validation_loss_has_zero_auxiliary_items_and_no_cache_requirement():
    model = VSFRMRDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).eval()

    with torch.no_grad():
        total, items = model.loss(synthetic_batch(), preds=(None, synthetic_predictions()))

    assert torch.isfinite(total)
    assert items.shape == (5,)
    torch.testing.assert_close(items[-2:], torch.zeros(2))
    assert model.vsf_rmr.peek_auxiliary_state() is None


def test_custom_trainer_get_model_constructs_vsf_model():
    trainer = object.__new__(VSFRMRTrainer)
    trainer.data = {"nc": 3, "channels": 3}
    trainer.lambda_vsf = 0.1

    model = trainer.get_model("rtdetr-l.yaml", weights=None, verbose=False)

    assert isinstance(model, VSFRMRDetectionModel)
    assert isinstance(model.vsf_rmr, VSFRMR)
    assert model.lambda_vsf == 0.1


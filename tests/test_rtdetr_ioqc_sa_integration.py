from __future__ import annotations

import torch
import pytest

from src.ioqc_sa_probe import P3SamplingStatistics
from src.rtdetr_ioqc_sa import IOQCSADetectionModel, prepare_matcher_inputs, regular_query_statistics


def test_stock_rtdetr_l_has_one_final_layer_probe_and_no_btdse():
    model = IOQCSADetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    decoder_head = model.model[-1]

    assert all(module.__class__.__name__ != "BTDSE" for module in model.modules())
    assert model.ioqc_probe.cross_attention is decoder_head.decoder.layers[-1].cross_attn
    assert model.ioqc_probe._hook is not None
    assert model.nc == 10


def test_regular_query_statistics_remove_denoising_prefix():
    statistics = P3SamplingStatistics(
        center=torch.arange(20, dtype=torch.float32).reshape(1, 10, 2),
        extent=torch.ones((1, 10, 2)),
        p3_mass=torch.ones((1, 10)),
        valid=torch.ones((1, 10), dtype=torch.bool),
        p3_shape=(8, 8),
    )

    regular = regular_query_statistics(statistics, dn_meta={"dn_num_split": [4, 6]})

    assert regular.center.shape == (1, 6, 2)
    torch.testing.assert_close(regular.center, statistics.center[:, 4:])
    assert regular.p3_shape == (8, 8)


def test_matcher_inputs_are_fp32_and_contiguous_after_query_slicing():
    boxes = torch.rand(2, 6, 8, dtype=torch.float16)[..., ::2]
    scores = torch.rand(2, 6, 20, dtype=torch.float16)[..., ::2]
    assert not boxes.is_contiguous()
    assert not scores.is_contiguous()

    matcher_boxes, matcher_scores = prepare_matcher_inputs(boxes, scores)

    assert matcher_boxes.dtype == torch.float32
    assert matcher_scores.dtype == torch.float32
    assert matcher_boxes.is_contiguous()
    assert matcher_scores.is_contiguous()


def test_eval_prediction_keeps_stock_output_and_does_not_capture_auxiliary_statistics():
    model = IOQCSADetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    # RT-DETR-L needs at least 300 encoder locations for its fixed Top-K query selection.
    image = torch.rand(1, 3, 160, 160)
    model.ioqc_probe.clear()

    with torch.no_grad():
        output = model.predict(image)

    assert output is not None
    assert model.ioqc_probe.last_statistics is None


def test_model_exposes_five_training_loss_names():
    model = IOQCSADetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)

    assert model.loss_names == (
        "giou_loss",
        "cls_loss",
        "l1_loss",
        "ioqc_comp_loss",
        "ioqc_align_loss",
    )


def synthetic_training_inputs(class_count: int = 2):
    query_count = 4
    layers = 6
    boxes = torch.tensor(
        [[0.40, 0.50, 0.10, 0.10], [0.52, 0.50, 0.10, 0.10], [0.40, 0.50, 0.10, 0.10], [0.9, 0.9, 0.05, 0.05]]
    )
    dec_boxes = boxes.view(1, 1, query_count, 4).repeat(layers, 1, 1, 1).requires_grad_()
    dec_scores = torch.full((layers, 1, query_count, class_count), -5.0)
    dec_scores[:, 0, :3, 0] = 5.0
    dec_scores.requires_grad_()
    enc_boxes = boxes.unsqueeze(0).clone().requires_grad_()
    enc_scores = torch.full((1, query_count, class_count), -5.0, requires_grad=True)
    batch = {
        "img": torch.zeros((1, 3, 160, 160)),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": boxes[:2].clone(),
        "batch_idx": torch.tensor([0.0, 0.0]),
    }
    scale = 0.10 / (12.0**0.5)
    statistics = P3SamplingStatistics(
        center=torch.tensor([[[0.40, 0.50], [0.52, 0.50], [0.40, 0.50], [0.9, 0.9]]], requires_grad=True),
        extent=torch.full((1, query_count, 2), scale, requires_grad=True),
        p3_mass=torch.ones((1, query_count)),
        valid=torch.ones((1, query_count), dtype=torch.bool),
        p3_shape=(80, 80),
    )
    return batch, (dec_boxes, dec_scores, enc_boxes, enc_scores, None), statistics


def test_model_loss_adds_two_finite_fp32_items_and_backpropagates():
    model = IOQCSADetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).train()
    model.set_ioqc_progress(20, 100)
    batch, predictions, statistics = synthetic_training_inputs()
    model.ioqc_probe.last_statistics = statistics

    total, items = model.loss(batch, preds=predictions)
    total.backward()

    assert total.dtype == torch.float32
    assert items.shape == (5,)
    assert torch.isfinite(total)
    assert torch.isfinite(items).all()
    assert statistics.center.grad is not None and torch.isfinite(statistics.center.grad).all()


def test_model_loss_rejects_nonfinite_detection_loss_before_optimizer_step():
    model = IOQCSADetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).train()
    batch, predictions, statistics = synthetic_training_inputs()
    predictions[1].data[0, 0, 0, 0] = float("nan")
    model.ioqc_probe.last_statistics = statistics

    with pytest.raises(FloatingPointError, match="NONFINITE_LOSS"):
        model.loss(batch, preds=predictions)

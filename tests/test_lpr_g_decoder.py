from __future__ import annotations

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.lpr_g_head import LPRGDeformableTransformerDecoder


def _stock_head():
    return RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).model[-1]


def test_training_decoder_returns_stock_boxes_and_stores_refined_side_output() -> None:
    torch.manual_seed(3)
    stock_head = _stock_head()
    wrapped_head = _stock_head()
    wrapped_head.load_state_dict(stock_head.state_dict())
    wrapped_head.decoder = LPRGDeformableTransformerDecoder.from_stock(
        wrapped_head.decoder,
        private_seed=10_000,
    )
    stock_head.train()
    wrapped_head.train()
    features = [torch.randn(1, 256, size, size) for size in (20, 10, 5)]
    batch = {
        "cls": torch.tensor([1]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0]),
        "gt_groups": [1],
    }

    torch.manual_seed(41)
    with torch.no_grad():
        stock = stock_head(features, batch)
    torch.manual_seed(41)
    with torch.no_grad():
        method = wrapped_head(features, batch)

    torch.testing.assert_close(method[0], stock[0], rtol=0, atol=0)
    torch.testing.assert_close(method[1], stock[1], rtol=0, atol=0)
    assert wrapped_head.decoder.last_refined_bboxes is not None
    assert wrapped_head.decoder.last_refined_bboxes.shape[-2:] == method[0][-1].shape[-2:]


def test_eval_output_mode_switches_and_rejects_unknown_mode() -> None:
    decoder = LPRGDeformableTransformerDecoder.from_stock(
        _stock_head().decoder,
        private_seed=10_000,
    )

    decoder.set_output_mode("stock")
    assert decoder.output_mode == "stock"
    decoder.set_output_mode("refined")
    assert decoder.output_mode == "refined"

    try:
        decoder.set_output_mode("candidate")
    except ValueError as error:
        assert "unsupported LPR-G output mode" in str(error)
    else:
        raise AssertionError("unknown output mode was accepted")

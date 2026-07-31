from __future__ import annotations

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.lpr_head import LPRDeformableTransformerDecoder
from src.rtdetr_lpr import LPRRTDETRDetectionModel, LPRTrainer


def test_lpr_model_replaces_only_decoder_container() -> None:
    model = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    head = model.model[-1]

    assert isinstance(head.decoder, LPRDeformableTransformerDecoder)
    assert len(head.decoder.lpr_refiners) == 6
    names = [type(module).__name__ for module in model.modules()]
    forbidden = ("BTDSE", "VSFRMR", "IOQC", "NWD", "P3SamplingProbe")
    assert all(token not in name for name in names for token in forbidden)


def test_zero_gate_model_matches_stock_eval_output_and_loads_stock_state() -> None:
    torch.manual_seed(0)
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    lpr = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()

    incompatible = lpr.load_state_dict(stock.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all("lpr_refiners" in key for key in incompatible.missing_keys)

    image = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        stock_output = stock.predict(image)
        lpr_output = lpr.predict(image)

    torch.testing.assert_close(lpr_output[0], stock_output[0], rtol=0, atol=0)


def test_lpr_state_dict_round_trips_without_missing_keys() -> None:
    source = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    target = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)

    incompatible = target.load_state_dict(source.state_dict(), strict=True)

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_lpr_trainer_constructs_custom_model() -> None:
    trainer = object.__new__(LPRTrainer)
    trainer.data = {"nc": 3, "channels": 3}
    trainer.max_logit_delta = 0.5

    model = trainer.get_model("rtdetr-l.yaml", weights=None, verbose=False)

    assert isinstance(model, LPRRTDETRDetectionModel)
    assert isinstance(model.model[-1].decoder, LPRDeformableTransformerDecoder)

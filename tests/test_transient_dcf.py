from __future__ import annotations

from copy import deepcopy

import pytest
from torch import nn

from src.fdr_head import (
    DistributionConditionedFeedback,
    FDRDeformableTransformerDecoder,
)
from src.transient_dcf import (
    apply_transient_dcf_state,
    find_distribution_feedback_decoder,
    transient_dcf_state,
)


class _Layer(nn.Module):
    pass


class _StockDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(6)])
        self.hidden_dim = 16
        self.num_layers = 6
        self.eval_idx = 5


def _model_with_feedback_decoder() -> nn.Module:
    model = nn.Module()
    model.backbone = nn.Linear(16, 16)
    model.decoder = FDRDeformableTransformerDecoder.from_stock(
        _StockDecoder(),
        pre_bbox_head=nn.Linear(16, 4),
        distribution_feedback=DistributionConditionedFeedback(
            16, private_seed=10_001
        ),
    )
    return model


def test_formal100_schedule_has_frozen_boundaries() -> None:
    assert transient_dcf_state(66, 100).scale == 1.0
    middle = [transient_dcf_state(epoch, 100).scale for epoch in range(67, 75)]
    assert all(0.0 < value < 1.0 for value in middle)
    assert all(left > right for left, right in zip(middle, middle[1:]))
    assert transient_dcf_state(67, 100).frozen is True
    assert transient_dcf_state(74, 100).scale > 0.0
    assert transient_dcf_state(75, 100).scale == 0.0
    assert transient_dcf_state(75, 100).checkpoint_eligible is True


@pytest.mark.parametrize("epoch,total", [(0, 100), (101, 100), (1, 0)])
def test_schedule_rejects_invalid_epoch_domain(epoch: int, total: int) -> None:
    with pytest.raises(ValueError, match="positive training horizon"):
        transient_dcf_state(epoch, total)


def test_apply_state_synchronizes_live_and_ema_and_freezes_only_dcf() -> None:
    live = _model_with_feedback_decoder()
    ema = deepcopy(live)
    state = transient_dcf_state(67, 100)

    apply_transient_dcf_state(live, ema, state)

    live_decoder = find_distribution_feedback_decoder(live)
    ema_decoder = find_distribution_feedback_decoder(ema)
    assert live_decoder.distribution_feedback_scale == state.scale
    assert ema_decoder.distribution_feedback_scale == state.scale
    assert all(
        not parameter.requires_grad
        for parameter in live_decoder.distribution_feedback.parameters()
    )
    assert any(
        parameter.requires_grad
        for name, parameter in live.named_parameters()
        if "distribution_feedback" not in name
    )
    assert all(
        parameter.requires_grad
        for parameter in ema_decoder.distribution_feedback.parameters()
    )


def test_schedule_scale_is_not_model_or_ema_state() -> None:
    live = _model_with_feedback_decoder()
    ema = deepcopy(live)
    apply_transient_dcf_state(live, ema, transient_dcf_state(75, 100))

    assert all(
        not name.endswith("distribution_feedback_scale")
        for name in live.state_dict()
    )
    assert all(
        not name.endswith("distribution_feedback_scale")
        for name in ema.state_dict()
    )


def test_decoder_lookup_requires_exactly_one_feedback_decoder() -> None:
    with pytest.raises(RuntimeError, match="expected one DCF decoder, found 0"):
        find_distribution_feedback_decoder(nn.Linear(4, 4))

    two = nn.ModuleList(
        [_model_with_feedback_decoder(), _model_with_feedback_decoder()]
    )
    with pytest.raises(RuntimeError, match="expected one DCF decoder, found 2"):
        find_distribution_feedback_decoder(two)

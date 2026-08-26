from copy import deepcopy

import pytest
from torch import nn

from src.fdr_head import (
    DistributionConditionedFeedback,
    FDRDeformableTransformerDecoder,
)
from src.persistent_dcf import (
    audit_persistent_dcf_state,
    persistent_dcf_state,
)
from src.rtdetr_fdr import FDRTrainer


class _Layer(nn.Module):
    pass


class _StockDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(6)])
        self.hidden_dim = 16
        self.num_layers = 6
        self.eval_idx = 5


def _model() -> nn.Module:
    model = nn.Module()
    model.backbone = nn.Linear(16, 16)
    model.head = nn.Module()
    model.head.decoder = FDRDeformableTransformerDecoder.from_stock(
        _StockDecoder(),
        pre_bbox_head=nn.Linear(16, 4),
        distribution_feedback=DistributionConditionedFeedback(
            16, private_seed=10_001
        ),
    )
    return model


def test_formal100_is_all_on_and_trainable_for_every_epoch() -> None:
    states = [persistent_dcf_state(epoch, 100) for epoch in range(1, 101)]
    assert all(state.scale == 1.0 for state in states)
    assert all(state.trainable for state in states)
    assert all(state.checkpoint_eligible for state in states)


@pytest.mark.parametrize("epoch,total", [(0, 100), (101, 100), (1, 0)])
def test_persistent_state_rejects_invalid_domain(epoch: int, total: int) -> None:
    with pytest.raises(ValueError, match="positive training horizon"):
        persistent_dcf_state(epoch, total)


def test_audit_requires_live_and_ema_scale_one_and_live_trainability() -> None:
    live = _model()
    ema = deepcopy(live)
    state = persistent_dcf_state(67, 100)
    record = audit_persistent_dcf_state(live, ema, state)
    assert record["live_scale"] == record["ema_scale"] == 1.0
    assert record["live_feedback_trainable"] is True

    live.head.decoder.set_distribution_feedback_scale(0.5)
    with pytest.raises(RuntimeError, match="scale must remain exactly 1.0"):
        audit_persistent_dcf_state(live, ema, state)


def test_dcf_parameters_are_exclusively_in_private_gradient_group() -> None:
    model = _model()
    holder = type("Holder", (), {"model": model})()
    groups = FDRTrainer.gradient_parameter_groups(holder)
    feedback = {
        id(p) for p in model.head.decoder.distribution_feedback.parameters()
    }
    private = {id(p) for p in groups["fdr_gradient_norm"]}
    common = {id(p) for p in groups["gradient_norm"]}
    assert feedback
    assert feedback <= private
    assert feedback.isdisjoint(common)

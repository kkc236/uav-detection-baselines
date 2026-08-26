"""Frozen schedule and live/EMA control for training-only DCF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from torch import nn

from src.fdr_head import FDRDeformableTransformerDecoder


@dataclass(frozen=True)
class TransientDCFState:
    """Exact state of the transient DCF schedule at one paper epoch."""

    paper_epoch: int
    total_epochs: int
    ratio: float
    scale: float
    frozen: bool
    checkpoint_eligible: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def transient_dcf_state(paper_epoch: int, total_epochs: int) -> TransientDCFState:
    """Return full-until-2/3, cosine-withdrawn-until-3/4 DCF state."""

    if total_epochs <= 0 or not 1 <= paper_epoch <= total_epochs:
        raise ValueError("paper epoch must be inside a positive training horizon")
    ratio = paper_epoch / total_epochs
    if ratio <= 2 / 3:
        scale = 1.0
    elif ratio >= 3 / 4:
        scale = 0.0
    else:
        phase = (ratio - 2 / 3) / (3 / 4 - 2 / 3)
        scale = 0.5 * (1.0 + math.cos(math.pi * phase))
    return TransientDCFState(
        paper_epoch=paper_epoch,
        total_epochs=total_epochs,
        ratio=ratio,
        scale=scale,
        frozen=ratio > 2 / 3,
        checkpoint_eligible=ratio >= 3 / 4,
    )


def find_distribution_feedback_decoder(
    model: nn.Module,
) -> FDRDeformableTransformerDecoder:
    """Require exactly one enabled DCF decoder in a model tree."""

    found = [
        module
        for module in model.modules()
        if isinstance(module, FDRDeformableTransformerDecoder)
        and module.distribution_feedback is not None
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one DCF decoder, found {len(found)}")
    return found[0]


def apply_transient_dcf_state(
    live_model: nn.Module,
    ema_model: nn.Module,
    state: TransientDCFState,
) -> None:
    """Synchronize an exact scale and freeze only live DCF parameters."""

    live = find_distribution_feedback_decoder(live_model)
    ema = find_distribution_feedback_decoder(ema_model)
    live.set_distribution_feedback_scale(state.scale)
    ema.set_distribution_feedback_scale(state.scale)
    if state.frozen:
        live.freeze_distribution_feedback()
    if live.distribution_feedback_scale != ema.distribution_feedback_scale:
        raise RuntimeError("live/EMA DCF scales diverged")


__all__ = [
    "TransientDCFState",
    "apply_transient_dcf_state",
    "find_distribution_feedback_decoder",
    "transient_dcf_state",
]

"""All-on state authority for gradient-decoupled persistent DCF."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from src.transient_dcf import find_distribution_feedback_decoder


@dataclass(frozen=True)
class PersistentDCFState:
    """Immutable expected DCF state at one paper epoch."""

    paper_epoch: int
    total_epochs: int
    scale: float = 1.0
    trainable: bool = True
    checkpoint_eligible: bool = True

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def persistent_dcf_state(
    paper_epoch: int, total_epochs: int
) -> PersistentDCFState:
    """Return the all-on state for a valid epoch in the training horizon."""

    if total_epochs <= 0 or not 1 <= paper_epoch <= total_epochs:
        raise ValueError("paper epoch must be inside a positive training horizon")
    return PersistentDCFState(paper_epoch=paper_epoch, total_epochs=total_epochs)


def audit_persistent_dcf_state(
    live_model: nn.Module,
    ema_model: nn.Module,
    state: PersistentDCFState,
) -> dict[str, int | float | bool]:
    """Verify that persistent DCF remains all-on without mutating either model."""

    live = find_distribution_feedback_decoder(live_model)
    ema = find_distribution_feedback_decoder(ema_model)
    live_scale = float(live.distribution_feedback_scale)
    ema_scale = float(ema.distribution_feedback_scale)
    if live_scale != 1.0 or ema_scale != 1.0 or state.scale != 1.0:
        raise RuntimeError("persistent DCF scale must remain exactly 1.0")
    trainable = all(
        parameter.requires_grad
        for parameter in live.distribution_feedback.parameters()
    )
    if not trainable or not state.trainable:
        raise RuntimeError("persistent DCF parameters must remain trainable")
    return {
        **state.to_dict(),
        "live_scale": live_scale,
        "ema_scale": ema_scale,
        "live_feedback_trainable": trainable,
    }


__all__ = [
    "PersistentDCFState",
    "audit_persistent_dcf_state",
    "persistent_dcf_state",
]

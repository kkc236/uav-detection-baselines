"""FDR decoder wrapper that privately refines only its incoming P3 tensor."""

from __future__ import annotations

from typing import Any
from torch import Tensor

from src.fdr_head import FDRRTDETRDecoder
from src.ra_glgm import RAGLGM


class RAFDRRTDETRDecoder(FDRRTDETRDecoder):
    """FDR head with RA-GLGM contained under the existing decoder graph node."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ra_glgm = RAGLGM(
            channels=256,
            hidden_channels=192,
            route_groups=8,
            max_residual_scale=0.5,
            private_seed=20_000,
        )

    def forward(
        self,
        x: list[Tensor] | tuple[Tensor, ...],
        batch: dict[str, Any] | None = None,
    ) -> tuple:
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("RA-FDR decoder requires exactly P3/P4/P5 inputs")
        if x[0].ndim != 4 or x[0].shape[1] != self.ra_glgm.channels:
            raise ValueError("RA-FDR decoder P3 feature contract changed")
        refined = list(x)
        refined[0] = self.ra_glgm(refined[0])
        return super().forward(refined, batch=batch)

__all__ = ["RAFDRRTDETRDecoder"]

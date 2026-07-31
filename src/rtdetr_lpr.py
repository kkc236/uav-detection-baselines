"""Ultralytics RT-DETR integration for localization-prior refinement."""

from __future__ import annotations

from pathlib import Path

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

from src.lpr_head import LPRDeformableTransformerDecoder
from src.rtdetr_vsf_rmr import apply_resume_runtime_overrides


class LPRRTDETRDetectionModel(RTDETRDetectionModel):
    """Stock RT-DETR whose decoder outputs are refined by LPR heads."""

    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        max_logit_delta: float = 0.5,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        head.decoder = LPRDeformableTransformerDecoder.from_stock(
            head.decoder,
            max_logit_delta=max_logit_delta,
        )
        self.nc = self.yaml["nc"]
        self.max_logit_delta = float(max_logit_delta)


class LPRTrainer(RTDETRTrainer):
    """RT-DETR trainer that constructs the repository-owned LPR model."""

    def __init__(self, *args, max_logit_delta: float = 0.5, **kwargs) -> None:
        self.max_logit_delta = float(max_logit_delta)
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = LPRRTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            max_logit_delta=self.max_logit_delta,
        )
        if weights:
            model.load(weights)
        return model

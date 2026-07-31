"""Ultralytics integration for the CSHC RT-DETR network module."""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Iterator

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

from src.cshc import C2CandidateFusion, CSHCRTDDETRDecoder, DySample
from src.cshc_loss import focal_binary_logits
from src.cshc_targets import build_tiny_center_targets


LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "c2_candidate_loss")


@contextmanager
def register_cshc_decoder() -> Iterator[None]:
    """Expose project-local components while parsing CSHC YAML without changing site-packages."""
    replaced = {
        "DySample": getattr(ultralytics_tasks, "DySample", None),
        "C2CandidateFusion": getattr(ultralytics_tasks, "C2CandidateFusion", None),
        "RTDETRDecoder": ultralytics_tasks.RTDETRDecoder,
    }
    ultralytics_tasks.DySample = DySample
    ultralytics_tasks.C2CandidateFusion = C2CandidateFusion
    ultralytics_tasks.RTDETRDecoder = CSHCRTDDETRDecoder
    try:
        yield
    finally:
        for name, previous in replaced.items():
            if previous is None:
                delattr(ultralytics_tasks, name)
            else:
                setattr(ultralytics_tasks, name, previous)


class CSHCDetectionModel(RTDETRDetectionModel):
    """RT-DETR detection model with C2 candidate-map supervision during training."""

    def __init__(
        self,
        cfg: str | Path = "configs/rtdetr-l-cshc.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        lambda_c2_candidate: float = 0.25,
    ) -> None:
        self.lambda_c2_candidate = float(lambda_c2_candidate)
        if self.lambda_c2_candidate < 0:
            raise ValueError("lambda_c2_candidate must be non-negative")
        with register_cshc_decoder():
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.nc = self.yaml["nc"]
        self.loss_names = LOSS_NAMES
        self.last_auxiliary_losses: dict[str, torch.Tensor] = {}

    @property
    def cshc_decoder(self) -> CSHCRTDDETRDecoder:
        decoder = self.model[-1]
        if not isinstance(decoder, CSHCRTDDETRDecoder):
            raise RuntimeError("CSHC model graph does not end with CSHCRTDDETRDecoder")
        return decoder

    def loss(self, batch: dict, preds=None):
        detection_loss, detection_items = super().loss(batch, preds=preds)
        candidates = self.cshc_decoder.last_candidates
        if candidates is None:
            raise RuntimeError("CSHC candidate map was not populated during the detection forward pass")
        target = build_tiny_center_targets(
            bboxes=batch["bboxes"].detach().to(device=candidates.objectness_logits.device, dtype=torch.float32),
            batch_idx=batch["batch_idx"].detach().to(device=candidates.objectness_logits.device),
            batch_size=batch["img"].shape[0],
            height=candidates.objectness_logits.shape[-2],
            width=candidates.objectness_logits.shape[-1],
        ).to(dtype=candidates.objectness_logits.dtype)
        candidate_loss = focal_binary_logits(candidates.objectness_logits, target)
        total = detection_loss + self.lambda_c2_candidate * candidate_loss
        self.last_auxiliary_losses = {"c2_candidate_loss": candidate_loss.detach()}
        items = torch.cat((detection_items, candidate_loss.detach().reshape(1)))
        return total, items


class CSHCTrainer(RTDETRTrainer):
    """Trainer that builds the project-owned model and reports its fourth loss item."""

    def __init__(self, *args, lambda_c2_candidate: float = 0.25, **kwargs) -> None:
        self.lambda_c2_candidate = float(lambda_c2_candidate)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = CSHCDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            lambda_c2_candidate=self.lambda_c2_candidate,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

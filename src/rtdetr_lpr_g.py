"""Ultralytics RT-DETR integration for isolated quality-gated refinement."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

from src.lpr_g_head import LPRGDeformableTransformerDecoder
from src.lpr_g_loss import MatchRecordingRTDETRDetectionLoss
from src.lpr_g_protocol import load_lpr_g_initial_state
from src.rtdetr_lpr import FixedPairedProtocolMixin
from src.rtdetr_vsf_rmr import apply_resume_runtime_overrides


class LPRGRTDETRDetectionModel(RTDETRDetectionModel):
    """RT-DETR with a detached last-layer refinement side branch."""

    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        max_logit_delta: float = 0.5,
        private_seed: int = 10_000,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        head.decoder = LPRGDeformableTransformerDecoder.from_stock(
            head.decoder,
            private_seed=private_seed,
            max_logit_delta=max_logit_delta,
        )
        self.nc = self.yaml["nc"]
        self.max_logit_delta = float(max_logit_delta)
        self.private_seed = int(private_seed)
        self.last_lpr_g_losses: dict[str, torch.Tensor] = {}
        self.last_lpr_g_loss_total: torch.Tensor | None = None

    def init_criterion(self) -> MatchRecordingRTDETRDetectionLoss:
        """Construct the stock RT-DETR criterion with match recording."""
        return MatchRecordingRTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def set_refinement_output(self, mode: str) -> None:
        """Choose stock or refined boxes for evaluation and inference."""
        self.model[-1].decoder.set_output_mode(mode)

    def loss(
        self,
        batch: dict,
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute unchanged stock losses plus private matched L1/GIoU losses."""
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        batch_size = img.shape[0]
        batch_index = batch["batch_idx"]
        ground_truth_groups = [(batch_index == index).sum().item() for index in range(batch_size)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_index.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": ground_truth_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        decoder = self.model[-1].decoder
        normal_refined = decoder.last_refined_bboxes
        if normal_refined is None:
            raise RuntimeError("LPR-G decoder did not produce a refinement side output")

        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
            _, normal_refined = torch.split(normal_refined, dn_meta["dn_num_split"], dim=1)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        stock_loss = self.criterion(
            (dec_bboxes, dec_scores),
            targets,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        refinement_loss = self.criterion.refinement_loss(normal_refined, targets)
        losses = {**stock_loss, **refinement_loss}
        self.last_lpr_g_losses = {
            name: value.detach() for name, value in refinement_loss.items()
        }
        self.last_lpr_g_loss_total = sum(refinement_loss.values())
        return sum(losses.values()), torch.as_tensor(
            [stock_loss[name].detach() for name in ("loss_giou", "loss_class", "loss_bbox")],
            device=img.device,
        )


class LPRGTrainer(FixedPairedProtocolMixin, RTDETRTrainer):
    """Strict paired trainer with independently clipped stock/private gradients."""

    def __init__(
        self,
        *args,
        max_logit_delta: float = 0.5,
        experiment_seed: int = 0,
        initial_state_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.max_logit_delta = float(max_logit_delta)
        self.experiment_seed = int(experiment_seed)
        self.initial_state_path = Path(initial_state_path) if initial_state_path is not None else None
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        common: list[torch.nn.Parameter] = []
        private: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (private if "lpr_g_refiner." in name else common).append(parameter)
        if not common or not private:
            raise RuntimeError("LPR-G stock/private parameter partition is incomplete")
        return {"gradient_norm": common, "lpr_g_gradient_norm": private}

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> LPRGRTDETRDetectionModel:
        model = LPRGRTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            max_logit_delta=self.max_logit_delta,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        elif self.initial_state_path is not None:
            artifact = torch.load(self.initial_state_path, map_location="cpu", weights_only=False)
            load_lpr_g_initial_state(model, artifact, variant="lprg")
        return model


class LPRGControlTrainer(FixedPairedProtocolMixin, RTDETRTrainer):
    """Stock control arm loaded from the same format-v2 common state."""

    def __init__(self, *args, initial_state_path: str | Path | None = None, **kwargs) -> None:
        self.initial_state_path = Path(initial_state_path) if initial_state_path is not None else None
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> RTDETRDetectionModel:
        model = RTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        elif self.initial_state_path is not None:
            artifact = torch.load(self.initial_state_path, map_location="cpu", weights_only=False)
            load_lpr_g_initial_state(model, artifact, variant="control")
        return model

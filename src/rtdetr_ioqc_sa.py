from __future__ import annotations

from copy import copy
from pathlib import Path

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

from src.ioqc_sa_loss import (
    IOQCSALossResult,
    IOQCSATargets,
    compute_ioqc_sa_loss,
    ensure_finite_losses,
    ioqc_ramp,
)
from src.ioqc_sa_probe import P3SamplingProbe, P3SamplingStatistics


LOSS_NAMES = (
    "giou_loss",
    "cls_loss",
    "l1_loss",
    "ioqc_comp_loss",
    "ioqc_align_loss",
)


def apply_resume_runtime_overrides(args: object, overrides: dict) -> None:
    for key in ("amp", "project", "name"):
        if key in overrides:
            setattr(args, key, bool(overrides[key]) if key == "amp" else overrides[key])


def regular_query_statistics(
    statistics: P3SamplingStatistics,
    dn_meta: dict | None,
) -> P3SamplingStatistics:
    denoising_queries = int(dn_meta["dn_num_split"][0]) if dn_meta is not None else 0
    selection = slice(denoising_queries, None)
    return P3SamplingStatistics(
        center=statistics.center[:, selection],
        extent=statistics.extent[:, selection],
        p3_mass=statistics.p3_mass[:, selection],
        valid=statistics.valid[:, selection],
        p3_shape=statistics.p3_shape,
    )


def prepare_matcher_inputs(
    boxes: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return boxes.float().contiguous(), scores.float().contiguous()


class IOQCSADetectionModel(RTDETRDetectionModel):
    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        lambda_competition: float = 0.05,
        lambda_alignment: float = 0.05,
        density_threshold: float = 1.0,
        duplicate_threshold: float = 0.10,
    ) -> None:
        self.lambda_competition = float(lambda_competition)
        self.lambda_alignment = float(lambda_alignment)
        self.density_threshold = float(density_threshold)
        self.duplicate_threshold = float(duplicate_threshold)
        self.ioqc_epoch = 0
        self.ioqc_total_epochs = 100
        self.last_ioqc_result: IOQCSALossResult | None = None
        self.last_ioqc_diagnostics: dict[str, torch.Tensor | float] = {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.nc = self.yaml["nc"]
        self.loss_names = LOSS_NAMES
        self.ioqc_probe = P3SamplingProbe()
        self.ioqc_probe.attach(self.model[-1].decoder)

    def set_ioqc_progress(self, epoch: int, total_epochs: int) -> None:
        self.ioqc_epoch = int(epoch)
        self.ioqc_total_epochs = int(total_epochs)

    def loss(self, batch: dict, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        image = batch["img"]
        batch_size = image.shape[0]
        batch_indices = batch["batch_idx"].to(image.device, dtype=torch.long).view(-1)
        groups = [(batch_indices == index).sum().item() for index in range(batch_size)]
        targets = {
            "cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=image.device),
            "batch_idx": batch_indices,
            "gt_groups": groups,
        }

        if preds is None:
            preds = self.predict(image, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]

        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        last_boxes = dec_bboxes[-1]
        last_scores = dec_scores[-1]
        detection_boxes = torch.cat((enc_bboxes.unsqueeze(0), dec_bboxes))
        detection_scores = torch.cat((enc_scores.unsqueeze(0), dec_scores))
        detection_losses = self.criterion(
            (detection_boxes, detection_scores),
            targets,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        detection_loss = sum(detection_losses.values())
        ensure_finite_losses(detection=detection_loss)
        detection_items = torch.stack(
            [detection_losses[key].detach() for key in ("loss_giou", "loss_class", "loss_bbox")]
        )
        if not self.training:
            return detection_loss, torch.cat((detection_items, detection_items.new_zeros(2)))

        captured = self.ioqc_probe.last_statistics
        if captured is None:
            raise RuntimeError("IOQC-SA P3 sampling statistics were not captured during the training forward pass")
        statistics = regular_query_statistics(captured, dn_meta)

        with torch.autocast(device_type=image.device.type, enabled=False):
            matcher_boxes, matcher_scores = prepare_matcher_inputs(last_boxes, last_scores)
            match_indices = self.criterion.matcher(
                matcher_boxes,
                matcher_scores,
                targets["bboxes"].float(),
                targets["cls"],
                groups,
            )
            ioqc_result = compute_ioqc_sa_loss(
                pred_boxes=last_boxes,
                pred_logits=last_scores,
                statistics=statistics,
                targets=IOQCSATargets(
                    boxes=targets["bboxes"],
                    classes=targets["cls"],
                    batch_indices=targets["batch_idx"],
                    groups=groups,
                ),
                match_indices=match_indices,
                density_threshold=self.density_threshold,
                duplicate_threshold=self.duplicate_threshold,
            )
            active_weight = ioqc_ramp(self.ioqc_epoch, self.ioqc_total_epochs)
            competition_contribution = active_weight * self.lambda_competition * ioqc_result.competition
            alignment_contribution = active_weight * self.lambda_alignment * ioqc_result.alignment
            total = detection_loss.float() + competition_contribution + alignment_contribution
            ensure_finite_losses(total=total)

        self.last_ioqc_result = ioqc_result
        self.last_ioqc_diagnostics = {
            "active_weight": active_weight,
            "competition_contribution": competition_contribution.detach(),
            "alignment_contribution": alignment_contribution.detach(),
        }
        loss_items = torch.cat(
            (
                detection_items,
                ioqc_result.competition.detach().reshape(1),
                ioqc_result.alignment.detach().reshape(1),
            )
        )
        return total, loss_items


class IOQCSATrainer(RTDETRTrainer):
    def __init__(
        self,
        *args,
        lambda_competition: float = 0.05,
        lambda_alignment: float = 0.05,
        density_threshold: float = 1.0,
        duplicate_threshold: float = 0.10,
        **kwargs,
    ) -> None:
        self.lambda_competition = lambda_competition
        self.lambda_alignment = lambda_alignment
        self.density_threshold = density_threshold
        self.duplicate_threshold = duplicate_threshold
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = IOQCSADetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            lambda_competition=self.lambda_competition,
            lambda_alignment=self.lambda_alignment,
            density_threshold=self.density_threshold,
            duplicate_threshold=self.duplicate_threshold,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

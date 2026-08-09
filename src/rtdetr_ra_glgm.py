"""Strict paired RT-DETR-L+FDR versus FDR+RA-GLGM integration."""

from __future__ import annotations

from pathlib import Path
import torch
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.utils import RANK, colorstr

from src.btd_se_dataset import BTDSEVisDroneDataset
from src.ra_glgm_head import RAFDRRTDETRDecoder
from src.ra_glgm_loss import (
    build_residual_difficulty_targets,
    residual_support_focal_loss,
)
from src.ra_glgm_protocol import load_ra_glgm_initial_state
from src.rtdetr_btdse import filter_detection_batch
from src.rtdetr_fdr import (
    FDRRTDETRDetectionModel,
    FDRTrainer,
    _MODEL_PARSE_LOCK,
)


ROOT = Path(__file__).resolve().parents[1]
RA_GLGM_MODEL_CFG = ROOT / "configs" / "rtdetr-l-fdr-ra-glgm.yaml"
RA_GLGM_CONTROL_CFG = ROOT / "configs" / "rtdetr-l-fdr-ra-glgm-control.yaml"
RA_GLGM_AUXILIARY_WEIGHT = 0.05


def register_ra_glgm_decoder() -> None:
    ultralytics_tasks.RAFDRRTDETRDecoder = RAFDRRTDETRDecoder


class _IgnoreAwareFDRModel(FDRRTDETRDetectionModel):
    """Apply identical ignore-sidecar filtering in both paired arms."""

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().loss(filter_detection_batch(batch), preds=preds)


class RAGLGMControlDetectionModel(_IgnoreAwareFDRModel):
    """Unmodified FDR graph under the RA experiment's shared data path."""


class RAGLGMDetectionModel(FDRRTDETRDetectionModel):
    """FDR model with private P3 RA refinement and residual supervision."""

    yaml_decoder_type = RAFDRRTDETRDecoder

    def __init__(
        self,
        cfg: str | Path | dict = RA_GLGM_MODEL_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
    ) -> None:
        with _MODEL_PARSE_LOCK:
            register_ra_glgm_decoder()
            super().__init__(
                cfg=cfg,
                ch=ch,
                nc=nc,
                verbose=verbose,
                private_seed=private_seed,
            )
        if not isinstance(self.model[-1], RAFDRRTDETRDecoder):
            raise TypeError("RA-GLGM YAML must end with RAFDRRTDETRDecoder")
        self.last_ra_glgm_losses: dict[str, torch.Tensor] = {}

    @property
    def ra_glgm(self):
        return self.model[-1].ra_glgm

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detection_batch = filter_detection_batch(batch)
        if preds is None:
            image = batch["img"]
            batch_index = detection_batch["batch_idx"]
            targets = {
                "cls": detection_batch["cls"].to(image.device, dtype=torch.long).view(-1),
                "bboxes": detection_batch["bboxes"].to(image.device),
                "batch_idx": batch_index.to(image.device, dtype=torch.long).view(-1),
                "gt_groups": [
                    int((batch_index == index).sum().item())
                    for index in range(int(image.shape[0]))
                ],
            }
            preds = self.predict(image, batch=targets)

        detection_loss, displayed = super().loss(detection_batch, preds=preds)
        if not self.training:
            self.last_ra_glgm_losses = {}
            return detection_loss, displayed
        if not isinstance(preds, tuple) or len(preds) != 5:
            raise RuntimeError("RA-GLGM training prediction contract changed")
        dec_bboxes, dec_scores, _, _, dn_meta = preds
        if dn_meta is not None:
            partition = dn_meta.get("dn_num_split")
            if not isinstance(partition, (list, tuple)) or len(partition) != 2:
                raise ValueError("RA-GLGM denoising partition is invalid")
            _, dec_bboxes = torch.split(dec_bboxes, tuple(map(int, partition)), dim=2)
            _, dec_scores = torch.split(dec_scores, tuple(map(int, partition)), dim=2)

        assignment = self.criterion.last_normal_decoder_assignment
        if assignment is None:
            raise RuntimeError("final normal FDR assignment is unavailable")
        support = self.ra_glgm.last_support_map
        if support is None:
            raise RuntimeError("RA-GLGM support map is unavailable")
        targets = build_residual_difficulty_targets(
            pred_bboxes=dec_bboxes[-1],
            pred_scores=dec_scores[-1],
            detection_bboxes=detection_batch["bboxes"],
            detection_classes=detection_batch["cls"],
            detection_batch_idx=detection_batch["batch_idx"],
            match_indices=assignment,
            all_bboxes=batch["bboxes"],
            all_classes=batch["cls"],
            all_batch_idx=batch["batch_idx"],
            height=int(support.shape[-2]),
            width=int(support.shape[-1]),
        )
        auxiliary = residual_support_focal_loss(support, targets)
        if not bool(torch.isfinite(auxiliary)):
            raise FloatingPointError("NONFINITE_RA_GLGM_SUPPORT_LOSS")
        self.last_ra_glgm_losses = {
            "loss_ra_support": auxiliary.detach(),
            "target_mean": targets.heatmap.mean().detach(),
            "valid_fraction": targets.valid_mask.float().mean().detach(),
        }
        return detection_loss + RA_GLGM_AUXILIARY_WEIGHT * auxiliary, displayed


class _IgnoreSidecarDatasetMixin:
    def build_dataset(
        self,
        img_path: str,
        mode: str = "val",
        batch: int | None = None,
    ):
        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)
        return BTDSEVisDroneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )


def _load_pair_state(model, path: Path | None, *, variant: str) -> None:
    if path is None:
        return
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    load_ra_glgm_initial_state(model, artifact, variant=variant)


class RAGLGMTrainer(_IgnoreSidecarDatasetMixin, FDRTrainer):
    """Method arm with separately clipped common, FDR, and RA gradients."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        ra_ids = {id(parameter) for parameter in self.model.ra_glgm.parameters()}
        common: list[torch.nn.Parameter] = []
        fdr: list[torch.nn.Parameter] = []
        ra: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in ra_ids:
                ra.append(parameter)
            elif ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name:
                fdr.append(parameter)
            else:
                common.append(parameter)
        if not common or not fdr or not ra:
            raise RuntimeError("RA-GLGM common/FDR/private gradient partition is incomplete")
        return {
            "gradient_norm": common,
            "fdr_gradient_norm": fdr,
            "ra_glgm_gradient_norm": ra,
        }

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> RAGLGMDetectionModel:
        model = RAGLGMDetectionModel(
            cfg or RA_GLGM_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_pair_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="ra_glgm",
            )
        return model


class RAGLGMControlTrainer(_IgnoreSidecarDatasetMixin, FDRTrainer):
    """FDR baseline arm under the identical ignore-sidecar pipeline."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> RAGLGMControlDetectionModel:
        model = RAGLGMControlDetectionModel(
            cfg or RA_GLGM_CONTROL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_pair_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="baseline",
            )
        return model


__all__ = [
    "RA_GLGM_AUXILIARY_WEIGHT",
    "RA_GLGM_CONTROL_CFG",
    "RA_GLGM_MODEL_CFG",
    "RAGLGMControlDetectionModel",
    "RAGLGMControlTrainer",
    "RAGLGMDetectionModel",
    "RAGLGMTrainer",
    "register_ra_glgm_decoder",
]

from __future__ import annotations

from copy import copy
from pathlib import Path

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from src.btd_se import BTDSE
from src.btd_se_dataset import BTDSEVisDroneDataset
from src.btd_se_loss import binary_focal_loss
from src.btd_se_targets import build_auxiliary_targets


LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "background_loss", "saliency_loss")


def register_btdse_module() -> None:
    """Expose the repository-owned layer to Ultralytics' YAML parser without editing site-packages."""
    ultralytics_tasks.BTDSE = BTDSE


def filter_detection_batch(batch: dict) -> dict:
    """Return a shallow batch copy with class -1 ignore instances removed from detection targets."""
    classes = batch["cls"].reshape(-1)
    detection_mask = classes >= 0
    filtered = dict(batch)
    for key in ("cls", "bboxes", "batch_idx"):
        filtered[key] = batch[key][detection_mask]
    return filtered


class BTDSEDetectionModel(RTDETRDetectionModel):
    def __init__(
        self,
        cfg: str | Path = "configs/rtdetr-l-btdse.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        lambda_background: float = 0.1,
        lambda_saliency: float = 0.1,
    ) -> None:
        register_btdse_module()
        self.lambda_background = float(lambda_background)
        self.lambda_saliency = float(lambda_saliency)
        self.last_auxiliary_losses: dict[str, torch.Tensor] = {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.nc = self.yaml["nc"]
        self.loss_names = LOSS_NAMES

    @property
    def btdse(self) -> BTDSE:
        return next(module for module in self.model if isinstance(module, BTDSE))

    def loss(self, batch: dict, preds=None):
        detection_batch = filter_detection_batch(batch)
        detection_loss, detection_items = super().loss(detection_batch, preds=preds)

        module = self.btdse
        reliability = module.last_background_reliability
        saliency = module.last_saliency
        if reliability is None or saliency is None:
            raise RuntimeError("BTD-SE auxiliary maps were not populated during the detection forward pass")

        cpu_targets = build_auxiliary_targets(
            bboxes=batch["bboxes"].detach().to(device="cpu", dtype=torch.float32),
            classes=batch["cls"].detach().to(device="cpu", dtype=torch.float32),
            batch_idx=batch["batch_idx"].detach().to(device="cpu", dtype=torch.float32),
            batch_size=batch["img"].shape[0],
            height=reliability.shape[-2],
            width=reliability.shape[-1],
        )
        background_target = cpu_targets.background.to(device=reliability.device, dtype=reliability.dtype)
        saliency_target = cpu_targets.saliency.to(device=saliency.device, dtype=saliency.dtype)
        saliency_valid = cpu_targets.saliency_valid.to(device=saliency.device)

        background_loss = binary_focal_loss(
            reliability,
            background_target,
            alpha=0.25,
            exponent=2.0,
        )
        saliency_loss = binary_focal_loss(
            saliency,
            saliency_target,
            alpha=0.25,
            exponent=2.0,
            valid_mask=saliency_valid,
        )
        total = (
            detection_loss
            + self.lambda_background * background_loss
            + self.lambda_saliency * saliency_loss
        )
        self.last_auxiliary_losses = {
            "background_loss": background_loss.detach(),
            "saliency_loss": saliency_loss.detach(),
        }
        loss_items = torch.cat(
            (
                detection_items,
                background_loss.detach().reshape(1),
                saliency_loss.detach().reshape(1),
            )
        )
        return total, loss_items


class BTDSETrainer(RTDETRTrainer):
    def __init__(
        self,
        *args,
        lambda_background: float = 0.1,
        lambda_saliency: float = 0.1,
        **kwargs,
    ) -> None:
        self.lambda_background = lambda_background
        self.lambda_saliency = lambda_saliency
        register_btdse_module()
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = BTDSEDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            lambda_background=self.lambda_background,
            lambda_saliency=self.lambda_saliency,
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return BTDSEVisDroneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

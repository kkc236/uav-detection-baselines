from __future__ import annotations

from copy import copy
from pathlib import Path

import torch
import ultralytics
import ultralytics.nn.modules.head as ultralytics_head
import ultralytics.nn.tasks as ultralytics_tasks
from ultralytics.models.rtdetr.train import RTDETRTrainer as UltralyticsRTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from src.ebc_qp_config import (
    SOURCE_SHA256,
    ULTRALYTICS_VERSION,
    EBCQPConfig,
    assert_ultralytics_source_lock,
)
from src.ebc_qp_decoder import EBCQPDecoder, register_ebc_qp_decoder


LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "p2_loss", "ebc_loss")
EBC_QP_IMPLEMENTATION_VERSION = "1.0"


def _assert_source_lock() -> None:
    package_root = Path(ultralytics.__file__).parent
    assert_ultralytics_source_lock(
        {
            "head.py": Path(ultralytics_head.__file__),
            "tasks.py": Path(ultralytics_tasks.__file__),
            "rtdetr-l.yaml": package_root / "cfg" / "models" / "rt-detr" / "rtdetr-l.yaml",
        }
    )


class EBCQPDetectionModel(RTDETRDetectionModel):
    def __init__(
        self,
        cfg: str | Path = "configs/rtdetr-l-ebc-qp.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        ebc_config: EBCQPConfig | None = None,
    ):
        _assert_source_lock()
        self.ebc_config = ebc_config or EBCQPConfig()
        original_decoder = ultralytics_tasks.RTDETRDecoder
        register_ebc_qp_decoder()
        try:
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        finally:
            ultralytics_tasks.RTDETRDecoder = original_decoder
        self.ebc_head.ebc_config = self.ebc_config
        self.nc = self.ebc_head.nc
        self.loss_names = LOSS_NAMES

    @property
    def ebc_head(self) -> EBCQPDecoder:
        head = self.model[-1]
        if not isinstance(head, EBCQPDecoder):
            raise TypeError("model does not end in EBCQPDecoder")
        return head

    def set_ebc_progress(self, epoch: int) -> None:
        self.ebc_head.set_progress(epoch)

    def loss(self, batch: dict, preds=None):
        stock_loss, stock_items = super().loss(batch, preds=preds)
        state = self.ebc_head.last_state
        if state is None:
            raise RuntimeError("EBC-QP forward state was not populated")
        ebc_loss = state.ebc_loss if state.competition_active else state.ebc_loss * 0.0
        total = (
            stock_loss.float()
            + self.ebc_config.lambda_p2 * state.p2_loss.float()
            + self.ebc_config.lambda_ebc * ebc_loss.float()
        )
        state.stock_loss = stock_loss.detach()
        items = torch.cat((stock_items, state.p2_loss.detach()[None], ebc_loss.detach()[None]))
        return total, items


def build_ebc_qp_checkpoint_metadata(config: EBCQPConfig, ebc_epoch: int) -> dict:
    return {
        "implementation_version": EBC_QP_IMPLEMENTATION_VERSION,
        "ultralytics_version": ULTRALYTICS_VERSION,
        "ebc_epoch": int(ebc_epoch),
        "config": config.as_dict(),
        "source_sha256": dict(SOURCE_SHA256),
    }


def validate_ebc_qp_checkpoint_metadata(metadata: dict, config: EBCQPConfig) -> None:
    if metadata.get("implementation_version") != EBC_QP_IMPLEMENTATION_VERSION:
        raise RuntimeError("EBC-QP implementation version mismatch")
    if metadata.get("ultralytics_version") != ULTRALYTICS_VERSION:
        raise RuntimeError("EBC-QP Ultralytics version mismatch")
    if metadata.get("config") != config.as_dict():
        raise RuntimeError("EBC-QP config mismatch")
    if metadata.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("EBC-QP source lock mismatch")


class EBCQPTrainer(UltralyticsRTDETRTrainer):
    def __init__(self, *args, ebc_config: EBCQPConfig | None = None, **kwargs):
        self.ebc_config = ebc_config or EBCQPConfig()
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._set_ebc_progress)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = EBCQPDetectionModel(
            cfg=cfg or "configs/rtdetr-l-ebc-qp.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            ebc_config=self.ebc_config,
        )
        if weights:
            model.load(weights)
        return model

    def _set_ebc_progress(self, _trainer=None) -> None:
        unwrap_model(self.model).set_ebc_progress(int(self.epoch))

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def save_model(self):
        saved = super().save_model()
        model = unwrap_model(self.model)
        metadata = build_ebc_qp_checkpoint_metadata(self.ebc_config, model.ebc_head.ebc_epoch)
        paths = {self.last, self.best}
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            paths.add(self.wdir / f"epoch{self.epoch}.pt")
        for path in paths:
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                checkpoint["ebc_qp"] = metadata
                torch.save(checkpoint, path)
        return saved

    def resume_training(self, checkpoint):
        if checkpoint is not None and self.resume:
            metadata = checkpoint.get("ebc_qp")
            if metadata is None:
                raise RuntimeError("EBC-QP checkpoint metadata is missing")
            validate_ebc_qp_checkpoint_metadata(metadata, self.ebc_config)
        super().resume_training(checkpoint)
        if checkpoint is not None and self.resume:
            unwrap_model(self.model).set_ebc_progress(int(checkpoint["ebc_qp"]["ebc_epoch"]))

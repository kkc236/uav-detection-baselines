from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class UpdateRecord:
    u_p2: float
    u_stock: float
    ratio: float
    step: int
    monitored: bool
    abort: bool


class NormalizedUpdateMonitor:
    def __init__(
        self,
        p2: list[torch.nn.Parameter],
        stock: list[torch.nn.Parameter],
        limit: float = 10.0,
        patience: int = 20,
        max_steps: int = 200,
        eps: float = 1e-12,
    ):
        self.p2 = [parameter for parameter in p2 if parameter.requires_grad]
        self.stock = [parameter for parameter in stock if parameter.requires_grad]
        self.limit = float(limit)
        self.patience = int(patience)
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.step = 0
        self.consecutive = 0
        self.trace: list[UpdateRecord] = []
        self._before_p2: list[torch.Tensor] | None = None
        self._before_stock: list[torch.Tensor] | None = None

    def snapshot(self) -> None:
        self._before_p2 = [parameter.detach().to(device="cpu", dtype=torch.float32).clone() for parameter in self.p2]
        self._before_stock = [
            parameter.detach().to(device="cpu", dtype=torch.float32).clone() for parameter in self.stock
        ]

    @staticmethod
    def _relative(
        before: list[torch.Tensor],
        after: list[torch.nn.Parameter],
        eps: float,
    ) -> float:
        delta_squared = sum(
            float((parameter.detach().float().cpu() - old).square().sum())
            for old, parameter in zip(before, after)
        )
        theta_squared = sum(float(old.square().sum()) for old in before)
        return delta_squared**0.5 / (theta_squared**0.5 + eps)

    def observe(self) -> UpdateRecord:
        if self._before_p2 is None or self._before_stock is None:
            raise RuntimeError("snapshot must be called before observe")
        self.step += 1
        if self.step > self.max_steps:
            return UpdateRecord(0.0, 0.0, 0.0, self.step, False, False)

        u_p2 = self._relative(self._before_p2, self.p2, self.eps)
        u_stock = self._relative(self._before_stock, self.stock, self.eps)
        ratio = u_p2 / (u_stock + self.eps)
        self.consecutive = self.consecutive + 1 if ratio > self.limit else 0
        record = UpdateRecord(
            u_p2=u_p2,
            u_stock=u_stock,
            ratio=ratio,
            step=self.step,
            monitored=True,
            abort=self.consecutive >= self.patience,
        )
        self.trace.append(record)
        return record


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

    def optimizer_step(self):
        if not hasattr(self, "update_monitor"):
            p2_parameters = []
            stock_parameters = []
            for name, parameter in unwrap_model(self.model).named_parameters():
                target = p2_parameters if ".p2_adapter." in name or ".p2_bbox_head." in name else stock_parameters
                target.append(parameter)
            self.update_monitor = NormalizedUpdateMonitor(
                p2_parameters,
                stock_parameters,
                limit=self.ebc_config.update_ratio_limit,
                patience=self.ebc_config.update_ratio_patience,
                max_steps=self.ebc_config.update_monitor_steps,
                eps=self.ebc_config.epsilon,
            )

        monitoring = self.update_monitor.step < self.update_monitor.max_steps
        if monitoring:
            self.update_monitor.snapshot()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        record = self.update_monitor.observe() if monitoring else None
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        if record is not None and record.abort:
            trace = [asdict(item) for item in self.update_monitor.trace[-self.update_monitor.patience :]]
            raise RuntimeError(f"EBC-QP normalized update ratio exceeded its limit: {trace}")

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

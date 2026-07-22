from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import ultralytics
import ultralytics.nn.modules.head as ultralytics_head
import ultralytics.nn.tasks as ultralytics_tasks
from ultralytics.models.rtdetr.train import RTDETRTrainer as UltralyticsRTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import init_seeds, unwrap_model

from src.ebc_qp_config import (
    SOURCE_SHA256,
    ULTRALYTICS_VERSION,
    EBCQPConfig,
    assert_ultralytics_source_lock,
)
from src.ebc_qp_decoder import EBCQPDecoder, register_ebc_qp_decoder
from src.ebc_qp_diagnostics import MechanismDiagnosticsAccumulator
from src.ebc_qp_metrics import TinyDetectionMetrics, validation_xy_gain
from src.ebc_qp_protocol import load_initial_state


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
        self._before_p2 = [parameter.detach().float().clone() for parameter in self.p2]
        self._before_stock = [parameter.detach().float().clone() for parameter in self.stock]

    @staticmethod
    def _relative(
        before: list[torch.Tensor],
        after: list[torch.nn.Parameter],
        eps: float,
    ) -> float:
        if not before:
            return 0.0
        delta_squared = torch.stack(
            [(parameter.detach().float() - old).square().sum() for old, parameter in zip(before, after)]
        ).sum()
        theta_squared = torch.stack([old.square().sum() for old in before]).sum()
        relative = delta_squared.sqrt() / (theta_squared.sqrt() + eps)
        return float(relative.item())

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
    stored_config = dict(metadata.get("config", {}))
    stored_config.setdefault("quality_weighted_ebc", False)
    if stored_config != config.as_dict():
        raise RuntimeError("EBC-QP config mismatch")
    if metadata.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("EBC-QP source lock mismatch")


class EBCQPValidator(RTDETRValidator):
    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self.tiny_metrics = TinyDetectionMetrics(self.iouv.cpu())

    def update_metrics(self, preds, batch) -> None:
        super().update_metrics(preds, batch)
        for image_index, prediction in enumerate(preds):
            prepared = self._prepare_batch(image_index, batch)
            prepared_prediction = self._prepare_pred(prediction)
            gain = validation_xy_gain(prepared["ori_shape"], prepared["imgsz"], prepared["ratio_pad"])
            self.tiny_metrics.update_from_prepared(prepared_prediction, prepared, gain)

    def get_stats(self) -> dict:
        results = super().get_stats()
        tiny = self.tiny_metrics.compute()
        results.update(
            {
                "metrics/AP-tiny": tiny.map,
                "metrics/Recall-tiny": tiny.recall,
                "metrics/AP-r<8": tiny.extreme_map,
                "metrics/AP-8<=r<=16": tiny.tiny_8_16_map,
            }
        )
        self.tiny_metrics.clear()
        return results

    def gather_stats(self) -> None:
        super().gather_stats()
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(self.tiny_metrics.state_dict(), gathered, dst=0)
            self.tiny_metrics.clear()
            for state in gathered:
                self.tiny_metrics.merge_state_dict(state)
        elif RANK > 0:
            dist.gather_object(self.tiny_metrics.state_dict(), None, dst=0)
            self.tiny_metrics.clear()


class EBCQPDiagnosticsValidator(EBCQPValidator):
    """Read-only validator that records query-replacement mechanism evidence."""

    def attach_diagnostic_model(self, model: EBCQPDetectionModel) -> None:
        self.diagnostic_head = model.ebc_head

    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        if not hasattr(self, "diagnostic_head"):
            heads = [module for module in model.modules() if isinstance(module, EBCQPDecoder)]
            if len(heads) != 1:
                raise RuntimeError(f"expected one EBC-QP decoder during diagnostics, found {len(heads)}")
            self.diagnostic_head = heads[0]
        self.mechanism_diagnostics = MechanismDiagnosticsAccumulator(
            tiny_radius=self.diagnostic_head.ebc_config.tiny_radius
        )
        self.mechanism_stats: dict[str, float | int] = {}

    def update_metrics(self, preds, batch) -> None:
        super().update_metrics(preds, batch)
        state = self.diagnostic_head.last_state
        if state is None:
            raise RuntimeError("EBC-QP forward state is missing during diagnostics")
        self.mechanism_diagnostics.update(state, batch)

    def get_stats(self) -> dict:
        results = super().get_stats()
        self.mechanism_stats = self.mechanism_diagnostics.compute()
        return results


class PairedProtocolOptimizerMixin:
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        name, lr, momentum = resolve_protocol_optimizer(name, lr=lr, momentum=momentum)
        return super().build_optimizer(
            model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )


def resolve_protocol_optimizer(name: str, *, lr: float, momentum: float) -> tuple[str, float, float]:
    if name.lower() == "auto":
        return "MuSGD", lr, momentum
    return name, lr, momentum


class EBCQPTrainer(PairedProtocolOptimizerMixin, UltralyticsRTDETRTrainer):
    def __init__(
        self,
        *args,
        ebc_config: EBCQPConfig | None = None,
        initial_state_path: str | Path | None = None,
        **kwargs,
    ):
        self.ebc_config = ebc_config or EBCQPConfig()
        self.initial_state_path = Path(initial_state_path) if initial_state_path is not None else None
        self.initial_state = _load_protocol_state(self.initial_state_path)
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
        if self.initial_state is not None:
            _validate_initial_state_seed(self.initial_state, self.args.seed)
            load_initial_state(model, self.initial_state, include_innovation=True)
        return model

    def _build_train_pipeline(self):
        _reset_paired_random_state(self.args.seed, self.args.deterministic)
        return super()._build_train_pipeline()

    def _set_ebc_progress(self, _trainer=None) -> None:
        epoch = int(self.epoch)
        unwrap_model(self.model).set_ebc_progress(epoch)
        ema_model = self.ema.ema if getattr(self, "ema", None) is not None else None
        if ema_model is not None:
            unwrap_model(ema_model).set_ebc_progress(epoch)

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return EBCQPValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

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


class PairedControlTrainer(PairedProtocolOptimizerMixin, UltralyticsRTDETRTrainer):
    def __init__(self, *args, initial_state_path: str | Path, **kwargs):
        self.initial_state_path = Path(initial_state_path)
        self.initial_state = _load_protocol_state(self.initial_state_path)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = RTDETRDetectionModel(
            cfg=cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        _validate_initial_state_seed(self.initial_state, self.args.seed)
        load_initial_state(model, self.initial_state, include_innovation=False)
        return model

    def _build_train_pipeline(self):
        _reset_paired_random_state(self.args.seed, self.args.deterministic)
        return super()._build_train_pipeline()

    def get_validator(self):
        self.loss_names = LOSS_NAMES[:3]
        return EBCQPValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


def _load_protocol_state(path: Path | None) -> dict | None:
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_initial_state_seed(artifact: dict, seed: int) -> None:
    artifact_seed = artifact.get("metadata", {}).get("seed")
    if artifact_seed != seed:
        raise ValueError(f"initial-state seed mismatch: expected {seed}, got {artifact_seed}")


def _reset_paired_random_state(seed: int, deterministic: bool) -> None:
    init_seeds(seed + 1 + RANK, deterministic=deterministic)

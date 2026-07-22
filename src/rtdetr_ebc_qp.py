from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any

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
from src.ebc_qp_stock_diagnostics import StockQueryProbe, StockTinyQueryAccumulator


LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "p2_loss", "ebc_loss")
QG_LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "p2_loss", "qg_loss", "ebc_loss")
EBC_QP_IMPLEMENTATION_VERSION = "1.0"
QG_P2_IMPLEMENTATION_VERSION = "1.1-qg-p2"
TSGR_P2_IMPLEMENTATION_VERSION = "2.0-tsgr-p2"


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


def _is_auxiliary_parameter_name(name: str) -> bool:
    auxiliary_modules = ("p2_adapter.", "p2_bbox_head.", "p2_quality_head.")
    return any(name.startswith(module) or f".{module}" in name for module in auxiliary_modules) or name.endswith(
        "p2_fusion_gamma"
    )


def is_tsgr_shallow_parameter_name(name: str) -> bool:
    parts = name.split(".")
    return len(parts) >= 3 and parts[0] == "model" and parts[1] in {"0", "1"}


def partition_optimizer_parameters(model, optimizer) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    named_by_id = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("optimizer contains duplicate trainable parameters")
    unknown_ids = set(optimizer_ids).difference(named_by_id)
    if unknown_ids:
        raise RuntimeError("optimizer contains trainable parameters that are not registered on the model")
    missing_ids = set(named_by_id).difference(optimizer_ids)
    if missing_ids:
        missing_names = sorted(named_by_id[parameter_id] for parameter_id in missing_ids)
        raise RuntimeError(f"trainable model parameters missing from optimizer: {missing_names[:5]}")

    auxiliary_parameters = [
        parameter for parameter in optimizer_parameters if _is_auxiliary_parameter_name(named_by_id[id(parameter)])
    ]
    auxiliary_ids = {id(parameter) for parameter in auxiliary_parameters}
    stock_parameters = [parameter for parameter in optimizer_parameters if id(parameter) not in auxiliary_ids]
    stock_ids = {id(parameter) for parameter in stock_parameters}
    if stock_ids.intersection(auxiliary_ids) or stock_ids.union(auxiliary_ids) != set(optimizer_ids):
        raise RuntimeError("stock/auxiliary optimizer parameter partition is not complete and disjoint")
    return stock_parameters, auxiliary_parameters


def _clip_coefficient(preclip_norm: float, max_norm: float) -> float:
    if preclip_norm <= 0.0:
        return 1.0
    return min(1.0, float(max_norm) / (float(preclip_norm) + 1e-6))


def add_clipped_gradient_contribution(
    contributions: list[tuple[torch.nn.Parameter, torch.Tensor]],
    *,
    max_norm: float,
) -> dict[str, float]:
    if not contributions:
        return {"preclip_norm": 0.0, "clip_coefficient": 1.0}
    norms = torch.stack([gradient.detach().float().norm(2) for _parameter, gradient in contributions])
    preclip_norm = float(norms.norm(2).item())
    if not torch.isfinite(norms).all():
        raise FloatingPointError("auxiliary gradient contribution is non-finite")
    coefficient = _clip_coefficient(preclip_norm, max_norm)
    for parameter, gradient in contributions:
        clipped = gradient.detach().to(dtype=parameter.dtype).mul(coefficient)
        if parameter.grad is None:
            parameter.grad = clipped.clone()
        else:
            parameter.grad.add_(clipped)
    return {"preclip_norm": preclip_norm, "clip_coefficient": coefficient}


def prepare_contribution_separated_gradients(
    model,
    optimizer,
    scaler,
    stock_parameters: list[torch.nn.Parameter],
    auxiliary_parameters: list[torch.nn.Parameter],
    *,
    max_norm: float,
    capture_tensors: bool = False,
) -> dict[str, Any]:
    scaled_contributions, contribution_scale = model.pop_isolated_auxiliary_gradients()
    scaler_scale = float(scaler.get_scale())
    if contribution_scale != scaler_scale:
        raise RuntimeError(
            f"isolated auxiliary gradient scale mismatch: buffered {contribution_scale}, scaler {scaler_scale}"
        )
    named_parameters = dict(model.named_parameters())
    parameter_names = {id(parameter): name for name, parameter in named_parameters.items()}
    auxiliary_ids = {id(parameter) for parameter in auxiliary_parameters}
    expected_names = {
        name
        for name, parameter in named_parameters.items()
        if id(parameter) in auxiliary_ids or is_tsgr_shallow_parameter_name(name)
    }
    unexpected = set(scaled_contributions).difference(expected_names)
    if unexpected:
        raise RuntimeError(f"isolated P2 gradient escaped its allowed boundary: {sorted(unexpected)[:5]}")

    shallow_scaled = {
        name: gradient
        for name, gradient in scaled_contributions.items()
        if is_tsgr_shallow_parameter_name(name)
    }
    private_scaled = {
        name: gradient
        for name, gradient in scaled_contributions.items()
        if name in expected_names and not is_tsgr_shallow_parameter_name(name)
    }
    shallow_scaled_finite = all(torch.isfinite(gradient).all() for gradient in shallow_scaled.values())

    installed_private: list[torch.nn.Parameter] = []
    for parameter in auxiliary_parameters:
        name = parameter_names[id(parameter)]
        gradient = private_scaled.get(name)
        if gradient is None:
            continue
        if parameter.grad is not None:
            raise RuntimeError(f"pure stock loss unexpectedly produced an auxiliary-private gradient: {name}")
        parameter.grad = gradient.detach().clone()
        installed_private.append(parameter)
    if not shallow_scaled_finite:
        if not installed_private:
            raise RuntimeError("cannot expose a non-finite routed gradient to GradScaler")
        installed_private[0].grad.reshape(-1)[0] = float("inf")

    scaler.unscale_(optimizer)
    private_unscaled: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for parameter in auxiliary_parameters:
        if parameter.grad is not None:
            private_unscaled.append((parameter, parameter.grad.detach().clone()))
            parameter.grad = None
    shallow_unscaled = [
        (named_parameters[name], gradient.detach() / contribution_scale)
        for name, gradient in shallow_scaled.items()
    ]
    auxiliary_finite = shallow_scaled_finite and all(
        torch.isfinite(gradient).all() for _parameter, gradient in private_unscaled
    )

    captured: dict[str, Any] = {}
    if capture_tensors:
        captured["pure_stock_preclip"] = {
            parameter_names[id(parameter)]: parameter.grad.detach().clone()
            for parameter in stock_parameters
            if parameter.grad is not None
        }
        captured["routed_shallow_preclip"] = {
            parameter_names[id(parameter)]: gradient.detach().clone()
            for parameter, gradient in shallow_unscaled
        }
        captured["aux_private_preclip"] = {
            parameter_names[id(parameter)]: gradient.detach().clone()
            for parameter, gradient in private_unscaled
        }

    shallow_stock_values = [
        parameter.grad.detach().float().norm(2)
        for parameter in stock_parameters
        if parameter.grad is not None and is_tsgr_shallow_parameter_name(parameter_names[id(parameter)])
    ]
    shallow_stock_norm = (
        float(torch.stack(shallow_stock_values).norm(2).item()) if shallow_stock_values else 0.0
    )
    stock_norm_tensor = torch.nn.utils.clip_grad_norm_(stock_parameters, max_norm=max_norm)
    stock_norm = float(stock_norm_tensor.detach().float().item())
    shallow_result = {"preclip_norm": float("nan"), "clip_coefficient": float("nan")}
    private_result = {"preclip_norm": float("nan"), "clip_coefficient": float("nan")}
    if auxiliary_finite:
        shallow_result = add_clipped_gradient_contribution(shallow_unscaled, max_norm=max_norm)
        private_result = add_clipped_gradient_contribution(private_unscaled, max_norm=max_norm)
    if capture_tensors:
        captured["merged_postclip"] = {
            parameter_names[id(parameter)]: parameter.grad.detach().clone()
            for parameter in [*stock_parameters, *auxiliary_parameters]
            if parameter.grad is not None
        }
    return {
        "gradient_clipping_mode": "contribution_separated",
        "scaler_scale": scaler_scale,
        "auxiliary_finite": auxiliary_finite,
        "pure_stock_preclip_norm": stock_norm,
        "pure_stock_shallow_preclip_norm": shallow_stock_norm,
        "pure_stock_clip_coefficient": _clip_coefficient(stock_norm, max_norm),
        "routed_shallow_preclip_norm": shallow_result["preclip_norm"],
        "routed_shallow_clip_coefficient": shallow_result["clip_coefficient"],
        "aux_private_preclip_norm": private_result["preclip_norm"],
        "aux_private_clip_coefficient": private_result["clip_coefficient"],
        **captured,
    }


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
        self._isolated_auxiliary_gradient_scale = 1.0
        self._isolated_auxiliary_gradient_buffer: dict[str, torch.Tensor] = {}
        self._isolated_auxiliary_buffer_scale: float | None = None
        original_decoder = ultralytics_tasks.RTDETRDecoder
        register_ebc_qp_decoder()
        try:
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        finally:
            ultralytics_tasks.RTDETRDecoder = original_decoder
        self.ebc_head.configure_ebc_config(self.ebc_config)
        self.nc = self.ebc_head.nc
        self.loss_names = QG_LOSS_NAMES if self.ebc_config.quality_gated_p2 else LOSS_NAMES

    @property
    def ebc_head(self) -> EBCQPDecoder:
        head = self.model[-1]
        if not isinstance(head, EBCQPDecoder):
            raise TypeError("model does not end in EBCQPDecoder")
        return head

    def set_ebc_progress(self, epoch: int) -> None:
        self.ebc_head.set_progress(epoch)

    def set_isolated_auxiliary_gradient_scale(self, scale: float) -> None:
        scale = float(scale)
        if scale <= 0.0:
            raise ValueError("isolated auxiliary gradient scale must be positive")
        if self._isolated_auxiliary_gradient_buffer and self._isolated_auxiliary_buffer_scale != scale:
            raise RuntimeError("GradScaler scale changed inside an accumulated optimizer step")
        self._isolated_auxiliary_gradient_scale = scale

    def clear_isolated_auxiliary_gradients(self) -> None:
        self._isolated_auxiliary_gradient_buffer.clear()
        self._isolated_auxiliary_buffer_scale = None

    def pop_isolated_auxiliary_gradients(self) -> tuple[dict[str, torch.Tensor], float]:
        if not self._isolated_auxiliary_gradient_buffer or self._isolated_auxiliary_buffer_scale is None:
            raise RuntimeError("isolated auxiliary gradient buffer is empty")
        gradients = self._isolated_auxiliary_gradient_buffer
        scale = self._isolated_auxiliary_buffer_scale
        self._isolated_auxiliary_gradient_buffer = {}
        self._isolated_auxiliary_buffer_scale = None
        return gradients, scale

    def _accumulate_isolated_p2_gradients(self, objective: torch.Tensor) -> None:
        targets = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and (_is_auxiliary_parameter_name(name) or is_tsgr_shallow_parameter_name(name))
        ]
        scale = self._isolated_auxiliary_gradient_scale
        gradients = torch.autograd.grad(
            objective * scale,
            [parameter for _name, parameter in targets],
            retain_graph=True,
            allow_unused=True,
        )
        if self._isolated_auxiliary_buffer_scale is None:
            self._isolated_auxiliary_buffer_scale = scale
        elif self._isolated_auxiliary_buffer_scale != scale:
            raise RuntimeError("GradScaler scale changed inside an accumulated optimizer step")
        for (name, _parameter), gradient in zip(targets, gradients):
            if gradient is None:
                continue
            detached = gradient.detach()
            previous = self._isolated_auxiliary_gradient_buffer.get(name)
            self._isolated_auxiliary_gradient_buffer[name] = (
                detached.clone() if previous is None else previous.add(detached)
            )

    def loss(self, batch: dict, preds=None):
        stock_loss, stock_items = super().loss(batch, preds=preds)
        state = self.ebc_head.last_state
        if state is None:
            raise RuntimeError("EBC-QP forward state was not populated")
        if self.ebc_config.contribution_separated_aux_gradients:
            if state.p2_entry_count != 0 or state.ordinary_query_count != self.ebc_config.query_budget:
                raise RuntimeError(
                    "TSGR query isolation failed for a training/validation batch: "
                    f"entries={state.p2_entry_count}, queries={state.ordinary_query_count}"
                )
        ebc_loss = state.ebc_loss if state.competition_active else state.ebc_loss * 0.0
        p2_objective = self.ebc_config.lambda_p2 * state.p2_loss.float()
        if self.ebc_config.contribution_separated_aux_gradients:
            if self.training and torch.is_grad_enabled():
                self._accumulate_isolated_p2_gradients(p2_objective)
            total = stock_loss.float()
        else:
            total = (
                stock_loss.float()
                + p2_objective
                + self.ebc_config.lambda_quality * state.quality_loss.float()
                + self.ebc_config.lambda_ebc * ebc_loss.float()
            )
        state.stock_loss = stock_loss.detach()
        side_items = [state.p2_loss.detach()[None]]
        if self.ebc_config.quality_gated_p2:
            side_items.append(state.quality_loss.detach()[None])
        side_items.append(ebc_loss.detach()[None])
        items = torch.cat((stock_items, *side_items))
        return total, items


def build_ebc_qp_checkpoint_metadata(config: EBCQPConfig, ebc_epoch: int) -> dict:
    return {
        "implementation_version": _implementation_version(config),
        "ultralytics_version": ULTRALYTICS_VERSION,
        "ebc_epoch": int(ebc_epoch),
        "config": config.as_dict(),
        "source_sha256": dict(SOURCE_SHA256),
    }


def validate_ebc_qp_checkpoint_metadata(metadata: dict, config: EBCQPConfig) -> None:
    if metadata.get("implementation_version") != _implementation_version(config):
        raise RuntimeError("EBC-QP implementation version mismatch")
    if metadata.get("ultralytics_version") != ULTRALYTICS_VERSION:
        raise RuntimeError("EBC-QP Ultralytics version mismatch")
    stored_config = dict(metadata.get("config", {}))
    stored_config.setdefault("quality_weighted_ebc", False)
    stored_config.setdefault("learnable_fusion_gamma", False)
    stored_config.setdefault("query_injection_enabled", True)
    stored_config.setdefault("quality_gated_p2", False)
    stored_config.setdefault("lambda_quality", 0.25)
    stored_config.setdefault("p2_c2_grad_scale", 0.0)
    stored_config.setdefault("contribution_separated_aux_gradients", False)
    if stored_config != config.as_dict():
        raise RuntimeError("EBC-QP config mismatch")
    if metadata.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("EBC-QP source lock mismatch")


def _implementation_version(config: EBCQPConfig) -> str:
    if config.contribution_separated_aux_gradients:
        return TSGR_P2_IMPLEMENTATION_VERSION
    return QG_P2_IMPLEMENTATION_VERSION if config.quality_gated_p2 else EBC_QP_IMPLEMENTATION_VERSION


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


class StockQueryDiagnosticsValidator(EBCQPValidator):
    """Shared read-only stock Top-K coverage and rank diagnostics for A0 and TSGR."""

    def attach_diagnostic_model(self, model) -> None:
        heads = [module for module in model.modules() if hasattr(module, "enc_score_head") and hasattr(module, "anchors")]
        if len(heads) != 1:
            raise RuntimeError(f"expected one RT-DETR decoder during stock diagnostics, found {len(heads)}")
        self.stock_query_probe = StockQueryProbe(heads[0])

    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        if not hasattr(self, "stock_query_probe"):
            self.attach_diagnostic_model(model)
        self.stock_query_accumulator = StockTinyQueryAccumulator(query_budget=300, tiny_radius=16.0)
        self.stock_query_stats: dict[str, float | int | list[int]] = {}

    def update_metrics(self, preds, batch) -> None:
        super().update_metrics(preds, batch)
        scores, centers = self.stock_query_probe.consume()
        self.stock_query_accumulator.update(scores, centers, batch)

    def get_stats(self) -> dict:
        results = super().get_stats()
        self.stock_query_stats = self.stock_query_accumulator.compute()
        return results


class PairedProtocolOptimizerMixin:
    controlled_amp_scale: float | None = None

    def _setup_train(self):
        super()._setup_train()
        if self.controlled_amp_scale is not None:
            if not bool(self.amp) or not torch.cuda.is_available():
                raise RuntimeError("controlled AMP requires CUDA AMP")
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
                init_scale=float(self.controlled_amp_scale),
                growth_interval=2**31 - 1,
            )
            if float(self.scaler.get_scale()) != float(self.controlled_amp_scale):
                raise RuntimeError("controlled AMP scale initialization failed")

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

    def _initialize_optimizer_evidence(self) -> None:
        self.optimizer_attempt = 0
        self.optimizer_evidence_path = Path(self.save_dir) / "optimizer-evidence.jsonl"
        if self.optimizer_evidence_path.exists():
            raise FileExistsError(f"refusing to append changed optimizer evidence: {self.optimizer_evidence_path}")

    def _record_optimizer_evidence(self, record: dict[str, Any]) -> None:
        if self.controlled_amp_scale is None:
            return
        self.optimizer_attempt += 1
        payload = {"optimizer_attempt": self.optimizer_attempt, **record}
        nonfinite_fields = sorted(
            key for key, value in payload.items() if isinstance(value, float) and not math.isfinite(value)
        )
        for key in nonfinite_fields:
            payload[key] = None
        payload["nonfinite_fields"] = nonfinite_fields
        self.optimizer_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with self.optimizer_evidence_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        if payload["amp_step_skipped"]:
            raise RuntimeError(f"controlled AMP skipped optimizer attempt {self.optimizer_attempt}")
        if nonfinite_fields:
            raise FloatingPointError(
                f"optimizer attempt {self.optimizer_attempt} contains non-finite fields: {nonfinite_fields}"
            )
        if payload.get("runtime_violation"):
            raise RuntimeError(
                f"optimizer attempt {self.optimizer_attempt} violated the E1 protocol: "
                f"{payload['runtime_violation']}"
            )

    def optimizer_step(self):
        scale_before = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        stock_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        stock_norm = float(stock_norm_tensor.detach().float().item())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self._record_optimizer_evidence(
            {
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
                "amp_step_skipped": scale_after < scale_before,
                "gradient_clipping_mode": "pure_stock",
                "pure_stock_preclip_norm": stock_norm,
                "pure_stock_clip_coefficient": _clip_coefficient(stock_norm, 10.0),
                "routed_shallow_preclip_norm": 0.0,
                "routed_shallow_clip_coefficient": 1.0,
                "aux_private_preclip_norm": 0.0,
                "aux_private_clip_coefficient": 1.0,
                "shallow_applied_ratio": 0.0,
                "p2_entry_count": None,
                "ordinary_query_count": None,
                "protocol_expected_p2_entry_count": 0,
                "protocol_expected_ordinary_query_count": 300,
                "runtime_violation": None,
            }
        )

    def _retain_e1_tail_checkpoint(self) -> Path | None:
        if self.controlled_amp_scale is None or int(self.args.epochs) != 10 or int(self.epoch) not in {7, 8, 9}:
            return None
        if int(self.args.save_period) != -1:
            raise RuntimeError("E1 tail checkpoint retention requires save_period=-1")
        if not self.last.is_file():
            raise RuntimeError("E1 last checkpoint is missing before tail retention")
        destination = self.wdir / f"epoch{self.epoch}.pt"
        if destination.exists():
            raise FileExistsError(f"refusing to replace retained E1 checkpoint: {destination}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"refusing to replace partial retained E1 checkpoint: {temporary}")
        shutil.copyfile(self.last, temporary)
        temporary.replace(destination)
        return destination


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
        controlled_amp_scale: float | None = None,
        **kwargs,
    ):
        self.ebc_config = ebc_config or EBCQPConfig()
        self.initial_state_path = Path(initial_state_path) if initial_state_path is not None else None
        self.initial_state = _load_protocol_state(self.initial_state_path)
        self.controlled_amp_scale = controlled_amp_scale
        super().__init__(*args, **kwargs)
        if self.controlled_amp_scale is not None:
            self._initialize_optimizer_evidence()
        self.add_callback("on_train_epoch_start", self._set_ebc_progress)
        self.add_callback("on_train_batch_start", self._set_isolated_auxiliary_gradient_scale)

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
            gamma_key = "model.28.p2_fusion_gamma"
            ignored_innovation_keys = set()
            if self.ebc_config.contribution_separated_aux_gradients and gamma_key in self.initial_state["innovation_state"]:
                ignored_innovation_keys.add(gamma_key)
            load_initial_state(
                model,
                self.initial_state,
                include_innovation=True,
                ignored_innovation_keys=ignored_innovation_keys,
            )
            model.ignored_initial_innovation_keys = tuple(sorted(ignored_innovation_keys))
        return model

    def _build_train_pipeline(self):
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            unwrapped = unwrap_model(model)
            if isinstance(unwrapped, EBCQPDetectionModel):
                unwrapped.clear_isolated_auxiliary_gradients()
        _reset_paired_random_state(self.args.seed, self.args.deterministic)
        return super()._build_train_pipeline()

    def _set_ebc_progress(self, _trainer=None) -> None:
        epoch = int(self.epoch)
        unwrap_model(self.model).set_ebc_progress(epoch)
        ema_model = self.ema.ema if getattr(self, "ema", None) is not None else None
        if ema_model is not None:
            unwrap_model(ema_model).set_ebc_progress(epoch)

    def _set_isolated_auxiliary_gradient_scale(self, _trainer=None) -> None:
        if not self.ebc_config.contribution_separated_aux_gradients:
            return
        if RANK != -1:
            raise RuntimeError("contribution-separated TSGR gradients currently require single-GPU training")
        unwrap_model(self.model).set_isolated_auxiliary_gradient_scale(float(self.scaler.get_scale()))

    def get_validator(self):
        self.loss_names = QG_LOSS_NAMES if self.ebc_config.quality_gated_p2 else LOSS_NAMES
        return EBCQPValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def optimizer_step(self):
        if not hasattr(self, "update_monitor"):
            stock_parameters, p2_parameters = partition_optimizer_parameters(unwrap_model(self.model), self.optimizer)
            self._p2_clip_parameters = p2_parameters
            self._stock_clip_parameters = stock_parameters
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
        scale_before = float(self.scaler.get_scale())
        if self.ebc_config.contribution_separated_aux_gradients:
            self.last_gradient_clip_diagnostics = prepare_contribution_separated_gradients(
                unwrap_model(self.model),
                self.optimizer,
                self.scaler,
                self._stock_clip_parameters,
                self._p2_clip_parameters,
                max_norm=10.0,
            )
        else:
            self.scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            total_norm_value = float(total_norm.detach().float().item())
            self.last_gradient_clip_diagnostics = {
                "gradient_clipping_mode": "legacy_combined",
                "combined_preclip_norm": total_norm_value,
                "combined_clip_coefficient": _clip_coefficient(total_norm_value, 10.0),
            }
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        record = self.update_monitor.observe() if monitoring else None
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        state = unwrap_model(self.model).ebc_head.last_state
        diagnostics = self.last_gradient_clip_diagnostics
        shallow_stock = float(diagnostics.get("pure_stock_shallow_preclip_norm", 0.0))
        shallow_route = float(diagnostics.get("routed_shallow_preclip_norm", 0.0))
        stock_coefficient = float(diagnostics.get("pure_stock_clip_coefficient", 1.0))
        route_coefficient = float(diagnostics.get("routed_shallow_clip_coefficient", 1.0))
        applied_ratio = (
            shallow_route * route_coefficient / (shallow_stock * stock_coefficient)
            if shallow_stock * stock_coefficient > 0.0
            else 0.0
        )
        violations = []
        if state is None:
            violations.append("missing_forward_state")
        else:
            if int(state.p2_entry_count) != 0:
                violations.append(f"p2_entry_count={int(state.p2_entry_count)}")
            if int(state.ordinary_query_count) != 300:
                violations.append(f"ordinary_query_count={int(state.ordinary_query_count)}")
        if not bool(diagnostics["auxiliary_finite"]):
            violations.append("nonfinite_auxiliary_gradient")
        self._record_optimizer_evidence(
            {
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
                "amp_step_skipped": scale_after < scale_before,
                "gradient_clipping_mode": diagnostics["gradient_clipping_mode"],
                "pure_stock_preclip_norm": float(diagnostics["pure_stock_preclip_norm"]),
                "pure_stock_shallow_preclip_norm": shallow_stock,
                "pure_stock_clip_coefficient": stock_coefficient,
                "routed_shallow_preclip_norm": shallow_route,
                "routed_shallow_clip_coefficient": route_coefficient,
                "aux_private_preclip_norm": float(diagnostics["aux_private_preclip_norm"]),
                "aux_private_clip_coefficient": float(diagnostics["aux_private_clip_coefficient"]),
                "shallow_applied_ratio": applied_ratio,
                "auxiliary_finite": bool(diagnostics["auxiliary_finite"]),
                "update_monitor_ratio": float(record.ratio) if record is not None else None,
                "update_monitor_consecutive": int(self.update_monitor.consecutive),
                "update_monitor_abort": bool(record.abort) if record is not None else False,
                "p2_entry_count": int(state.p2_entry_count) if state is not None else -1,
                "ordinary_query_count": int(state.ordinary_query_count) if state is not None else -1,
                "runtime_violation": ";".join(violations) if violations else None,
            }
        )
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
        self._retain_e1_tail_checkpoint()
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
    def __init__(
        self,
        *args,
        initial_state_path: str | Path,
        controlled_amp_scale: float | None = None,
        **kwargs,
    ):
        self.initial_state_path = Path(initial_state_path)
        self.initial_state = _load_protocol_state(self.initial_state_path)
        self.controlled_amp_scale = controlled_amp_scale
        super().__init__(*args, **kwargs)
        if self.controlled_amp_scale is not None:
            self._initialize_optimizer_evidence()

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

    def save_model(self):
        saved = super().save_model()
        self._retain_e1_tail_checkpoint()
        return saved


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

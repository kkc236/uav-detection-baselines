from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ebc_qp_causal_audit import (  # noqa: E402
    capture_grouped_gradients,
    capture_grouped_tensor_mapping,
    capture_grouped_parameter_deltas,
    capture_grouped_values,
    clone_named_parameters,
    capture_parameter_delta_signatures,
    capture_parameter_delta_sha256,
    capture_parameter_signatures,
    capture_tensor_sha256,
    validate_audit_attempt,
)
from src.ebc_qp_protocol import state_fingerprint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one arm of the frozen 100-step A0/AUX E0 causal audit.")
    parser.add_argument("--arm", required=True, choices=("a0", "a0-repeat", "aux-audit", "h0", "h1"))
    parser.add_argument("--initial-state", required=True, type=Path)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100, choices=(100,))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--controlled-amp-scale", type=float, choices=(256.0,))
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("/mnt/uav/runs/ebc-qp-e0-causal-audit"))
    return parser


def resolved_audit_steps(args: argparse.Namespace) -> int:
    if args.controlled_amp_scale is not None:
        return 32
    return 1 if args.smoke else int(args.steps)


def resolved_run_name(args: argparse.Namespace) -> str:
    if args.controlled_amp_scale is not None:
        scale = int(args.controlled_amp_scale)
        return f"e0-{args.arm}-seed0-controlled-amp{scale}-32step"
    suffix = "-smoke" if args.smoke else ""
    return f"e0-{args.arm}-seed0{suffix}"


def controlled_amp_config(args: argparse.Namespace) -> dict[str, Any]:
    enabled = args.controlled_amp_scale is not None
    return {
        "enabled": enabled,
        "init_scale": float(args.controlled_amp_scale) if enabled else None,
        "growth_interval": 1000 if enabled else None,
        "require_zero_skips": enabled,
    }


def build_audit_ebc_config(arm: str):
    from src.ebc_qp_config import EBCQPConfig

    if arm in {"a0", "a0-repeat"}:
        return None
    if arm == "aux-audit":
        return EBCQPConfig(
            lambda_ebc=0.0,
            learnable_fusion_gamma=True,
            query_injection_enabled=False,
            quality_gated_p2=False,
        )
    if arm in {"h0", "h1"}:
        return EBCQPConfig(
            lambda_p2=0.1,
            lambda_quality=0.0,
            lambda_ebc=0.0,
            learnable_fusion_gamma=False,
            query_injection_enabled=False,
            quality_gated_p2=False,
            p2_c2_grad_scale=0.0 if arm == "h0" else 0.1,
            contribution_separated_aux_gradients=True,
        )
    raise ValueError(f"unsupported audit arm: {arm}")


def validate_controlled_amp_runtime(amp_enabled: bool, scaler: Any, config: Mapping[str, Any]) -> None:
    if not config.get("enabled"):
        return
    if not amp_enabled or not scaler.is_enabled():
        raise RuntimeError("controlled AMP audit requires AMP to remain enabled")
    actual_scale = float(scaler.get_scale())
    expected_scale = float(config["init_scale"])
    if actual_scale != expected_scale:
        raise RuntimeError(f"controlled AMP scale mismatch: expected {expected_scale}, got {actual_scale}")


def build_audit_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Return the arm-independent training protocol used by both E0 traces."""
    return {
        "data": str(Path(args.data).resolve()),
        "epochs": 10,
        "fraction": 1.0,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": args.device,
        "project": str(Path(args.project).resolve()),
        "name": "e0-causal-audit",
        "exist_ok": False,
        "resume": False,
        "pretrained": False,
        "cache": False,
        "amp": True,
        "deterministic": True,
        "seed": args.seed,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": False,
        "save_period": -1,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "plots": False,
        "val": False,
    }


def batch_fingerprint(batch: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest().upper()


def tensor_structure_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")

    update(value)
    return digest.hexdigest().upper()


def optimizer_common_manifest(
    optimizer: torch.optim.Optimizer,
    common_parameters: Mapping[str, torch.nn.Parameter],
) -> dict[str, dict[str, Any]]:
    by_id = {id(parameter): name for name, parameter in common_parameters.items()}
    result: dict[str, dict[str, Any]] = {}
    keys = ("param_group", "lr", "initial_lr", "momentum", "weight_decay", "nesterov", "use_muon")
    for group in optimizer.param_groups:
        metadata = {key: _json_scalar(group[key]) for key in keys if key in group}
        for parameter in group["params"]:
            name = by_id.get(id(parameter))
            if name is not None:
                if name in result:
                    raise ValueError(f"common parameter appears in multiple optimizer groups: {name}")
                result[name] = dict(metadata)
    missing = set(common_parameters) - set(result)
    if missing:
        raise ValueError(f"common parameters missing from optimizer: {sorted(missing)[:5]}")
    return dict(sorted(result.items()))


def rng_fingerprint() -> str:
    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest().upper()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    """Encode non-finite AMP-attempt evidence without producing invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in (
        (args.initial_state, "initial state"),
        (args.protocol_manifest, "protocol manifest"),
        (args.data, "data YAML"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    manifest = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    expected_state = str(args.initial_state.resolve())
    expected_data = str(args.data.resolve())
    if manifest.get("initial_state", {}).get("path") != expected_state:
        raise SystemExit("protocol manifest initial-state path mismatch")
    if manifest.get("initial_state", {}).get("sha256") != _sha256(args.initial_state):
        raise SystemExit("protocol manifest initial-state hash mismatch")
    if manifest.get("data", {}).get("path") != expected_data:
        raise SystemExit("protocol manifest data path mismatch")
    if manifest.get("seed") != args.seed:
        raise SystemExit("protocol manifest seed mismatch")
    return manifest


def _make_trainers():
    from ultralytics.utils.torch_utils import unwrap_model

    from src.ebc_qp_config import EBCQPConfig
    from src.rtdetr_ebc_qp import (
        EBCQPTrainer,
        PairedControlTrainer,
        partition_optimizer_parameters,
        prepare_contribution_separated_gradients,
    )

    class CausalAuditMixin:
        def __init__(
            self,
            *trainer_args,
            audit_arm: str,
            audit_output: Path,
            audit_steps: int,
            audit_metadata: dict[str, Any],
            audit_controlled_amp: dict[str, Any],
            **trainer_kwargs,
        ):
            self.audit_arm = audit_arm
            self.audit_output = Path(audit_output)
            self.audit_target_steps = int(audit_steps)
            self.audit_steps: list[dict[str, Any]] = []
            self.audit_successful_updates = 0
            self.audit_pending_batches: list[str] = []
            self.audit_pending_rng: list[str] = []
            self.audit_probe_mode = False
            self.audit_p2_only_stock_grad_l2 = 0.0
            self.audit_p2_only_stock_grad_grouped: dict[str, dict[str, float | int]] = {}
            self.audit_p2_only_stock_grad_parameters: dict[str, dict[str, float | int]] = {}
            self.audit_p2_only_aux_private_grad_l2 = 0.0
            self.audit_p2_only_aux_private_grad_parameters: dict[str, dict[str, float | int]] = {}
            self.audit_initial_probe: dict[str, Any] = {}
            self.audit_metadata = audit_metadata
            self.audit_controlled_amp = dict(audit_controlled_amp)
            super().__init__(*trainer_args, **trainer_kwargs)

        def _setup_train(self):
            super()._setup_train()
            (
                self.audit_initial_probe,
                self.audit_p2_only_stock_grad_l2,
                self.audit_p2_only_aux_private_grad_l2,
            ) = self._run_initial_probe(unwrap_model(self.model))
            for loader in (self.train_loader, self.test_loader):
                if hasattr(loader, "close"):
                    loader.close()
            self._build_train_pipeline()
            if self.audit_controlled_amp["enabled"]:
                self.scaler = torch.amp.GradScaler(
                    "cuda",
                    enabled=bool(self.amp),
                    init_scale=self.audit_controlled_amp["init_scale"],
                    growth_interval=self.audit_controlled_amp["growth_interval"],
                )
                validate_controlled_amp_runtime(bool(self.amp), self.scaler, self.audit_controlled_amp)
            self.validator = self.get_validator()
            self.scheduler.last_epoch = self.start_epoch - 1

            model = unwrap_model(self.model)
            common_names = set(self.initial_state["common_state"])
            named_parameters = dict(model.named_parameters())
            missing = common_names - set(named_parameters) - set(dict(model.named_buffers()))
            if missing:
                raise RuntimeError(f"common state missing from audit model: {sorted(missing)[:5]}")
            self.audit_common_parameters = {
                name: named_parameters[name] for name in sorted(common_names & set(named_parameters))
            }
            self.audit_common_buffer_names = common_names & set(dict(model.named_buffers()))
            self.audit_optimizer_manifest = optimizer_common_manifest(self.optimizer, self.audit_common_parameters)
            if getattr(getattr(self, "ebc_config", None), "contribution_separated_aux_gradients", False):
                self.audit_stock_parameters, self.audit_auxiliary_parameters = partition_optimizer_parameters(
                    model, self.optimizer
                )
            self.audit_common_fingerprint = state_fingerprint(self.initial_state["common_state"])
            self.audit_initial_state_sha256 = _sha256(self.initial_state_path)
            self._write_audit_trace()

        def _run_initial_probe(self, model) -> tuple[dict[str, Any], float, float]:
            self.audit_probe_mode = True
            raw_batch = next(iter(self.train_loader))
            raw_fingerprint = batch_fingerprint(raw_batch)
            batch = self.preprocess_batch(raw_batch)
            probe = deepcopy(model).train()
            probe.zero_grad(set_to_none=True)
            head = probe.model[-1]
            captured: dict[str, Any] = {}

            def capture_inputs(_module, args):
                captured["inputs"] = args[0]

            def capture_output(_module, _args, output):
                captured["output"] = output

            pre_hook = head.register_forward_pre_hook(capture_inputs)
            output_hook = head.register_forward_hook(capture_output)
            random_state = rng_fingerprint()
            with torch.autocast(device_type="cuda", enabled=bool(self.amp)):
                total, _items = probe(batch)
            pre_hook.remove()
            output_hook.remove()
            output = captured.get("output")
            inputs = captured.get("inputs")
            if not isinstance(output, tuple) or len(output) < 4 or not isinstance(inputs, list):
                raise RuntimeError("initial probe failed to capture RT-DETR head inputs/outputs")
            topk_indices = _stock_topk_indices(head, inputs)

            grad_norm = 0.0
            state = getattr(head, "last_state", None)
            if self.audit_arm in {"aux-audit", "h0", "h1"}:
                if state is None:
                    raise RuntimeError("AUX probe did not populate EBC-QP state")
                weight = self.ebc_config.lambda_p2 if self.audit_arm in {"h0", "h1"} else 1.0
                (weight * state.p2_loss).backward()
            common_names = set(self.initial_state["common_state"])
            common = {name: parameter for name, parameter in probe.named_parameters() if name in common_names}
            auxiliary_private = {
                name: parameter for name, parameter in probe.named_parameters() if name not in common_names
            }
            grouped = capture_grouped_gradients(common)
            self.audit_p2_only_stock_grad_grouped = grouped
            self.audit_p2_only_stock_grad_parameters = capture_parameter_signatures(_gradient_tensors(common))
            auxiliary_private_gradients = _gradient_tensors(auxiliary_private)
            self.audit_p2_only_aux_private_grad_parameters = capture_parameter_signatures(
                auxiliary_private_gradients
            )
            grad_norm = math.sqrt(sum(record["l2"] ** 2 for record in grouped.values()))
            auxiliary_private_grad_norm = _tensor_mapping_l2(auxiliary_private_gradients)
            probe_record = {
                "batch_fingerprint": raw_fingerprint,
                "rng_before_forward": random_state,
                "stock_topk_fingerprint": tensor_structure_fingerprint(topk_indices),
                "decoder_output_fingerprint": tensor_structure_fingerprint(output[:2]),
                "stock_output_fingerprint": tensor_structure_fingerprint(output[2:4]),
                "stock_loss": float((state.stock_loss if state is not None else total).detach().float().item()),
                "p2_loss": float(state.p2_loss.detach().float().item()) if state is not None else None,
                "p2_entry_count": int(state.p2_entry_count) if state is not None else 0,
                "ordinary_query_count": int(state.ordinary_query_count) if state is not None else 300,
                "stock_topk_shape": list(topk_indices.shape),
            }
            del probe, batch, raw_batch
            torch.cuda.empty_cache()
            self.audit_probe_mode = False
            self.audit_pending_batches.clear()
            self.audit_pending_rng.clear()
            return probe_record, grad_norm, auxiliary_private_grad_norm

        def preprocess_batch(self, batch):
            fingerprint = batch_fingerprint(batch)
            prepared = super().preprocess_batch(batch)
            if not self.audit_probe_mode:
                self.audit_pending_batches.append(fingerprint)
                self.audit_pending_rng.append(rng_fingerprint())
            return prepared

        def optimizer_step(self):
            model = unwrap_model(self.model)
            before = clone_named_parameters(self.audit_common_parameters)
            scale_before = float(self.scaler.get_scale())
            all_named_parameters = dict(model.named_parameters())
            auxiliary_parameters = {
                name: parameter for name, parameter in all_named_parameters.items() if name not in self.audit_common_parameters
            }
            contribution_separated = bool(
                getattr(getattr(self, "ebc_config", None), "contribution_separated_aux_gradients", False)
            )
            routed_shallow_grad_tensors: dict[str, torch.Tensor] = {}
            routed_shallow_grad_total_norm = 0.0
            routed_shallow_clip_coefficient = 1.0
            aux_private_clip_coefficient = 1.0
            combined_hypothetical_clip_coefficient = None
            if contribution_separated:
                clip_diagnostics = prepare_contribution_separated_gradients(
                    model,
                    self.optimizer,
                    self.scaler,
                    self.audit_stock_parameters,
                    self.audit_auxiliary_parameters,
                    max_norm=10.0,
                    capture_tensors=True,
                )
                stock_grad_tensors = {
                    name: tensor
                    for name, tensor in clip_diagnostics["pure_stock_preclip"].items()
                    if name in self.audit_common_parameters
                }
                auxiliary_grad_tensors = dict(clip_diagnostics["aux_private_preclip"])
                routed_shallow_grad_tensors = dict(clip_diagnostics["routed_shallow_preclip"])
                stock_grad_preclip = capture_grouped_tensor_mapping(stock_grad_tensors)
                stock_grad_total_norm = float(clip_diagnostics["pure_stock_preclip_norm"])
                aux_private_grad_total_norm = float(clip_diagnostics["aux_private_preclip_norm"])
                routed_shallow_grad_total_norm = float(clip_diagnostics["routed_shallow_preclip_norm"])
                stock_only_clip_coefficient = float(clip_diagnostics["pure_stock_clip_coefficient"])
                clip_coefficient = stock_only_clip_coefficient
                routed_shallow_clip_coefficient = float(
                    clip_diagnostics["routed_shallow_clip_coefficient"]
                )
                aux_private_clip_coefficient = float(clip_diagnostics["aux_private_clip_coefficient"])
                total_norm_value = stock_grad_total_norm
                clip_total_norm_finite = math.isfinite(total_norm_value)
                clip_norm_partition_relative_error = 0.0 if clip_total_norm_finite else float("nan")
                hypothetical_norm = math.sqrt(
                    stock_grad_total_norm**2
                    + routed_shallow_grad_total_norm**2
                    + aux_private_grad_total_norm**2
                )
                combined_hypothetical_clip_coefficient = _clip_coefficient_from_norm(hypothetical_norm)
                stock_grad_postclip_tensors = {
                    name: tensor
                    for name, tensor in clip_diagnostics["merged_postclip"].items()
                    if name in self.audit_common_parameters
                }
            else:
                self.scaler.unscale_(self.optimizer)
                stock_grad_tensors = _gradient_tensors(self.audit_common_parameters)
                stock_grad_preclip = capture_grouped_gradients(self.audit_common_parameters)
                stock_grad_total_norm = math.sqrt(
                    sum(record["l2"] ** 2 for record in stock_grad_preclip.values())
                )
                auxiliary_grad_tensors = _gradient_tensors(auxiliary_parameters)
                aux_private_grad_total_norm = _tensor_mapping_l2(auxiliary_grad_tensors)
                stock_only_clip_coefficient = _clip_coefficient_from_norm(stock_grad_total_norm)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                total_norm_value = float(total_norm.detach().float().item())
                clip_total_norm_finite = math.isfinite(total_norm_value)
                clip_coefficient = _clip_coefficient_from_norm(total_norm_value)
                reconstructed_norm = math.sqrt(stock_grad_total_norm**2 + aux_private_grad_total_norm**2)
                clip_norm_partition_relative_error = (
                    abs(total_norm_value - reconstructed_norm)
                    / max(total_norm_value, reconstructed_norm, 1e-12)
                    if math.isfinite(total_norm_value) and math.isfinite(reconstructed_norm)
                    else float("nan")
                )
                stock_grad_postclip_tensors = _gradient_tensors(self.audit_common_parameters)
            stock_grad_preclip_parameters = capture_parameter_signatures(stock_grad_tensors)
            stock_grad_preclip_sha256 = capture_tensor_sha256(stock_grad_tensors)
            stock_grad_preclip_finite = _tensor_mapping_all_finite(stock_grad_tensors)
            aux_private_grad_finite = _tensor_mapping_all_finite(auxiliary_grad_tensors)
            routed_shallow_grad_finite = _tensor_mapping_all_finite(routed_shallow_grad_tensors)
            all_grad_preclip_finite = (
                stock_grad_preclip_finite and aux_private_grad_finite and routed_shallow_grad_finite
            )
            routed_shallow_grad_parameters = capture_parameter_signatures(routed_shallow_grad_tensors)
            routed_shallow_grad_sha256 = capture_tensor_sha256(routed_shallow_grad_tensors)
            stock_grad_postclip = capture_grouped_tensor_mapping(stock_grad_postclip_tensors)
            stock_grad_postclip_parameters = capture_parameter_signatures(stock_grad_postclip_tensors)
            stock_grad_postclip_finite = _tensor_mapping_all_finite(stock_grad_postclip_tensors)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = float(self.scaler.get_scale())
            step_skipped = scale_after < scale_before
            if not step_skipped:
                self.audit_successful_updates += 1
            stock_delta = capture_grouped_parameter_deltas(before, self.audit_common_parameters)
            stock_delta_parameters = capture_parameter_delta_signatures(before, self.audit_common_parameters)
            stock_delta_sha256 = capture_parameter_delta_sha256(before, self.audit_common_parameters)
            stock_delta_tensors = {
                name: self.audit_common_parameters[name].detach().float() - before[name]
                for name in self.audit_common_parameters
            }
            stock_delta_finite = _tensor_mapping_all_finite(stock_delta_tensors)
            stock_delta_zero = _tensor_mapping_all_zero(stock_delta_tensors)
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

            buffers = dict(model.named_buffers())
            stock_bn_tensors = {
                name: buffers[name]
                for name in sorted(self.audit_common_buffer_names)
                if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
            }
            stock_bn = capture_grouped_values(stock_bn_tensors)
            stock_bn_parameters = capture_parameter_signatures(stock_bn_tensors)
            stock_bn_sha256 = capture_tensor_sha256(stock_bn_tensors)
            ema_parameters = dict(unwrap_model(self.ema.ema).named_parameters()) if self.ema else {}
            stock_ema_tensors = {
                name: ema_parameters[name] for name in self.audit_common_parameters if name in ema_parameters
            }
            stock_ema = capture_grouped_values(stock_ema_tensors)
            stock_ema_parameters = capture_parameter_signatures(stock_ema_tensors)
            stock_state = capture_grouped_values(self.audit_common_parameters)
            stock_state_parameters = capture_parameter_signatures(self.audit_common_parameters)
            optimizer_state_tensors = _optimizer_state_tensors(self.optimizer, self.audit_common_parameters)
            optimizer_state = capture_grouped_values(optimizer_state_tensors)
            optimizer_state_parameters = capture_parameter_signatures(optimizer_state_tensors)

            if step_skipped:
                clip_coefficient = None
                stock_only_clip_coefficient = None
                routed_shallow_clip_coefficient = None
                aux_private_clip_coefficient = None
                combined_hypothetical_clip_coefficient = None
                clip_norm_partition_relative_error = None

            loss_value = float(self.loss.detach().float().item())
            loss_item_values = [
                float(value) for value in self.loss_items.detach().float().cpu().reshape(-1)
            ]
            forward_state = getattr(getattr(model, "ebc_head", None), "last_state", None)

            step = len(self.audit_steps) + 1
            record = {
                    "optimizer_step": step,
                    "optimizer_attempt": step,
                    "successful_update": not step_skipped,
                    "successful_update_index": self.audit_successful_updates if not step_skipped else None,
                    "epoch": int(self.epoch),
                    "batch_fingerprints": list(self.audit_pending_batches),
                    "rng_before_forward": list(self.audit_pending_rng),
                    "stock_grad_preclip": stock_grad_preclip,
                    "stock_grad_preclip_parameters": stock_grad_preclip_parameters,
                    "stock_grad_preclip_sha256": stock_grad_preclip_sha256,
                    "stock_grad_preclip_finite": stock_grad_preclip_finite,
                    "stock_grad_postclip": stock_grad_postclip,
                    "stock_grad_postclip_parameters": stock_grad_postclip_parameters,
                    "stock_grad_postclip_finite": stock_grad_postclip_finite,
                    "gradient_clipping_mode": (
                        "contribution_separated" if contribution_separated else "legacy_combined"
                    ),
                    "clip_total_norm": total_norm_value,
                    "clip_total_norm_finite": clip_total_norm_finite,
                    "clip_coefficient": clip_coefficient,
                    "stock_only_clip_coefficient": stock_only_clip_coefficient,
                    "stock_grad_total_norm": stock_grad_total_norm,
                    "aux_private_grad_total_norm": aux_private_grad_total_norm,
                    "aux_private_grad_finite": aux_private_grad_finite,
                    "aux_private_clip_coefficient": aux_private_clip_coefficient,
                    "routed_shallow_grad_total_norm": routed_shallow_grad_total_norm,
                    "routed_shallow_grad_parameters": routed_shallow_grad_parameters,
                    "routed_shallow_grad_sha256": routed_shallow_grad_sha256,
                    "routed_shallow_grad_finite": routed_shallow_grad_finite,
                    "routed_shallow_clip_coefficient": routed_shallow_clip_coefficient,
                    "combined_hypothetical_clip_coefficient": combined_hypothetical_clip_coefficient,
                    "all_grad_preclip_finite": all_grad_preclip_finite,
                    "clip_norm_partition_relative_error": clip_norm_partition_relative_error,
                    "amp_scale_before": scale_before,
                    "amp_scale_after": scale_after,
                    "amp_step_skipped": step_skipped,
                    "stock_delta": stock_delta,
                    "stock_delta_parameters": stock_delta_parameters,
                    "stock_delta_sha256": stock_delta_sha256,
                    "stock_delta_finite": stock_delta_finite,
                    "stock_delta_zero": stock_delta_zero,
                    "stock_state": stock_state,
                    "stock_state_parameters": stock_state_parameters,
                    "model_parameters_finite": _tensor_mapping_all_finite(all_named_parameters),
                    "stock_bn": stock_bn,
                    "stock_bn_parameters": stock_bn_parameters,
                    "stock_bn_sha256": stock_bn_sha256,
                    "stock_bn_finite": _tensor_mapping_all_finite(stock_bn_tensors),
                    "stock_ema": stock_ema,
                    "stock_ema_parameters": stock_ema_parameters,
                    "stock_ema_finite": _tensor_mapping_all_finite(stock_ema_tensors),
                    "optimizer_state": optimizer_state,
                    "optimizer_state_parameters": optimizer_state_parameters,
                    "optimizer_state_finite": _tensor_mapping_all_finite(optimizer_state_tensors),
                    "optimizer_groups": _optimizer_group_runtime(self.optimizer),
                    "loss": loss_value,
                    "loss_items": loss_item_values,
                    "loss_finite": math.isfinite(loss_value),
                    "loss_items_finite": all(math.isfinite(value) for value in loss_item_values),
                    "p2_entry_count": int(forward_state.p2_entry_count) if forward_state is not None else 0,
                    "ordinary_query_count": (
                        int(forward_state.ordinary_query_count) if forward_state is not None else 300
                    ),
                }
            self.audit_steps.append(record)
            self.audit_pending_batches.clear()
            self.audit_pending_rng.clear()
            self._write_audit_trace()
            violations = validate_audit_attempt(record)
            if violations:
                raise RuntimeError(f"invalid audit attempt {step}: {'; '.join(violations)}")
            if self.audit_controlled_amp["require_zero_skips"] and step_skipped:
                raise RuntimeError(f"controlled AMP audit invalidated by a skipped step at attempt {step}")
            skipped_attempts = sum(bool(item["amp_step_skipped"]) for item in self.audit_steps)
            consecutive_skips = 0
            for item in reversed(self.audit_steps):
                if not item["amp_step_skipped"]:
                    break
                consecutive_skips += 1
            if skipped_attempts > 32 or consecutive_skips > 16:
                raise RuntimeError(
                    f"AMP overflow limit exceeded: total={skipped_attempts}, consecutive={consecutive_skips}"
                )
            if self.audit_successful_updates >= self.audit_target_steps:
                self.stop = True

        def _write_audit_trace(self):
            payload = {
                "format_version": 1,
                "evidence": self.audit_metadata,
                "arm": self.audit_arm,
                "target_optimizer_steps": self.audit_target_steps,
                "attempted_optimizer_steps": len(self.audit_steps),
                "completed_successful_updates": self.audit_successful_updates,
                "common_initial_fingerprint": self.audit_common_fingerprint,
                "initial_state_path": str(self.initial_state_path.resolve()),
                "initial_state_sha256": self.audit_initial_state_sha256,
                "optimizer_common_manifest": getattr(self, "audit_optimizer_manifest", {}),
                "controlled_amp": self.audit_controlled_amp,
                "p2_only_stock_grad_l2": self.audit_p2_only_stock_grad_l2,
                "p2_only_stock_grad_grouped": self.audit_p2_only_stock_grad_grouped,
                "p2_only_stock_grad_parameters": self.audit_p2_only_stock_grad_parameters,
                "p2_only_aux_private_grad_l2": self.audit_p2_only_aux_private_grad_l2,
                "p2_only_aux_private_grad_parameters": self.audit_p2_only_aux_private_grad_parameters,
                "initial_probe": self.audit_initial_probe,
                "ebc_config": getattr(getattr(self, "ebc_config", None), "as_dict", lambda: None)(),
                "ignored_initial_innovation_keys": list(
                    getattr(unwrap_model(self.model), "ignored_initial_innovation_keys", ())
                ),
                "steps": self.audit_steps,
            }
            _atomic_write_json(self.audit_output, payload)

    class CausalAuditControlTrainer(CausalAuditMixin, PairedControlTrainer):
        pass

    class CausalAuditAuxTrainer(CausalAuditMixin, EBCQPTrainer):
        pass

    return CausalAuditControlTrainer, CausalAuditAuxTrainer, EBCQPConfig


def _optimizer_state_tensors(
    optimizer: torch.optim.Optimizer,
    common_parameters: Mapping[str, torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in common_parameters.items():
        for state_name, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                tensors[f"{name}.{state_name}"] = value
    return tensors


def _gradient_tensors(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: parameter.grad for name, parameter in parameters.items() if parameter.grad is not None}


def _tensor_mapping_l2(tensors: Mapping[str, torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    total = torch.stack([value.detach().float().square().sum() for value in tensors.values()]).sum().sqrt()
    return float(total.item())


def _tensor_mapping_all_finite(tensors: Mapping[str, torch.Tensor]) -> bool:
    if not tensors:
        return True
    checks = torch.stack([torch.isfinite(value.detach()).all() for value in tensors.values()])
    return bool(checks.all().item())


def _tensor_mapping_all_zero(tensors: Mapping[str, torch.Tensor]) -> bool:
    if not tensors:
        return True
    checks = torch.stack([torch.count_nonzero(value.detach()) == 0 for value in tensors.values()])
    return bool(checks.all().item())


def _clip_coefficient_from_norm(total_norm: float, *, max_norm: float = 10.0) -> float | None:
    if not math.isfinite(total_norm):
        return None
    return min(1.0, max_norm / (total_norm + 1e-6))


def _stock_topk_indices(head, inputs: list[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        if hasattr(head, "_stock_query_set"):
            feats, shapes, _projected_p3 = head._project_stock_inputs(inputs[1:])
            _stock, indices = head._stock_query_set(feats, shapes)
            return indices.detach()
        feats, shapes = head._get_encoder_input(inputs)
        if head.dynamic or head.shapes != shapes:
            head.anchors, head.valid_mask = head._generate_anchors(
                shapes,
                dtype=feats.dtype,
                device=feats.device,
            )
            head.shapes = shapes
        features = head.enc_output(head.valid_mask * feats)
        scores = head.enc_score_head(features)
        return torch.topk(scores.max(-1).values, head.num_queries, dim=1).indices.detach()


def _optimizer_group_runtime(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    keys = ("param_group", "lr", "initial_lr", "momentum", "weight_decay", "nesterov", "use_muon")
    return [{key: _json_scalar(group[key]) for key in keys if key in group} for group in optimizer.param_groups]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return str(value)


def _build_evidence_metadata(args: argparse.Namespace, settings: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    import ultralytics

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise SystemExit("refusing causal audit with tracked repository changes")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "src" / "ebc_qp_causal_audit.py",
            ROOT / "src" / "rtdetr_ebc_qp.py",
            ROOT / "src" / "ebc_qp_decoder.py",
            ROOT / "src" / "ebc_qp_config.py",
        )
    }
    return {
        "git_commit": commit,
        "sources": sources,
        "command": list(sys.argv),
        "settings": settings,
        "controlled_amp": controlled_amp_config(args),
        "protocol_manifest_path": str(args.protocol_manifest.resolve()),
        "protocol_manifest_sha256": _sha256(args.protocol_manifest),
        "protocol_signature": manifest.get("signature"),
        "data_path": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke and args.controlled_amp_scale is not None:
        raise SystemExit("--smoke and --controlled-amp-scale are mutually exclusive")
    manifest = _validate_protocol(args)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing audit trace: {args.output}")

    settings = build_audit_settings(args)
    settings["name"] = resolved_run_name(args)
    settings["model"] = (
        "rtdetr-l.yaml"
        if args.arm in {"a0", "a0-repeat"}
        else str(ROOT / "configs" / "rtdetr-l-ebc-qp.yaml")
    )
    evidence = _build_evidence_metadata(args, settings, manifest)
    target_steps = resolved_audit_steps(args)
    ControlTrainer, AuxTrainer, EBCQPConfig = _make_trainers()
    common = {
        "overrides": settings,
        "initial_state_path": args.initial_state,
        "audit_arm": args.arm,
        "audit_output": args.output,
        "audit_steps": target_steps,
        "audit_metadata": evidence,
        "audit_controlled_amp": controlled_amp_config(args),
    }
    ebc_config = build_audit_ebc_config(args.arm)
    if ebc_config is None:
        trainer = ControlTrainer(**common)
    else:
        trainer = AuxTrainer(**common, ebc_config=ebc_config)
    trainer.train()
    if trainer.audit_successful_updates != target_steps:
        raise RuntimeError(
            f"audit stopped after {trainer.audit_successful_updates} successful updates, expected {target_steps}"
        )


if __name__ == "__main__":
    main()

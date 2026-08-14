"""P3-only PR-IRA graph integration for the mature FDR+BPDD detector."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from src.fdr_protocol import initialize_private_module, validate_fdr_initial_state
from src.pr_ira import PRIRA
from src.pr_ira_protocol import (
    pr_ira_private_update_enabled,
    validate_resume_authority,
)
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, FDRBPDDTrainer


BPDD_PR_IRA_MODEL_CFG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "rtdetr-l-fdr-bpdd-pr-ira.yaml"
)
PR_IRA_MODEL_INDEX = 22
PR_IRA_STATE_PREFIX = f"model.{PR_IRA_MODEL_INDEX}."
PR_IRA_PRIVATE_LR_MULTIPLIER = 0.1
_MODEL_KEY = re.compile(r"^model\.(\d+)\.(.+)$")


@dataclass
class _PRIRAFirewallEntry:
    parameter: nn.Parameter
    parameter_shape: torch.Size
    parameter_dtype: torch.dtype
    parameter_device: torch.device
    gradient: torch.Tensor


def remap_bpdd_pr_ira_shared_key(name: str) -> str:
    """Shift every post-P3 mature state key past the inserted PR-IRA layer."""

    match = _MODEL_KEY.match(name)
    if match is None:
        return name
    index = int(match.group(1))
    if index < PR_IRA_MODEL_INDEX:
        return name
    return f"model.{index + 1}.{match.group(2)}"


def load_fdr_bpdd_pr_ira_initial_state(
    model: FDRBPDDDetectionModel,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Load every mature FDR/BPDD tensor while preserving only PR-IRA state."""

    validate_fdr_initial_state(artifact)
    source = {
        **artifact["fdr_public_state"],
        **artifact["private_state"],
    }
    mapped = {
        remap_bpdd_pr_ira_shared_key(name): value
        for name, value in source.items()
    }
    if len(mapped) != len(source):
        raise ValueError("BPDD PR-IRA state alias produced duplicate target keys")

    target = model.state_dict()
    missing_shared = sorted(set(mapped) - set(target))
    pr_ira_private_keys = sorted(set(target) - set(mapped))
    if missing_shared:
        raise ValueError(
            f"FDR shared keys missing after PR-IRA insertion: {missing_shared[:5]}"
        )
    if not pr_ira_private_keys or any(
        not name.startswith(PR_IRA_STATE_PREFIX)
        for name in pr_ira_private_keys
    ):
        raise ValueError("only model.22 PR-IRA tensors may be private state")

    for name, expected in mapped.items():
        actual = target[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"FDR shared tensor contract changed: {name}")

    incompatible = model.load_state_dict(mapped, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected FDR shared keys: {sorted(incompatible.unexpected_keys)[:5]}"
        )
    missing_keys = sorted(incompatible.missing_keys)
    if missing_keys != pr_ira_private_keys:
        raise ValueError("FDR BPDD PR-IRA initial-state load was not isolated")

    loaded = model.state_dict()
    shared_mismatch = sum(
        not torch.equal(loaded[name].detach().cpu(), expected.detach().cpu())
        for name, expected in mapped.items()
    )
    if shared_mismatch:
        raise ValueError(f"FDR shared initialization mismatch: {shared_mismatch}")
    return {
        "shared_tensor_count": len(mapped),
        "shared_mismatch_count": shared_mismatch,
        "missing_keys": missing_keys,
        "pr_ira_private_keys": pr_ira_private_keys,
    }


def _exact_combined_resume_state(weights: Any) -> Mapping[str, torch.Tensor]:
    """Normalize only an exact combined module or explicit state mapping."""

    candidate = weights
    if isinstance(candidate, Mapping):
        if candidate.get("model") is not None:
            candidate = candidate["model"]
        elif candidate.get("ema") is not None:
            candidate = candidate["ema"]
        elif "state_dict" in candidate:
            candidate = candidate["state_dict"]

    if isinstance(candidate, nn.Module):
        if not isinstance(candidate, FDRBPDDPRIRADetectionModel):
            raise ValueError(
                "resume requires an exact combined FDR+BPDD+PR-IRA model/state"
            )
        return candidate.state_dict()
    if isinstance(candidate, Mapping) and all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in candidate.items()
    ):
        return candidate
    raise ValueError(
        "resume requires an exact combined FDR+BPDD+PR-IRA model/state"
    )


def load_exact_fdr_bpdd_pr_ira_resume_state(
    model: "FDRBPDDPRIRADetectionModel",
    weights: Any,
) -> None:
    """Strictly restore a combined checkpoint without key intersection."""

    source = _exact_combined_resume_state(weights)
    target = model.state_dict()
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        raise ValueError(
            "resume requires an exact combined FDR+BPDD+PR-IRA model/state: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    normalized: dict[str, torch.Tensor] = {}
    for name, expected in target.items():
        actual = source[name]
        if actual.shape != expected.shape:
            raise ValueError(
                "resume requires an exact combined FDR+BPDD+PR-IRA model/state: "
                f"shape mismatch for {name}"
            )
        if actual.dtype != expected.dtype:
            if not (actual.is_floating_point() and expected.is_floating_point()):
                raise ValueError(
                    "resume requires an exact combined FDR+BPDD+PR-IRA model/state: "
                    f"dtype mismatch for {name}"
                )
            actual = actual.to(dtype=expected.dtype)
        normalized[name] = actual

    try:
        model.load_state_dict(normalized, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "resume requires an exact combined FDR+BPDD+PR-IRA model/state"
        ) from error


def validate_pr_ira_resume_payload(
    weights: Any,
    expected_run_identity: Mapping[str, Any] | None,
) -> None:
    """Require exact source/protocol/data/run authority before tensor loading."""

    if not isinstance(expected_run_identity, Mapping):
        raise ValueError("resume requires an expected PR-IRA run identity")
    if not isinstance(weights, Mapping):
        raise ValueError("resume checkpoint is missing its PR-IRA run identity")
    checkpoint_identity = weights.get("pr_ira_run_identity")
    if not isinstance(checkpoint_identity, Mapping):
        train_args = weights.get("train_args")
        if isinstance(train_args, Mapping):
            checkpoint_identity = train_args.get("pr_ira_run_identity")
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError("resume checkpoint is missing its PR-IRA run identity")
    validate_resume_authority(checkpoint_identity, expected_run_identity)


class FDRBPDDPRIRADetectionModel(FDRBPDDDetectionModel):
    """FDR+BPDD model whose decoder alone sees one identity-safe P3 PR-IRA."""

    def __init__(
        self,
        cfg: str | Path | dict = BPDD_PR_IRA_MODEL_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
        experiment_seed: int = 0,
    ) -> None:
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            private_seed=private_seed,
        )
        if (
            len(self.model) != 30
            or not isinstance(self.model[PR_IRA_MODEL_INDEX], PRIRA)
            or sum(isinstance(module, PRIRA) for module in self.modules()) != 1
        ):
            raise TypeError(
                "PR-IRA must be the only standalone YAML layer at model index 22"
            )
        if self.model[PR_IRA_MODEL_INDEX].f != 21:
            raise ValueError("PR-IRA must consume the stock P3 RepC3 output at index 21")
        if self.model[23].f != 21:
            raise ValueError("stock P4 must bypass PR-IRA and consume model index 21")
        if self.model[-1].f != [22, 25, 28]:
            raise ValueError("FDR decoder must consume PR-IRA-P3 plus stock P4/P5")

        self.experiment_seed = int(experiment_seed)
        self.pr_ira_private_seed = 20_000 + self.experiment_seed
        adapter = self.pr_ira
        initialize_private_module(
            adapter,
            private_seed=self.pr_ira_private_seed,
            zero_final_layers=(adapter.channel_gate[-1], adapter.spatial_gate),
        )
        with torch.no_grad():
            adapter.amplitude.zero_()
        self.last_main_loss: torch.Tensor | None = None
        self.last_bpdd_loss: torch.Tensor | None = None
        private_parameters = tuple(adapter.parameters())
        self._pr_ira_private_parameter_ids = tuple(
            id(parameter) for parameter in private_parameters
        )
        self._pr_ira_firewall_buffer: dict[int, _PRIRAFirewallEntry] = {}
        self._pr_ira_firewall_subtracted = False

    @property
    def pr_ira(self) -> PRIRA:
        module = self.model[PR_IRA_MODEL_INDEX]
        if not isinstance(module, PRIRA):
            raise RuntimeError("PR-IRA graph layer was unexpectedly replaced")
        return module

    def pr_ira_private_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the identity-frozen private parameter tuple."""

        parameters = tuple(self.pr_ira.parameters())
        identifiers = tuple(id(parameter) for parameter in parameters)
        if identifiers != self._pr_ira_private_parameter_ids:
            raise RuntimeError("PR-IRA private parameter count or identity changed")
        return parameters

    @property
    def pr_ira_firewall_buffer_empty(self) -> bool:
        return not self._pr_ira_firewall_buffer and not self._pr_ira_firewall_subtracted

    @property
    def pr_ira_firewall_buffer_size(self) -> int:
        return len(self._pr_ira_firewall_buffer)

    def clear_pr_ira_firewall_buffer(self) -> None:
        """Reset the complete private-gradient optimizer-window state."""

        self._pr_ira_firewall_buffer.clear()
        self._pr_ira_firewall_subtracted = False

    def validate_pr_ira_firewall_buffer(self) -> None:
        """Validate parameter identity and every accumulated FP32 contribution."""

        parameters = self.pr_ira_private_parameters()
        if len(self._pr_ira_firewall_buffer) != len(parameters):
            raise RuntimeError("PR-IRA firewall buffer count mismatch")
        for parameter in parameters:
            entry = self._pr_ira_firewall_buffer.get(id(parameter))
            if entry is None or entry.parameter is not parameter:
                raise RuntimeError("PR-IRA firewall parameter identity changed")
            if parameter.shape != entry.parameter_shape:
                raise RuntimeError("PR-IRA firewall parameter shape changed")
            if parameter.dtype != entry.parameter_dtype:
                raise RuntimeError("PR-IRA firewall parameter dtype changed")
            if parameter.device != entry.parameter_device:
                raise RuntimeError("PR-IRA firewall parameter device changed")
            if entry.gradient.shape != parameter.shape:
                raise RuntimeError("PR-IRA firewall gradient shape mismatch")
            if entry.gradient.dtype != torch.float32:
                raise RuntimeError("PR-IRA firewall gradient dtype must be FP32")
            if entry.gradient.device != parameter.device:
                raise RuntimeError("PR-IRA firewall gradient device mismatch")
            if not bool(torch.isfinite(entry.gradient).all().item()):
                raise FloatingPointError("non-finite PR-IRA firewall gradient")

    def capture_pr_ira_firewall_gradient(self, loss_bpdd: torch.Tensor) -> None:
        """Accumulate one unscaled BPDD-private gradient contribution in FP32."""

        if self._pr_ira_firewall_subtracted:
            raise RuntimeError("PR-IRA firewall optimizer step is half-complete")
        parameters = self.pr_ira_private_parameters()
        try:
            gradients = torch.autograd.grad(
                loss_bpdd,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if len(gradients) != len(parameters):
                raise RuntimeError("PR-IRA firewall gradient count mismatch")

            contributions: list[torch.Tensor] = []
            for parameter, gradient in zip(parameters, gradients, strict=True):
                if gradient is None:
                    contribution = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                        memory_format=torch.preserve_format,
                    )
                else:
                    if gradient.shape != parameter.shape:
                        raise RuntimeError("PR-IRA firewall gradient shape mismatch")
                    if gradient.dtype != parameter.dtype:
                        raise RuntimeError("PR-IRA firewall gradient dtype mismatch")
                    if gradient.device != parameter.device:
                        raise RuntimeError("PR-IRA firewall gradient device mismatch")
                    contribution = gradient.detach().to(dtype=torch.float32)
                if not bool(torch.isfinite(contribution).all().item()):
                    raise FloatingPointError("non-finite PR-IRA firewall gradient")
                contributions.append(contribution)

            if self._pr_ira_firewall_buffer:
                self.validate_pr_ira_firewall_buffer()
                accumulated = []
                for parameter, contribution in zip(
                    parameters,
                    contributions,
                    strict=True,
                ):
                    candidate = (
                        self._pr_ira_firewall_buffer[id(parameter)].gradient
                        + contribution
                    )
                    if not bool(torch.isfinite(candidate).all().item()):
                        raise FloatingPointError(
                            "non-finite accumulated PR-IRA firewall gradient"
                        )
                    accumulated.append(candidate)
                for parameter, candidate in zip(
                    parameters,
                    accumulated,
                    strict=True,
                ):
                    self._pr_ira_firewall_buffer[id(parameter)].gradient = candidate
            else:
                self._pr_ira_firewall_buffer = {
                    id(parameter): _PRIRAFirewallEntry(
                        parameter=parameter,
                        parameter_shape=parameter.shape,
                        parameter_dtype=parameter.dtype,
                        parameter_device=parameter.device,
                        gradient=contribution,
                    )
                    for parameter, contribution in zip(
                        parameters,
                        contributions,
                        strict=True,
                    )
                }
        except BaseException:
            self.clear_pr_ira_firewall_buffer()
            raise

    def subtract_pr_ira_firewall_buffer(self) -> None:
        """Subtract accumulated BPDD-private gradients after AMP unscale."""

        if self._pr_ira_firewall_subtracted:
            raise RuntimeError("PR-IRA firewall optimizer step is already half-complete")
        if not self._pr_ira_firewall_buffer:
            raise RuntimeError("PR-IRA firewall buffer is empty")
        self.validate_pr_ira_firewall_buffer()
        actions: list[tuple[torch.Tensor, torch.Tensor]] = []
        for parameter in self.pr_ira_private_parameters():
            contribution = self._pr_ira_firewall_buffer[id(parameter)].gradient
            gradient = parameter.grad
            if gradient is None:
                if torch.count_nonzero(contribution).item() != 0:
                    raise RuntimeError(
                        "PR-IRA parameter gradient is missing for a BPDD contribution"
                    )
                continue
            if gradient.shape != parameter.shape:
                raise RuntimeError("PR-IRA parameter gradient shape changed")
            if gradient.dtype != parameter.dtype:
                raise RuntimeError("PR-IRA parameter gradient dtype changed")
            if gradient.device != parameter.device:
                raise RuntimeError("PR-IRA parameter gradient device changed")
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError("non-finite unscaled PR-IRA gradient")
            converted = contribution.to(dtype=gradient.dtype)
            candidate = gradient.detach().float() - contribution
            if not bool(torch.isfinite(converted).all().item()) or not bool(
                torch.isfinite(candidate).all().item()
            ):
                raise FloatingPointError("non-finite PR-IRA firewall subtraction")
            actions.append((gradient, converted))

        with torch.no_grad():
            for gradient, contribution in actions:
                gradient.sub_(contribution)
        self._pr_ira_firewall_subtracted = True

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the unchanged mature loss as exact main and BPDD components."""

        try:
            result = super().loss(batch, preds)
            bpdd_names = [
                name for name in self.last_fdr_losses if name == "loss_bpdd"
            ]
            if not self.training and not bpdd_names:
                self.last_main_loss = result[0]
                self.last_bpdd_loss = None
                return result
            if len(bpdd_names) != 1:
                raise RuntimeError(
                    "combined training loss requires exactly one loss_bpdd"
                )

            self.last_bpdd_loss = self.last_fdr_losses["loss_bpdd"]
            self.last_main_loss = sum(
                value
                for name, value in self.last_fdr_losses.items()
                if name != "loss_bpdd"
            )
            self.capture_pr_ira_firewall_gradient(self.last_bpdd_loss)
            return result
        except BaseException:
            self.clear_pr_ira_firewall_buffer()
            raise


class FDRBPDDPRIRATrainer(FDRBPDDTrainer):
    """FDR+BPDD trainer with strict combined graph construction and resume."""

    def __init__(
        self,
        *args: Any,
        pr_ira_run_identity: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.pr_ira_run_identity = (
            dict(pr_ira_run_identity)
            if pr_ira_run_identity is not None
            else None
        )
        super().__init__(*args, **kwargs)
        if self.pr_ira_run_identity is not None:
            setattr(
                self.args,
                "pr_ira_run_identity",
                dict(self.pr_ira_run_identity),
            )

    def build_optimizer(
        self,
        model: nn.Module,
        name: str = "MuSGD",
        lr: float = 0.01,
        momentum: float = 0.937,
        decay: float = 0.0005,
        iterations: float = 1e5,
    ) -> torch.optim.Optimizer:
        """Preserve stock groups while applying the frozen private LR ratio."""

        optimizer = super().build_optimizer(
            model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )
        combined = unwrap_model(model)
        if not isinstance(combined, FDRBPDDPRIRADetectionModel):
            raise TypeError("optimizer requires the combined PR-IRA model")
        private_ids = {
            id(parameter)
            for parameter in combined.pr_ira_private_parameters()
            if parameter.requires_grad
        }
        if not private_ids:
            raise RuntimeError("PR-IRA private optimizer group is empty")

        rebuilt: list[dict[str, Any]] = []
        seen: list[int] = []
        private_seen: set[int] = set()
        for original in optimizer.param_groups:
            public_parameters: list[nn.Parameter] = []
            private_parameters: list[nn.Parameter] = []
            for parameter in original["params"]:
                identifier = id(parameter)
                seen.append(identifier)
                if identifier in private_ids:
                    private_parameters.append(parameter)
                    private_seen.add(identifier)
                else:
                    public_parameters.append(parameter)

            if public_parameters:
                public_group = dict(original)
                public_group["params"] = public_parameters
                public_group.pop("pr_ira_private", None)
                rebuilt.append(public_group)
            if private_parameters:
                private_group = dict(original)
                private_group["params"] = private_parameters
                private_group["lr"] = (
                    float(original["lr"]) * PR_IRA_PRIVATE_LR_MULTIPLIER
                )
                if "initial_lr" in private_group:
                    private_group["initial_lr"] = (
                        float(original["initial_lr"])
                        * PR_IRA_PRIVATE_LR_MULTIPLIER
                    )
                private_group["pr_ira_private"] = True
                rebuilt.append(private_group)

        if len(seen) != len(set(seen)):
            raise RuntimeError("optimizer contains duplicate parameters")
        if private_seen != private_ids:
            raise RuntimeError("optimizer omitted PR-IRA private parameters")
        optimizer.param_groups[:] = rebuilt
        return optimizer

    def _firewall_model(self) -> FDRBPDDPRIRADetectionModel:
        model = unwrap_model(self.model)
        if not isinstance(model, FDRBPDDPRIRADetectionModel):
            raise TypeError("trainer requires the combined FDR+BPDD+PR-IRA model")
        return model

    def _model_train(self) -> None:
        """Apply the immutable relative-progress schedule once per epoch."""

        super()._model_train()
        self._firewall_model().pr_ira.set_training_progress(
            int(self.epoch) + 1,
            int(self.epochs),
        )

    def suppress_pr_ira_inactive_gradients(self) -> bool:
        """Prevent momentum or decay from moving inactive private state."""

        if hasattr(self, "epoch") and hasattr(self, "epochs"):
            current_epoch = int(self.epoch) + 1
            if pr_ira_private_update_enabled(current_epoch, int(self.epochs)):
                return False
        for parameter in self._firewall_model().pr_ira_private_parameters():
            parameter.grad = None
        return True

    def _clear_memory(self, threshold: float | None = None) -> None:
        """Reset accumulation only for OOM/final cleanup, never per epoch."""

        candidate = getattr(self, "model", None)
        if candidate is not None and threshold is None:
            unwrapped = unwrap_model(candidate)
            if isinstance(unwrapped, FDRBPDDPRIRADetectionModel):
                for parameter in unwrapped.parameters():
                    parameter.grad = None
                unwrapped.clear_pr_ira_firewall_buffer()
        super()._clear_memory(threshold)

    def reset_pr_ira_firewall_state(self) -> None:
        """Clear gradients and pending private contributions before a retry."""

        self.optimizer.zero_grad()
        self._firewall_model().clear_pr_ira_firewall_buffer()

    def optimizer_step(self) -> None:
        """Subtract BPDD-private gradients within the frozen stock step order."""

        model = self._firewall_model()
        scale_before = float(self.scaler.get_scale())
        try:
            self.scaler.unscale_(self.optimizer)
            model.subtract_pr_ira_firewall_buffer()
            self.suppress_pr_ira_inactive_gradients()
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=10.0,
            )
            value = float(norm.detach().float().cpu())
            finite_value = value if math.isfinite(value) else None
            self.last_gradient_norms = {"gradient_norm": finite_value}
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            model.clear_pr_ira_firewall_buffer()
            if self.ema:
                self.ema.update(self.model)
        except BaseException:
            self.optimizer.zero_grad()
            model.clear_pr_ira_firewall_buffer()
            raise

        scale_after = float(self.scaler.get_scale())
        skipped = scale_after < scale_before
        gradient_finite = finite_value is not None
        self._record_optimizer_evidence(
            {
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
                "amp_step_skipped": skipped,
                "gradient_norm": finite_value,
                "gradient_norm_finite": gradient_finite,
            }
        )
        if (
            scale_before != self.controlled_amp_scale
            or scale_after != self.controlled_amp_scale
        ):
            raise RuntimeError(
                f"fixed AMP scale changed: before={scale_before}, "
                f"after={scale_after}, expected={self.controlled_amp_scale}"
            )
        if skipped:
            raise RuntimeError("fixed AMP skipped an optimizer attempt")
        if not gradient_finite:
            raise FloatingPointError("non-finite paired gradient norm")

    def save_model(self) -> Any:
        """Allow checkpoints only between complete optimizer windows."""

        if not self._firewall_model().pr_ira_firewall_buffer_empty:
            raise RuntimeError("PR-IRA firewall buffer must be empty before save")
        return super().save_model()

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> FDRBPDDPRIRADetectionModel:
        del cfg
        model = FDRBPDDPRIRADetectionModel(
            BPDD_PR_IRA_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
            experiment_seed=self.experiment_seed,
        )
        if weights:
            validate_pr_ira_resume_payload(
                weights,
                getattr(self, "pr_ira_run_identity", None),
            )
            load_exact_fdr_bpdd_pr_ira_resume_state(model, weights)
        elif self.initial_state_path is not None:
            artifact = torch.load(
                Path(self.initial_state_path),
                map_location="cpu",
                weights_only=False,
            )
            load_fdr_bpdd_pr_ira_initial_state(model, artifact)
        return model


__all__ = [
    "BPDD_PR_IRA_MODEL_CFG",
    "FDRBPDDPRIRADetectionModel",
    "FDRBPDDPRIRATrainer",
    "PR_IRA_PRIVATE_LR_MULTIPLIER",
    "load_exact_fdr_bpdd_pr_ira_resume_state",
    "load_fdr_bpdd_pr_ira_initial_state",
    "remap_bpdd_pr_ira_shared_key",
    "validate_pr_ira_resume_payload",
]

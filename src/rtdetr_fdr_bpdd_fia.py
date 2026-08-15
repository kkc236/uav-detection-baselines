"""P3-only FIA integration for the FDR detector with training-only BPDD."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from ultralytics.utils import RANK

from src.fdr_protocol import initialize_private_module, validate_fdr_initial_state
from src.fia import FIA
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, FDRBPDDTrainer


BPDD_FIA_MODEL_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-bpdd-fia.yaml"
)
FIA_MODEL_INDEX = 22
FIA_STATE_PREFIX = f"model.{FIA_MODEL_INDEX}."
_MODEL_KEY = re.compile(r"^model\.(\d+)\.(.+)$")


def remap_bpdd_fia_shared_key(name: str) -> str:
    """Shift every post-P3 BPDD state key past the inserted FIA layer."""

    match = _MODEL_KEY.match(name)
    if match is None:
        return name
    index = int(match.group(1))
    if index < FIA_MODEL_INDEX:
        return name
    return f"model.{index + 1}.{match.group(2)}"


def load_fdr_bpdd_fia_initial_state(
    model: FDRBPDDDetectionModel,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Load all frozen FDR tensors exactly while preserving only FIA state."""

    validate_fdr_initial_state(artifact)
    source = {
        **artifact["fdr_public_state"],
        **artifact["private_state"],
    }
    mapped = {remap_bpdd_fia_shared_key(name): value for name, value in source.items()}
    if len(mapped) != len(source):
        raise ValueError("BPDD FIA state alias produced duplicate target keys")

    target = model.state_dict()
    missing_shared = sorted(set(mapped) - set(target))
    fia_private_keys = sorted(set(target) - set(mapped))
    if missing_shared:
        raise ValueError(
            f"FDR shared keys missing after FIA insertion: {missing_shared[:5]}"
        )
    if not fia_private_keys or any(
        not name.startswith(FIA_STATE_PREFIX) for name in fia_private_keys
    ):
        raise ValueError("only model.22 FIA tensors may be private state")

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
    if missing_keys != fia_private_keys:
        raise ValueError("FDR BPDD FIA initial-state load was not isolated")

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
        "fia_private_keys": fia_private_keys,
    }


def _exact_combined_resume_state(weights: Any) -> Mapping[str, torch.Tensor]:
    """Normalize only an exact combined module or explicit state mapping."""

    candidate = weights
    if isinstance(candidate, Mapping):
        if "model" in candidate:
            candidate = candidate["model"]
        elif "state_dict" in candidate:
            candidate = candidate["state_dict"]

    if isinstance(candidate, nn.Module):
        if not isinstance(candidate, FDRBPDDFIADetectionModel):
            raise ValueError(
                "resume requires an exact combined FDR+BPDD+FIA model/state"
            )
        return candidate.state_dict()
    if isinstance(candidate, Mapping) and all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in candidate.items()
    ):
        return candidate
    raise ValueError("resume requires an exact combined FDR+BPDD+FIA model/state")


def load_exact_fdr_bpdd_fia_resume_state(
    model: "FDRBPDDFIADetectionModel",
    weights: Any,
) -> None:
    """Strictly restore a combined checkpoint without key intersection."""

    source = _exact_combined_resume_state(weights)
    target = model.state_dict()
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        raise ValueError(
            "resume requires an exact combined FDR+BPDD+FIA model/state: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    normalized: dict[str, torch.Tensor] = {}
    for name, expected in target.items():
        actual = source[name]
        if actual.shape != expected.shape:
            raise ValueError(
                "resume requires an exact combined FDR+BPDD+FIA model/state: "
                f"shape mismatch for {name}"
            )
        if actual.dtype != expected.dtype:
            if not (actual.is_floating_point() and expected.is_floating_point()):
                raise ValueError(
                    "resume requires an exact combined FDR+BPDD+FIA model/state: "
                    f"dtype mismatch for {name}"
                )
            actual = actual.to(dtype=expected.dtype)
        normalized[name] = actual

    try:
        model.load_state_dict(normalized, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "resume requires an exact combined FDR+BPDD+FIA model/state"
        ) from error


class FDRBPDDFIADetectionModel(FDRBPDDDetectionModel):
    """FDR+BPDD model whose decoder alone sees one identity-safe P3 FIA."""

    def __init__(
        self,
        cfg: str | Path | dict = BPDD_FIA_MODEL_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
        fia_private_seed: int = 20_000,
    ) -> None:
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            private_seed=private_seed,
        )
        if len(self.model) != 30 or not isinstance(self.model[FIA_MODEL_INDEX], FIA):
            raise TypeError("FIA must be the standalone YAML layer at model index 22")
        if self.model[FIA_MODEL_INDEX].f != 21:
            raise ValueError("FIA must consume the stock P3 RepC3 output at index 21")
        if self.model[23].f != 21:
            raise ValueError("stock P4 must bypass FIA and consume model index 21")
        if self.model[-1].f != [22, 25, 28]:
            raise ValueError("FDR decoder must consume FIA-P3 plus stock P4/P5")

        self.fia_private_seed = int(fia_private_seed)
        cuda_devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            initialize_private_module(
                self.model[FIA_MODEL_INDEX],
                private_seed=self.fia_private_seed,
            )
        with torch.no_grad():
            self.model[FIA_MODEL_INDEX].residual_scale.zero_()

    @property
    def fia(self) -> FIA:
        module = self.model[FIA_MODEL_INDEX]
        if not isinstance(module, FIA):
            raise RuntimeError("FIA graph layer was unexpectedly replaced")
        return module


class FDRBPDDFIATrainer(FDRBPDDTrainer):
    """Strict FDR+BPDD trainer with an independently clipped FIA group."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        common: list[torch.nn.Parameter] = []
        fdr_private: list[torch.nn.Parameter] = []
        fia_private: list[torch.nn.Parameter] = []
        fia_ids = {id(parameter) for parameter in self.model.model[FIA_MODEL_INDEX].parameters()}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in fia_ids:
                fia_private.append(parameter)
            elif ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name:
                fdr_private.append(parameter)
            else:
                common.append(parameter)
        if not common or not fdr_private or not fia_private:
            raise RuntimeError("FDR BPDD FIA gradient partition is incomplete")
        return {
            "gradient_norm": common,
            "fdr_gradient_norm": fdr_private,
            "fia_gradient_norm": fia_private,
        }

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> FDRBPDDFIADetectionModel:
        del cfg
        model = FDRBPDDFIADetectionModel(
            BPDD_FIA_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
            fia_private_seed=20_000 + self.experiment_seed,
        )
        if weights:
            load_exact_fdr_bpdd_fia_resume_state(model, weights)
        elif self.initial_state_path is not None:
            artifact = torch.load(
                Path(self.initial_state_path),
                map_location="cpu",
                weights_only=False,
            )
            load_fdr_bpdd_fia_initial_state(model, artifact)
        return model


__all__ = [
    "BPDD_FIA_MODEL_CFG",
    "FDRBPDDFIADetectionModel",
    "FDRBPDDFIATrainer",
    "load_exact_fdr_bpdd_fia_resume_state",
    "load_fdr_bpdd_fia_initial_state",
    "remap_bpdd_fia_shared_key",
]

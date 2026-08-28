"""Isolated LRS system integrations for VisDrone arms G, H, and I."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from ultralytics.utils import RANK

from src.fdr_protocol import initialize_private_module, validate_fdr_initial_state
from src.fia import FIA
from src.rtdetr_fdr import FDRRTDETRDetectionModel, FDRTrainer, _load_initial_state
from src.rtdetr_fdr_bpdd import (
    FDRBPDDDetectionModel,
    FDRBPDDTrainer,
)


ROOT = Path(__file__).resolve().parents[1]
ARM_CONFIGS = {
    "g": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd.yaml",
    "h": ROOT / "configs" / "rtdetr-l-lrs-fdr-fia.yaml",
    "i": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd-fia.yaml",
}
FIA_MODEL_INDEX = 22
FIA_STATE_PREFIX = f"model.{FIA_MODEL_INDEX}."
_MODEL_KEY = re.compile(r"^model\.(\d+)\.(.+)$")
_FDR_PRIVATE_MARKERS = (
    ".dec_bbox_head.",
    ".decoder.pre_bbox_head.",
    ".decoder.distribution_feedback.",
)


def remap_fia_shared_key(name: str) -> str:
    """Shift every state key at or after the inserted FIA graph position."""

    match = _MODEL_KEY.match(name)
    if match is None:
        return name
    index = int(match.group(1))
    if index < FIA_MODEL_INDEX:
        return name
    return f"model.{index + 1}.{match.group(2)}"


def load_fia_initial_state(
    model: nn.Module,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly load a rebuilt non-FIA FDR artifact into an FIA graph."""

    validate_fdr_initial_state(artifact)
    source = {
        **artifact["fdr_public_state"],
        **artifact["private_state"],
    }
    mapped = {remap_fia_shared_key(name): value for name, value in source.items()}
    if len(mapped) != len(source):
        raise ValueError("FIA state remapping produced duplicate target keys")

    target = model.state_dict()
    unexpected = sorted(set(mapped) - set(target))
    private = sorted(set(target) - set(mapped))
    if unexpected:
        raise ValueError(
            f"FDR shared keys are missing after FIA insertion: {unexpected[:5]}"
        )
    if not private or any(
        not name.startswith(FIA_STATE_PREFIX) for name in private
    ):
        raise ValueError("only model.22 FIA tensors may remain private")

    for name, expected in mapped.items():
        actual = target[name]
        if actual.shape != expected.shape:
            raise ValueError(f"FDR shared tensor shape changed: {name}")
        if actual.dtype != expected.dtype:
            raise ValueError(f"FDR shared tensor dtype changed: {name}")

    incompatible = model.load_state_dict(mapped, strict=False)
    actual_missing = sorted(incompatible.missing_keys)
    actual_unexpected = sorted(incompatible.unexpected_keys)
    if actual_unexpected:
        raise ValueError(f"unexpected FDR shared keys: {actual_unexpected[:5]}")
    if actual_missing != private:
        raise ValueError("FIA initial-state missing-key contract changed")

    loaded = model.state_dict()
    mismatch_count = sum(
        not torch.equal(
            loaded[name].detach().cpu(),
            expected.detach().cpu(),
        )
        for name, expected in mapped.items()
    )
    if mismatch_count:
        raise ValueError(f"FDR shared initialization mismatch: {mismatch_count}")
    return {
        "shared_tensor_count": len(mapped),
        "shared_mismatch_count": mismatch_count,
        "missing_keys": actual_missing,
        "fia_private_keys": private,
    }


def load_fdr_initial_state_artifact(path: str | Path) -> Mapping[str, Any]:
    """Safely deserialize and validate an operator-supplied FDR artifact."""

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping):
        raise TypeError("FDR initial state must be a checkpoint mapping")
    validate_fdr_initial_state(artifact)
    return artifact


def initialize_fia_graph(model: nn.Module, *, private_seed: int) -> FIA:
    """Validate the P3-only FIA topology and initialize only its private layer."""

    graph = getattr(model, "model", None)
    if not isinstance(graph, nn.Sequential) or len(graph) != 30:
        raise ValueError("FIA graph must contain exactly 30 modules")
    fia = graph[FIA_MODEL_INDEX]
    if not isinstance(fia, FIA):
        raise TypeError("FIA must be the standalone YAML layer at model index 22")
    if fia.f != 21:
        raise ValueError("FIA must consume the stock P3 output at model index 21")
    if graph[23].f != 21:
        raise ValueError("stock P4 must bypass FIA and consume model index 21")
    if graph[-1].f != [22, 25, 28]:
        raise ValueError("FDR decoder must consume FIA-P3 plus stock P4/P5")

    initialize_private_module(fia, private_seed=int(private_seed))
    with torch.no_grad():
        fia.residual_scale.zero_()
    return fia


class LRSFDRBPDDDetectionModel(FDRBPDDDetectionModel):
    """Arm G: LRS-FDR with parameter-free BPDD and no FIA."""

    def __init__(
        self,
        cfg: str | Path | dict = ARM_CONFIGS["g"],
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            private_seed=private_seed,
        )


class LRSFDRFIADetectionModel(FDRRTDETRDetectionModel):
    """Arm H: LRS-FDR with one independently initialized P3 FIA."""

    def __init__(
        self,
        cfg: str | Path | dict = ARM_CONFIGS["h"],
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
        self.fia_private_seed = int(fia_private_seed)
        initialize_fia_graph(self, private_seed=self.fia_private_seed)

    @property
    def fia(self) -> FIA:
        module = self.model[FIA_MODEL_INDEX]
        if not isinstance(module, FIA):
            raise RuntimeError("FIA graph layer was unexpectedly replaced")
        return module


class LRSFDRBPDDFIADetectionModel(FDRBPDDDetectionModel):
    """Arm I: LRS-FDR with both training-only BPDD and P3 FIA."""

    def __init__(
        self,
        cfg: str | Path | dict = ARM_CONFIGS["i"],
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
        self.fia_private_seed = int(fia_private_seed)
        initialize_fia_graph(self, private_seed=self.fia_private_seed)

    @property
    def fia(self) -> FIA:
        module = self.model[FIA_MODEL_INDEX]
        if not isinstance(module, FIA):
            raise RuntimeError("FIA graph layer was unexpectedly replaced")
        return module


MODEL_TYPES = {
    "g": LRSFDRBPDDDetectionModel,
    "h": LRSFDRFIADetectionModel,
    "i": LRSFDRBPDDFIADetectionModel,
}


def _fia_gradient_parameter_groups(
    model: nn.Module,
) -> dict[str, list[torch.nn.Parameter]]:
    graph = getattr(model, "model", None)
    if not isinstance(graph, nn.Sequential) or len(graph) <= FIA_MODEL_INDEX:
        raise RuntimeError("FIA gradient partition requires the validated FIA graph")
    fia_ids = {id(parameter) for parameter in graph[FIA_MODEL_INDEX].parameters()}
    common: list[torch.nn.Parameter] = []
    fdr_private: list[torch.nn.Parameter] = []
    fia_private: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in fia_ids:
            fia_private.append(parameter)
        elif any(marker in name for marker in _FDR_PRIVATE_MARKERS):
            fdr_private.append(parameter)
        else:
            common.append(parameter)

    groups = {
        "gradient_norm": common,
        "fdr_gradient_norm": fdr_private,
        "fia_gradient_norm": fia_private,
    }
    identifiers = [[id(parameter) for parameter in group] for group in groups.values()]
    if any(not group for group in identifiers):
        raise RuntimeError("FDR/FIA gradient partition is incomplete")
    flattened = [identifier for group in identifiers for identifier in group]
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise RuntimeError("FDR/FIA gradient partition is not disjoint and exhaustive")
    return groups


def _load_fia_artifact(model: nn.Module, path: str | Path | None) -> None:
    if path is None:
        return
    artifact = load_fdr_initial_state_artifact(path)
    load_fia_initial_state(model, artifact)


class LRSFDRBPDDTrainer(FDRBPDDTrainer):
    """Arm G trainer with the normal strict FDR artifact loader."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> LRSFDRBPDDDetectionModel:
        del cfg
        model = LRSFDRBPDDDetectionModel(
            ARM_CONFIGS["g"],
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_initial_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="fdr",
            )
        return model


class LRSFDRFIATrainer(FDRTrainer):
    """Arm H trainer with independently clipped FDR and FIA parameters."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        return _fia_gradient_parameter_groups(self.model)

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> LRSFDRFIADetectionModel:
        del cfg
        if weights is not None:
            raise ValueError("FIA arms are fresh-only and reject checkpoint weights")
        model = LRSFDRFIADetectionModel(
            ARM_CONFIGS["h"],
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
            fia_private_seed=20_000 + self.experiment_seed,
        )
        _load_fia_artifact(model, getattr(self, "initial_state_path", None))
        return model


class LRSFDRBPDDFIATrainer(FDRBPDDTrainer):
    """Arm I trainer with independently clipped FDR and FIA parameters."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        return _fia_gradient_parameter_groups(self.model)

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> LRSFDRBPDDFIADetectionModel:
        del cfg
        if weights is not None:
            raise ValueError("FIA arms are fresh-only and reject checkpoint weights")
        model = LRSFDRBPDDFIADetectionModel(
            ARM_CONFIGS["i"],
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
            fia_private_seed=20_000 + self.experiment_seed,
        )
        _load_fia_artifact(model, getattr(self, "initial_state_path", None))
        return model


TRAINER_TYPES = {
    "g": LRSFDRBPDDTrainer,
    "h": LRSFDRFIATrainer,
    "i": LRSFDRBPDDFIATrainer,
}


__all__ = [
    "ARM_CONFIGS",
    "FIA_MODEL_INDEX",
    "FIA_STATE_PREFIX",
    "LRSFDRBPDDFIADetectionModel",
    "LRSFDRBPDDFIATrainer",
    "LRSFDRBPDDDetectionModel",
    "LRSFDRBPDDTrainer",
    "LRSFDRFIADetectionModel",
    "LRSFDRFIATrainer",
    "MODEL_TYPES",
    "ROOT",
    "TRAINER_TYPES",
    "initialize_fia_graph",
    "load_fdr_initial_state_artifact",
    "load_fia_initial_state",
    "remap_fia_shared_key",
]

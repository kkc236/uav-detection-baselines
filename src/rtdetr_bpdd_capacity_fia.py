"""Three-module LRS-GFDR + capacity-v2 AC-BPDD + FIA integration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from ultralytics.utils import RANK

from src.fdr_protocol import initialize_private_module, validate_fdr_initial_state
from src.fia import FIA
from src.rtdetr_bpdd_capacity import CapacityBPDDDetectionModel, CapacityBPDDTrainer


ROOT = Path(__file__).resolve().parents[1]
CAPACITY_BPDD_FIA_CFG = ROOT / "configs" / "rtdetr-l-lrs-gfdr-capacity-v2-bpdd-fia.yaml"
FIA_MODEL_INDEX = 22
FIA_STATE_PREFIX = f"model.{FIA_MODEL_INDEX}."
_MODEL_KEY = re.compile(r"^model\.(\d+)\.(.+)$")
_FDR_PRIVATE_MARKERS = (
    ".dec_bbox_head.",
    ".decoder.pre_bbox_head.",
    ".decoder.distribution_feedback.",
)


def remap_capacity_fia_shared_key(name: str) -> str:
    """Shift every source graph key at/after the inserted FIA layer by one."""

    match = _MODEL_KEY.match(name)
    if match is None:
        return name
    index = int(match.group(1))
    if index < FIA_MODEL_INDEX:
        return name
    return f"model.{index + 1}.{match.group(2)}"


def initialize_capacity_fia_graph(model: nn.Module, *, private_seed: int) -> FIA:
    """Validate the 30-layer graph and initialize only FIA's private state."""

    graph = getattr(model, "model", None)
    if not isinstance(graph, nn.Sequential) or len(graph) != 30:
        raise ValueError("capacity-v2 FIA graph must contain exactly 30 modules")
    fia = graph[FIA_MODEL_INDEX]
    if not isinstance(fia, FIA):
        raise TypeError("FIA must be the standalone YAML layer at model index 22")
    if fia.f != 21:
        raise ValueError("FIA must consume stock P3 output at model index 21")
    if graph[23].f != 21:
        raise ValueError("stock P4 must bypass FIA and consume model index 21")
    if graph[-1].f != [1, 22, 25, 28]:
        raise ValueError("capacity-v2 FDR decoder must consume P2/FIA-P3/P4/P5")
    initialize_private_module(fia, private_seed=int(private_seed))
    with torch.no_grad():
        fia.residual_scale.zero_()
    return fia


def load_capacity_fia_initial_state(
    model: nn.Module,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the frozen FDR artifact after the single FIA graph insertion."""

    validate_fdr_initial_state(artifact)
    source = {**artifact["fdr_public_state"], **artifact["private_state"]}
    mapped = {remap_capacity_fia_shared_key(name): value for name, value in source.items()}
    if len(mapped) != len(source):
        raise ValueError("capacity-v2 FIA remapping produced duplicate target keys")
    target = model.state_dict()
    missing = sorted(set(mapped) - set(target))
    extra = sorted(set(target) - set(mapped))
    allowed_prefixes = (FIA_STATE_PREFIX, "model.29.decoder.local_expert.")
    allowed_extra = [name for name in extra if any(name.startswith(prefix) for prefix in allowed_prefixes)]
    if missing or set(extra) != set(allowed_extra):
        raise ValueError(
            "capacity-v2 FIA initial-state keys do not match target model: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    expected = {**mapped, **{name: target[name] for name in allowed_extra}}
    incompatible = model.load_state_dict(expected, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("capacity-v2 FIA strict state loading produced incompatible keys")
    return {
        "shared_tensor_count": len(mapped),
        "missing_private_prefixes": sorted({
            prefix for prefix in allowed_prefixes
            if any(name.startswith(prefix) for name in allowed_extra)
        }),
    }


class CapacityBPDDFIADetectionModel(CapacityBPDDDetectionModel):
    """Capacity-v2 AC-BPDD detector with identity-safe P3 FIA enabled."""

    capacity_method_revision = "v2-fp32-extent-logspace-capacity-bpdd-fia"

    def __init__(
        self,
        cfg: str | Path | dict = CAPACITY_BPDD_FIA_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
        fia_private_seed: int = 20_000,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, private_seed=private_seed)
        self.fia_private_seed = int(fia_private_seed)
        initialize_capacity_fia_graph(self, private_seed=self.fia_private_seed)

    @property
    def fia(self) -> FIA:
        module = self.model[FIA_MODEL_INDEX]
        if not isinstance(module, FIA):
            raise RuntimeError("FIA graph layer was unexpectedly replaced")
        return module


class CapacityBPDDFIATrainer(CapacityBPDDTrainer):
    """Fresh-only trainer with disjoint common/FDR/expert/FIA diagnostics."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        graph = getattr(self.model, "model", None)
        if not isinstance(graph, nn.Sequential) or len(graph) <= FIA_MODEL_INDEX:
            raise RuntimeError("capacity-v2 FIA gradient partition requires validated graph")
        fia_ids = {id(parameter) for parameter in graph[FIA_MODEL_INDEX].parameters()}
        groups: dict[str, list[torch.nn.Parameter]] = {
            "gradient_norm": [], "fdr_gradient_norm": [],
            "expert_gradient_norm": [], "fia_gradient_norm": [],
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in fia_ids:
                groups["fia_gradient_norm"].append(parameter)
            elif ".decoder.local_expert." in name:
                groups["expert_gradient_norm"].append(parameter)
            elif any(marker in name for marker in _FDR_PRIVATE_MARKERS):
                groups["fdr_gradient_norm"].append(parameter)
            else:
                groups["gradient_norm"].append(parameter)
        if any(not values for values in groups.values()):
            raise RuntimeError("capacity-v2 FIA gradient partition is incomplete")
        identifiers = [id(value) for values in groups.values() for value in values]
        expected = {id(value) for value in self.model.parameters() if value.requires_grad}
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
            raise RuntimeError("capacity-v2 FIA gradient partition is not exhaustive")
        return groups

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> CapacityBPDDFIADetectionModel:
        del cfg
        if weights is not None:
            raise ValueError("capacity-v2 FIA arm is fresh-only")
        model = CapacityBPDDFIADetectionModel(
            CAPACITY_BPDD_FIA_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
            fia_private_seed=20_000 + self.experiment_seed,
        )
        path = getattr(self, "initial_state_path", None)
        if path is not None:
            artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
            if not isinstance(artifact, Mapping):
                raise TypeError("initial state must be a checkpoint mapping")
            load_capacity_fia_initial_state(model, artifact)
        return model


__all__ = [
    "CAPACITY_BPDD_FIA_CFG", "FIA_MODEL_INDEX", "FIA_STATE_PREFIX",
    "CapacityBPDDFIADetectionModel", "CapacityBPDDFIATrainer",
    "initialize_capacity_fia_graph", "load_capacity_fia_initial_state",
    "remap_capacity_fia_shared_key",
]

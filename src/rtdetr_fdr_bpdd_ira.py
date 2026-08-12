"""P3-only IRA integration for the FDR detector with training-only BPDD."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from ultralytics.utils import RANK

from src.fdr_protocol import initialize_private_module, validate_fdr_initial_state
from src.ira import IRA
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, FDRBPDDTrainer


BPDD_IRA_MODEL_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-bpdd-ira.yaml"
)
IRA_MODEL_INDEX = 22
IRA_STATE_PREFIX = f"model.{IRA_MODEL_INDEX}."
_MODEL_KEY = re.compile(r"^model\.(\d+)\.(.+)$")


def remap_bpdd_ira_shared_key(name: str) -> str:
    """Shift every post-P3 BPDD state key past the inserted IRA layer."""

    match = _MODEL_KEY.match(name)
    if match is None:
        return name
    index = int(match.group(1))
    if index < IRA_MODEL_INDEX:
        return name
    return f"model.{index + 1}.{match.group(2)}"


def load_fdr_bpdd_ira_initial_state(
    model: FDRBPDDDetectionModel,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Load all frozen FDR tensors exactly while preserving only IRA state."""

    validate_fdr_initial_state(artifact)
    source = {
        **artifact["fdr_public_state"],
        **artifact["private_state"],
    }
    mapped = {remap_bpdd_ira_shared_key(name): value for name, value in source.items()}
    if len(mapped) != len(source):
        raise ValueError("BPDD IRA state alias produced duplicate target keys")

    target = model.state_dict()
    missing_shared = sorted(set(mapped) - set(target))
    ira_private_keys = sorted(set(target) - set(mapped))
    if missing_shared:
        raise ValueError(
            f"FDR shared keys missing after IRA insertion: {missing_shared[:5]}"
        )
    if not ira_private_keys or any(
        not name.startswith(IRA_STATE_PREFIX) for name in ira_private_keys
    ):
        raise ValueError("only model.22 IRA tensors may be private state")

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
    if missing_keys != ira_private_keys:
        raise ValueError("FDR BPDD IRA initial-state load was not isolated")

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
        "ira_private_keys": ira_private_keys,
    }


class FDRBPDDIRADetectionModel(FDRBPDDDetectionModel):
    """FDR+BPDD model whose decoder alone sees one identity-safe P3 IRA."""

    def __init__(
        self,
        cfg: str | Path | dict = BPDD_IRA_MODEL_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
        ira_private_seed: int = 20_000,
    ) -> None:
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            private_seed=private_seed,
        )
        if len(self.model) != 30 or not isinstance(self.model[IRA_MODEL_INDEX], IRA):
            raise TypeError("IRA must be the standalone YAML layer at model index 22")
        if self.model[IRA_MODEL_INDEX].f != 21:
            raise ValueError("IRA must consume the stock P3 RepC3 output at index 21")
        if self.model[23].f != 21:
            raise ValueError("stock P4 must bypass IRA and consume model index 21")
        if self.model[-1].f != [22, 25, 28]:
            raise ValueError("FDR decoder must consume IRA-P3 plus stock P4/P5")

        self.ira_private_seed = int(ira_private_seed)
        initialize_private_module(
            self.model[IRA_MODEL_INDEX],
            private_seed=self.ira_private_seed,
        )
        with torch.no_grad():
            self.model[IRA_MODEL_INDEX].residual_scale.zero_()

    @property
    def ira(self) -> IRA:
        module = self.model[IRA_MODEL_INDEX]
        if not isinstance(module, IRA):
            raise RuntimeError("IRA graph layer was unexpectedly replaced")
        return module


class FDRBPDDIRATrainer(FDRBPDDTrainer):
    """Strict FDR+BPDD trainer with an independently clipped IRA group."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        common: list[torch.nn.Parameter] = []
        fdr_private: list[torch.nn.Parameter] = []
        ira_private: list[torch.nn.Parameter] = []
        ira_ids = {id(parameter) for parameter in self.model.model[IRA_MODEL_INDEX].parameters()}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in ira_ids:
                ira_private.append(parameter)
            elif ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name:
                fdr_private.append(parameter)
            else:
                common.append(parameter)
        if not common or not fdr_private or not ira_private:
            raise RuntimeError("FDR BPDD IRA gradient partition is incomplete")
        return {
            "gradient_norm": common,
            "fdr_gradient_norm": fdr_private,
            "ira_gradient_norm": ira_private,
        }

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> FDRBPDDIRADetectionModel:
        del cfg
        model = FDRBPDDIRADetectionModel(
            BPDD_IRA_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
            ira_private_seed=20_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        elif self.initial_state_path is not None:
            artifact = torch.load(
                Path(self.initial_state_path),
                map_location="cpu",
                weights_only=False,
            )
            load_fdr_bpdd_ira_initial_state(model, artifact)
        return model


__all__ = [
    "BPDD_IRA_MODEL_CFG",
    "FDRBPDDIRADetectionModel",
    "FDRBPDDIRATrainer",
    "load_fdr_bpdd_ira_initial_state",
    "remap_bpdd_ira_shared_key",
]

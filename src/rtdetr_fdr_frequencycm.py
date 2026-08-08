"""Strict FDR integration with one removable YAML-visible FrequencyCM layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from ultralytics.utils import RANK

from src.fdr_protocol import validate_fdr_initial_state
from src.frequency_cm import FrequencyCM
from src.rtdetr_fdr import FDRRTDETRDetectionModel, FDRTrainer


FDR_FREQUENCYCM_MODEL_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-frequencycm.yaml"
)
OLD_DECODER_PREFIX = "model.28."
NEW_DECODER_PREFIX = "model.29."
FREQUENCYCM_PREFIX = "model.28."


def remap_fdr_decoder_key(name: str) -> str:
    """Map the sole decoder-index shift introduced by the new YAML layer."""

    if name.startswith(OLD_DECODER_PREFIX):
        return NEW_DECODER_PREFIX + name[len(OLD_DECODER_PREFIX) :]
    return name


def load_fdr_frequencycm_initial_state(
    model: FDRRTDETRDetectionModel,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Load every FDR tensor exactly and leave only FrequencyCM privately initialized."""

    validate_fdr_initial_state(artifact)
    source = {
        **artifact["fdr_public_state"],
        **artifact["private_state"],
    }
    mapped = {remap_fdr_decoder_key(name): value for name, value in source.items()}
    if len(mapped) != len(source):
        raise ValueError("FDR decoder alias produced duplicate target keys")

    target = model.state_dict()
    missing_shared = sorted(set(mapped) - set(target))
    private_keys = sorted(set(target) - set(mapped))
    if missing_shared:
        raise ValueError(f"FDR shared keys missing after decoder alias: {missing_shared[:5]}")
    if not private_keys or any(not name.startswith(FREQUENCYCM_PREFIX) for name in private_keys):
        raise ValueError("only model.28 FrequencyCM tensors may be new private state")

    for name, expected in mapped.items():
        actual = target[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"FDR shared tensor contract changed: {name}")

    incompatible = model.load_state_dict(mapped, strict=False)
    if incompatible.unexpected_keys or sorted(incompatible.missing_keys) != private_keys:
        raise ValueError("FDR FrequencyCM initial-state load was not isolated")

    shared_mismatch = 0
    loaded = model.state_dict()
    for name, expected in mapped.items():
        if not torch.equal(loaded[name].detach().cpu(), expected.detach().cpu()):
            shared_mismatch += 1
    if shared_mismatch:
        raise ValueError(f"FDR shared initialization mismatch: {shared_mismatch}")
    return {
        "shared_tensor_count": len(mapped),
        "shared_mismatch_count": shared_mismatch,
        "private_keys": private_keys,
    }


class FDRFrequencyCMDetectionModel(FDRRTDETRDetectionModel):
    """FDR detector whose graph contains one YAML-visible FrequencyCM layer."""

    def __init__(
        self,
        cfg: str | Path | dict = FDR_FREQUENCYCM_MODEL_CFG,
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
        if len(self.model) != 30 or not isinstance(self.model[28], FrequencyCM):
            raise TypeError("FrequencyCM must be the standalone YAML layer at model index 28")


class FDRFrequencyCMTrainer(FDRTrainer):
    """Strict formal trainer for the isolated FDR + FrequencyCM arm."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        common: list[torch.nn.Parameter] = []
        fdr_private: list[torch.nn.Parameter] = []
        frequency_private: list[torch.nn.Parameter] = []
        frequency_ids = {id(parameter) for parameter in self.model.model[28].parameters()}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in frequency_ids:
                frequency_private.append(parameter)
            elif ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name:
                fdr_private.append(parameter)
            else:
                common.append(parameter)
        if not common or not fdr_private or not frequency_private:
            raise RuntimeError("FDR FrequencyCM gradient partition is incomplete")
        return {
            "gradient_norm": common,
            "fdr_gradient_norm": fdr_private,
            "frequencycm_gradient_norm": frequency_private,
        }

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> FDRFrequencyCMDetectionModel:
        model = FDRFrequencyCMDetectionModel(
            cfg or FDR_FREQUENCYCM_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        elif self.initial_state_path is not None:
            artifact = torch.load(
                Path(self.initial_state_path),
                map_location="cpu",
                weights_only=False,
            )
            load_fdr_frequencycm_initial_state(model, artifact)
        return model


__all__ = [
    "FDR_FREQUENCYCM_MODEL_CFG",
    "FDRFrequencyCMDetectionModel",
    "FDRFrequencyCMTrainer",
    "load_fdr_frequencycm_initial_state",
    "remap_fdr_decoder_key",
]

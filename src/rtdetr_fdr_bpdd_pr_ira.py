"""P3-only PR-IRA graph integration for the mature FDR+BPDD detector."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from ultralytics.utils import RANK

from src.fdr_protocol import initialize_private_module, validate_fdr_initial_state
from src.pr_ira import PRIRA
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, FDRBPDDTrainer


BPDD_PR_IRA_MODEL_CFG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "rtdetr-l-fdr-bpdd-pr-ira.yaml"
)
PR_IRA_MODEL_INDEX = 22
PR_IRA_STATE_PREFIX = f"model.{PR_IRA_MODEL_INDEX}."
_MODEL_KEY = re.compile(r"^model\.(\d+)\.(.+)$")


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
        if "model" in candidate:
            candidate = candidate["model"]
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

    @property
    def pr_ira(self) -> PRIRA:
        module = self.model[PR_IRA_MODEL_INDEX]
        if not isinstance(module, PRIRA):
            raise RuntimeError("PR-IRA graph layer was unexpectedly replaced")
        return module

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the unchanged mature loss as exact main and BPDD components."""

        result = super().loss(batch, preds)
        bpdd_names = [
            name for name in self.last_fdr_losses if name == "loss_bpdd"
        ]
        if not self.training and not bpdd_names:
            self.last_main_loss = result[0]
            self.last_bpdd_loss = None
            return result
        if len(bpdd_names) != 1:
            raise RuntimeError("combined training loss requires exactly one loss_bpdd")

        self.last_bpdd_loss = self.last_fdr_losses["loss_bpdd"]
        self.last_main_loss = sum(
            value
            for name, value in self.last_fdr_losses.items()
            if name != "loss_bpdd"
        )
        return result


class FDRBPDDPRIRATrainer(FDRBPDDTrainer):
    """FDR+BPDD trainer with strict combined graph construction and resume."""

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
    "load_exact_fdr_bpdd_pr_ira_resume_state",
    "load_fdr_bpdd_pr_ira_initial_state",
    "remap_bpdd_pr_ira_shared_key",
]

"""Minimal FDR+BPDD+RA-GLGM integration on the locked RA v1.1 graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch import nn
from ultralytics.utils import RANK

from src.bpdd_loss import BPDDDetectionLoss, BPDDOptions
from src.rtdetr_ra_glgm import (
    RAGLGMDetectionModel,
    RAGLGMTrainer,
    _load_pair_state,
)


FDR_BPDD_RA_GLGM_MODEL_CFG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "rtdetr-l-fdr-bpdd-ra-glgm.yaml"
)
_BPDD_OPTION_KEYS = {
    "enabled",
    "weight",
    "temperature",
    "margin",
    "eps",
    "matched_layer",
    "include_dn",
}


def _parse_bpdd_options(payload: dict[str, Any]) -> BPDDOptions:
    unknown = set(payload) - _BPDD_OPTION_KEYS
    if unknown:
        raise ValueError(f"unknown BPDD loss options: {sorted(unknown)}")
    if payload.get("matched_layer", "final") != "final":
        raise ValueError("BPDD v1 requires the final stock assignment")
    if bool(payload.get("include_dn", False)):
        raise ValueError("BPDD v1 excludes denoising Queries")
    return BPDDOptions(
        enabled=bool(payload.get("enabled", True)),
        weight=float(payload.get("weight", 0.5)),
        temperature=float(payload.get("temperature", 0.5)),
        margin=float(payload.get("margin", 0.02)),
        eps=float(payload.get("eps", 1e-6)),
    )


class _BPDDEpochStatistics:
    """Aggregate BPDD diagnostics by their exact eligible-edge denominator."""

    _WEIGHTED_FIELDS = (
        "active_edge_ratio",
        "mean_reliability",
        "mean_teacher_improvement",
        "mixture_beats_final_ratio",
        "mean_mixture_advantage_over_final",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._weighted_sums: dict[str, Tensor] = {}
        self._eligible_edges = 0
        self._matched_queries = 0

    @torch.no_grad()
    def update(self, statistics: dict[str, Tensor]) -> None:
        if not statistics:
            return
        eligible = int(statistics["eligible_edges"].detach().item())
        matched = int(statistics["matched_queries"].detach().item())
        if eligible < 0 or matched < 0:
            raise ValueError("BPDD epoch statistics contain a negative count")
        self._eligible_edges += eligible
        self._matched_queries += matched
        for name in self._WEIGHTED_FIELDS:
            value = statistics[name].detach().float()
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("NONFINITE_BPDD_EPOCH_STATISTIC")
            contribution = value * eligible
            self._weighted_sums[name] = (
                contribution
                if name not in self._weighted_sums
                else self._weighted_sums[name] + contribution
            )

    def values(self) -> dict[str, Tensor]:
        if not self._weighted_sums:
            raise RuntimeError("BPDD epoch contains no recorded training batch")
        sample = next(iter(self._weighted_sums.values()))
        denominator = max(self._eligible_edges, 1)
        return {
            **{
                name: value / denominator
                for name, value in self._weighted_sums.items()
            },
            "matched_queries": sample.new_tensor(float(self._matched_queries)),
            "eligible_edges": sample.new_tensor(float(self._eligible_edges)),
        }


class FDRBPDDRAGLGMDetectionModel(RAGLGMDetectionModel):
    """RA v1.1 model whose training criterion adds parameter-free BPDD."""

    def __init__(
        self,
        cfg: str | Path | dict = FDR_BPDD_RA_GLGM_MODEL_CFG,
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
        raw_options = self.yaml.get("bpdd_loss")
        if not isinstance(raw_options, dict):
            raise TypeError("FDR+BPDD+RA-GLGM YAML requires a bpdd_loss mapping")
        self.bpdd_options = _parse_bpdd_options(dict(raw_options))
        self.last_bpdd_statistics: dict[str, Tensor] = {}
        self._bpdd_epoch_statistics = _BPDDEpochStatistics()

    def reset_bpdd_epoch_statistics(self) -> None:
        self._bpdd_epoch_statistics.reset()

    def bpdd_epoch_statistics(self) -> dict[str, Tensor]:
        return self._bpdd_epoch_statistics.values()

    def init_criterion(self) -> BPDDDetectionLoss:
        return BPDDDetectionLoss(
            nc=self.nc,
            use_vfl=True,
            fgl_weight=float(self.fdr_loss_options.get("fgl_weight", 0.15)),
            supervise_pre_boxes=bool(
                self.fdr_loss_options.get("supervise_pre_boxes", True)
                and self.fdr.preliminary_box
            ),
            bpdd_options=self.bpdd_options,
        )

    def loss(
        self,
        batch: dict[str, Tensor],
        preds: tuple | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        if not isinstance(self.criterion, BPDDDetectionLoss):
            raise TypeError("BPDD model criterion was unexpectedly replaced")
        self.criterion.bpdd_runtime_enabled = bool(self.training)
        result = super().loss(batch, preds=preds)
        self.last_bpdd_statistics = dict(self.criterion.last_bpdd_statistics)
        if self.training:
            self._bpdd_epoch_statistics.update(self.last_bpdd_statistics)
        return result


class FDRBPDDRAGLGMTrainer(RAGLGMTrainer):
    """Keep the complete RA trainer contract while constructing the combo model."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | nn.Module | None = None,
        verbose: bool = True,
    ) -> FDRBPDDRAGLGMDetectionModel:
        model = FDRBPDDRAGLGMDetectionModel(
            cfg or FDR_BPDD_RA_GLGM_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            if not isinstance(weights, nn.Module):
                raise TypeError("combo resume weights must be a loaded checkpoint model")
            expected = model.state_dict()
            received = weights.state_dict()
            if set(expected) != set(received):
                missing = sorted(set(expected) - set(received))
                unexpected = sorted(set(received) - set(expected))
                raise ValueError(
                    "combo resume state keys differ: "
                    f"missing={missing[:3]}, unexpected={unexpected[:3]}"
                )
            mismatched = [
                name
                for name, value in expected.items()
                if value.shape != received[name].shape
            ]
            if mismatched:
                raise ValueError(f"combo resume state shapes differ: {mismatched[:3]}")
            model.load_state_dict(received, strict=True)
        else:
            _load_pair_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="ra_glgm",
            )
        return model


__all__ = [
    "FDR_BPDD_RA_GLGM_MODEL_CFG",
    "_BPDDEpochStatistics",
    "FDRBPDDRAGLGMDetectionModel",
    "FDRBPDDRAGLGMTrainer",
]

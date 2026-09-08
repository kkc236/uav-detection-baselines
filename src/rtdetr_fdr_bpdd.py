"""Training-only BPDD integration for the declarative FDR detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch import Tensor
from ultralytics.utils import RANK

from src.bpdd_loss import BPDDDetectionLoss, BPDDOptions
from src.rtdetr_fdr import (
    FDRRTDETRDetectionModel,
    FDRTrainer,
    _load_initial_state,
)


BPDD_MODEL_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-bpdd.yaml"
)


def _resolve_bpdd_model_cfg(cfg: dict | str | Path | None) -> dict | str | Path:
    """Use explicit BPDD candidates while preserving plain-FDR resume behavior."""

    if cfg is None:
        return BPDD_MODEL_CFG
    if isinstance(cfg, dict):
        return cfg if isinstance(cfg.get("bpdd_loss"), dict) else BPDD_MODEL_CFG
    path = Path(cfg)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return BPDD_MODEL_CFG
    return path if isinstance(payload, dict) and isinstance(payload.get("bpdd_loss"), dict) else BPDD_MODEL_CFG
_BPDD_OPTION_KEYS = {
    "enabled",
    "weight",
    "temperature",
    "margin",
    "eps",
    "matched_layer",
    "assignment_mode",
    "include_dn",
    "decoded_iou_gate",
    "iou_margin",
    "residual_gradient_only",
    "distribution_objective",
}


def _parse_bpdd_options(payload: dict[str, Any]) -> BPDDOptions:
    unknown = set(payload) - _BPDD_OPTION_KEYS
    if unknown:
        raise ValueError(f"unknown BPDD loss options: {sorted(unknown)}")
    if "assignment_mode" in payload and "matched_layer" in payload:
        raise ValueError("BPDD assignment mode must have one authority")
    assignment_mode = payload.get("assignment_mode")
    if assignment_mode is None:
        if payload.get("matched_layer", "final") != "final":
            raise ValueError("legacy BPDD requires the final stock assignment")
        assignment_mode = "final"
    if bool(payload.get("include_dn", False)):
        raise ValueError("BPDD excludes denoising Queries")
    return BPDDOptions(
        enabled=bool(payload.get("enabled", True)),
        weight=float(payload.get("weight", 0.5)),
        temperature=float(payload.get("temperature", 0.5)),
        margin=float(payload.get("margin", 0.02)),
        eps=float(payload.get("eps", 1e-6)),
        assignment_mode=str(assignment_mode),
        decoded_iou_gate=payload.get("decoded_iou_gate", False),
        iou_margin=float(payload.get("iou_margin", 0.0)),
        residual_gradient_only=payload.get("residual_gradient_only", False),
        distribution_objective=str(payload.get("distribution_objective", "kl")),
    )


class FDRBPDDDetectionModel(FDRRTDETRDetectionModel):
    """FDR model whose training criterion adds parameter-free BPDD."""

    def __init__(
        self,
        cfg: str | Path | dict = BPDD_MODEL_CFG,
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
            raise TypeError("BPDD model YAML requires a bpdd_loss mapping")
        self.bpdd_options = _parse_bpdd_options(dict(raw_options))
        self.last_bpdd_statistics: dict[str, Tensor] = {}

    def init_criterion(self) -> BPDDDetectionLoss:
        return BPDDDetectionLoss(
            nc=self.nc,
            use_vfl=True,
            fgl_weight=float(self.fdr_loss_options.get("fgl_weight", 0.15)),
            supervise_pre_boxes=bool(
                self.fdr_loss_options.get("supervise_pre_boxes", True)
                and self.fdr.preliminary_box
            ),
            supervise_dn_fdr=bool(
                self.fdr_loss_options.get("supervise_dn_fdr", True)
            ),
            edge_adaptive_fgl=bool(
                self.fdr_loss_options.get("edge_adaptive_fgl", False)
            ),
            reliability_shrinkage_alpha=float(
                self.fdr_loss_options.get("reliability_shrinkage_alpha", 0.0)
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
        result = super().loss(batch, preds)
        self.last_bpdd_statistics = dict(self.criterion.last_bpdd_statistics)
        return result


class FDRBPDDTrainer(FDRTrainer):
    """Use the exact FDR trainer and initialization with the BPDD criterion."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> FDRBPDDDetectionModel:
        # Resume checkpoints may carry the plain FDR YAML. BPDD has an identical
        # state contract, so normalize the graph authority to the candidate YAML
        # before strictly loading those tensors.
        model_cfg = _resolve_bpdd_model_cfg(cfg)
        model = FDRBPDDDetectionModel(
            model_cfg,
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


__all__ = [
    "BPDD_MODEL_CFG",
    "FDRBPDDDetectionModel",
    "FDRBPDDTrainer",
]

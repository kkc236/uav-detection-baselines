"""FDR+BPDD bridge on the RA Full100 synchronized FDR baseline."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor, nn
from ultralytics.utils import RANK

from src.bpdd_loss import BPDDDetectionLoss
from src.rtdetr_fdr import FDRRTDETRDetectionModel
from src.rtdetr_fdr_bpdd_ra_glgm import _BPDDEpochStatistics, _parse_bpdd_options
from src.rtdetr_ra_glgm import RAGLGMControlTrainer, _load_pair_state


FDR_BPDD_BRIDGE_MODEL_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-bpdd-bridge.yaml"
)


class FDRBPDDBridgeDetectionModel(FDRRTDETRDetectionModel):
    """RA-authority FDR graph with parameter-free training-only BPDD."""

    def __init__(
        self,
        cfg: str | Path | dict = FDR_BPDD_BRIDGE_MODEL_CFG,
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
            raise TypeError("FDR+BPDD bridge YAML requires a bpdd_loss mapping")
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
            raise TypeError("BPDD bridge criterion was unexpectedly replaced")
        self.criterion.bpdd_runtime_enabled = bool(self.training)
        result = super().loss(batch, preds=preds)
        self.last_bpdd_statistics = dict(self.criterion.last_bpdd_statistics)
        if self.training:
            self._bpdd_epoch_statistics.update(self.last_bpdd_statistics)
        return result


class FDRBPDDBridgeTrainer(RAGLGMControlTrainer):
    """Use A's dataset/gradient contract while constructing the B graph."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | nn.Module | None = None,
        verbose: bool = True,
    ) -> FDRBPDDBridgeDetectionModel:
        model = FDRBPDDBridgeDetectionModel(
            cfg or FDR_BPDD_BRIDGE_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            if not isinstance(weights, nn.Module):
                raise TypeError("BPDD bridge resume weights must be a loaded model")
            expected = model.state_dict()
            received = weights.state_dict()
            if set(expected) != set(received):
                raise ValueError("BPDD bridge resume state keys differ")
            mismatched = [
                name
                for name, value in expected.items()
                if value.shape != received[name].shape
            ]
            if mismatched:
                raise ValueError(
                    f"BPDD bridge resume state shapes differ: {mismatched[:3]}"
                )
            model.load_state_dict(received, strict=True)
        else:
            _load_pair_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="baseline",
            )
        return model


__all__ = [
    "FDR_BPDD_BRIDGE_MODEL_CFG",
    "FDRBPDDBridgeDetectionModel",
    "FDRBPDDBridgeTrainer",
]

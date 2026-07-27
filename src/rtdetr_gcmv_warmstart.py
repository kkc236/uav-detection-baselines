"""Mature-baseline calibration and matched fine-tuning trainers for GCMV-EI."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import RANK

from src.gcmv_warmstart import (
    PLEC_EXTRA_PREFIXES,
    load_baseline_checkpoint,
    load_module_artifact,
    open_residual_scalar,
    split_warmstart_optimizer_groups,
)
from src.rtdetr_gcmv_plec import (
    GCMVPLECControlTrainer,
    GCMVPLECDetectionModel,
    GCMVPLECTrainer,
    PLEC_DETECTION_LOSS_NAMES,
    PLEC_LOSS_NAMES,
)


DETECTOR_LR = 1e-4
MODULE_LR = 1e-3
RHO_LR = 1e-3
OPEN_GAMMA = 0.02


class GCMVWarmStartTrainer(GCMVPLECTrainer):
    """Common weights-only baseline loader with fresh fine-tuning state."""

    control_arm = False
    calibration_arm = False

    def __init__(
        self,
        *args,
        baseline_checkpoint_path: str | Path,
        calibrated_module_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        overrides = kwargs.get("overrides")
        if not isinstance(overrides, dict):
            raise ValueError("warm-start overrides must be a dict")
        if int(overrides.get("batch", 0)) != 8:
            raise ValueError("warm-start diagnostic requires batch=8")
        if int(overrides.get("seed", -1)) != 0:
            raise ValueError("warm-start diagnostic requires seed=0")
        self.baseline_checkpoint_path = Path(
            baseline_checkpoint_path
        ).resolve()
        if not self.baseline_checkpoint_path.is_file():
            raise FileNotFoundError(self.baseline_checkpoint_path)
        self.calibrated_module_path = (
            None
            if calibrated_module_path is None
            else Path(calibrated_module_path).resolve()
        )
        if (
            self.calibrated_module_path is not None
            and not self.calibrated_module_path.is_file()
        ):
            raise FileNotFoundError(self.calibrated_module_path)
        self.baseline_summary: dict = {}
        self.plec_optimizer_attempts = 0
        self.plec_amp_scale_min = 128.0
        self.plec_amp_scale_max = 128.0
        RTDETRTrainer.__init__(self, *args, **kwargs)

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights=None,
        verbose: bool = True,
    ):
        if weights is not None:
            raise ValueError(
                "warm-start checkpoint must be supplied explicitly"
            )
        model = GCMVPLECDetectionModel(
            cfg or "configs/rtdetr-l-gcmv-plec.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        self.baseline_summary = load_baseline_checkpoint(
            model,
            self.baseline_checkpoint_path,
        )
        if self.calibrated_module_path is not None:
            artifact = torch.load(
                self.calibrated_module_path,
                map_location="cpu",
                weights_only=False,
            )
            load_module_artifact(model, artifact)
        if self.control_arm:
            model.gcmv_enabled = False
            model.loss_names = PLEC_DETECTION_LOSS_NAMES
        elif self.calibration_arm:
            model.calibration_only = True
            model.gcmv_injector.peg.rho.data.zero_()
            for name, parameter in model.named_parameters():
                trainable = (
                    name.startswith(PLEC_EXTRA_PREFIXES)
                    and name != "gcmv_injector.peg.rho"
                )
                parameter.requires_grad_(trainable)
        else:
            if self.calibrated_module_path is None:
                raise ValueError(
                    "method fine-tuning requires a calibrated module"
                )
            open_residual_scalar(model, gamma=OPEN_GAMMA)
        return model

    def _apply_parameter_policy(self) -> None:
        """Reapply calibration freezing after Ultralytics' freeze pass.

        BaseTrainer deliberately re-enables floating-point parameters that are
        not covered by its index-based ``freeze`` argument.  Calibration uses a
        name-based module boundary, so this policy must run immediately before
        the optimizer is constructed.
        """

        if not self.calibration_arm:
            return
        for name, parameter in self.model.named_parameters():
            trainable = (
                name.startswith(PLEC_EXTRA_PREFIXES)
                and name != "gcmv_injector.peg.rho"
            )
            parameter.requires_grad_(trainable)

    def _build_train_pipeline(self) -> None:
        self._apply_parameter_policy()
        super()._build_train_pipeline()

    def build_optimizer(
        self,
        model,
        name="MuSGD",
        lr=DETECTOR_LR,
        momentum=0.937,
        decay=0.0005,
        iterations=1e5,
    ):
        optimizer = super().build_optimizer(
            model,
            name="MuSGD",
            lr=DETECTOR_LR,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )
        split_warmstart_optimizer_groups(
            optimizer,
            model=model,
            detector_lr=DETECTOR_LR,
            module_lr=MODULE_LR,
            rho_lr=RHO_LR,
            include_detector=not self.calibration_arm,
            include_module=not self.control_arm,
            include_rho=not self.calibration_arm,
        )
        return optimizer


class GCMVWarmStartControlTrainer(GCMVWarmStartTrainer):
    """Matched detector-only fine-tuning arm."""

    control_arm = True

    def _setup_train(self) -> None:
        super()._setup_train()
        self.loss_names = PLEC_DETECTION_LOSS_NAMES

    preprocess_batch = GCMVPLECControlTrainer.preprocess_batch


class GCMVWarmStartCalibrationTrainer(GCMVWarmStartTrainer):
    """One full-data auxiliary-only calibration epoch at exact gamma zero."""

    calibration_arm = True

    def _setup_train(self) -> None:
        super()._setup_train()
        self.loss_names = PLEC_LOSS_NAMES

    def save_model(self) -> bool:
        """Calibration persists a module-only artifact, never a detector."""

        return False

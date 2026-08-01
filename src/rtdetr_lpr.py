"""Ultralytics RT-DETR integration for localization-prior refinement."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

from src.lpr_head import LPRDeformableTransformerDecoder
from src.lpr_protocol import load_initial_state
from src.rtdetr_vsf_rmr import apply_resume_runtime_overrides


class LPRRTDETRDetectionModel(RTDETRDetectionModel):
    """Stock RT-DETR whose decoder outputs are refined by LPR heads."""

    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        max_logit_delta: float = 0.5,
        lpr_seed: int = 3407,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        head.decoder = LPRDeformableTransformerDecoder.from_stock(
            head.decoder,
            max_logit_delta=max_logit_delta,
            private_seed=lpr_seed,
        )
        self.nc = self.yaml["nc"]
        self.max_logit_delta = float(max_logit_delta)
        self.lpr_seed = int(lpr_seed)


class FixedPairedProtocolMixin:
    """Enforce MuSGD and fixed AMP128 for both paired experiment arms."""

    controlled_amp_scale = 128.0
    controlled_amp_growth_interval = 2**31 - 1

    def _resume_optimizer_attempt(self) -> int:
        path = self.optimizer_evidence_path
        if not path.exists():
            if self.resume:
                raise ValueError(f"resume optimizer evidence is missing: {path}")
            return 0
        if not self.resume:
            raise FileExistsError(f"refusing to append changed optimizer evidence: {path}")
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"resume optimizer evidence is unreadable: {path}") from error
        if not records:
            raise ValueError(f"resume optimizer evidence is empty: {path}")
        for attempt, record in enumerate(records, start=1):
            valid = (
                record.get("optimizer_attempt") == attempt
                and record.get("amp_scale_before") == self.controlled_amp_scale
                and record.get("amp_scale_after") == self.controlled_amp_scale
                and record.get("amp_step_skipped") is False
                and record.get("gradient_norm_finite") is True
            )
            if not valid:
                raise ValueError(f"resume optimizer evidence is invalid at attempt {attempt}: {path}")
        return len(records)

    def _setup_train(self):
        super()._setup_train()
        if not bool(self.amp) or not torch.cuda.is_available():
            raise RuntimeError("fixed paired AMP requires CUDA AMP")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=self.controlled_amp_scale,
            growth_interval=self.controlled_amp_growth_interval,
        )
        if float(self.scaler.get_scale()) != self.controlled_amp_scale:
            raise RuntimeError("fixed AMP scale initialization failed")
        self.optimizer_evidence_path = Path(self.save_dir) / "optimizer-evidence.jsonl"
        self.optimizer_attempt = self._resume_optimizer_attempt()

    def build_optimizer(self, model, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005, iterations=1e5):
        actual = {"name": name, "lr": lr, "momentum": momentum, "decay": decay}
        expected = {"name": "MuSGD", "lr": 0.01, "momentum": 0.937, "decay": 0.0005}
        if actual != expected:
            raise ValueError(f"paired optimizer must use exact MuSGD contract: expected={expected}, actual={actual}")
        return super().build_optimizer(
            model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )

    def _record_optimizer_evidence(self, record: dict) -> None:
        self.optimizer_attempt += 1
        payload = {"optimizer_attempt": self.optimizer_attempt, **record}
        self.optimizer_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with self.optimizer_evidence_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Return independently clipped parameter groups for one optimizer step."""
        return {
            "gradient_norm": [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
        }

    def optimizer_step(self):
        scale_before = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        norms: dict[str, float | None] = {}
        for name, parameters in self.gradient_parameter_groups().items():
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            value = float(norm.detach().float().cpu())
            norms[name] = value if math.isfinite(value) else None
        gradient_finite = all(value is not None for value in norms.values())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        skipped = scale_after < scale_before
        self._record_optimizer_evidence(
            {
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
                "amp_step_skipped": skipped,
                **norms,
                "gradient_norm_finite": gradient_finite,
            }
        )
        if scale_before != self.controlled_amp_scale or scale_after != self.controlled_amp_scale:
            raise RuntimeError(
                f"fixed AMP scale changed: before={scale_before}, after={scale_after}, "
                f"expected={self.controlled_amp_scale}"
            )
        if skipped:
            raise RuntimeError("fixed AMP skipped an optimizer attempt")
        if not gradient_finite:
            raise FloatingPointError("non-finite paired gradient norm")


def _load_frozen_state(model, initial_state_path: str | Path | None, *, variant: str) -> None:
    if initial_state_path is None:
        return
    artifact = torch.load(Path(initial_state_path), map_location="cpu", weights_only=False)
    load_initial_state(model, artifact, variant=variant)


class LPRTrainer(FixedPairedProtocolMixin, RTDETRTrainer):
    """Strict paired trainer that constructs the repository-owned LPR model."""

    def __init__(
        self,
        *args,
        max_logit_delta: float = 0.5,
        experiment_seed: int = 0,
        initial_state_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.max_logit_delta = float(max_logit_delta)
        self.experiment_seed = int(experiment_seed)
        self.initial_state_path = Path(initial_state_path) if initial_state_path is not None else None
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = LPRRTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            max_logit_delta=self.max_logit_delta,
            lpr_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_frozen_state(model, getattr(self, "initial_state_path", None), variant="lpr")
        return model


class FixedPairedControlTrainer(FixedPairedProtocolMixin, RTDETRTrainer):
    """Stock RT-DETR arm using exactly the same paired runtime contract."""

    def __init__(self, *args, initial_state_path: str | Path | None = None, **kwargs) -> None:
        self.initial_state_path = Path(initial_state_path) if initial_state_path is not None else None
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = RTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        else:
            _load_frozen_state(model, getattr(self, "initial_state_path", None), variant="control")
        return model

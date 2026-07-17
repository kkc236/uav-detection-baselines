from __future__ import annotations

from copy import copy
from pathlib import Path

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.plotting import feature_visualization

from src.vsf_rmr import VSFRMR, ordered_scale_weights
from src.vsf_rmr_loss import VSFRMRLossResult, compute_vsf_rmr_loss, ensure_finite_vsf_losses


LOSS_NAMES = (
    "giou_loss",
    "cls_loss",
    "l1_loss",
    "vsf_local_loss",
    "vsf_global_loss",
)


def apply_resume_runtime_overrides(args: object, overrides: dict) -> None:
    for key in ("amp", "project", "name", "optimizer", "lr0", "momentum"):
        if key in overrides:
            setattr(args, key, bool(overrides[key]) if key == "amp" else overrides[key])
    if "project" in overrides or "name" in overrides:
        project = Path(getattr(args, "project"))
        name = str(getattr(args, "name"))
        setattr(args, "save_dir", str((project / name).resolve()))


class VSFRMRDetectionModel(RTDETRDetectionModel):
    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        lambda_vsf: float = 0.1,
    ) -> None:
        self.lambda_vsf = float(lambda_vsf)
        self.last_vsf_result: VSFRMRLossResult | None = None
        self.last_vsf_diagnostics: dict[str, torch.Tensor | float] = {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.nc = self.yaml["nc"]
        self.loss_names = LOSS_NAMES
        self.vsf_rmr = VSFRMR(channels=256, route_channels=32)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for module in self.model[:-1]:
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
            if profile:
                self._profile_one_layer(module, x, dt)
            x = module(x)
            y.append(x if module.i in self.save else None)
            if visualize:
                feature_visualization(x, module.type, module.i, save_dir=visualize)
            if module.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
                if module.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        head = self.model[-1]
        decoder_features = self.vsf_rmr([y[index] for index in head.f])
        return head(decoder_features, batch)

    def loss(self, batch: dict, preds=None):
        detection_loss, detection_items = super().loss(batch, preds=preds)
        if not self.training:
            self.vsf_rmr.pop_auxiliary_state()
            return detection_loss, torch.cat((detection_items, detection_items.new_zeros(2)))

        state = self.vsf_rmr.pop_auxiliary_state()
        if state is None:
            raise RuntimeError("VSF-RMR auxiliary state was not captured during the training forward pass")

        result = compute_vsf_rmr_loss(
            scale_field=state.scale_field,
            global_scale=state.global_scale,
            bboxes=batch["bboxes"],
            batch_indices=batch["batch_idx"],
            image_size=tuple(int(value) for value in batch["img"].shape[-2:]),
        )
        auxiliary = result.local + result.global_
        total = detection_loss.float() + self.lambda_vsf * auxiliary
        ensure_finite_vsf_losses(detection=detection_loss.float(), total=total)

        alpha3, alpha4, alpha5 = ordered_scale_weights(state.scale_field.detach())
        self.last_vsf_result = result
        self.last_vsf_diagnostics = {
            "field_mean": state.scale_field.detach().mean(),
            "field_std": state.scale_field.detach().std(unbiased=False),
            "field_min": state.scale_field.detach().min(),
            "field_max": state.scale_field.detach().max(),
            "global_mean": state.global_scale.detach().mean(),
            "alpha3_mean": alpha3.mean(),
            "alpha4_mean": alpha4.mean(),
            "alpha5_mean": alpha5.mean(),
            "center_correlation": result.center_correlation.detach(),
            "gamma3_norm": self.vsf_rmr.gamma[0].detach().norm(),
            "gamma4_norm": self.vsf_rmr.gamma[1].detach().norm(),
            "gamma5_norm": self.vsf_rmr.gamma[2].detach().norm(),
        }
        loss_items = torch.cat(
            (
                detection_items,
                result.local.detach().reshape(1),
                result.global_.detach().reshape(1),
            )
        )
        return total, loss_items


class VSFRMRTrainer(RTDETRTrainer):
    def __init__(self, *args, lambda_vsf: float = 0.1, **kwargs) -> None:
        self.lambda_vsf = float(lambda_vsf)
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = VSFRMRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            lambda_vsf=self.lambda_vsf,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


class MatchedBaselineTrainer(RTDETRTrainer):
    """Stock RT-DETR trainer whose runtime controls remain adjustable after resume."""

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)

"""Strict paired RT-DETR-L+FDR versus FDR+RA-GLGM integration."""

from __future__ import annotations

from pathlib import Path
import torch
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.utils import RANK, colorstr

from src.btd_se_dataset import BTDSEVisDroneDataset
from src.ra_glgm_head import RAFDRRTDETRDecoder
from src.ra_glgm_loss import (
    ResidualDifficultyTargets,
    build_residual_difficulty_targets,
    instance_scale_predictions,
    residual_support_focal_loss,
    scale_conditioning_loss,
)
from src.ra_glgm_protocol import load_ra_glgm_initial_state
from src.rtdetr_btdse import filter_detection_batch
from src.rtdetr_fdr import (
    FDRRTDETRDetectionModel,
    FDRTrainer,
    _MODEL_PARSE_LOCK,
)


ROOT = Path(__file__).resolve().parents[1]
RA_GLGM_MODEL_CFG = ROOT / "configs" / "rtdetr-l-fdr-ra-glgm.yaml"
RA_GLGM_CONTROL_CFG = ROOT / "configs" / "rtdetr-l-fdr-ra-glgm-control.yaml"
RA_GLGM_AUXILIARY_WEIGHT = 0.05
RA_GLGM_SCALE_AUXILIARY_WEIGHT = 0.05


class _RAEpochScaleStatistics:
    """Accumulate exact instance-balanced scale/router statistics for one epoch."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._routes: list[torch.Tensor] = []
        self._loss_sum: torch.Tensor | None = None
        self._instances = 0
        self._route_delta_sum: torch.Tensor | None = None
        self._route_delta_max: torch.Tensor | None = None
        self._route_delta_count = 0
        self._scale_slope_rms: torch.Tensor | None = None
        self._scale_slope_max_abs: torch.Tensor | None = None

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: ResidualDifficultyTargets,
        scale_loss: torch.Tensor,
        route_weights: torch.Tensor,
        scale_slopes: torch.Tensor,
    ) -> None:
        predicted, target, routes = instance_scale_predictions(
            predictions.detach(), targets, route_weights.detach()
        )
        if not len(predicted):
            return
        if routes is None:
            raise RuntimeError("RA route diagnostics are unavailable")
        if scale_slopes.shape != (1, route_weights.shape[2], 1, 1):
            raise ValueError("RA scale slopes do not match router groups")
        count = len(predicted)
        self._predictions.append(predicted.detach().float())
        self._targets.append(target.detach().float())
        self._routes.append(routes.detach().float())
        loss_term = scale_loss.detach().float() * count
        self._loss_sum = (
            loss_term if self._loss_sum is None else self._loss_sum + loss_term
        )
        self._instances += count

        # Reconstruct the exact same-logit route obtained after removing only
        # the scale-conditioned antisymmetric bias.  This prevents an ordinary
        # content router from masquerading as active scale modulation.
        bounded_slopes = scale_slopes.detach().float().tanh().view(-1)
        centered_scale = predictions.detach().float().sub(0.5).mul(2.0)
        scale_bias = bounded_slopes.view(1, -1, 1, 1) * centered_scale
        actual_global = route_weights[:, 1].detach().float().clamp(1e-6, 1.0 - 1e-6)
        actual_log_odds = torch.log(actual_global) - torch.log1p(-actual_global)
        neutral_global = torch.sigmoid(actual_log_odds - 2.0 * scale_bias)
        route_delta = (actual_global - neutral_global).abs()
        delta_sum = route_delta.sum()
        delta_max = route_delta.max()
        self._route_delta_sum = (
            delta_sum
            if self._route_delta_sum is None
            else self._route_delta_sum + delta_sum
        )
        self._route_delta_max = (
            delta_max
            if self._route_delta_max is None
            else torch.maximum(self._route_delta_max, delta_max)
        )
        self._route_delta_count += route_delta.numel()
        self._scale_slope_rms = bounded_slopes.square().mean().sqrt()
        self._scale_slope_max_abs = bounded_slopes.abs().max()

    @staticmethod
    def _pearson(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = first - first.mean()
        second = second - second.mean()
        denominator = (first.square().sum() * second.square().sum()).sqrt()
        return torch.where(
            denominator > 0,
            (first * second).sum() / denominator.clamp_min(1e-12),
            torch.zeros_like(denominator),
        )

    @staticmethod
    def _average_ranks(values: torch.Tensor) -> torch.Tensor:
        """Return zero-based average ranks with exact tie handling."""

        ordered, order = torch.sort(values)
        boundaries = torch.ones_like(ordered, dtype=torch.bool)
        boundaries[1:] = ordered[1:] != ordered[:-1]
        groups = boundaries.cumsum(0) - 1
        counts = torch.bincount(groups)
        starts = counts.cumsum(0) - counts
        average = starts.float() + (counts.float() - 1.0) / 2.0
        ranked = torch.empty_like(values, dtype=torch.float32)
        ranked[order] = average[groups]
        return ranked

    def values(self) -> dict[str, torch.Tensor]:
        if (
            not self._instances
            or self._loss_sum is None
            or self._route_delta_sum is None
            or self._route_delta_max is None
            or self._scale_slope_rms is None
            or self._scale_slope_max_abs is None
        ):
            raise RuntimeError("RA scale epoch contains no instance supervision")
        predicted = torch.cat(self._predictions)
        target = torch.cat(self._targets)
        routes = torch.cat(self._routes)
        error = predicted - target
        route_load = routes.mean(dim=0)
        route_probabilities = torch.stack((1.0 - routes, routes), dim=1)
        route_entropy = -(
            route_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
            * route_probabilities
        ).sum(dim=1).mean()
        route_correlations = torch.stack(
            [self._pearson(target, routes[:, group]) for group in range(routes.shape[1])]
        )
        return {
            "loss_ra_scale": self._loss_sum / self._instances,
            "scale_instances": predicted.new_tensor(float(self._instances)),
            "scale_mae": error.abs().mean(),
            "scale_rmse": error.square().mean().sqrt(),
            "scale_prediction_mean": predicted.mean(),
            "scale_prediction_std": predicted.std(unbiased=False),
            "scale_target_mean": target.mean(),
            "scale_target_std": target.std(unbiased=False),
            "scale_pearson": self._pearson(predicted, target),
            "scale_spearman": self._pearson(
                self._average_ranks(predicted), self._average_ranks(target)
            ),
            "route_entropy": route_entropy,
            "route_global_mean": routes.mean(),
            "route_global_std": routes.std(unbiased=False),
            "route_load_min": route_load.min(),
            "route_load_max": route_load.max(),
            "scale_route_correlation_mean_abs": route_correlations.abs().mean(),
            "scale_route_correlation_max_abs": route_correlations.abs().max(),
            "scale_slope_rms": self._scale_slope_rms,
            "scale_slope_max_abs": self._scale_slope_max_abs,
            "scale_modulation_route_delta_mean": (
                self._route_delta_sum / self._route_delta_count
            ),
            "scale_modulation_route_delta_max": self._route_delta_max,
        }


def register_ra_glgm_decoder() -> None:
    ultralytics_tasks.RAFDRRTDETRDecoder = RAFDRRTDETRDecoder


class _IgnoreAwareFDRModel(FDRRTDETRDetectionModel):
    """Apply identical ignore-sidecar filtering in both paired arms."""

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().loss(filter_detection_batch(batch), preds=preds)


class RAGLGMControlDetectionModel(_IgnoreAwareFDRModel):
    """Unmodified FDR graph under the RA experiment's shared data path."""


class RAGLGMDetectionModel(FDRRTDETRDetectionModel):
    """FDR model with private P3 RA refinement and residual supervision."""

    yaml_decoder_type = RAFDRRTDETRDecoder

    def __init__(
        self,
        cfg: str | Path | dict = RA_GLGM_MODEL_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
    ) -> None:
        with _MODEL_PARSE_LOCK:
            register_ra_glgm_decoder()
            super().__init__(
                cfg=cfg,
                ch=ch,
                nc=nc,
                verbose=verbose,
                private_seed=private_seed,
            )
        if not isinstance(self.model[-1], RAFDRRTDETRDecoder):
            raise TypeError("RA-GLGM YAML must end with RAFDRRTDETRDecoder")
        self.last_ra_glgm_losses: dict[str, torch.Tensor] = {}
        self._ra_epoch_scale_statistics = _RAEpochScaleStatistics()

    @property
    def ra_glgm(self):
        return self.model[-1].ra_glgm

    def reset_ra_glgm_epoch_statistics(self) -> None:
        self._ra_epoch_scale_statistics.reset()

    def ra_glgm_epoch_statistics(self) -> dict[str, torch.Tensor]:
        return self._ra_epoch_scale_statistics.values()

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detection_batch = filter_detection_batch(batch)
        if preds is None:
            image = batch["img"]
            batch_index = detection_batch["batch_idx"]
            targets = {
                "cls": detection_batch["cls"].to(image.device, dtype=torch.long).view(-1),
                "bboxes": detection_batch["bboxes"].to(image.device),
                "batch_idx": batch_index.to(image.device, dtype=torch.long).view(-1),
                "gt_groups": [
                    int((batch_index == index).sum().item())
                    for index in range(int(image.shape[0]))
                ],
            }
            preds = self.predict(image, batch=targets)

        detection_loss, displayed = super().loss(detection_batch, preds=preds)
        if not self.training:
            self.last_ra_glgm_losses = {}
            return detection_loss, displayed
        if not isinstance(preds, tuple) or len(preds) != 5:
            raise RuntimeError("RA-GLGM training prediction contract changed")
        dec_bboxes, dec_scores, _, _, dn_meta = preds
        if dn_meta is not None:
            partition = dn_meta.get("dn_num_split")
            if not isinstance(partition, (list, tuple)) or len(partition) != 2:
                raise ValueError("RA-GLGM denoising partition is invalid")
            _, dec_bboxes = torch.split(dec_bboxes, tuple(map(int, partition)), dim=2)
            _, dec_scores = torch.split(dec_scores, tuple(map(int, partition)), dim=2)

        assignment = self.criterion.last_normal_decoder_assignment
        if assignment is None:
            raise RuntimeError("final normal FDR assignment is unavailable")
        support = self.ra_glgm.last_support_map
        if support is None:
            raise RuntimeError("RA-GLGM support map is unavailable")
        scale_values = self.ra_glgm.last_scale_values
        route_weights = self.ra_glgm.last_route_weights
        if scale_values is None:
            raise RuntimeError("RA-GLGM continuous scale values are unavailable")
        if route_weights is None:
            raise RuntimeError("RA-GLGM route weights are unavailable")
        targets = build_residual_difficulty_targets(
            pred_bboxes=dec_bboxes[-1],
            pred_scores=dec_scores[-1],
            detection_bboxes=detection_batch["bboxes"],
            detection_classes=detection_batch["cls"],
            detection_batch_idx=detection_batch["batch_idx"],
            match_indices=assignment,
            all_bboxes=batch["bboxes"],
            all_classes=batch["cls"],
            all_batch_idx=batch["batch_idx"],
            height=int(support.shape[-2]),
            width=int(support.shape[-1]),
        )
        auxiliary = residual_support_focal_loss(support, targets)
        scale_auxiliary = scale_conditioning_loss(scale_values, targets)
        if not bool(torch.isfinite(auxiliary)) or not bool(torch.isfinite(scale_auxiliary)):
            raise FloatingPointError("NONFINITE_RA_GLGM_AUXILIARY_LOSS")
        self._ra_epoch_scale_statistics.update(
            scale_values,
            targets,
            scale_auxiliary,
            route_weights,
            self.ra_glgm.scale_expert_slopes,
        )
        self.last_ra_glgm_losses = {
            "loss_ra_support": auxiliary.detach(),
            "loss_ra_scale": scale_auxiliary.detach(),
            "target_mean": targets.heatmap.mean().detach(),
            "valid_fraction": targets.valid_mask.float().mean().detach(),
        }
        return (
            detection_loss
            + RA_GLGM_AUXILIARY_WEIGHT * auxiliary
            + RA_GLGM_SCALE_AUXILIARY_WEIGHT * scale_auxiliary,
            displayed,
        )


class _IgnoreSidecarDatasetMixin:
    def build_dataset(
        self,
        img_path: str,
        mode: str = "val",
        batch: int | None = None,
    ):
        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)
        return BTDSEVisDroneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )


def _load_pair_state(model, path: Path | None, *, variant: str) -> None:
    if path is None:
        return
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    load_ra_glgm_initial_state(model, artifact, variant=variant)


class RAGLGMTrainer(_IgnoreSidecarDatasetMixin, FDRTrainer):
    """Method arm with separately clipped common, FDR, and RA gradients."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        ra_ids = {id(parameter) for parameter in self.model.ra_glgm.parameters()}
        common: list[torch.nn.Parameter] = []
        fdr: list[torch.nn.Parameter] = []
        ra: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in ra_ids:
                ra.append(parameter)
            elif ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name:
                fdr.append(parameter)
            else:
                common.append(parameter)
        if not common or not fdr or not ra:
            raise RuntimeError("RA-GLGM common/FDR/private gradient partition is incomplete")
        return {
            "gradient_norm": common,
            "fdr_gradient_norm": fdr,
            "ra_glgm_gradient_norm": ra,
        }

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> RAGLGMDetectionModel:
        model = RAGLGMDetectionModel(
            cfg or RA_GLGM_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_pair_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="ra_glgm",
            )
        return model


class RAGLGMControlTrainer(_IgnoreSidecarDatasetMixin, FDRTrainer):
    """FDR baseline arm under the identical ignore-sidecar pipeline."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> RAGLGMControlDetectionModel:
        model = RAGLGMControlDetectionModel(
            cfg or RA_GLGM_CONTROL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_pair_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="baseline",
            )
        return model


__all__ = [
    "RA_GLGM_AUXILIARY_WEIGHT",
    "RA_GLGM_CONTROL_CFG",
    "RA_GLGM_MODEL_CFG",
    "RA_GLGM_SCALE_AUXILIARY_WEIGHT",
    "RAGLGMControlDetectionModel",
    "RAGLGMControlTrainer",
    "RAGLGMDetectionModel",
    "RAGLGMTrainer",
    "register_ra_glgm_decoder",
]

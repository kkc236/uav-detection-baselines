"""Full-model Ultralytics RT-DETR integration for signed IBER-BE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

from src.iber_head import IBEROutput, IBERRefiner
from src.itber_geometry import cxcywh_to_xyxy
from src.itber_loss import ITBERLosses, itber_private_loss
from src.lpr_g_loss import MatchRecordingRTDETRDetectionLoss
from src.rtdetr_iber import IBERRecordingDecoder
from src.rtdetr_lpr import FixedPairedProtocolMixin
from src.rtdetr_vsf_rmr import apply_resume_runtime_overrides


PRIVATE_PARAMETER_MARKER = "iber_refiner."


class IBERFullRTDETRDetectionModel(RTDETRDetectionModel):
    """RT-DETR with an isolated signed boundary refiner on final normal queries."""

    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int = 10_000,
        image_size: int = 640,
        rho: float = 0.05,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        if int(head.num_queries) != 300:
            raise ValueError("formal IBER-BE requires exactly 300 RT-DETR queries")
        stock_decoder = head.decoder
        head.decoder = IBERRecordingDecoder.from_stock(
            stock_decoder,
            normal_query_count=300,
        )
        first_projection = head.input_proj[0][0]
        parameter = next(self.parameters())
        self.iber_refiner = IBERRefiner(
            hidden_dim=int(head.decoder.hidden_dim),
            f3_channels=int(first_projection.in_channels),
            private_seed=int(private_seed),
            probe="b3",
            image_size=int(image_size),
            rho=float(rho),
        ).to(device=parameter.device, dtype=parameter.dtype)
        self.nc = self.yaml["nc"]
        self.private_seed = int(private_seed)
        self.image_size = int(image_size)
        self.rho = float(rho)
        self.output_mode = "refined"
        self.last_iber_output: IBEROutput | None = None
        self.last_iber_losses: dict[str, torch.Tensor] = {}
        self.last_iber_loss_total: torch.Tensor | None = None
        self._last_f3: torch.Tensor | None = None
        self._head_input_hook = head.register_forward_pre_hook(self._capture_head_input)

    def _capture_head_input(self, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], (list, tuple)) or not inputs[0]:
            raise RuntimeError("RT-DETR head did not receive a feature pyramid")
        self._last_f3 = inputs[0][0].detach()

    def init_criterion(self) -> MatchRecordingRTDETRDetectionLoss:
        """Use the stock criterion while recording its normal-query assignment."""
        return MatchRecordingRTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def set_refinement_output(self, mode: str) -> None:
        """Select stock, signed-refined, or boundary-disabled inference boxes."""
        if mode not in {"stock", "refined", "boundary_off"}:
            raise ValueError(f"unsupported IBER-BE output mode: {mode}")
        self.output_mode = mode

    def _refine_last_prediction(self, image: torch.Tensor) -> IBEROutput:
        decoder = self.model[-1].decoder
        hidden = decoder.last_hidden
        scores = decoder.last_stock_scores
        boxes = decoder.last_stock_boxes
        if hidden is None or scores is None or boxes is None or self._last_f3 is None:
            raise RuntimeError("full-model IBER-BE evidence capture is incomplete")
        output = self.iber_refiner(
            hidden,
            boxes,
            scores,
            self._last_f3,
            image.detach(),
        )
        self.last_iber_output = output
        return output

    def predict(
        self,
        x: torch.Tensor,
        profile: bool = False,
        visualize: bool = False,
        batch: dict | None = None,
        augment: bool = False,
        embed: list[int] | None = None,
    ):
        """Run stock RT-DETR once and apply only the detached IBER side branch."""
        if embed is not None:
            raise ValueError("formal IBER-BE does not support embedding-mode prediction")
        self._last_f3 = None
        stock_prediction = super().predict(
            x,
            profile=profile,
            visualize=visualize,
            batch=batch,
            augment=augment,
            embed=embed,
        )
        output = self._refine_last_prediction(x)
        if self.training:
            return stock_prediction

        head = self.model[-1]
        selected = output.select_boxes(self.output_mode)
        detections = head.postprocess(selected, output.stock_scores.sigmoid())
        if head.export:
            return detections
        raw = stock_prediction[1]
        return detections, raw

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add isolated IBER supervision while preserving every stock loss."""
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        image = batch["img"]
        batch_index = batch["batch_idx"]
        targets = {
            "cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=image.device),
            "batch_idx": batch_index.to(image.device, dtype=torch.long).view(-1),
            "gt_groups": [
                (batch_index == index).sum().item()
                for index in range(image.shape[0])
            ],
        }

        if preds is None:
            preds = self.predict(image, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = (
            preds if self.training else preds[1]
        )
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(
                dec_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_scores, dec_scores = torch.split(
                dec_scores, dn_meta["dn_num_split"], dim=2
            )

        dec_bboxes = torch.cat((enc_bboxes.unsqueeze(0), dec_bboxes))
        dec_scores = torch.cat((enc_scores.unsqueeze(0), dec_scores))
        stock_losses = self.criterion(
            (dec_bboxes, dec_scores),
            targets,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        matches = self.criterion.last_stock_match_indices
        if matches is None:
            raise RuntimeError("stock normal-query assignment is unavailable")
        if self.last_iber_output is None:
            raise RuntimeError("IBER-BE output is unavailable for private loss")
        private_losses: ITBERLosses = itber_private_loss(
            self.last_iber_output,
            target_edges=cxcywh_to_xyxy(targets["bboxes"]),
            match_indices=matches,
            rho=self.rho,
        )
        self.last_iber_losses = {
            name: value.detach()
            for name, value in vars(private_losses).items()
            if isinstance(value, torch.Tensor)
        }
        self.last_iber_loss_total = private_losses.total
        return sum(stock_losses.values()) + private_losses.total, torch.as_tensor(
            [
                stock_losses[name].detach()
                for name in ("loss_giou", "loss_class", "loss_bbox")
            ],
            device=image.device,
        )


class IBERFullTrainer(FixedPairedProtocolMixin, RTDETRTrainer):
    """Full RT-DETR-L trainer with one MuSGD optimizer and isolated clipping."""

    def __init__(self, *args, experiment_seed: int = 0, **kwargs) -> None:
        if int(experiment_seed) != 0:
            raise ValueError("formal IBER-BE training is frozen to seed0")
        self.experiment_seed = 0
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            self.args.epochs = 100

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        public: list[torch.nn.Parameter] = []
        private: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (private if PRIVATE_PARAMETER_MARKER in name else public).append(parameter)
        if not public or not private:
            raise RuntimeError("IBER-BE public/private parameter partition is incomplete")
        return {"gradient_norm": public, "iber_gradient_norm": private}

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> IBERFullRTDETRDetectionModel:
        model = IBERFullRTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000,
            image_size=640,
            rho=0.05,
        )
        if weights:
            model.load(weights)
        return model


__all__ = [
    "IBERFullRTDETRDetectionModel",
    "IBERFullTrainer",
    "PRIVATE_PARAMETER_MARKER",
]

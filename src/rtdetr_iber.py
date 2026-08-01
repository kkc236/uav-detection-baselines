"""Frozen Ultralytics RT-DETR evidence adapter for IBER-BE."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.modules.utils import inverse_sigmoid

from src.iber_head import IBEROutput, IBERRefiner
from src.itber_geometry import cxcywh_to_xyxy
from src.itber_loss import itber_private_loss


class IBERRecordingDecoder(nn.Module):
    """Reproduce the stock decoder and detach only its final evidence."""

    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        eval_idx: int,
        *,
        normal_query_count: int = 300,
    ) -> None:
        super().__init__()
        if normal_query_count < 1:
            raise ValueError("normal query count must be positive")
        self.layers = layers
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.eval_idx = int(eval_idx)
        self.normal_query_count = int(normal_query_count)
        self.last_hidden: torch.Tensor | None = None
        self.last_stock_scores: torch.Tensor | None = None
        self.last_stock_boxes: torch.Tensor | None = None

    @classmethod
    def from_stock(
        cls,
        stock: nn.Module,
        *,
        normal_query_count: int = 300,
    ) -> "IBERRecordingDecoder":
        """Reuse every stock decoder layer without changing its state keys."""
        wrapped = cls(
            layers=stock.layers,
            hidden_dim=stock.hidden_dim,
            num_layers=stock.num_layers,
            eval_idx=stock.eval_idx,
            normal_query_count=normal_query_count,
        )
        wrapped.train(stock.training)
        return wrapped

    def _record(
        self,
        hidden: torch.Tensor,
        score: torch.Tensor,
        stock_box: torch.Tensor,
    ) -> None:
        normal = min(self.normal_query_count, hidden.shape[1])
        selection = slice(hidden.shape[1] - normal, hidden.shape[1])
        self.last_hidden = hidden[:, selection].detach()
        self.last_stock_scores = score[:, selection].detach()
        self.last_stock_boxes = stock_box[:, selection].detach()

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the stock equations while recording only the final state."""
        self.last_hidden = None
        self.last_stock_scores = None
        self.last_stock_boxes = None
        output = embed
        decoded_boxes: list[torch.Tensor] = []
        decoded_scores: list[torch.Tensor] = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()

        for index, layer in enumerate(self.layers):
            output = layer(
                output,
                refer_bbox,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(refer_bbox),
            )
            bbox = bbox_head[index](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                score = score_head[index](output)
                stock_box = (
                    refined_bbox
                    if index == 0
                    else torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))
                )
                decoded_scores.append(score)
                decoded_boxes.append(stock_box)
                if index == self.num_layers - 1:
                    self._record(output, score, stock_box)
            elif index == self.eval_idx:
                score = score_head[index](output)
                decoded_scores.append(score)
                decoded_boxes.append(refined_bbox)
                self._record(output, score, refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(decoded_boxes), torch.stack(decoded_scores)


class FrozenIBERAdapter(nn.Module):
    """Keep RT-DETR immutable and train only the private IBER refiner."""

    def __init__(
        self,
        detector: nn.Module,
        refiner: IBERRefiner,
        criterion: RTDETRDetectionLoss,
        *,
        rho: float,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.refiner = refiner
        self.criterion = criterion
        self.rho = float(rho)
        self.output_mode = "refined"
        self.last_match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.last_output: IBEROutput | None = None
        self._last_f3: torch.Tensor | None = None
        self.detector.requires_grad_(False)
        self.detector.eval()
        self._head_hook = self.detector.model[-1].register_forward_pre_hook(
            self._capture_head_input
        )

    @classmethod
    def from_detector(
        cls,
        detector: nn.Module,
        *,
        private_seed: int,
        probe: str = "b3",
        image_size: int = 640,
        rho: float = 0.05,
        normal_query_count: int = 300,
    ) -> "FrozenIBERAdapter":
        head = detector.model[-1]
        if not isinstance(head.decoder, IBERRecordingDecoder):
            head.decoder = IBERRecordingDecoder.from_stock(
                head.decoder,
                normal_query_count=normal_query_count,
            )
        first_projection = head.input_proj[0][0]
        f3_channels = int(first_projection.in_channels)
        hidden_dim = int(head.decoder.hidden_dim)
        device_parameter = next(detector.parameters())
        refiner = IBERRefiner(
            hidden_dim=hidden_dim,
            f3_channels=f3_channels,
            private_seed=private_seed,
            probe=probe,
            image_size=image_size,
            rho=rho,
        ).to(device=device_parameter.device, dtype=device_parameter.dtype)
        nc = int(getattr(detector, "nc", detector.yaml["nc"]))
        criterion = RTDETRDetectionLoss(nc=nc, use_vfl=True)
        return cls(detector, refiner, criterion, rho=rho)

    def _capture_head_input(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], (list, tuple)) or not inputs[0]:
            raise RuntimeError("RT-DETR head did not receive a feature pyramid")
        self._last_f3 = inputs[0][0].detach()

    def train(self, mode: bool = True) -> "FrozenIBERAdapter":
        """Toggle the private module while permanently locking detector eval."""
        super().train(mode)
        self.detector.requires_grad_(False)
        self.detector.eval()
        return self

    def set_output_mode(self, mode: str) -> None:
        if mode not in {"stock", "refined", "boundary_off"}:
            raise ValueError("IBER output mode must be stock, refined, or boundary_off")
        self.output_mode = mode

    def selected_boxes(self, output: IBEROutput) -> torch.Tensor:
        return output.select_boxes(self.output_mode)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return stock-score Top-300 detections with the selected boxes."""
        output = self.forward_evidence(image)
        return self.detector.model[-1].postprocess(
            self.selected_boxes(output),
            output.stock_scores.sigmoid(),
        )

    def forward_evidence(self, image: torch.Tensor) -> IBEROutput:
        """Run one immutable detector forward and invoke the private head."""
        self.detector.eval()
        self._last_f3 = None
        detached_image = image.detach()
        with torch.no_grad():
            self.detector.predict(detached_image)
        decoder = self.detector.model[-1].decoder
        last_hidden = getattr(decoder, "last_hidden", None)
        last_stock_scores = getattr(decoder, "last_stock_scores", None)
        last_stock_boxes = getattr(decoder, "last_stock_boxes", None)
        if (
            last_hidden is None
            or last_stock_scores is None
            or last_stock_boxes is None
            or self._last_f3 is None
        ):
            raise RuntimeError("frozen RT-DETR evidence capture is incomplete")
        return self.refiner(
            last_hidden,
            last_stock_boxes,
            last_stock_scores,
            self._last_f3,
            detached_image,
        )

    def training_step(self, batch: dict[str, torch.Tensor]):
        """Match stock outputs once and compute only the private objective."""
        image = batch["img"]
        output = self.forward_evidence(image)
        self.last_output = output
        target_boxes = batch["bboxes"].detach().to(
            device=image.device, dtype=output.stock_boxes.dtype
        )
        target_classes = batch["cls"].detach().to(
            device=image.device, dtype=torch.long
        ).view(-1)
        batch_index = batch["batch_idx"].detach().to(
            device=image.device, dtype=torch.long
        ).view(-1)
        groups = [
            int((batch_index == index).sum()) for index in range(image.shape[0])
        ]
        self.last_match_indices = self.criterion.matcher(
            output.stock_boxes.detach(),
            output.stock_scores.detach(),
            target_boxes,
            target_classes,
            groups,
        )
        return itber_private_loss(
            output,
            target_edges=cxcywh_to_xyxy(target_boxes),
            match_indices=self.last_match_indices,
            rho=self.rho,
        )

"""Frozen Ultralytics RT-DETR evidence adapter for I-TBER v1.1."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.modules.utils import inverse_sigmoid

from src.itber_geometry import cxcywh_to_xyxy
from src.itber_head import ITBEROutput, ITBERRefiner
from src.itber_loss import ITBERLosses, itber_private_loss


class ITBERRecordingDecoder(nn.Module):
    """Reproduce the stock decoder and detach its final evidence for I-TBER."""

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
        self.last_three_boxes: torch.Tensor | None = None

    @classmethod
    def from_stock(
        cls,
        stock: nn.Module,
        *,
        normal_query_count: int = 300,
    ) -> "ITBERRecordingDecoder":
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
        trajectory: list[torch.Tensor],
    ) -> None:
        if len(trajectory) < 3:
            raise RuntimeError("I-TBER requires at least three decoder box states")
        normal = min(self.normal_query_count, hidden.shape[1])
        selection = slice(hidden.shape[1] - normal, hidden.shape[1])
        self.last_hidden = hidden[:, selection].detach()
        self.last_stock_scores = score[:, selection].detach()
        self.last_three_boxes = torch.stack(
            [boxes[:, selection].detach() for boxes in trajectory[-3:]], dim=0
        )

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
        """Run the byte-equivalent stock equations while recording evidence."""
        output = embed
        decoded_boxes: list[torch.Tensor] = []
        decoded_scores: list[torch.Tensor] = []
        trajectory: list[torch.Tensor] = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        self.last_hidden = None
        self.last_stock_scores = None
        self.last_three_boxes = None

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
                trajectory.append(stock_box)
                if index == self.num_layers - 1:
                    self._record(output, score, trajectory)
            elif index == self.eval_idx:
                score = score_head[index](output)
                trajectory.append(refined_bbox)
                decoded_scores.append(score)
                decoded_boxes.append(refined_bbox)
                self._record(output, score, trajectory)
                break
            else:
                trajectory.append(refined_bbox)

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(decoded_boxes), torch.stack(decoded_scores)


class FrozenITBERAdapter(nn.Module):
    """Keep RT-DETR immutable and train only the private I-TBER refiner."""

    def __init__(
        self,
        detector: nn.Module,
        refiner: ITBERRefiner,
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
        self.last_output: ITBEROutput | None = None
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
        probe: str = "p3",
        image_size: int = 640,
        rho: float = 0.05,
        normal_query_count: int = 300,
    ) -> "FrozenITBERAdapter":
        head = detector.model[-1]
        if not isinstance(head.decoder, ITBERRecordingDecoder):
            head.decoder = ITBERRecordingDecoder.from_stock(
                head.decoder,
                normal_query_count=normal_query_count,
            )
        first_projection = head.input_proj[0][0]
        f3_channels = int(first_projection.in_channels)
        hidden_dim = int(head.decoder.hidden_dim)
        device_parameter = next(detector.parameters())
        refiner = ITBERRefiner(
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

    def train(self, mode: bool = True) -> "FrozenITBERAdapter":
        """Allow private train/eval changes while permanently locking detector eval."""
        super().train(mode)
        self.detector.eval()
        return self

    def set_output_mode(self, mode: str) -> None:
        if mode not in {"stock", "refined"}:
            raise ValueError("I-TBER output mode must be stock or refined")
        self.output_mode = mode

    def selected_boxes(self, output: ITBEROutput) -> torch.Tensor:
        return output.stock_boxes if self.output_mode == "stock" else output.refined_boxes

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return stock-score Top-300 detections with the selected box mode."""
        output = self.forward_evidence(image)
        scores = self.detector.model[-1].decoder.last_stock_scores
        if scores is None:
            raise RuntimeError("I-TBER forward did not capture stock scores")
        return self.detector.model[-1].postprocess(
            self.selected_boxes(output),
            scores.sigmoid(),
        )

    def forward_evidence(self, image: torch.Tensor) -> ITBEROutput:
        """Run one immutable detector forward and invoke the private head."""
        self.detector.eval()
        self._last_f3 = None
        with torch.no_grad():
            self.detector.predict(image)
        decoder = self.detector.model[-1].decoder
        if (
            decoder.last_hidden is None
            or decoder.last_stock_scores is None
            or decoder.last_three_boxes is None
            or self._last_f3 is None
        ):
            raise RuntimeError("frozen RT-DETR evidence capture is incomplete")
        box_l2, box_l1, stock_boxes = decoder.last_three_boxes.unbind(dim=0)
        return self.refiner(
            decoder.last_hidden,
            box_l2,
            box_l1,
            stock_boxes,
            decoder.last_stock_scores,
            self._last_f3,
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> ITBERLosses:
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
            self.detector.model[-1].decoder.last_stock_scores.detach(),
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

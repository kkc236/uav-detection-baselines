"""YAML-declared, query-level ACR-EG integration for Ultralytics RT-DETR.

The module is intentionally attached to ``RTDETRDetectionModel`` itself.  A
global forward and four shared local forwards expose final decoder queries;
``GCQF`` consumes those queries before the RT-DETR loss is evaluated and its
learned global-retention logit is injected into the final decoder-query class
scores.  Thus the module is not a prediction postprocessor: its output is an
input to the detector criterion and its parameters are ordinary model
parameters owned by checkpoints and the optimizer.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from src.acr_eg_integration import ACREGConfig, load_acr_eg_config
from src.ascv_loc import preserve_batchnorm_buffers
from src.gcqf import GCQF, GCQFOutput
from src.gcmv_data import GCMVRTDETRDataset
from src.gcte_types import QueryEvidence, ViewGeometry
from src.gcte_views import build_frozen_view_geometry, transform_xywh_homography
from src.gcte_targets import build_tiny_anchor_mask
from src.gcte_formal_trainer import GCTEFormalTrainer


ACR_EG_LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "acr_eg_gate")
ACR_EG_EXTRA_PREFIX = "acr_eg."


def inject_query_retention_logits(
    decoder_scores: torch.Tensor,
    retention_logits: torch.Tensor,
    *,
    num_queries: int,
    gain: float,
) -> torch.Tensor:
    """Inject one learned retain logit into final non-denoising query scores."""

    if decoder_scores.ndim != 4:
        raise ValueError("decoder_scores must have shape [L,B,Q,C]")
    if num_queries <= 0 or num_queries > decoder_scores.shape[2]:
        raise ValueError("num_queries must select final decoder queries")
    if not 0.0 < float(gain) <= 1.0:
        raise ValueError("gain must be in (0,1]")
    expected = (decoder_scores.shape[1], num_queries, 1)
    if retention_logits.shape != expected:
        raise ValueError(
            "retention_logits must match [B,num_queries,1]"
        )
    final = decoder_scores[-1]
    prefix = final[:, :-num_queries]
    residual = retention_logits.tanh().to(final.dtype) * float(gain)
    fused_final = final[:, -num_queries:] + residual
    return torch.cat(
        (decoder_scores[:-1], torch.cat((prefix, fused_final), dim=1).unsqueeze(0)),
        dim=0,
    )


class _LiveFinalQueryCapture:
    """Capture the final decoder query tensor without detaching it."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None
        self.handle: torch.utils.hooks.RemovableHandle | None = None

    def attach(self, layer: nn.Module) -> None:
        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("ACR-EG final decoder layer returned non-tensor")
            self.value = output

        self.handle = layer.register_forward_hook(capture)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@contextmanager
def _capture_final_decoder_queries(model: RTDETRDetectionModel) -> Iterator[_LiveFinalQueryCapture]:
    head = model.model[-1]
    decoder = getattr(head, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or not layers:
        raise RuntimeError("ACR-EG cannot find RT-DETR decoder layers")
    capture = _LiveFinalQueryCapture()
    capture.attach(layers[-1])
    try:
        yield capture
    finally:
        capture.remove()


def _require_raw_training_output(value: object) -> tuple[torch.Tensor, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != 5
        or not all(isinstance(item, torch.Tensor) or item is None for item in value)
    ):
        raise RuntimeError("ACR-EG RT-DETR training output contract drift")
    return value


class ACREGDetectionModel(RTDETRDetectionModel):
    """RT-DETR-L whose final decoder queries are fused by registered ACR-EG."""

    def __init__(
        self,
        cfg: str | Path = "configs/rtdetr-l-acr-eg.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.acr_eg_config: ACREGConfig = load_acr_eg_config(cfg)
        self.acr_eg = GCQF(
            query_dim=self.acr_eg_config.query_dim,
            num_classes=int(self.yaml["nc"]),
            num_heads=self.acr_eg_config.num_heads,
            num_views=self.acr_eg_config.num_views,
            residual_eta=self.acr_eg_config.residual_eta,
        )
        self.acr_eg_gain = 0.2
        self.acr_eg_num_queries = int(self.model[-1].num_queries)
        if self.acr_eg_num_queries != 300:
            raise RuntimeError("ACR-EG requires RT-DETR num_queries=300")
        self.nc = int(self.yaml["nc"])
        self.loss_names = ACR_EG_LOSS_NAMES
        self.last_acr_eg_output: GCQFOutput | None = None

    @property
    def acr_eg_enabled(self) -> bool:
        return bool(
            self.acr_eg_config.enabled
            and self.acr_eg_config.forward_integration
            and not self.acr_eg_config.gcte_off
            and not self.acr_eg_config.acr_eg_off
        )

    def _stock_forward_with_queries(
        self, image: torch.Tensor, *, batch: dict | None
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        with _capture_final_decoder_queries(self) as capture:
            raw = super().predict(image, batch=batch)
        raw_output = _require_raw_training_output(raw)
        if capture.value is None:
            raise RuntimeError("ACR-EG final query capture did not execute")
        queries = capture.value
        if queries.ndim != 3 or queries.shape[1] < self.acr_eg_num_queries:
            raise RuntimeError("ACR-EG decoder-query layout drift")
        return raw_output, queries[:, -self.acr_eg_num_queries :]

    def _local_evidence(self, local_views: torch.Tensor) -> QueryEvidence:
        expected = (4, 3, 640, 640)
        if local_views.ndim != 5 or tuple(local_views.shape[1:]) != expected:
            raise ValueError(
                "local_views must have shape [B,4,3,640,640] for ACR-EG"
            )
        query_parts: list[torch.Tensor] = []
        box_parts: list[torch.Tensor] = []
        logit_parts: list[torch.Tensor] = []
        for view_index in range(4):
            with torch.no_grad(), preserve_batchnorm_buffers(self):
                raw, queries = self._stock_forward_with_queries(
                    local_views[:, view_index], batch=None
                )
            boxes, logits = raw[0][-1], raw[1][-1]
            query_parts.append(queries.detach())
            box_parts.append(boxes[:, -self.acr_eg_num_queries :].detach())
            logit_parts.append(logits[:, -self.acr_eg_num_queries :].detach())
        local_queries = torch.cat(query_parts, dim=1)
        local_logits = torch.cat(logit_parts, dim=1)
        local_boxes = torch.cat(box_parts, dim=1)
        return QueryEvidence(
            queries=local_queries,
            logits=local_logits,
            boxes=local_boxes,
            quality=local_logits.sigmoid().amax(dim=-1, keepdim=True),
        )

    @staticmethod
    def _live_geometry(source_shapes: torch.Tensor, *, device: torch.device) -> ViewGeometry:
        if source_shapes.ndim != 2 or source_shapes.shape[1] != 2:
            raise ValueError("source_shape must have shape [B,2]")
        shapes = [tuple(int(value) for value in row) for row in source_shapes.detach().cpu().tolist()]
        geometry = build_frozen_view_geometry(source_shapes=shapes, queries_per_view=300)
        return ViewGeometry(
            homography=geometry.homography.to(device=device),
            crop_metadata=geometry.crop_metadata.to(device=device),
            view_index=geometry.view_index.to(device=device),
            valid_mask=geometry.valid_mask.to(device=device),
        )

    def predict(
        self,
        x,
        profile=False,
        visualize=False,
        batch=None,
        augment=False,
        embed=None,
        *,
        local_views: torch.Tensor | None = None,
        source_shapes: torch.Tensor | None = None,
    ):
        if (
            not self.acr_eg_enabled
            or local_views is None
            or source_shapes is None
        ):
            return super().predict(
                x, profile=profile, visualize=visualize, batch=batch, augment=augment, embed=embed
            )
        if profile or visualize or embed is not None or augment:
            raise ValueError("ACR-EG paired prediction does not support profiling flags")
        raw, global_queries = self._stock_forward_with_queries(x, batch=batch)
        global_boxes = raw[0][-1][:, -self.acr_eg_num_queries :]
        global_logits = raw[1][-1][:, -self.acr_eg_num_queries :]
        global_evidence = QueryEvidence(
            queries=global_queries,
            logits=global_logits,
            boxes=global_boxes,
            quality=global_logits.sigmoid().amax(dim=-1, keepdim=True),
        )
        local_evidence = self._local_evidence(local_views)
        geometry = self._live_geometry(source_shapes, device=x.device)
        canonical_local_boxes = transform_xywh_homography(
            local_evidence.boxes, geometry.homography, clip=True
        )
        output = self.acr_eg(
            global_evidence,
            local_evidence,
            geometry,
            anchor_mask=build_tiny_anchor_mask(canonical_local_boxes),
            residual_enabled=self.acr_eg_config.residual_enabled,
        )
        self.last_acr_eg_output = output
        fused_scores = inject_query_retention_logits(
            raw[1],
            output.global_retain_logits,
            num_queries=self.acr_eg_num_queries,
            gain=self.acr_eg_gain,
        )
        return (raw[0], fused_scores, raw[2], raw[3], raw[4])

    def loss(self, batch: dict, preds=None):
        if (
            preds is not None
            or not self.training
            or not self.acr_eg_enabled
            or "local_views" not in batch
            or "source_shape" not in batch
        ):
            return super().loss(batch, preds=preds)
        image = batch["img"]
        batch_indices = batch["batch_idx"].to(image.device, dtype=torch.long).view(-1)
        targets = {
            "cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(image.device),
            "batch_idx": batch_indices,
            "gt_groups": [int((batch_indices == index).sum().item()) for index in range(image.shape[0])],
        }
        predictions = self.predict(
            image,
            batch=targets,
            local_views=batch["local_views"],
            source_shapes=batch["source_shape"],
        )
        detection_loss, detection_items = super().loss(batch, preds=predictions)
        if self.last_acr_eg_output is None:
            raise RuntimeError("ACR-EG forward did not produce gate output")
        gate_item = self.last_acr_eg_output.global_retain_logits.detach().abs().mean().reshape(1)
        return detection_loss, torch.cat((detection_items, gate_item))


def load_mature_baseline(model: ACREGDetectionModel, path: str | Path) -> None:
    """Load only stock RT-DETR parameters, preserving the new ACR-EG module."""

    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    candidate = payload.get("ema") or payload.get("model") if isinstance(payload, dict) else payload
    if not isinstance(candidate, nn.Module):
        raise RuntimeError("mature baseline checkpoint has no detector module")
    source = candidate.float().state_dict()
    destination = model.state_dict()
    stock_names = {name for name in destination if not name.startswith(ACR_EG_EXTRA_PREFIX)}
    missing = stock_names - set(source)
    if missing:
        raise RuntimeError(f"mature baseline misses stock keys: {sorted(missing)[:3]}")
    incompatible = model.load_state_dict(
        {name: source[name] for name in stock_names}, strict=False
    )
    if set(incompatible.missing_keys) != {name for name in destination if name.startswith(ACR_EG_EXTRA_PREFIX)}:
        raise RuntimeError("ACR-EG mature-baseline load drift")
    if incompatible.unexpected_keys:
        raise RuntimeError("ACR-EG mature-baseline unexpected keys")


class ACREGFormalTrainer(GCTEFormalTrainer):
    """Frozen-protocol trainer that owns the integrated RT-DETR model."""

    def __init__(self, *args, **kwargs) -> None:
        self.baseline_checkpoint = Path(os.environ["GCTE_ACR_EG_BASELINE"]).resolve()
        self.model_yaml = Path(os.environ["GCTE_ACR_EG_YAML"]).resolve()
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        if weights is not None:
            raise ValueError("ACR-EG formal run must load only the sealed mature baseline")
        model = ACREGDetectionModel(
            cfg or self.model_yaml,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        load_mature_baseline(model, self.baseline_checkpoint)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return GCMVRTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            local_imgsz=640,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        processed = RTDETRTrainer.preprocess_batch(self, batch)
        processed["local_views"] = (
            processed["local_views"].to(self.device, non_blocking=self.device.type == "cuda").float() / 255
        )
        processed["source_shape"] = processed["source_shape"].to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        return processed

    def validate(self):
        return {}, float("-inf")

    def final_eval(self):
        return None


__all__ = [
    "ACREGDetectionModel",
    "ACREGFormalTrainer",
    "ACR_EG_LOSS_NAMES",
    "inject_query_retention_logits",
    "load_mature_baseline",
]

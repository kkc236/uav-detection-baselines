"""Read-only final-query extraction from Ultralytics RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import nn

from src.gcte_types import QueryEvidence


@dataclass(frozen=True)
class DecoderEvidenceExtraction:
    evidence: QueryEvidence
    postprocessed: torch.Tensor
    selected_query_indices: torch.Tensor


class FinalDecoderQueryProbe:
    """Capture one decoder layer output without changing the forward value."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def attach(self, layer: nn.Module) -> None:
        self.remove()
        self.value = None
        self._handle = layer.register_forward_hook(self._capture)

    def _capture(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("final RT-DETR decoder layer returned non-tensor")
        self.value = output.detach().clone()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _batchnorm_fingerprint(module: nn.Module) -> str:
    digest = sha256()
    for name, child in module.named_modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        for field in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(child, field, None)
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{name}.{field}".encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def freeze_detector(model: nn.Module) -> nn.Module:
    """Put the shared detector in immutable inference mode."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def stock_postprocess_with_query_indices(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    *,
    num_queries: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce RTDETRDecoder.postprocess and retain source query indices."""

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must be [B,Q,4]")
    if logits.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
        raise ValueError("logits must share [B,Q] with boxes")
    if num_queries <= 0 or num_queries > logits.shape[1] * logits.shape[2]:
        raise ValueError("num_queries is outside the flattened score range")
    probabilities = logits.sigmoid()
    scores, flat_indices = probabilities.flatten(1).topk(num_queries)
    query_indices = torch.div(
        flat_indices,
        logits.shape[-1],
        rounding_mode="floor",
    )
    selected_boxes = boxes.gather(
        dim=1,
        index=query_indices.unsqueeze(-1).expand(-1, -1, 4).long(),
    )
    classes = flat_indices - query_indices * logits.shape[-1]
    predictions = torch.cat(
        (
            selected_boxes,
            scores.unsqueeze(-1),
            classes.unsqueeze(-1).to(selected_boxes.dtype),
        ),
        dim=-1,
    )
    return predictions, query_indices


def _decoder_head(model: nn.Module) -> nn.Module:
    modules = getattr(model, "model", None)
    if not isinstance(modules, (nn.ModuleList, nn.Sequential, list, tuple)):
        raise RuntimeError("RT-DETR model does not expose its module graph")
    if not modules:
        raise RuntimeError("RT-DETR module graph is empty")
    head = modules[-1]
    decoder = getattr(head, "decoder", None)
    layers = getattr(decoder, "layers", None)
    eval_index = getattr(decoder, "eval_idx", None)
    if (
        not isinstance(layers, (nn.ModuleList, nn.Sequential))
        or not layers
        or not isinstance(eval_index, int)
        or not 0 <= eval_index < len(layers)
    ):
        raise RuntimeError("RT-DETR final decoder layer is unavailable")
    return head


def extract_decoder_query_evidence(
    model: nn.Module,
    images: torch.Tensor,
    *,
    expected_query_count: int = 300,
) -> DecoderEvidenceExtraction:
    """Run a frozen eval forward and return final decoder query evidence."""

    if model.training:
        raise RuntimeError("RT-DETR query extraction requires eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("RT-DETR query extraction requires frozen parameters")
    if not isinstance(images, torch.Tensor) or images.ndim != 3:
        # The real model receives [B,3,H,W]; compact fake decoders in unit
        # tests use [B,Q,C].  Both are rank >= 3 tensor batches.
        if not isinstance(images, torch.Tensor) or images.ndim < 3:
            raise ValueError("images must be a batched tensor")
    head = _decoder_head(model)
    decoder = head.decoder
    layer = decoder.layers[decoder.eval_idx]
    probe = FinalDecoderQueryProbe()
    before = _batchnorm_fingerprint(model)
    probe.attach(layer)
    try:
        with torch.inference_mode():
            raw = model.predict(images, batch=None)
    finally:
        probe.remove()
    after = _batchnorm_fingerprint(model)
    if before != after:
        raise RuntimeError("RT-DETR extraction mutated BatchNorm buffers")
    if probe.value is None:
        raise RuntimeError("final decoder query hook did not execute")
    if (
        not isinstance(raw, tuple)
        or len(raw) != 2
        or not isinstance(raw[1], tuple)
        or len(raw[1]) != 5
    ):
        raise RuntimeError("unexpected RT-DETR eval output contract")
    stock_predictions, auxiliary = raw
    dec_boxes, dec_logits, _enc_boxes, _enc_logits, dn_meta = auxiliary
    if dn_meta is not None:
        raise RuntimeError("eval extraction unexpectedly produced DN metadata")
    boxes = dec_boxes[-1].detach()
    logits = dec_logits[-1].detach()
    queries = probe.value.detach()
    if (
        boxes.shape[1] != expected_query_count
        or logits.shape[1] != expected_query_count
        or queries.shape[1] != expected_query_count
    ):
        raise RuntimeError("RT-DETR decoder query count drift")
    if boxes.shape[:2] != logits.shape[:2] or boxes.shape[:2] != queries.shape[:2]:
        raise RuntimeError("RT-DETR decoder evidence shape drift")
    reproduced, query_indices = stock_postprocess_with_query_indices(
        boxes,
        logits,
        num_queries=expected_query_count,
    )
    if not torch.equal(reproduced, stock_predictions):
        raise RuntimeError("RT-DETR stock postprocess reproduction drift")
    evidence = QueryEvidence(
        queries=queries,
        logits=logits,
        boxes=boxes,
        quality=logits.sigmoid().amax(dim=-1, keepdim=True),
    )
    return DecoderEvidenceExtraction(
        evidence=evidence.detached(),
        postprocessed=stock_predictions.detach(),
        selected_query_indices=query_indices.detach(),
    )


__all__ = [
    "DecoderEvidenceExtraction",
    "FinalDecoderQueryProbe",
    "extract_decoder_query_evidence",
    "freeze_detector",
    "stock_postprocess_with_query_indices",
]

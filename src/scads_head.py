"""YAML-visible SCADS extension of the validated FDR decoder box path."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from ultralytics.nn.modules.utils import inverse_sigmoid

from src.fdr_head import (
    FDR_DECODER_LAYERS,
    FDRDeformableTransformerDecoder,
    FDRRTDETRDecoder,
)
from src.fdr_math import REG_MAX, REG_SCALE, distance2bbox
from src.scads import (
    AdaptiveIntegral,
    DEFAULT_ROUTER_HIDDEN,
    DEFAULT_ROUTER_SEED,
    DEFAULT_ROUTER_TEMPERATURE,
    DEFAULT_SUPPORT_UPS,
    ScaleConditionedSupportRouter,
)


class SCADSFDRRTDETRDecoder(FDRRTDETRDecoder):
    """RT-DETR FDR head with trainable query-conditioned support routing."""

    _OPTION_DEFAULTS = {
        **FDRRTDETRDecoder._OPTION_DEFAULTS,
        "support_up_values": list(DEFAULT_SUPPORT_UPS),
        "support_router_hidden": DEFAULT_ROUTER_HIDDEN,
        "support_temperature": DEFAULT_ROUTER_TEMPERATURE,
        "support_private_seed": DEFAULT_ROUTER_SEED,
    }

    def __init__(
        self,
        nc: int = 80,
        ch: tuple[int, int, int] | list[int] = (256, 256, 256),
        declared_ch_or_options: tuple[int, int, int] | list[int] | dict | None = None,
        options: dict | None = None,
    ) -> None:
        super().__init__(nc, ch, declared_ch_or_options, options)
        resolved = self.fdr_options
        support_ups = tuple(float(value) for value in resolved["support_up_values"])
        if not bool(resolved["preliminary_box"]):
            raise ValueError("SCADS requires preliminary_box=true")
        base_index = support_ups.index(float(resolved["up"]))
        previous = self.decoder
        if not isinstance(previous, FDRDeformableTransformerDecoder):
            raise TypeError("SCADS expected the validated FDR decoder")
        self.decoder = SCADSFDRDeformableTransformerDecoder.from_fdr(
            previous,
            support_ups=support_ups,
            router_hidden=int(resolved["support_router_hidden"]),
            temperature=float(resolved["support_temperature"]),
            private_seed=int(resolved["support_private_seed"]),
            base_support_index=base_index,
        )
        self.decoder.reg_max = int(resolved["reg_max"])
        self.decoder.final_layers = [
            module.layers[-1] for module in self.dec_bbox_head
        ]
        self.decoder.cumulative = bool(resolved["cumulative"])
        self.decoder.preliminary_box = True


class SCADSFDRDeformableTransformerDecoder(FDRDeformableTransformerDecoder):
    """Use one layer-0 support decision for all cumulative FDR layers."""

    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        eval_idx: int,
        pre_bbox_head: nn.Module,
        *,
        support_ups: Sequence[float],
        router_hidden: int,
        temperature: float,
        private_seed: int,
        base_support_index: int,
    ) -> None:
        super().__init__(layers, hidden_dim, num_layers, eval_idx, pre_bbox_head)
        self.support_router = ScaleConditionedSupportRouter(
            hidden_dim,
            router_hidden,
            len(tuple(support_ups)),
            temperature=temperature,
            private_seed=private_seed,
            base_support_index=base_support_index,
        )
        self.adaptive_integral = AdaptiveIntegral(
            support_ups,
            reg_max=REG_MAX,
            reg_scale=REG_SCALE,
        )
        self.last_support_logits: Tensor | None = None
        self.last_support_weights: Tensor | None = None

    @classmethod
    def from_fdr(
        cls,
        decoder: FDRDeformableTransformerDecoder,
        *,
        support_ups: Sequence[float],
        router_hidden: int,
        temperature: float,
        private_seed: int,
        base_support_index: int,
    ) -> "SCADSFDRDeformableTransformerDecoder":
        if decoder.num_layers != FDR_DECODER_LAYERS:
            raise ValueError("SCADS requires exactly six FDR decoder layers")
        return cls(
            decoder.layers,
            decoder.hidden_dim,
            decoder.num_layers,
            decoder.eval_idx,
            decoder.pre_bbox_head,
            support_ups=support_ups,
            router_hidden=router_hidden,
            temperature=temperature,
            private_seed=private_seed,
            base_support_index=base_support_index,
        )

    def _clear_evidence(self) -> None:
        super()._clear_evidence()
        self.last_support_logits = None
        self.last_support_weights = None

    def forward(
        self,
        embed: Tensor,
        refer_bbox: Tensor,
        feats: Tensor,
        shapes: list,
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if len(bbox_head) != self.num_layers or len(score_head) != self.num_layers:
            raise ValueError("SCADS decoder requires one box and score head per layer")
        self._clear_evidence()
        output = embed
        output_detach: Tensor | int = 0
        cumulative_corners: Tensor | int = 0
        reference = refer_bbox.sigmoid()

        decoded_boxes: list[Tensor] = []
        class_logits: list[Tensor] = []
        corner_logits: list[Tensor] = []
        references: list[Tensor] = []
        preliminary: Tensor | None = None
        initial_reference: Tensor | None = None
        support_logits: Tensor | None = None
        support_weights: Tensor | None = None

        for index, layer in enumerate(self.layers):
            output = layer(
                output,
                reference,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(reference),
            )
            if index == 0:
                preliminary = torch.sigmoid(
                    self.pre_bbox_head(output) + inverse_sigmoid(reference)
                )
                initial_reference = preliminary.detach()
                support_logits, support_weights = self.support_router(
                    output,
                    preliminary,
                )

            if initial_reference is None or support_weights is None:
                raise RuntimeError("SCADS support reference was not initialized")
            delta_corners = bbox_head[index](output + output_detach)
            cumulative_corners = (
                delta_corners + cumulative_corners
                if self.cumulative
                else delta_corners
            )
            refined = distance2bbox(
                initial_reference,
                self.adaptive_integral(cumulative_corners, support_weights),
                self.reg_scale,
            )

            if self.training or index == self.eval_idx:
                decoded_boxes.append(refined)
                class_logits.append(score_head[index](output))
                corner_logits.append(cumulative_corners)
                references.append(initial_reference)
                if not self.training:
                    break

            reference = refined.detach() if self.training else refined
            output_detach = output.detach()

        if preliminary is None or support_logits is None or not decoded_boxes:
            raise RuntimeError("SCADS decoder produced no output")
        self.last_pre_bboxes = preliminary
        self.last_corner_logits = torch.stack(corner_logits)
        self.last_references = torch.stack(references)
        self.last_support_logits = support_logits
        self.last_support_weights = support_weights
        return torch.stack(decoded_boxes), torch.stack(class_logits)


__all__ = [
    "SCADSFDRDeformableTransformerDecoder",
    "SCADSFDRRTDETRDecoder",
]

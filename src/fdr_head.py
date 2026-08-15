"""FDR-only decoder box path for Ultralytics RT-DETR.

The stock decoder layers and classification heads remain untouched.  This
module owns only the preliminary four-coordinate box head contract and the
six cumulative 33-bin-per-edge distribution heads used by FDR.
"""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import Tensor, nn
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import MLP
from ultralytics.nn.modules.utils import inverse_sigmoid

from src.fdr_math import Integral, REG_MAX, REG_SCALE, UP, distance2bbox


FDR_OUTPUT_DIM = 4 * (REG_MAX + 1)
FDR_DECODER_LAYERS = 6


class FDRRTDETRDecoder(RTDETRDecoder):
    """YAML-visible RT-DETR head with the validated FDR box contract."""

    _OPTION_DEFAULTS = {
        "hidden_dim": 256,
        "num_queries": 300,
        "num_decoder_layers": FDR_DECODER_LAYERS,
        "reg_max": REG_MAX,
        "reg_scale": REG_SCALE,
        "up": UP,
        "cumulative": True,
        "preliminary_box": True,
        "private_seed": 10_000,
    }

    def __init__(
        self,
        nc: int = 80,
        ch: tuple[int, int, int] | list[int] = (256, 256, 256),
        declared_ch_or_options: tuple[int, int, int] | list[int] | dict | None = None,
        options: dict | None = None,
    ) -> None:
        if options is None and isinstance(declared_ch_or_options, dict):
            options = declared_ch_or_options
            declared_channels = tuple(int(channel) for channel in ch)
        else:
            declared_channels = tuple(
                int(channel)
                for channel in (
                    declared_ch_or_options
                    if declared_ch_or_options is not None
                    else ch
                )
            )
        parsed_channels = tuple(int(channel) for channel in ch)
        if declared_channels != parsed_channels:
            raise ValueError(
                "FDR YAML channels do not match parsed feature channels: "
                f"declared={declared_channels}, parsed={parsed_channels}"
            )
        supplied = dict(options or {})
        unknown = set(supplied) - set(self._OPTION_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown FDR decoder options: {sorted(unknown)}")
        resolved = {**self._OPTION_DEFAULTS, **supplied}
        if int(resolved["reg_max"]) != REG_MAX:
            raise ValueError(f"formal FDR requires reg_max={REG_MAX}")
        if float(resolved["reg_scale"]) != REG_SCALE:
            raise ValueError(f"formal FDR requires reg_scale={REG_SCALE}")
        if float(resolved["up"]) != UP:
            raise ValueError(f"formal FDR requires up={UP}")

        hidden_dim = int(resolved["hidden_dim"])
        num_queries = int(resolved["num_queries"])
        num_layers = int(resolved["num_decoder_layers"])
        private_seed = int(resolved["private_seed"])
        super().__init__(
            nc=int(nc),
            ch=parsed_channels,
            hd=hidden_dim,
            nq=num_queries,
            ndl=num_layers,
        )

        stock_pre_bbox_head = self.dec_bbox_head[0]
        distribution_heads = build_distribution_heads(
            hidden_dim,
            num_layers,
            private_seed=private_seed,
        )
        self.decoder = FDRDeformableTransformerDecoder.from_stock(
            self.decoder,
            pre_bbox_head=stock_pre_bbox_head,
        )
        self.dec_bbox_head = distribution_heads
        self.decoder.reg_max = int(resolved["reg_max"])
        self.decoder.final_layers = [
            module.layers[-1] for module in distribution_heads
        ]
        self.decoder.cumulative = bool(resolved["cumulative"])
        self.decoder.preliminary_box = bool(resolved["preliminary_box"])
        self.fdr_options = resolved


def cumulative_distribution_logits(deltas: Tensor) -> Tensor:
    """Return official cumulative residual distribution logits by layer."""

    if deltas.ndim < 2 or deltas.shape[-1] != FDR_OUTPUT_DIM:
        raise ValueError(
            f"deltas must have a layer axis and last dimension {FDR_OUTPUT_DIM}"
        )
    return deltas.cumsum(dim=0)


def build_distribution_heads(
    hidden_dim: int,
    num_layers: int = FDR_DECODER_LAYERS,
    *,
    private_seed: int,
) -> nn.ModuleList:
    """Build deterministic private FDR heads without consuming public RNG."""

    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers != FDR_DECODER_LAYERS:
        raise ValueError(f"FDR-only requires exactly {FDR_DECODER_LAYERS} decoder layers")

    private_generator = torch.Generator(device="cpu")
    private_generator.manual_seed(int(private_seed))
    with torch.device("meta"):
        heads = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, FDR_OUTPUT_DIM, num_layers=3) for _ in range(num_layers)]
        )
    heads.to_empty(device=torch.device("cpu"))
    for head in heads:
        for layer in head.layers:
            nn.init.kaiming_uniform_(
                layer.weight,
                a=math.sqrt(5),
                generator=private_generator,
            )
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(
                    layer.bias,
                    -bound,
                    bound,
                    generator=private_generator,
                )
    for head in heads:
        nn.init.zeros_(head.layers[-1].weight)
        nn.init.zeros_(head.layers[-1].bias)
    return heads


class FDRDeformableTransformerDecoder(nn.Module):
    """Run stock decoder layers with the pinned D-FINE FDR box representation."""

    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        eval_idx: int,
        pre_bbox_head: nn.Module,
    ) -> None:
        super().__init__()
        if num_layers != FDR_DECODER_LAYERS or len(layers) != FDR_DECODER_LAYERS:
            raise ValueError(f"FDR-only requires exactly six decoder layers")
        self.layers = layers
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.eval_idx = int(eval_idx)
        self.pre_bbox_head = pre_bbox_head
        self.cumulative = True
        self.preliminary_box = True
        self.integral = Integral(REG_MAX, torch.tensor([UP]), torch.tensor([REG_SCALE]))
        self.register_buffer("up", torch.tensor([UP], dtype=torch.float32))
        self.register_buffer("reg_scale", torch.tensor([REG_SCALE], dtype=torch.float32))

        self.last_corner_logits: Tensor | None = None
        self.last_references: Tensor | None = None
        self.last_pre_bboxes: Tensor | None = None

    def __setstate__(self, state: dict) -> None:
        """Restore exact pinned defaults in pre-declarative pickled checkpoints."""

        super().__setstate__(state)
        if not hasattr(self, "cumulative"):
            self.cumulative = True
        if not hasattr(self, "preliminary_box"):
            self.preliminary_box = True

    @classmethod
    def from_stock(
        cls,
        stock: nn.Module,
        *,
        pre_bbox_head: nn.Module,
    ) -> "FDRDeformableTransformerDecoder":
        """Wrap the exact stock layers and privately copy the preliminary head."""

        required = ("layers", "hidden_dim", "num_layers", "eval_idx")
        missing = [name for name in required if not hasattr(stock, name)]
        if missing:
            raise TypeError(f"stock decoder is missing required fields: {missing}")
        if int(stock.num_layers) != FDR_DECODER_LAYERS or len(stock.layers) != FDR_DECODER_LAYERS:
            raise ValueError("FDR-only requires exactly six decoder layers")
        return cls(
            layers=stock.layers,
            hidden_dim=stock.hidden_dim,
            num_layers=stock.num_layers,
            eval_idx=stock.eval_idx,
            pre_bbox_head=deepcopy(pre_bbox_head),
        )

    def _clear_evidence(self) -> None:
        self.last_corner_logits = None
        self.last_references = None
        self.last_pre_bboxes = None

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
        """Return stock-compatible box/class stacks and retain training evidence."""

        if len(bbox_head) != self.num_layers or len(score_head) != self.num_layers:
            raise ValueError("FDR decoder requires one box and score head per decoder layer")
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
                initial_reference = (
                    preliminary.detach()
                    if self.preliminary_box
                    else reference.detach()
                )

            if initial_reference is None:
                raise RuntimeError("preliminary FDR reference was not initialized")
            delta_corners = bbox_head[index](output + output_detach)
            cumulative_corners = (
                delta_corners + cumulative_corners
                if self.cumulative
                else delta_corners
            )
            refined = distance2bbox(
                initial_reference,
                self.integral(cumulative_corners),
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

        if preliminary is None or not decoded_boxes:
            raise RuntimeError("FDR decoder produced no output")
        self.last_pre_bboxes = preliminary
        self.last_corner_logits = torch.stack(corner_logits)
        self.last_references = torch.stack(references)
        return torch.stack(decoded_boxes), torch.stack(class_logits)


__all__ = [
    "FDR_DECODER_LAYERS",
    "FDR_OUTPUT_DIM",
    "FDRDeformableTransformerDecoder",
    "FDRRTDETRDecoder",
    "build_distribution_heads",
    "cumulative_distribution_logits",
]

"""FDR-only decoder box path for Ultralytics RT-DETR.

The stock decoder layers and classification heads remain untouched.  This
module owns only the preliminary four-coordinate box head contract and the
six cumulative 33-bin-per-edge distribution heads used by FDR.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import MLP
from ultralytics.nn.modules.utils import inverse_sigmoid

from src.fdr_math import (
    Integral,
    REG_MAX,
    REG_SCALE,
    UP,
    decode_feasible_fdr_boxes,
    distance2bbox,
)
from src.bpdd_capacity import LocalBoundaryExpert


FDR_OUTPUT_DIM = 4 * (REG_MAX + 1)
FDR_DECODER_LAYERS = 6


@dataclass(frozen=True)
class ExpertPrediction:
    """Normal-query prediction produced by the independent local expert."""

    boxes: Tensor
    classes: Tensor
    corners: Tensor


class DistributionConditionedFeedback(nn.Module):
    """Encode a detached preceding 4x33 FDR distribution for one next residual."""

    def __init__(self, hidden_dim: int, *, private_seed: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(private_seed))
        with torch.device("meta"):
            self.edge_encoder = nn.Linear(REG_MAX + 1, 16)
            self.output = nn.Linear(4 * 16, hidden_dim)
        self.to_empty(device=torch.device("cpu"))
        nn.init.kaiming_uniform_(
            self.edge_encoder.weight, a=math.sqrt(5), generator=generator
        )
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.edge_encoder.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.edge_encoder.bias, -bound, bound, generator=generator)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cumulative_logits: Tensor) -> Tensor:
        if cumulative_logits.ndim != 3 or cumulative_logits.shape[-1] != FDR_OUTPUT_DIM:
            raise ValueError(
                f"cumulative_logits must be rank 3 with last dimension {FDR_OUTPUT_DIM}"
            )
        probabilities = cumulative_logits.detach().reshape(
            *cumulative_logits.shape[:-1], 4, REG_MAX + 1
        ).softmax(dim=-1)
        encoded = torch.nn.functional.silu(self.edge_encoder(probabilities))
        return self.output(encoded.flatten(start_dim=-2))


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
        "distribution_feedback": False,
        "feasible_geometry": True,
        "local_expert": False,
        "local_expert_preserve_base": False,
        "local_expert_hidden": 256,
        "local_expert_heads": 8,
        "local_expert_ff": 1024,
        "local_expert_seed": 30_000,
        "local_expert_chunk_size": 32,
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
        preserve_base = bool(resolved["local_expert_preserve_base"])
        if preserve_base and not bool(resolved["local_expert"]):
            raise ValueError("local_expert_preserve_base requires local_expert=True")
        if preserve_base and len(parsed_channels) != 4:
            raise ValueError("preserved local expert requires P2/P3/P4/P5 inputs")
        decoder_channels = parsed_channels[1:] if preserve_base else parsed_channels
        if len(decoder_channels) != 3:
            raise ValueError("RT-DETR decoder requires exactly P3/P4/P5 inputs")
        super().__init__(
            nc=int(nc),
            ch=decoder_channels,
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
            distribution_feedback=(
                DistributionConditionedFeedback(
                    hidden_dim,
                    private_seed=private_seed + 1,
                )
                if bool(resolved["distribution_feedback"])
                else None
            ),
            feasible_geometry=bool(resolved["feasible_geometry"]),
        )
        self.dec_bbox_head = distribution_heads
        self.decoder.reg_max = int(resolved["reg_max"])
        self.decoder.final_layers = [
            module.layers[-1] for module in distribution_heads
        ]
        self.decoder.cumulative = bool(resolved["cumulative"])
        self.decoder.preliminary_box = bool(resolved["preliminary_box"])
        self.decoder.feasible_geometry = bool(resolved["feasible_geometry"])
        self.decoder.local_expert_preserve_base = preserve_base
        if bool(resolved["local_expert"]):
            self.decoder.local_expert = LocalBoundaryExpert(
                channels=(parsed_channels[:3] if preserve_base else (hidden_dim,) * 3),
                hidden=int(resolved["local_expert_hidden"]),
                nc=int(nc),
                heads=int(resolved["local_expert_heads"]),
                ff=int(resolved["local_expert_ff"]),
                private_seed=int(resolved["local_expert_seed"]),
                chunk_size=int(resolved["local_expert_chunk_size"]),
            )
        else:
            self.decoder.local_expert = None
        self.fdr_options = resolved

    def forward(self, x: list[Tensor], batch: dict | None = None) -> tuple | Tensor:
        """Route raw P2/P3/P4 to the expert and projected P3/P4/P5 to RT-DETR."""

        if not bool(self.fdr_options["local_expert_preserve_base"]):
            return super().forward(x, batch)

        from ultralytics.models.utils.ops import get_cdn_group

        self.decoder._clear_evidence()
        if len(x) != 4:
            raise ValueError("preserved local expert requires four P2/P3/P4/P5 inputs")
        local_features = x[:3]
        feats, shapes = self._get_encoder_input(x[1:])
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox
        )
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            local_features=local_features,
            normal_query_count=self.num_queries,
        )
        if self.training and dn_meta is None:
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        raw = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return raw
        processed = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return processed if self.export else (processed, raw)


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
        distribution_feedback: DistributionConditionedFeedback | None = None,
        feasible_geometry: bool = True,
    ) -> None:
        super().__init__()
        if num_layers != FDR_DECODER_LAYERS or len(layers) != FDR_DECODER_LAYERS:
            raise ValueError(f"FDR-only requires exactly six decoder layers")
        self.layers = layers
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.eval_idx = int(eval_idx)
        self.pre_bbox_head = pre_bbox_head
        self.distribution_feedback = distribution_feedback
        self.local_expert: LocalBoundaryExpert | None = None
        self.local_expert_preserve_base = False
        self.feasible_geometry = bool(feasible_geometry)
        # This must remain a plain Python float. ModelEMA would smooth a tensor
        # buffer and leave residual feedback after the exact-off boundary.
        self.distribution_feedback_scale = 1.0
        self.cumulative = True
        self.preliminary_box = True
        self.integral = Integral(REG_MAX, torch.tensor([UP]), torch.tensor([REG_SCALE]))
        self.register_buffer("up", torch.tensor([UP], dtype=torch.float32))
        self.register_buffer("reg_scale", torch.tensor([REG_SCALE], dtype=torch.float32))

        self.last_corner_logits: Tensor | None = None
        self.last_references: Tensor | None = None
        self.last_pre_bboxes: Tensor | None = None
        self.last_geometry_statistics: dict[str, Tensor] = {}
        self.last_expert_prediction: ExpertPrediction | None = None

    @staticmethod
    def _feature_maps(feats: Tensor, shapes: list) -> list[Tensor]:
        """Unflatten the three projected encoder maps for local sampling."""
        if feats.ndim != 3 or feats.shape[-1] != 256:
            raise ValueError("encoder features must have shape [B,N,256]")
        maps: list[Tensor] = []
        offset = 0
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("encoder shape entries must be [height,width]")
            height, width = int(shape[0]), int(shape[1])
            count = height * width
            chunk = feats[:, offset : offset + count]
            if chunk.shape[1] != count:
                raise ValueError("encoder feature sequence does not match shapes")
            maps.append(chunk.transpose(1, 2).reshape(feats.shape[0], feats.shape[2], height, width))
            offset += count
        if offset != feats.shape[1] or len(maps) != 3:
            raise ValueError("local expert requires exactly three encoder feature maps")
        return maps

    def __setstate__(self, state: dict) -> None:
        """Restore exact pinned defaults in pre-declarative pickled checkpoints."""

        super().__setstate__(state)
        if not hasattr(self, "cumulative"):
            self.cumulative = True
        if not hasattr(self, "preliminary_box"):
            self.preliminary_box = True
        if not hasattr(self, "distribution_feedback"):
            self.distribution_feedback = None
        if not hasattr(self, "distribution_feedback_scale"):
            self.distribution_feedback_scale = 1.0
        if not hasattr(self, "feasible_geometry"):
            self.feasible_geometry = True
        if not hasattr(self, "local_expert"):
            self.local_expert = None
        if not hasattr(self, "local_expert_preserve_base"):
            self.local_expert_preserve_base = False
        if not hasattr(self, "last_expert_prediction"):
            self.last_expert_prediction = None

    def set_distribution_feedback_scale(self, value: float) -> None:
        """Set the exact adapter multiplier without registering EMA state."""

        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("distribution feedback scale must be finite and in [0, 1]")
        self.distribution_feedback_scale = value

    def freeze_distribution_feedback(self) -> None:
        """Freeze only the training-time DCF adapter parameters."""

        if self.distribution_feedback is not None:
            self.distribution_feedback.requires_grad_(False)

    @classmethod
    def from_stock(
        cls,
        stock: nn.Module,
        *,
        pre_bbox_head: nn.Module,
        distribution_feedback: DistributionConditionedFeedback | None = None,
        feasible_geometry: bool = True,
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
            distribution_feedback=distribution_feedback,
            feasible_geometry=feasible_geometry,
        )

    def _clear_evidence(self) -> None:
        self.last_corner_logits = None
        self.last_references = None
        self.last_pre_bboxes = None
        self.last_geometry_statistics = {}
        self.last_expert_prediction = None

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
        local_features: list[Tensor] | None = None,
        normal_query_count: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return stock-compatible box/class stacks and retain training evidence."""

        if len(bbox_head) != self.num_layers or len(score_head) != self.num_layers:
            raise ValueError("FDR decoder requires one box and score head per decoder layer")
        self._clear_evidence()
        output = embed
        output_detach: Tensor | int = 0
        cumulative_corners: Tensor | int = 0
        reference = refer_bbox.float().sigmoid()

        decoded_boxes: list[Tensor] = []
        class_logits: list[Tensor] = []
        corner_logits: list[Tensor] = []
        references: list[Tensor] = []
        geometry_records: list[dict[str, Tensor]] = []
        preliminary: Tensor | None = None
        initial_reference: Tensor | None = None

        for index, layer in enumerate(self.layers):
            # Geometry stays FP32, while attention/MLP inputs follow the network
            # dtype (including a fully half-converted model without autocast).
            layer_reference = reference.to(dtype=output.dtype)
            output = layer(
                output,
                layer_reference,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(layer_reference),
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
            regression_input = output + output_detach
            if (
                index > 0
                and self.distribution_feedback is not None
                and self.distribution_feedback_scale != 0.0
            ):
                if not isinstance(cumulative_corners, Tensor):
                    raise RuntimeError("DCF requires a preceding cumulative distribution")
                regression_input = regression_input + (
                    self.distribution_feedback_scale
                    * self.distribution_feedback(cumulative_corners)
                )
            delta_corners = bbox_head[index](regression_input)
            cumulative_corners = (
                delta_corners + cumulative_corners
                if self.cumulative
                else delta_corners
            )
            current_classes = score_head[index](output)
            # Casting alone is insufficient: autocast would lower F.linear again.
            with torch.autocast(device_type=cumulative_corners.device.type, enabled=False):
                raw_distance = self.integral(cumulative_corners.float())
                if self.feasible_geometry:
                    refined, geometry = decode_feasible_fdr_boxes(
                        initial_reference, raw_distance, self.reg_scale
                    )
                else:
                    refined = distance2bbox(
                        initial_reference, raw_distance, self.reg_scale
                    )
                    scale = self.reg_scale.abs().reshape(())
                    raw_pair = scale + raw_distance[..., :2] + raw_distance[..., 2:]
                    raw_extent = raw_pair * initial_reference[..., 2:] / scale
                    geometry = {
                        "total": raw_pair[..., 0].new_tensor(raw_pair[..., 0].numel(), dtype=torch.long),
                        "horizontal_infeasible": (raw_pair[..., 0] < 0).sum().detach(),
                        "vertical_infeasible": (raw_pair[..., 1] < 0).sum().detach(),
                        "minimum_raw_horizontal": raw_pair[..., 0].detach().amin(),
                        "minimum_raw_vertical": raw_pair[..., 1].detach().amin(),
                        "minimum_extent": raw_distance.new_zeros(()),
                        "minimum_decoded_width": raw_extent[..., 0].detach().amin(),
                        "minimum_decoded_height": raw_extent[..., 1].detach().amin(),
                        "numerical_floor_edges": raw_distance.new_zeros((), dtype=torch.long),
                    }
            reported_corners = cumulative_corners
            reported_classes = current_classes
            reported_boxes = refined
            if self.local_expert is not None and index == self.num_layers - 1:
                if self.local_expert_preserve_base:
                    if local_features is None or normal_query_count is None:
                        raise ValueError("preserved local expert requires raw features and query count")
                    normal_count = int(normal_query_count)
                    if normal_count <= 0 or normal_count > output.shape[1]:
                        raise ValueError("normal_query_count is outside the decoder query range")
                    normal_slice = slice(output.shape[1] - normal_count, output.shape[1])
                    expert_corners, expert_classes = self.local_expert(
                        output[:, normal_slice],
                        cumulative_corners[:, normal_slice],
                        current_classes[:, normal_slice],
                        refined[:, normal_slice].detach(),
                        local_features,
                    )
                    with torch.autocast(device_type=expert_corners.device.type, enabled=False):
                        expert_distance = self.integral(expert_corners.float())
                        expert_reference = initial_reference[:, normal_slice]
                        if self.feasible_geometry:
                            expert_boxes, _ = decode_feasible_fdr_boxes(
                                expert_reference, expert_distance, self.reg_scale
                            )
                        else:
                            expert_boxes = distance2bbox(
                                expert_reference, expert_distance, self.reg_scale
                            )
                    self.last_expert_prediction = ExpertPrediction(
                        boxes=expert_boxes,
                        classes=expert_classes,
                        corners=expert_corners,
                    )
                    if not self.training:
                        reported_corners = expert_corners
                        reported_classes = expert_classes
                        reported_boxes = expert_boxes
                else:
                    cumulative_corners, current_classes = self.local_expert(
                        output,
                        cumulative_corners,
                        current_classes,
                        refined.detach(),
                        self._feature_maps(feats, shapes),
                    )
                    with torch.autocast(device_type=cumulative_corners.device.type, enabled=False):
                        raw_distance = self.integral(cumulative_corners.float())
                        if self.feasible_geometry:
                            refined, geometry = decode_feasible_fdr_boxes(
                                initial_reference, raw_distance, self.reg_scale
                            )
                        else:
                            refined = distance2bbox(initial_reference, raw_distance, self.reg_scale)
                    reported_corners = cumulative_corners
                    reported_classes = current_classes
                    reported_boxes = refined
            geometry_records.append(geometry)

            if self.training or index == self.eval_idx:
                decoded_boxes.append(reported_boxes)
                class_logits.append(reported_classes)
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
        self.last_geometry_statistics = {
            "total": torch.stack([item["total"] for item in geometry_records]).sum(),
            "horizontal_infeasible": torch.stack(
                [item["horizontal_infeasible"] for item in geometry_records]
            ).sum(),
            "vertical_infeasible": torch.stack(
                [item["vertical_infeasible"] for item in geometry_records]
            ).sum(),
            "minimum_raw_horizontal": torch.stack(
                [item["minimum_raw_horizontal"] for item in geometry_records]
            ).amin(),
            "minimum_raw_vertical": torch.stack(
                [item["minimum_raw_vertical"] for item in geometry_records]
            ).amin(),
            "minimum_extent": geometry_records[0]["minimum_extent"],
            "minimum_decoded_width": torch.stack(
                [item["minimum_decoded_width"] for item in geometry_records]
            ).amin(),
            "minimum_decoded_height": torch.stack(
                [item["minimum_decoded_height"] for item in geometry_records]
            ).amin(),
            "numerical_floor_edges": torch.stack(
                [item["numerical_floor_edges"] for item in geometry_records]
            ).sum(),
        }
        return torch.stack(decoded_boxes), torch.stack(class_logits)


__all__ = [
    "FDR_DECODER_LAYERS",
    "FDR_OUTPUT_DIM",
    "FDRDeformableTransformerDecoder",
    "FDRRTDETRDecoder",
    "ExpertPrediction",
    "DistributionConditionedFeedback",
    "build_distribution_heads",
    "cumulative_distribution_logits",
]

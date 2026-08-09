"""Repository-owned Ultralytics RT-DETR integration for FDR-only boxes.

Only the decoder box representation changes.  The stock backbone, encoder,
query selection, decoder layers, classification heads, denoising builder and
postprocess implementation remain owned by Ultralytics 8.4.90.
"""

from __future__ import annotations

import re
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.nn.tasks import RTDETRDetectionModel, yaml_model_load
from ultralytics.utils import RANK

from src.fdr_head import (
    FDR_OUTPUT_DIM,
    FDRDeformableTransformerDecoder,
    FDRRTDETRDecoder,
)
from src.fdr_loss import FDRDetectionLoss
from src.fdr_protocol import load_fdr_initial_state
from src.rtdetr_lpr import FixedPairedProtocolMixin
from src.rtdetr_vsf_rmr import apply_resume_runtime_overrides


FDR_MODEL_CFG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr.yaml"
_MODEL_PARSE_LOCK = threading.RLock()


def register_fdr_module() -> None:
    """Expose the repository-owned head to Ultralytics' YAML parser."""

    ultralytics_tasks.FDRRTDETRDecoder = FDRRTDETRDecoder


def build_stock_rtdetr_model(*args: Any, **kwargs: Any) -> RTDETRDetectionModel:
    """Build a stock model without racing the temporary FDR parser alias."""

    with _MODEL_PARSE_LOCK:
        return RTDETRDetectionModel(*args, **kwargs)


def _legacy_fdr_state_signature(weights: Any) -> bool:
    """Recognize the exact structural signature of pre-declarative FDR weights."""

    if weights is None or not callable(getattr(weights, "state_dict", None)):
        return False
    state = weights.state_dict()
    has_pre_box = any(
        name.endswith(".decoder.pre_bbox_head.layers.2.weight")
        and tensor.ndim >= 1
        and int(tensor.shape[0]) == 4
        for name, tensor in state.items()
    )
    distribution_layers: set[int] = set()
    pattern = re.compile(r"(?:^|\.)dec_bbox_head\.(\d+)\.layers\.2\.weight$")
    for name, tensor in state.items():
        match = pattern.search(name)
        if match and tensor.ndim >= 1 and int(tensor.shape[0]) == FDR_OUTPUT_DIM:
            distribution_layers.add(int(match.group(1)))
    return has_pre_box and distribution_layers == set(range(6))


def _normalise_legacy_fdr_cfg(
    cfg: str | Path | dict,
    weights: Any,
) -> tuple[str | Path | dict, bool]:
    """Upgrade only signature-proven legacy FDR YAML to the declarative graph."""

    payload = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
    final_layer = payload.get("head", [None])[-1]
    if not isinstance(final_layer, list) or len(final_layer) != 4:
        return cfg, False
    module = final_layer[2]
    module_name = module if isinstance(module, str) else getattr(module, "__name__", "")
    if module_name == "FDRRTDETRDecoder":
        return cfg, False
    if module_name != "RTDETRDecoder" or not _legacy_fdr_state_signature(weights):
        return cfg, False

    declarative = yaml_model_load(FDR_MODEL_CFG)
    if "nc" in payload:
        declarative["nc"] = payload["nc"]
    options = declarative["head"][-1][3][-1]
    options["private_seed"] = int(getattr(weights, "private_seed", 10_000))
    return declarative, True


def _cfg_with_private_seed(
    cfg: str | Path | dict,
    private_seed: int | None,
    *,
    decoder_name: str = "FDRRTDETRDecoder",
) -> str | Path | dict:
    """Return an FDR YAML payload with an optional deterministic seed override."""

    if private_seed is None:
        return cfg
    payload = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
    final_layer = payload.get("head", [None])[-1]
    if len(final_layer) != 4 or final_layer[2] != decoder_name:
        raise TypeError("FDR model YAML must end with FDRRTDETRDecoder")
    arguments = final_layer[3]
    if len(arguments) != 3 or not isinstance(arguments[2], dict):
        raise TypeError("FDRRTDETRDecoder YAML arguments must end with an options mapping")
    arguments[2]["private_seed"] = int(private_seed)
    return payload


@dataclass(frozen=True)
class FDRTrainingEvidence:
    """Normal-query evidence plus an optional denoising-query partition."""

    corner_logits: Tensor
    references: Tensor
    pre_boxes: Tensor
    dn_corner_logits: Tensor | None = None
    dn_references: Tensor | None = None
    dn_pre_boxes: Tensor | None = None


def _dn_partition(dn_meta: dict[str, Any] | None) -> tuple[int, int] | None:
    if dn_meta is None:
        return None
    split = dn_meta.get("dn_num_split")
    if not isinstance(split, (list, tuple)) or len(split) != 2:
        raise ValueError("dn_meta must contain a two-element dn_num_split partition")
    denoising, normal = (int(split[0]), int(split[1]))
    if denoising < 0 or normal <= 0:
        raise ValueError("dn_num_split partition must be non-negative/positive")
    return denoising, normal


def split_fdr_evidence(
    corner_logits: Tensor,
    references: Tensor,
    pre_boxes: Tensor,
    dn_meta: dict[str, Any] | None,
) -> FDRTrainingEvidence:
    """Validate and split cached FDR tensors into normal and DN queries."""

    if corner_logits.ndim != 4 or corner_logits.shape[-1] != FDR_OUTPUT_DIM:
        raise ValueError("corner_logits must have shape [layers,batch,queries,132]")
    if references.shape != (*corner_logits.shape[:-1], 4):
        raise ValueError("references must match corner logits and end in four coordinates")
    if pre_boxes.shape != (*corner_logits.shape[1:3], 4):
        raise ValueError("pre_boxes must have shape [batch,queries,4]")

    partition = _dn_partition(dn_meta)
    if partition is None:
        return FDRTrainingEvidence(corner_logits, references, pre_boxes)

    denoising, normal = partition
    if denoising + normal != corner_logits.shape[2]:
        raise ValueError("dn_num_split partition does not match FDR query count")
    dn_corners, normal_corners = torch.split(
        corner_logits, (denoising, normal), dim=2
    )
    dn_references, normal_references = torch.split(
        references, (denoising, normal), dim=2
    )
    dn_pre, normal_pre = torch.split(pre_boxes, (denoising, normal), dim=1)
    return FDRTrainingEvidence(
        corner_logits=normal_corners,
        references=normal_references,
        pre_boxes=normal_pre,
        dn_corner_logits=dn_corners,
        dn_references=dn_references,
        dn_pre_boxes=dn_pre,
    )


class FDRRTDETRDetectionModel(RTDETRDetectionModel):
    """Ultralytics RT-DETR-L with only its decoder box path replaced."""

    # Subclasses may pin a YAML-visible FDR subtype. ``None`` deliberately
    # resolves the module global at construction time so tests and audited
    # runtime probes can replace the baseline head without stale references.
    yaml_decoder_type = None

    def __init__(
        self,
        cfg: str | Path | dict = FDR_MODEL_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
    ) -> None:
        with _MODEL_PARSE_LOCK:
            register_fdr_module()
            decoder_type = self.yaml_decoder_type or FDRRTDETRDecoder
            cfg = _cfg_with_private_seed(
                cfg,
                private_seed,
                decoder_name=decoder_type.__name__,
            )
            stock_decoder_type = ultralytics_tasks.RTDETRDecoder
            ultralytics_tasks.RTDETRDecoder = decoder_type
            try:
                super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
            finally:
                ultralytics_tasks.RTDETRDecoder = stock_decoder_type
        head = self.model[-1]
        if not isinstance(head, FDRRTDETRDecoder):
            raise TypeError("FDR model YAML must end with FDRRTDETRDecoder")
        if int(head.num_queries) != 300:
            raise ValueError("the frozen FDR protocol requires exactly 300 queries")
        if int(head.num_decoder_layers) != 6:
            raise ValueError("the frozen FDR protocol requires exactly six decoder layers")
        self.private_seed = int(head.fdr_options["private_seed"])
        self.fdr_loss_options = dict(self.yaml.get("fdr_loss", {}))
        self.nc = int(self.yaml["nc"])
        self.last_fdr_evidence: FDRTrainingEvidence | None = None
        self.last_fdr_losses: dict[str, Tensor] = {}

    @property
    def fdr(self) -> FDRDeformableTransformerDecoder:
        """Expose the repository-owned FDR box path without double-registering it."""

        decoder = self.model[-1].decoder
        if not isinstance(decoder, FDRDeformableTransformerDecoder):
            raise RuntimeError("FDR decoder was unexpectedly replaced")
        return decoder

    def _capture_fdr_evidence(
        self, dn_meta: dict[str, Any] | None
    ) -> FDRTrainingEvidence:
        decoder = self.fdr
        if (
            decoder.last_corner_logits is None
            or decoder.last_references is None
            or decoder.last_pre_bboxes is None
        ):
            raise RuntimeError("FDR decoder did not retain training evidence")
        evidence = split_fdr_evidence(
            decoder.last_corner_logits,
            decoder.last_references,
            decoder.last_pre_bboxes,
            dn_meta,
        )
        self.last_fdr_evidence = evidence
        return evidence

    def init_criterion(self) -> FDRDetectionLoss:
        """Build stock RT-DETR loss extended only by FGL and pre-box losses."""

        return FDRDetectionLoss(
            nc=self.nc,
            use_vfl=True,
            fgl_weight=float(self.fdr_loss_options.get("fgl_weight", 0.15)),
            supervise_pre_boxes=bool(
                self.fdr_loss_options.get("supervise_pre_boxes", True)
                and self.fdr.preliminary_box
            ),
        )

    def loss(
        self,
        batch: dict[str, Tensor],
        preds: tuple | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute seven-layer stock loss plus isolated decoder FDR losses."""

        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        image = batch["img"]
        batch_size = int(image.shape[0])
        batch_index = batch["batch_idx"]
        targets: dict[str, Any] = {
            "cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=image.device),
            "batch_idx": batch_index.to(image.device, dtype=torch.long).view(-1),
            "gt_groups": [
                int((batch_index == index).sum().item())
                for index in range(batch_size)
            ],
        }

        if preds is None:
            preds = self.predict(image, batch=targets)
        if not self.training and isinstance(preds, tuple) and len(preds) == 2:
            # Ultralytics validation wraps the five training tensors as
            # ``(postprocessed, auxiliary)`` before asking the model for loss.
            preds = preds[1]
        if not isinstance(preds, tuple) or len(preds) != 5:
            raise RuntimeError("stock RT-DETR loss prediction contract changed")
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds

        evidence = self.last_fdr_evidence
        if evidence is None:
            # Supports callers that supplied predictions from this model directly.
            evidence = self._capture_fdr_evidence(dn_meta)

        if dn_meta is None:
            dn_bboxes = None
            dn_scores = None
        else:
            partition = _dn_partition(dn_meta)
            if partition is None:
                raise RuntimeError("denoising partition unexpectedly disappeared")
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, partition, dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, partition, dim=2)

        # Preserve the exact Ultralytics stock contract: encoder first, then
        # all six decoder layers.  FDRDetectionLoss explicitly skips encoder
        # when consuming the six-layer corner/pre evidence.
        stock_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        stock_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        fgl_reference = (
            evidence.pre_boxes if self.fdr.preliminary_box else evidence.references[0]
        )
        dn_fgl_reference = evidence.dn_pre_boxes
        if not self.fdr.preliminary_box and evidence.dn_references is not None:
            dn_fgl_reference = evidence.dn_references[0]
        losses = self.criterion.stock_plus_fgl(
            (stock_bboxes, stock_scores),
            targets,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
            corner_logits=evidence.corner_logits,
            pre_boxes=fgl_reference,
            dn_corner_logits=evidence.dn_corner_logits,
            dn_pre_boxes=dn_fgl_reference,
        )
        self.last_fdr_losses = losses
        total = sum(losses.values())
        displayed = torch.stack(
            [
                losses[name].detach()
                for name in ("loss_giou", "loss_class", "loss_bbox")
            ]
        ).to(image.device)
        return total, displayed

    def predict(
        self,
        x: Tensor,
        profile: bool = False,
        visualize: bool = False,
        batch: dict[str, Any] | None = None,
        augment: bool = False,
        embed: list[int] | None = None,
    ) -> tuple | Tensor:
        """Run stock prediction and retain isolated FDR training evidence."""

        output = super().predict(
            x,
            profile=profile,
            visualize=visualize,
            batch=batch,
            augment=augment,
            embed=embed,
        )
        if self.training:
            if not isinstance(output, tuple) or len(output) != 5:
                raise RuntimeError("stock RT-DETR training output contract changed")
            self._capture_fdr_evidence(output[-1])
        else:
            self.last_fdr_evidence = None
        return output


def _load_initial_state(
    model: RTDETRDetectionModel,
    path: str | Path | None,
    *,
    variant: str,
) -> None:
    if path is None:
        return
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    load_fdr_initial_state(model, artifact, variant=variant)


class FDRFixedPairedProtocolMixin(FixedPairedProtocolMixin):
    """Use the already-passed real F3 AMP gate without a network model download."""

    def _setup_train(self) -> None:
        import ultralytics.engine.trainer as engine_trainer

        original = engine_trainer.check_amp
        engine_trainer.check_amp = lambda _model: True
        try:
            super()._setup_train()
        finally:
            engine_trainer.check_amp = original


class FDRTrainer(FDRFixedPairedProtocolMixin, RTDETRTrainer):
    """Strict paired trainer for the isolated FDR-only detector arm."""

    def __init__(
        self,
        *args: Any,
        experiment_seed: int = 0,
        initial_state_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.experiment_seed = int(experiment_seed)
        self.initial_state_path = (
            Path(initial_state_path) if initial_state_path is not None else None
        )
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides: dict[str, Any]) -> None:
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        common: list[torch.nn.Parameter] = []
        private: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = (
                private
                if ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name
                else common
            )
            destination.append(parameter)
        if not common or not private:
            raise RuntimeError("FDR stock/private parameter partition is incomplete")
        return {"gradient_norm": common, "fdr_gradient_norm": private}

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> FDRRTDETRDetectionModel:
        model_cfg, legacy_resume = _normalise_legacy_fdr_cfg(
            cfg or FDR_MODEL_CFG,
            weights,
        )
        model = FDRRTDETRDetectionModel(
            model_cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=None if legacy_resume else 10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_initial_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="fdr",
            )
        return model


class FDRControlTrainer(FDRFixedPairedProtocolMixin, RTDETRTrainer):
    """Stock RT-DETR control arm under the identical FDR paired protocol."""

    def __init__(
        self,
        *args: Any,
        initial_state_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.initial_state_path = (
            Path(initial_state_path) if initial_state_path is not None else None
        )
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides: dict[str, Any]) -> None:
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> RTDETRDetectionModel:
        model = build_stock_rtdetr_model(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        else:
            _load_initial_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="control",
            )
        return model


def run_f1_preflight(context: Any) -> dict[str, Any]:
    """Run the real CPU equivalence gate through the stable loader API."""
    from src.fdr_runtime_preflight import run_f1

    return run_f1(context)


def run_f2_preflight(context: Any) -> dict[str, Any]:
    """Run the real FDR tensor and edge-case gate through the stable loader API."""
    from src.fdr_runtime_preflight import run_f2

    return run_f2(context)


def run_f3_preflight(context: Any) -> dict[str, Any]:
    """Run the real RTX 4090 single-step integration gate."""
    from src.fdr_runtime_preflight import run_f3

    return run_f3(context)


def run_f4_representation_preflight(context: Any) -> dict[str, Any]:
    """Run the mature-baseline FDR representation audit."""
    from src.fdr_runtime_preflight import run_f4

    return run_f4(context)


__all__ = [
    "FDRControlTrainer",
    "FDRFixedPairedProtocolMixin",
    "FDRRTDETRDetectionModel",
    "FDRTrainer",
    "FDRTrainingEvidence",
    "build_stock_rtdetr_model",
    "run_f1_preflight",
    "run_f2_preflight",
    "run_f3_preflight",
    "run_f4_representation_preflight",
    "split_fdr_evidence",
]

"""PLEC stage integration at RT-DETR's pre-decoder P3 boundary."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from src.ascv_loc import preserve_batchnorm_buffers
from src.gcmv_data import (
    GCMV_GLOBAL_IMAGE_SIZE,
    GCMV_LOCAL_IMAGE_SIZE,
    GCMVRTDETRDataset,
)
from src.gcmv_geometry import PLECGeometry, build_plec_geometry
from src.gcmv_plec import (
    PLECOutput,
    PhasePreservingLocalEvidenceCanonicalizer,
)
from src.sbr_geometry import LetterboxTransform, overlapping_tiles


PLEC_CHANNELS = 256
PLEC_LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss")


def batchnorm_buffer_fingerprint(module: nn.Module) -> str:
    """Return a stable fingerprint of every BatchNorm running-statistic buffer."""

    digest = sha256()
    for name, child in module.named_modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(child, buffer_name, None)
            if value is None:
                continue
            digest.update(f"{name}.{buffer_name}\0".encode())
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class PLECReferenceAdapter(nn.Module):
    """Common non-contribution adapter used before the full PEG exists."""

    def __init__(self, channels: int = PLEC_CHANNELS) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.project = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.gamma_ref = nn.Parameter(torch.zeros(()))

    def forward(
        self, global_p3: torch.Tensor, local_canonical: torch.Tensor
    ) -> torch.Tensor:
        if global_p3.shape != local_canonical.shape:
            raise ValueError("global and canonical P3 tensors must share shape")
        return global_p3 + self.gamma_ref * self.project(local_canonical)


def _require_gcmv_config(payload: object) -> dict:
    expected = {
        "enabled": True,
        "module": "PLEC",
        "semantic_p3_index": 21,
        "decoder_feature_indices": [21, 24, 27],
        "global_imgsz": 640,
        "local_imgsz": 1088,
        "tile_ratio": 0.6,
        "views": ["TL", "TR", "BL", "BR"],
        "reference_adapter": {
            "projection": "Conv2d-1x1",
            "gamma_init": 0.0,
        },
    }
    if payload != expected:
        raise ValueError(f"GCMV PLEC configuration drift: {payload}")
    return expected


class GCMVPLECDetectionModel(RTDETRDetectionModel):
    """RT-DETR-L with four shared local feature passes and trainable PLEC."""

    def __init__(
        self,
        cfg: str | Path = "configs/rtdetr-l-gcmv-plec.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        config = _require_gcmv_config(self.yaml.get("gcmv"))
        self.gcmv_enabled = bool(config["enabled"])
        self.semantic_p3_index = int(config["semantic_p3_index"])
        self.decoder_feature_indices = tuple(
            int(index) for index in config["decoder_feature_indices"]
        )
        self.global_imgsz = int(config["global_imgsz"])
        self.local_imgsz = int(config["local_imgsz"])
        self.plec = PhasePreservingLocalEvidenceCanonicalizer(
            channels=PLEC_CHANNELS
        )
        self.reference_adapter = PLECReferenceAdapter(PLEC_CHANNELS)
        self.audit_local_batchnorm = False
        self.capture_local_feature_gradients = False
        self.last_local_p3: list[torch.Tensor] | None = None
        self.last_plec_diagnostics: dict[str, torch.Tensor | bool] = {}
        self.last_local_bn_preserved = True
        self.nc = self.yaml["nc"]
        self.loss_names = PLEC_LOSS_NAMES
        if self.model[-1].f != list(self.decoder_feature_indices):
            raise RuntimeError("GCMV decoder feature boundary drift")

    def _forward_until(
        self, x: torch.Tensor, *, stop_index: int
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        saved: list[torch.Tensor | None] = []
        for module in self.model:
            if module.i > stop_index:
                break
            if module.f != -1:
                x = (
                    saved[module.f]
                    if isinstance(module.f, int)
                    else [
                        x if source == -1 else saved[source]
                        for source in module.f
                    ]
                )
            x = module(x)
            saved.append(x if module.i in self.save else None)
        return x, saved

    def _extract_local_p3(self, local_view: torch.Tensor) -> torch.Tensor:
        feature, _ = self._forward_until(
            local_view, stop_index=self.semantic_p3_index
        )
        return feature

    def _build_plec_geometry(
        self,
        *,
        source_shapes: torch.Tensor,
        global_p3: torch.Tensor,
        local_p3: Sequence[torch.Tensor],
    ) -> PLECGeometry:
        if (
            not isinstance(source_shapes, torch.Tensor)
            or source_shapes.ndim != 2
            or source_shapes.shape[1] != 2
            or source_shapes.shape[0] != global_p3.shape[0]
        ):
            raise ValueError("source_shapes must have shape [B,2]")
        shapes = [
            (int(height), int(width))
            for height, width in source_shapes.detach().cpu().tolist()
        ]
        tiles = [
            overlapping_tiles(width=width, height=height)
            for height, width in shapes
        ]
        global_transforms = [
            LetterboxTransform.from_view(
                width=width,
                height=height,
                imgsz=self.global_imgsz,
            )
            for height, width in shapes
        ]
        local_transforms = [
            [
                LetterboxTransform.from_view(
                    width=tile.width,
                    height=tile.height,
                    imgsz=self.local_imgsz,
                )
                for tile in image_tiles
            ]
            for image_tiles in tiles
        ]
        return build_plec_geometry(
            source_shapes=shapes,
            tiles=tiles,
            global_transforms=global_transforms,
            local_transforms=local_transforms,
            global_feature_shape=tuple(global_p3.shape[-2:]),
            local_feature_shape=tuple(local_p3[0].shape[-2:]),
            device=global_p3.device,
            dtype=torch.float32,
        )

    def inject_local_p3(
        self,
        *,
        global_p3: torch.Tensor,
        local_p3: Sequence[torch.Tensor],
        source_shapes: torch.Tensor,
    ) -> torch.Tensor:
        geometry = self._build_plec_geometry(
            source_shapes=source_shapes,
            global_p3=global_p3,
            local_p3=local_p3,
        )
        plec_output: PLECOutput = self.plec(local_p3, geometry)
        self.last_plec_diagnostics = {
            "valid_count": plec_output.valid_count.detach(),
            "edge_prior": plec_output.edge_prior.detach(),
            "overlap_weights": plec_output.overlap_weights.detach(),
        }
        return self.reference_adapter(global_p3, plec_output.canonical)

    def _local_feature_passes(
        self, local_views: torch.Tensor
    ) -> list[torch.Tensor]:
        if (
            local_views.ndim != 5
            or local_views.shape[1:] != (
                4,
                3,
                self.local_imgsz,
                self.local_imgsz,
            )
        ):
            raise ValueError(
                "local_views must have shape "
                f"[B,4,3,{self.local_imgsz},{self.local_imgsz}]"
            )
        before = (
            batchnorm_buffer_fingerprint(self)
            if self.audit_local_batchnorm
            else None
        )
        features: list[torch.Tensor] = []
        for view_index in range(4):
            local_view = local_views[:, view_index]
            if self.training and torch.is_grad_enabled():
                feature = checkpoint(
                    self._extract_local_p3,
                    local_view,
                    use_reentrant=False,
                    context_fn=lambda: (
                        preserve_batchnorm_buffers(self),
                        preserve_batchnorm_buffers(self),
                    ),
                )
            else:
                with preserve_batchnorm_buffers(self):
                    feature = self._extract_local_p3(local_view)
            if self.capture_local_feature_gradients and feature.requires_grad:
                feature.retain_grad()
            features.append(feature)
        after = (
            batchnorm_buffer_fingerprint(self)
            if self.audit_local_batchnorm
            else before
        )
        self.last_local_bn_preserved = before == after
        if not self.last_local_bn_preserved:
            raise RuntimeError("GCMV local passes mutated BatchNorm buffers")
        self.last_local_p3 = (
            features if self.capture_local_feature_gradients else None
        )
        return features

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
            not self.gcmv_enabled
            or local_views is None
            or source_shapes is None
        ):
            return super().predict(
                x,
                profile=profile,
                visualize=visualize,
                batch=batch,
                augment=augment,
                embed=embed,
            )
        if profile or visualize or embed is not None:
            raise ValueError(
                "GCMV paired prediction does not support profile/visualize/embed"
            )

        global_output, saved = self._forward_until(x, stop_index=27)
        del global_output
        local_p3 = self._local_feature_passes(local_views)
        global_p3 = saved[self.decoder_feature_indices[0]]
        if not isinstance(global_p3, torch.Tensor):
            raise RuntimeError("global P3 was not retained")
        fused_p3 = self.inject_local_p3(
            global_p3=global_p3,
            local_p3=local_p3,
            source_shapes=source_shapes,
        )
        decoder_features = [
            fused_p3,
            saved[self.decoder_feature_indices[1]],
            saved[self.decoder_feature_indices[2]],
        ]
        if not all(isinstance(value, torch.Tensor) for value in decoder_features):
            raise RuntimeError("GCMV decoder features were not retained")
        return self.model[-1](decoder_features, batch)

    def loss(self, batch: dict, preds=None):
        if (
            preds is not None
            or not self.training
            or not self.gcmv_enabled
            or "local_views" not in batch
        ):
            return super().loss(batch, preds=preds)
        image = batch["img"]
        batch_indices = batch["batch_idx"].to(
            image.device, dtype=torch.long
        ).view(-1)
        targets = {
            "cls": batch["cls"].to(
                image.device, dtype=torch.long
            ).view(-1),
            "bboxes": batch["bboxes"].to(image.device),
            "batch_idx": batch_indices,
            "gt_groups": [
                int((batch_indices == index).sum().item())
                for index in range(image.shape[0])
            ],
        }
        predictions = self.predict(
            image,
            batch=targets,
            local_views=batch["local_views"],
            source_shapes=batch["source_shape"],
        )
        return super().loss(batch, preds=predictions)


class GCMVPLECTrainer(RTDETRTrainer):
    """Bounded train-only trainer for the first PLEC screen."""

    def _setup_train(self) -> None:
        requested_amp = bool(self.args.amp)
        self.args.amp = False
        super()._setup_train()
        if not requested_amp:
            return
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("GCMV PLEC AMP requires CUDA")
        self.args.amp = True
        self.amp = True
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=128.0,
            growth_interval=2**31 - 1,
        )

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights=None,
        verbose: bool = True,
    ):
        model = GCMVPLECDetectionModel(
            cfg or "configs/rtdetr-l-gcmv-plec.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights is not None:
            model.load(weights)
        return model

    def build_dataset(
        self, img_path: str, mode: str = "val", batch: int | None = None
    ):
        return GCMVRTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            local_imgsz=GCMV_LOCAL_IMAGE_SIZE,
            batch_size=batch,
            augment=False,
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
        processed = super().preprocess_batch(batch)
        processed["local_views"] = (
            processed["local_views"]
            .to(self.device, non_blocking=self.device.type == "cuda")
            .float()
            / 255
        )
        processed["source_shape"] = processed["source_shape"].to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        return processed

    def validate(self):
        return {}, float("-inf")

    def final_eval(self):
        return None


class GCMVPLECControlTrainer(GCMVPLECTrainer):
    """Matched stock arm using the same global data path and optimizer setup."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights=None,
        verbose: bool = True,
    ):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.gcmv_enabled = False
        return model

    def preprocess_batch(self, batch: dict) -> dict:
        global_batch = {
            key: value
            for key, value in batch.items()
            if key not in {"local_views", "source_shape"}
        }
        return RTDETRTrainer.preprocess_batch(self, global_batch)

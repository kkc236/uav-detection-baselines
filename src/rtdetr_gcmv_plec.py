"""PLEC stage integration at RT-DETR's pre-decoder P3 boundary."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from src.ascv_loc import preserve_batchnorm_buffers
from src.ascv_loc_protocol import state_fingerprint
from src.gcmv_data import (
    GCMV_GLOBAL_IMAGE_SIZE,
    GCMV_LOCAL_IMAGE_SIZE,
    GCMVRTDETRDataset,
)
from src.gcmv_geometry import PLECGeometry, build_plec_geometry
from src.gcmv_fusion import GCMVEvidenceInjectionModule
from src.gcmv_loss import (
    build_gcmv_scale_targets,
    gcmv_auxiliary_loss,
)
from src.gcmv_plec_protocol import validate_plec_initial_state_artifact
from src.gcmv_plec import (
    PLECOutput,
    PhasePreservingLocalEvidenceCanonicalizer,
)
from src.sbr_geometry import LetterboxTransform, overlapping_tiles


PLEC_CHANNELS = 256
PLEC_DETECTION_LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss")
PLEC_LOSS_NAMES = (
    *PLEC_DETECTION_LOSS_NAMES,
    "gcmv_tiny_loss",
    "gcmv_gate_loss",
    "gcmv_protect_loss",
)
PLEC_EXTRA_PREFIXES = ("plec.", "gcmv_injector.")


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


def load_plec_initial_state(
    model: nn.Module,
    artifact: dict,
    *,
    seed: int,
) -> None:
    """Load the sealed stock state while preserving new PLEC parameters."""

    if artifact.get("metadata", {}).get("seed") != seed:
        raise ValueError("PLEC_INITIAL_STATE_SEED_MISMATCH")
    common = artifact.get("common_state")
    expected = artifact.get("fingerprints", {}).get("common")
    if not isinstance(common, dict) or state_fingerprint(common) != expected:
        raise ValueError("PLEC_INITIAL_STATE_FINGERPRINT_MISMATCH")
    model_names = set(model.state_dict())
    extra_names = {
        name
        for name in model_names
        if name.startswith(PLEC_EXTRA_PREFIXES)
    }
    stock_names = model_names - extra_names
    common_names = set(common)
    if common_names != stock_names:
        raise ValueError(
            "PLEC_INITIAL_STATE_KEYS_MISMATCH: "
            f"missing={sorted(stock_names - common_names)}, "
            f"unexpected={sorted(common_names - stock_names)}"
        )
    incompatible = model.load_state_dict(common, strict=False)
    if set(incompatible.missing_keys) != extra_names:
        raise RuntimeError("PLEC extra-parameter initialization drift")
    if incompatible.unexpected_keys:
        raise RuntimeError("PLEC initial state has unexpected keys")


def _require_gcmv_config(payload: object) -> dict:
    expected = {
        "enabled": True,
        "module": "GCMV-EI",
        "semantic_p3_index": 21,
        "decoder_feature_indices": [21, 24, 27],
        "global_imgsz": 640,
        "local_imgsz": 1088,
        "tile_ratio": 0.6,
        "views": ["TL", "TR", "BL", "BR"],
        "gglf": {
            "interaction_channels": 64,
            "num_heads": 4,
            "window_size": 3,
        },
        "peg": {
            "residual_scalar_init": 0.0,
            "gate_logit_init": 0.0,
        },
    }
    if payload != expected:
        raise ValueError(f"GCMV-EI configuration drift: {payload}")
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
        self.gcmv_injector = GCMVEvidenceInjectionModule(
            channels=PLEC_CHANNELS,
            interaction_channels=int(
                config["gglf"]["interaction_channels"]
            ),
            num_heads=int(config["gglf"]["num_heads"]),
            window_size=int(config["gglf"]["window_size"]),
        )
        self.calibration_only = False
        self.audit_local_batchnorm = False
        self.capture_local_feature_gradients = False
        self.last_local_p3: list[torch.Tensor] | None = None
        self.last_plec_diagnostics: dict[str, torch.Tensor | bool] = {}
        self.last_gcmv_aux: dict[str, torch.Tensor] = {}
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
        global_to_source: torch.Tensor | None = None,
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
            global_to_source=(
                None
                if global_to_source is None
                else list(global_to_source)
            ),
            global_feature_shape=tuple(global_p3.shape[-2:]),
            local_feature_shape=tuple(local_p3[0].shape[-2:]),
            device=global_p3.device,
            dtype=torch.float32,
        )

    def inject_gcmv_evidence(
        self,
        *,
        global_p3: torch.Tensor,
        local_p3: Sequence[torch.Tensor],
        source_shapes: torch.Tensor,
        global_to_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        geometry = self._build_plec_geometry(
            source_shapes=source_shapes,
            global_to_source=global_to_source,
            global_p3=global_p3,
            local_p3=local_p3,
        )
        plec_output: PLECOutput = self.plec(local_p3, geometry)
        self.last_plec_diagnostics = {
            "valid_count": plec_output.valid_count.detach(),
            "edge_prior": plec_output.edge_prior.detach(),
            "overlap_weights": plec_output.overlap_weights.detach(),
        }
        injection = self.gcmv_injector(global_p3, plec_output)
        self.last_plec_diagnostics.update(
            {
                "gglf_confidence": injection.confidence.detach(),
                "gglf_attention_entropy": (
                    injection.attention_entropy.detach()
                ),
                "gglf_tiny_map": injection.tiny_map.detach(),
                "peg_gate_hat": injection.gate_hat.detach(),
                "peg_gate": injection.gate.detach(),
                "peg_gamma": injection.gamma.detach(),
            }
        )
        self.last_gcmv_aux = {
            "tiny_map": injection.tiny_map,
            "gate_hat": injection.gate_hat,
            "gate": injection.gate,
            "coverage": (plec_output.valid_count > 0).to(
                injection.tiny_map.dtype
            ),
        }
        return injection.enhanced

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
            with torch.no_grad(), preserve_batchnorm_buffers(self):
                feature = self._extract_local_p3(local_view).detach()
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
        global_to_source: torch.Tensor | None = None,
    ):
        if (
            not self.gcmv_enabled
            or local_views is None
            or source_shapes is None
            or global_to_source is None
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
        fused_p3 = self.inject_gcmv_evidence(
            global_p3=global_p3,
            local_p3=local_p3,
            source_shapes=source_shapes,
            global_to_source=global_to_source,
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
            global_to_source=batch["global_to_source"],
        )
        detection_loss, detection_items = super().loss(
            batch,
            preds=predictions,
        )
        required_aux = {"tiny_map", "gate_hat", "gate", "coverage"}
        if set(self.last_gcmv_aux) != required_aux:
            raise RuntimeError("GCMV auxiliary forward state is incomplete")
        tiny_target, non_tiny_mask = build_gcmv_scale_targets(
            bboxes=batch["bboxes"].to(
                image.device,
                dtype=torch.float32,
            ),
            batch_idx=batch_indices,
            batch_size=int(image.shape[0]),
            feature_shape=tuple(
                int(value)
                for value in self.last_gcmv_aux["tiny_map"].shape[-2:]
            ),
            image_shape=tuple(int(value) for value in image.shape[-2:]),
            tiny_max_size=16.0,
        )
        auxiliary = gcmv_auxiliary_loss(
            tiny_map=self.last_gcmv_aux["tiny_map"],
            gate_hat=self.last_gcmv_aux["gate_hat"],
            gate=self.last_gcmv_aux["gate"],
            coverage=self.last_gcmv_aux["coverage"],
            tiny_target=tiny_target,
            non_tiny_mask=non_tiny_mask,
        )
        auxiliary_items = torch.stack(
            (
                auxiliary.tiny.detach(),
                auxiliary.gate.detach(),
                auxiliary.protect.detach(),
            )
        )
        total_loss = (
            auxiliary.total
            if self.calibration_only
            else detection_loss + auxiliary.total
        )
        return total_loss, torch.cat((detection_items, auxiliary_items))


class GCMVPLECTrainer(RTDETRTrainer):
    """Bounded train-only trainer for the first GCMV-EI screen."""

    def __init__(
        self,
        *args,
        initial_state_path: str | Path,
        **kwargs,
    ) -> None:
        overrides = kwargs.get("overrides")
        if not isinstance(overrides, dict):
            raise ValueError("GCMV-EI overrides must be a dict")
        if int(overrides.get("batch", 0)) != 8:
            raise ValueError("GCMV-EI requires frozen batch=8")
        if int(overrides.get("seed", -1)) != 0:
            raise ValueError("GCMV-EI first screen requires seed=0")
        self.initial_state_path = Path(initial_state_path).resolve()
        self.initial_state = torch.load(
            self.initial_state_path,
            map_location="cpu",
            weights_only=False,
        )
        validate_plec_initial_state_artifact(self.initial_state, seed=0)
        self.plec_optimizer_attempts = 0
        self.plec_amp_scale_min = 128.0
        self.plec_amp_scale_max = 128.0
        super().__init__(*args, **kwargs)

    def _setup_train(self) -> None:
        requested_amp = bool(self.args.amp)
        self.args.amp = False
        super()._setup_train()
        self.loss_names = PLEC_LOSS_NAMES
        if not requested_amp:
            return
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("GCMV-EI AMP requires CUDA")
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
        if weights is not None:
            raise ValueError("GCMV-EI pretrained weights are forbidden")
        model = GCMVPLECDetectionModel(
            cfg or "configs/rtdetr-l-gcmv-plec.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        load_plec_initial_state(model, self.initial_state, seed=0)
        return model

    def build_dataset(
        self, img_path: str, mode: str = "val", batch: int | None = None
    ):
        return GCMVRTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            local_imgsz=GCMV_LOCAL_IMAGE_SIZE,
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
        for key in ("source_to_global", "global_to_source"):
            processed[key] = processed[key].to(
                self.device,
                non_blocking=self.device.type == "cuda",
                dtype=torch.float32,
            )
        return processed

    def validate(self):
        return {}, float("-inf")

    def final_eval(self):
        return None

    def _build_train_pipeline(self) -> None:
        if int(self.batch_size) != 8:
            raise RuntimeError(
                f"PLEC_BATCH_DRIFT: frozen=8 runtime={self.batch_size}"
            )
        super()._build_train_pipeline()
        if (
            int(getattr(self.train_loader, "batch_size", -1)) != 8
            or int(getattr(self.train_loader, "num_workers", -1)) != 8
        ):
            raise RuntimeError("PLEC train-loader contract drift")

    def optimizer_step(self) -> None:
        before = float(self.scaler.get_scale())
        if before != 128.0:
            raise FloatingPointError(
                f"PLEC AMP scale drift before optimizer step: {before}"
            )
        super().optimizer_step()
        after = float(self.scaler.get_scale())
        self.plec_optimizer_attempts += 1
        self.plec_amp_scale_min = min(self.plec_amp_scale_min, before, after)
        self.plec_amp_scale_max = max(self.plec_amp_scale_max, before, after)
        if after != 128.0:
            raise FloatingPointError(
                f"PLEC AMP scale drift after optimizer step: {after}"
            )


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
        model.loss_names = PLEC_DETECTION_LOSS_NAMES
        return model

    def _setup_train(self) -> None:
        super()._setup_train()
        self.loss_names = PLEC_DETECTION_LOSS_NAMES

    def preprocess_batch(self, batch: dict) -> dict:
        global_batch = {
            key: value
            for key, value in batch.items()
            if key
            not in {
                "local_views",
                "source_shape",
                "source_to_global",
                "global_to_source",
            }
        }
        return RTDETRTrainer.preprocess_batch(self, global_batch)

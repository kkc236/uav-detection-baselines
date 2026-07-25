"""Matched Ultralytics RT-DETR integration for the T-ASCV tiny expert."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOCAL_RANK, RANK
from ultralytics.utils.torch_utils import init_seeds

from src.ascv_loc import (
    ASCV_LAMBDA,
    add_preflight_checkpoint_probe,
    ascv_warmup,
    build_local_targets,
    canonical_image_id,
    crop_and_resize,
    join_matches_by_target_id,
    preserve_batchnorm_buffers,
    select_target_anchored_crops,
)
from src.ascv_loc_protocol import (
    state_fingerprint,
    training_batch_sha256,
    validate_initial_state_artifact,
)
from src.tascv import TASCVLossResult, compute_tascv_loss
from src.tascv_stage import TASCVStage, allowed_seeds, stage_policy


LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "tascv_loss")
CONTROL_LOSS_NAMES = LOSS_NAMES[:3]
MATCHED_BATCH_SIZE = 8
MATCHED_AMP_SCALE = 128.0
MATCHED_AMP_GROWTH_INTERVAL = 2**31 - 1


@dataclass(frozen=True)
class RegularPredictions:
    boxes: torch.Tensor
    scores: torch.Tensor


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"TASCV_NONFINITE_{name.upper()}")


def _strict_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"T-ASCV {name} must be an exact integer")
    return int(value)


def _batchnorm_buffer_fingerprint(module: torch.nn.Module) -> str:
    buffers = {}
    for module_name, child in module.named_modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            for buffer_name in (
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ):
                value = getattr(child, buffer_name, None)
                if value is not None:
                    buffers[f"{module_name}.{buffer_name}"] = value
    return state_fingerprint(buffers)


def _full_targets(batch: dict, image: torch.Tensor) -> dict:
    batch_indices = batch["batch_idx"].to(
        image.device,
        dtype=torch.long,
    ).view(-1)
    groups = [
        int((batch_indices == index).sum().item())
        for index in range(image.shape[0])
    ]
    return {
        "cls": batch["cls"].to(
            image.device,
            dtype=torch.long,
        ).view(-1),
        "bboxes": batch["bboxes"].to(image.device),
        "batch_idx": batch_indices,
        "gt_groups": groups,
    }


def _regular_predictions(predictions) -> RegularPredictions:
    dec_boxes, dec_scores, _enc_boxes, _enc_scores, dn_meta = predictions
    if dn_meta is not None:
        _dn_boxes, dec_boxes = torch.split(
            dec_boxes,
            dn_meta["dn_num_split"],
            dim=2,
        )
        _dn_scores, dec_scores = torch.split(
            dec_scores,
            dn_meta["dn_num_split"],
            dim=2,
        )
    boxes = dec_boxes[-1].float().contiguous()
    scores = dec_scores[-1].float().contiguous()
    _require_finite("regular_boxes", boxes)
    _require_finite("regular_scores", scores)
    return RegularPredictions(boxes=boxes, scores=scores)


def _image_keys(
    batch: dict,
    batch_size: int,
    *,
    dataset_root: str | Path | None,
) -> list[str]:
    values = batch.get("im_file")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != batch_size
    ):
        raise RuntimeError(
            "T-ASCV training requires one stable im_file identity per image"
        )
    keys: list[str] = []
    for value in values:
        candidate = Path(str(value))
        if candidate.is_absolute():
            if dataset_root is None:
                raise RuntimeError(
                    "T-ASCV requires the sealed dataset root for absolute "
                    "im_file values"
                )
            keys.append(
                canonical_image_id(
                    candidate,
                    dataset_root=dataset_root,
                )
            )
        else:
            if ".." in candidate.parts:
                raise RuntimeError(
                    "T-ASCV received a non-canonical im_file identity: "
                    f"{value}"
                )
            keys.append(candidate.as_posix())
    return keys


class TASCVDetectionModel(RTDETRDetectionModel):
    """Stock RT-DETR plus a training-only tiny consistency loss."""

    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.tascv_epoch = 0
        self.last_tascv_result: TASCVLossResult | None = None
        self.last_tascv_diagnostics: dict[
            str,
            torch.Tensor | float | int,
        ] = {}
        self.tascv_dataset_root: Path | None = None
        self.last_local_forward_calls = 0
        self.last_local_bn_preserved = False
        self.tascv_preflight_probe = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.nc = self.yaml["nc"]
        self.loss_names = LOSS_NAMES

    def set_tascv_progress(self, epoch: int) -> None:
        resolved = _strict_integer("epoch", epoch)
        if resolved < 0:
            raise ValueError("T-ASCV epoch must be nonnegative")
        self.tascv_epoch = resolved

    def loss(self, batch: dict, preds=None):
        if not self.training:
            detection_loss, detection_items = super().loss(
                batch,
                preds=preds,
            )
            return detection_loss, torch.cat(
                (detection_items, detection_items.new_zeros(1))
            )

        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        image = batch["img"]
        targets = _full_targets(batch, image)
        if preds is None:
            preds = self.predict(image, batch=targets)

        # The stock full-view criterion is the only detection criterion call.
        detection_loss, detection_items = super().loss(
            batch,
            preds=preds,
        )
        _require_finite(
            "stock_detection_loss",
            detection_loss.float(),
        )
        full_regular = _regular_predictions(preds)

        crops = select_target_anchored_crops(
            boxes=targets["bboxes"],
            batch_indices=targets["batch_idx"],
            batch_size=image.shape[0],
            image_hw=tuple(image.shape[-2:]),
            image_keys=_image_keys(
                batch,
                image.shape[0],
                dataset_root=self.tascv_dataset_root,
            ),
        )
        local_targets = build_local_targets(
            full_boxes=targets["bboxes"],
            classes=targets["cls"],
            batch_indices=targets["batch_idx"],
            crops=crops,
            image_hw=tuple(image.shape[-2:]),
        )
        local_images = crop_and_resize(image, crops)
        local_bn_fingerprint = _batchnorm_buffer_fingerprint(self)
        self.last_local_forward_calls = 0
        self.last_local_bn_preserved = True

        def local_forward(
            local_input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.last_local_forward_calls += 1
            regular = _regular_predictions(
                self.predict(local_input, batch=None)
            )
            if (
                _batchnorm_buffer_fingerprint(self)
                != local_bn_fingerprint
            ):
                self.last_local_bn_preserved = False
                raise RuntimeError(
                    "TASCV_LOCAL_BRANCH_MUTATED_BATCHNORM_BUFFERS"
                )
            checkpoint_probe = regular.boxes.float().square().mean()
            return regular.boxes, regular.scores, checkpoint_probe

        local_boxes, local_scores, checkpoint_probe = checkpoint(
            local_forward,
            local_images,
            use_reentrant=False,
            context_fn=lambda: (
                preserve_batchnorm_buffers(self),
                preserve_batchnorm_buffers(self),
            ),
        )
        # The local expert is a teacher only. Detach before matching or any
        # auxiliary computation so future refactors cannot leak local grads.
        local_regular = RegularPredictions(
            boxes=local_boxes.detach(),
            scores=local_scores.detach(),
        )

        full_matches = self.criterion.matcher(
            full_regular.boxes,
            full_regular.scores,
            targets["bboxes"].float().contiguous(),
            targets["cls"],
            targets["gt_groups"],
        )
        local_matches = self.criterion.matcher(
            local_regular.boxes,
            local_regular.scores,
            local_targets.boxes.float().contiguous(),
            local_targets.classes.to(dtype=torch.long),
            local_targets.groups,
        )
        joined = join_matches_by_target_id(
            full_matches=full_matches,
            local_matches=local_matches,
            local_gt_ids=local_targets.gt_ids,
        )
        selected_full = full_regular.boxes[
            joined.batch_indices,
            joined.full_query_indices,
        ]
        selected_local = local_regular.boxes[
            joined.batch_indices,
            joined.local_query_indices,
        ]
        selected_targets = targets["bboxes"][joined.gt_ids]
        pair_crops = crops[joined.batch_indices]
        tascv_result = compute_tascv_loss(
            full_pred_boxes=selected_full,
            local_pred_boxes=selected_local,
            full_gt_boxes=selected_targets,
            pair_crops=pair_crops,
            image_hw=tuple(image.shape[-2:]),
        )
        active_weight = ASCV_LAMBDA * ascv_warmup(
            self.tascv_epoch
        )
        contribution = active_weight * tascv_result.loss
        total = detection_loss.float() + contribution
        total = add_preflight_checkpoint_probe(
            total,
            checkpoint_probe,
            enabled=self.tascv_preflight_probe,
        )
        _require_finite("total_loss", total)

        self.last_tascv_result = tascv_result
        self.last_tascv_diagnostics = {
            "epoch": self.tascv_epoch,
            "active_weight": active_weight,
            "contribution": contribution.detach(),
            "eligible_local_targets": len(local_targets.gt_ids),
            "auxiliary_non_tiny_pair_count": 0,
        }
        return total, torch.cat(
            (
                detection_items,
                tascv_result.loss.detach().reshape(1),
            )
        )


class _NoValidation:
    def __init__(self) -> None:
        self.metrics = SimpleNamespace(keys=[])

    def __call__(self, *args, **kwargs):
        raise RuntimeError("TASCV_INTERNAL_VALIDATION_FORBIDDEN")


def load_matched_initial_state(
    model: torch.nn.Module,
    artifact: dict,
    seed: int,
) -> None:
    if artifact.get("metadata", {}).get("seed") != seed:
        raise ValueError("TASCV_INITIAL_STATE_SEED_MISMATCH")
    common = artifact.get("common_state")
    expected = artifact.get("fingerprints", {}).get("common")
    if not isinstance(common, dict) or state_fingerprint(common) != expected:
        raise ValueError("TASCV_INITIAL_STATE_FINGERPRINT_MISMATCH")
    model_names = set(model.state_dict())
    common_names = set(common)
    if model_names != common_names:
        raise ValueError(
            "TASCV_INITIAL_STATE_KEYS_MISMATCH: "
            f"missing={sorted(model_names - common_names)}, "
            f"unexpected={sorted(common_names - model_names)}"
        )
    model.load_state_dict(common, strict=True)


class _MatchedTrainOnlyMixin:
    def __init__(
        self,
        *args,
        stage: TASCVStage | str,
        initial_state_path: str | Path,
        **kwargs,
    ) -> None:
        self.tascv_stage = TASCVStage(stage)
        overrides = kwargs.get("overrides")
        if not isinstance(overrides, dict):
            raise ValueError("T-ASCV overrides must be a dict")
        batch = _strict_integer("batch", overrides.get("batch"))
        if batch != MATCHED_BATCH_SIZE:
            raise ValueError("T-ASCV requires matched batch=8")
        seed = _strict_integer("seed", overrides.get("seed"))
        if seed not in allowed_seeds(self.tascv_stage):
            raise ValueError(
                f"T-ASCV seed {seed} is not allowed for stage "
                f"{self.tascv_stage.value}"
            )
        self.tascv_policy = stage_policy(self.tascv_stage)
        self.tascv_successful_batches = 0
        self.internal_validation_bypass_count = 0
        self.tascv_optimizer_attempts = 0
        self.tascv_amp_scale_min = MATCHED_AMP_SCALE
        self.tascv_amp_scale_max = MATCHED_AMP_SCALE
        self.tascv_optimizer_observation: dict = {}
        self.tascv_loader_observation: dict = {}
        self.tascv_observed_tensor_batch_sizes: set[int] = set()
        self.tascv_batch_canaries: list[dict[str, int | str]] = []
        self.tascv_preprocessed_batch_count = 0
        self.tascv_epoch_one_canary_recorded = False
        self.initial_state_path = Path(initial_state_path).resolve()
        self.initial_state = torch.load(
            self.initial_state_path,
            map_location="cpu",
            weights_only=False,
        )
        validate_initial_state_artifact(
            self.initial_state,
            seed=seed,
        )
        self.frozen_batch_size = MATCHED_BATCH_SIZE
        super().__init__(*args, **kwargs)

    def preprocess_batch(self, batch):
        processed = super().preprocess_batch(batch)
        observed_batch = int(processed["img"].shape[0])
        self.tascv_observed_tensor_batch_sizes.add(observed_batch)
        self.tascv_preprocessed_batch_count += 1
        epoch = int(getattr(self, "epoch", 0))
        should_bind = self.tascv_preprocessed_batch_count <= 2
        if epoch == 1 and not self.tascv_epoch_one_canary_recorded:
            should_bind = True
            self.tascv_epoch_one_canary_recorded = True
        if should_bind:
            self.tascv_batch_canaries.append(
                {
                    "epoch": epoch,
                    "batch": self.tascv_preprocessed_batch_count,
                    "sha256": training_batch_sha256(processed),
                }
            )
        return processed

    def _setup_train(self):
        self.args.amp = False
        super()._setup_train()
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("TASCV_MATCHED_AMP_REQUIRES_CUDA")
        self.args.amp = True
        self.amp = True
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=MATCHED_AMP_SCALE,
            growth_interval=MATCHED_AMP_GROWTH_INTERVAL,
        )

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        if name not in {"auto", "MuSGD"}:
            raise ValueError(f"TASCV_OPTIMIZER_DRIFT: {name}")
        optimizer = super().build_optimizer(
            model,
            name="MuSGD",
            lr=0.01,
            momentum=0.937,
            decay=decay,
            iterations=iterations,
        )
        self.tascv_optimizer_observation = {
            "class": type(optimizer).__name__,
            "requested_lr0": 0.01,
            "requested_momentum": 0.937,
            "groups": [
                {
                    "param_group": group.get("param_group"),
                    "lr": float(group["lr"]),
                    "momentum": float(
                        group.get("momentum", 0.0)
                    ),
                    "weight_decay": float(
                        group.get("weight_decay", 0.0)
                    ),
                    "use_muon": bool(
                        group.get("use_muon", False)
                    ),
                    "parameter_count": len(group["params"]),
                }
                for group in optimizer.param_groups
            ],
        }
        if self.tascv_optimizer_observation["class"] != "MuSGD":
            raise RuntimeError("TASCV_OPTIMIZER_CLASS_DRIFT")
        if any(
            group["momentum"] != 0.937
            for group in self.tascv_optimizer_observation["groups"]
        ):
            raise RuntimeError("TASCV_OPTIMIZER_MOMENTUM_DRIFT")
        return optimizer

    def optimizer_step(self):
        scale_before = float(self.scaler.get_scale())
        if scale_before != MATCHED_AMP_SCALE:
            raise RuntimeError(
                "TASCV_AMP_SCALE_DRIFT_BEFORE_STEP: "
                f"{scale_before}"
            )
        super().optimizer_step()
        scale_after = float(self.scaler.get_scale())
        self.tascv_optimizer_attempts += 1
        self.tascv_amp_scale_min = min(
            self.tascv_amp_scale_min,
            scale_before,
            scale_after,
        )
        self.tascv_amp_scale_max = max(
            self.tascv_amp_scale_max,
            scale_before,
            scale_after,
        )
        if scale_after != MATCHED_AMP_SCALE:
            raise FloatingPointError(
                "TASCV_AMP_SCALE_DRIFT_AFTER_STEP: "
                f"{scale_after}"
            )

    def _build_train_pipeline(self):
        if self.batch_size != self.frozen_batch_size:
            raise RuntimeError(
                "TASCV_BATCH_DRIFT: "
                f"frozen={self.frozen_batch_size}, "
                f"runtime={self.batch_size}"
            )
        init_seeds(
            int(self.args.seed) + 1 + RANK,
            deterministic=bool(self.args.deterministic),
        )
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"],
            batch_size=batch_size,
            rank=LOCAL_RANK,
            mode="train",
        )
        self.tascv_loader_observation = {
            "trainer_batch_size": int(self.batch_size),
            "per_rank_batch_size": int(batch_size),
            "loader_batch_size": int(
                getattr(self.train_loader, "batch_size", batch_size)
            ),
            "loader_num_workers": int(
                getattr(self.train_loader, "num_workers", -1)
            ),
        }
        expected_loader = {
            "trainer_batch_size": 8,
            "per_rank_batch_size": 8,
            "loader_batch_size": 8,
            "loader_num_workers": 8,
        }
        if self.tascv_loader_observation != expected_loader:
            raise RuntimeError(
                "TASCV_LOADER_CONTRACT_DRIFT: "
                f"{self.tascv_loader_observation}"
            )
        self.test_loader = None
        self.accumulate = max(
            round(self.args.nbs / self.batch_size),
            1,
        )
        weight_decay = (
            self.args.weight_decay
            * self.batch_size
            * self.accumulate
            / self.args.nbs
        )
        iterations = (
            math.ceil(
                len(self.train_loader.dataset)
                / max(self.batch_size, self.args.nbs)
            )
            * self.epochs
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def get_validator(self):
        return _NoValidation()

    def validate(self):
        self.internal_validation_bypass_count += 1
        return {}, float("-inf")

    def final_eval(self):
        return None

    def record_successful_batch(self) -> None:
        self.tascv_successful_batches += 1
        maximum = self.tascv_policy.max_train_batches
        if (
            maximum is not None
            and self.tascv_successful_batches >= maximum
        ):
            self.stop = True


class TASCVTrainer(_MatchedTrainOnlyMixin, RTDETRTrainer):
    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ):
        if weights:
            raise ValueError("TASCV_PRETRAINED_WEIGHTS_FORBIDDEN")
        model = TASCVDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model.tascv_dataset_root = Path(self.data["path"]).resolve()
        model.tascv_preflight_probe = (
            self.tascv_stage is TASCVStage.PREFLIGHT_1
        )
        load_matched_initial_state(
            model,
            self.initial_state,
            int(self.args.seed),
        )
        return model

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return super().get_validator()


class TASCVControlTrainer(_MatchedTrainOnlyMixin, RTDETRTrainer):
    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ):
        if weights:
            raise ValueError("TASCV_PRETRAINED_WEIGHTS_FORBIDDEN")
        model = RTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model.nc = model.yaml["nc"]
        load_matched_initial_state(
            model,
            self.initial_state,
            int(self.args.seed),
        )
        return model

    def get_validator(self):
        self.loss_names = CONTROL_LOSS_NAMES
        return super().get_validator()


__all__ = [
    "CONTROL_LOSS_NAMES",
    "LOSS_NAMES",
    "MATCHED_AMP_GROWTH_INTERVAL",
    "MATCHED_AMP_SCALE",
    "MATCHED_BATCH_SIZE",
    "RegularPredictions",
    "TASCVControlTrainer",
    "TASCVDetectionModel",
    "TASCVTrainer",
    "load_matched_initial_state",
]

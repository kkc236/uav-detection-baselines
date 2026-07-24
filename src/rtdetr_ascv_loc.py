from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOCAL_RANK, RANK
from ultralytics.utils.torch_utils import init_seeds
from torch.utils.checkpoint import checkpoint

from src.ascv_loc import (
    ASCV_LAMBDA,
    ASCVLocLossResult,
    ascv_warmup,
    build_local_targets,
    canonical_image_id,
    compute_ascv_loc_loss,
    crop_and_resize,
    join_matches_by_target_id,
    preserve_batchnorm_buffers,
    select_target_anchored_crops,
)
from src.ascv_loc_stage import ASCVStage, ASCVStagePolicy, stage_policy
from src.ascv_loc_protocol import state_fingerprint, validate_initial_state_artifact


LOSS_NAMES = ("giou_loss", "cls_loss", "l1_loss", "ascv_loc_loss")
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
        raise FloatingPointError(f"ASCV_LOC_NONFINITE_{name.upper()}")


def _full_targets(batch: dict, image: torch.Tensor) -> dict:
    batch_indices = batch["batch_idx"].to(image.device, dtype=torch.long).view(-1)
    groups = [int((batch_indices == index).sum().item()) for index in range(image.shape[0])]
    return {
        "cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
        "bboxes": batch["bboxes"].to(image.device),
        "batch_idx": batch_indices,
        "gt_groups": groups,
    }


def _regular_predictions(predictions) -> RegularPredictions:
    dec_boxes, dec_scores, _enc_boxes, _enc_scores, dn_meta = predictions
    if dn_meta is not None:
        _dn_boxes, dec_boxes = torch.split(dec_boxes, dn_meta["dn_num_split"], dim=2)
        _dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
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
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != batch_size:
        raise RuntimeError("ASCV-Loc training requires one stable im_file identity per image")
    keys: list[str] = []
    for value in values:
        candidate = Path(str(value))
        if candidate.is_absolute():
            if dataset_root is None:
                raise RuntimeError("ASCV-Loc requires the sealed dataset root for absolute im_file values")
            keys.append(canonical_image_id(candidate, dataset_root=dataset_root))
        else:
            normalized = candidate.as_posix()
            if ".." in candidate.parts:
                raise RuntimeError(f"ASCV-Loc received a non-canonical im_file identity: {value}")
            keys.append(normalized)
    return keys


class ASCVLocDetectionModel(RTDETRDetectionModel):
    """Stock RT-DETR plus a training-only, localization-only cross-view loss."""

    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.ascv_epoch = 0
        self.last_ascv_result: ASCVLocLossResult | None = None
        self.last_ascv_diagnostics: dict[str, torch.Tensor | float] = {}
        self.ascv_dataset_root: Path | None = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # Ultralytics 8.4.90 does not persist nc on RTDETRDetectionModel even
        # though init_criterion() reads it on the first loss call.
        self.nc = self.yaml["nc"]
        self.loss_names = LOSS_NAMES

    def set_ascv_progress(self, epoch: int) -> None:
        self.ascv_epoch = int(epoch)

    def loss(self, batch: dict, preds=None):
        # Validation/evaluation remains the exact stock branch. It receives a
        # fourth zero item only so trainer logging has a stable schema.
        if not self.training:
            detection_loss, detection_items = super().loss(batch, preds=preds)
            return detection_loss, torch.cat((detection_items, detection_items.new_zeros(1)))

        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        image = batch["img"]
        targets = _full_targets(batch, image)
        if preds is None:
            preds = self.predict(image, batch=targets)

        # The stock criterion is called exactly once and only with the original
        # full-view batch/predictions.
        detection_loss, detection_items = super().loss(batch, preds=preds)
        _require_finite("stock_detection_loss", detection_loss.float())
        full_regular = _regular_predictions(preds)

        crops = select_target_anchored_crops(
            boxes=targets["bboxes"],
            batch_indices=targets["batch_idx"],
            batch_size=image.shape[0],
            image_hw=tuple(image.shape[-2:]),
            image_keys=_image_keys(
                batch,
                image.shape[0],
                dataset_root=self.ascv_dataset_root,
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

        # No local targets are passed to the decoder: no DN queries and no
        # local detection/classification loss are created. Activation
        # checkpointing preserves the matched batch=8 contract on 24GB while
        # recomputing under the same frozen-BN context during backward.
        def local_forward(local_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            regular = _regular_predictions(self.predict(local_input, batch=None))
            return regular.boxes, regular.scores

        local_boxes, local_scores = checkpoint(
            local_forward,
            local_images,
            use_reentrant=False,
            context_fn=lambda: (preserve_batchnorm_buffers(self), preserve_batchnorm_buffers(self)),
        )
        local_regular = RegularPredictions(boxes=local_boxes, scores=local_scores)

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
        selected_full = full_regular.boxes[joined.batch_indices, joined.full_query_indices]
        selected_local = local_regular.boxes[joined.batch_indices, joined.local_query_indices]
        selected_targets = targets["bboxes"][joined.gt_ids]
        pair_crops = crops[joined.batch_indices]
        ascv_result = compute_ascv_loc_loss(
            full_pred_boxes=selected_full,
            local_pred_boxes=selected_local,
            full_gt_boxes=selected_targets,
            pair_crops=pair_crops,
            image_hw=tuple(image.shape[-2:]),
        )
        active_weight = ASCV_LAMBDA * ascv_warmup(self.ascv_epoch)
        contribution = active_weight * ascv_result.loss
        total = detection_loss.float() + contribution
        _require_finite("total_loss", total)

        self.last_ascv_result = ascv_result
        self.last_ascv_diagnostics = {
            "epoch": self.ascv_epoch,
            "active_weight": active_weight,
            "contribution": contribution.detach(),
            "eligible_local_targets": len(local_targets.gt_ids),
        }
        return total, torch.cat((detection_items, ascv_result.loss.detach().reshape(1)))


class _NoValidation:
    """Sentinel used to make accidental internal validation fail closed."""

    def __init__(self) -> None:
        self.metrics = SimpleNamespace(keys=[])

    def __call__(self, *args, **kwargs):
        raise RuntimeError("ASCV_LOC_INTERNAL_VALIDATION_FORBIDDEN")


def load_matched_initial_state(model: torch.nn.Module, artifact: dict, seed: int) -> None:
    if artifact.get("metadata", {}).get("seed") != seed:
        raise ValueError("ASCV_LOC_INITIAL_STATE_SEED_MISMATCH")
    common = artifact.get("common_state")
    expected_fingerprint = artifact.get("fingerprints", {}).get("common")
    if not isinstance(common, dict) or state_fingerprint(common) != expected_fingerprint:
        raise ValueError("ASCV_LOC_INITIAL_STATE_FINGERPRINT_MISMATCH")
    model_names = set(model.state_dict())
    common_names = set(common)
    if model_names != common_names:
        raise ValueError(
            f"ASCV_LOC_INITIAL_STATE_KEYS_MISMATCH: "
            f"missing={sorted(model_names - common_names)}, "
            f"unexpected={sorted(common_names - model_names)}"
        )
    model.load_state_dict(common, strict=True)


class _MatchedTrainOnlyMixin:
    """Exact matched-baseline optimizer, scaler, initialization and data path."""

    def __init__(
        self,
        *args,
        stage: ASCVStage | str,
        initial_state_path: str | Path,
        **kwargs,
    ) -> None:
        self.ascv_stage = ASCVStage(stage)
        self.ascv_policy = stage_policy(self.ascv_stage)
        self.ascv_successful_batches = 0
        self.internal_validation_bypass_count = 0
        self.ascv_optimizer_attempts = 0
        self.ascv_amp_scale_min = MATCHED_AMP_SCALE
        self.ascv_amp_scale_max = MATCHED_AMP_SCALE
        self.initial_state_path = Path(initial_state_path).resolve()
        self.initial_state = torch.load(self.initial_state_path, map_location="cpu", weights_only=False)
        overrides = kwargs.get("overrides")
        if not isinstance(overrides, dict) or int(overrides.get("batch", 0)) != MATCHED_BATCH_SIZE:
            raise ValueError("ASCV-Loc requires matched batch=8")
        validate_initial_state_artifact(self.initial_state, seed=int(overrides["seed"]))
        self.frozen_batch_size = MATCHED_BATCH_SIZE
        super().__init__(*args, **kwargs)

    def _setup_train(self):
        # Skip Ultralytics' unrelated YOLO26 download-based AMP admission.
        # The frozen matched protocol enables CUDA AMP with a fixed scale.
        self.args.amp = False
        super()._setup_train()
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("ASCV_LOC_MATCHED_AMP_REQUIRES_CUDA")
        self.args.amp = True
        self.amp = True
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=MATCHED_AMP_SCALE,
            growth_interval=MATCHED_AMP_GROWTH_INTERVAL,
        )

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        if name not in {"auto", "MuSGD"}:
            raise ValueError(f"ASCV_LOC_OPTIMIZER_DRIFT: {name}")
        return super().build_optimizer(
            model,
            name="MuSGD",
            lr=0.01,
            momentum=0.937,
            decay=decay,
            iterations=iterations,
        )

    def optimizer_step(self):
        scale_before = float(self.scaler.get_scale())
        if scale_before != MATCHED_AMP_SCALE:
            raise RuntimeError(f"ASCV_LOC_AMP_SCALE_DRIFT_BEFORE_STEP: {scale_before}")
        super().optimizer_step()
        scale_after = float(self.scaler.get_scale())
        self.ascv_optimizer_attempts += 1
        self.ascv_amp_scale_min = min(self.ascv_amp_scale_min, scale_before, scale_after)
        self.ascv_amp_scale_max = max(self.ascv_amp_scale_max, scale_before, scale_after)
        if scale_after != MATCHED_AMP_SCALE:
            raise FloatingPointError(f"ASCV_LOC_AMP_SCALE_DRIFT_AFTER_STEP: {scale_after}")

    def _build_train_pipeline(self):
        """Build only the train loader; never resolve a val/test dataset."""

        if self.batch_size != self.frozen_batch_size:
            raise RuntimeError(
                f"ASCV_LOC_BATCH_DRIFT: frozen={self.frozen_batch_size}, runtime={self.batch_size}"
            )
        init_seeds(int(self.args.seed) + 1 + RANK, deterministic=bool(self.args.deterministic))
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"],
            batch_size=batch_size,
            rank=LOCAL_RANK,
            mode="train",
        )
        self.test_loader = None
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
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
        # Ultralytics 8.4.90 calls validate at final_epoch even with val=False.
        # Return a non-selecting sentinel without constructing or reading val.
        self.internal_validation_bypass_count += 1
        return {}, float("-inf")

    def final_eval(self):
        # Scientific SBR evaluation is a separate, once-per-stage process.
        return None

    def record_successful_batch(self) -> None:
        self.ascv_successful_batches += 1
        maximum = self.ascv_policy.max_train_batches
        if maximum is not None and self.ascv_successful_batches >= maximum:
            self.stop = True


class ASCVLocTrainer(_MatchedTrainOnlyMixin, RTDETRTrainer):
    """Matched ASCV treatment trainer."""

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        if weights:
            raise ValueError("ASCV_LOC_PRETRAINED_WEIGHTS_FORBIDDEN")
        model = ASCVLocDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model.ascv_dataset_root = Path(self.data["path"]).resolve()
        load_matched_initial_state(model, self.initial_state, int(self.args.seed))
        return model

    def get_validator(self):
        self.loss_names = LOSS_NAMES
        return super().get_validator()


class ASCVControlTrainer(_MatchedTrainOnlyMixin, RTDETRTrainer):
    """Exact stock paired control with the same train-only runtime."""

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        if weights:
            raise ValueError("ASCV_LOC_PRETRAINED_WEIGHTS_FORBIDDEN")
        model = RTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model.nc = model.yaml["nc"]
        load_matched_initial_state(model, self.initial_state, int(self.args.seed))
        return model

    def get_validator(self):
        self.loss_names = CONTROL_LOSS_NAMES
        return super().get_validator()

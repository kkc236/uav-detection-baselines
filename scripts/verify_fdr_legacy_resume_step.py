"""Verify that a legacy FDR checkpoint can resume and execute one train step.

This is intentionally a verification utility, not a training entry point.  It
uses the production :class:`FDRTrainer` reconstruction and resume methods, then
performs one small CPU step and publishes evidence only after every invariant
has passed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from ultralytics.utils.torch_utils import ModelEMA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_fdr import FDRTrainer  # noqa: E402


FROZEN_OPTIMIZER = {
    "name": "MuSGD",
    "lr": 0.01,
    "momentum": 0.937,
    "decay": 0.0005,
}
FROZEN_AMP_SCALE = 128.0
FROZEN_AMP_GROWTH_INTERVAL = 2**31 - 1
EXPECTED_STATE_TENSORS = 950


@dataclass(frozen=True)
class LegacyResumeContext:
    trainer: Any
    checkpoint: dict[str, Any]
    saved_yaml_head: str
    normalized_yaml_head: str
    state_tensor_count: int


def checkpoint_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace ``path`` without exposing a partial success report."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _yaml_head_name(model: Any) -> str:
    yaml = getattr(model, "yaml", None)
    if not isinstance(yaml, dict):
        raise TypeError("checkpoint model does not contain an Ultralytics YAML mapping")
    try:
        module = yaml["head"][-1][2]
    except (KeyError, IndexError, TypeError) as error:
        raise TypeError("checkpoint model YAML does not contain a decoder head") from error
    if isinstance(module, str):
        return module
    return getattr(module, "__name__", type(module).__name__)


def _checkpoint_model(checkpoint: dict[str, Any]) -> Any:
    for field in ("ema", "model"):
        model = checkpoint.get(field)
        if model is not None:
            return model
    raise ValueError("checkpoint contains neither a non-null 'ema' nor 'model'")


def prepare_legacy_resume(
    *, checkpoint: str | Path, nc: int = 10
) -> LegacyResumeContext:
    """Reconstruct a legacy checkpoint through production trainer logic."""

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"legacy FDR checkpoint does not exist: {checkpoint_path}")

    # Avoid BaseTrainer.__init__: it creates run directories and dataset state
    # that are irrelevant to this isolated compatibility proof. setup_model is
    # still the real production method and therefore exercises load_checkpoint,
    # legacy signature detection, YAML normalization, and strict weight loading.
    trainer = object.__new__(FDRTrainer)
    trainer.model = str(checkpoint_path)
    trainer.args = SimpleNamespace(
        pretrained=False,
        close_mosaic=0,
        model=str(checkpoint_path),
    )
    trainer.resume = True
    trainer.data = {"nc": int(nc), "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None

    loaded_checkpoint = trainer.setup_model()
    if not isinstance(loaded_checkpoint, dict):
        raise TypeError("FDRTrainer.setup_model did not return a checkpoint mapping")
    source_model = _checkpoint_model(loaded_checkpoint)
    saved_yaml_head = _yaml_head_name(source_model)
    normalized_yaml_head = _yaml_head_name(trainer.model)
    if saved_yaml_head != "RTDETRDecoder":
        raise TypeError(
            "resume-step verifier requires a legacy RTDETRDecoder checkpoint; "
            f"got {saved_yaml_head}"
        )
    if normalized_yaml_head != "FDRRTDETRDecoder":
        raise TypeError(
            "legacy FDR YAML was not normalized to FDRRTDETRDecoder; "
            f"got {normalized_yaml_head}"
        )
    state_tensor_count = sum(
        isinstance(value, Tensor) for value in trainer.model.state_dict().values()
    )
    if state_tensor_count != EXPECTED_STATE_TENSORS:
        raise RuntimeError(
            "legacy FDR state contract changed: "
            f"expected {EXPECTED_STATE_TENSORS}, got {state_tensor_count}"
        )
    return LegacyResumeContext(
        trainer=trainer,
        checkpoint=loaded_checkpoint,
        saved_yaml_head=saved_yaml_head,
        normalized_yaml_head=normalized_yaml_head,
        state_tensor_count=state_tensor_count,
    )


def _new_cpu_scaler() -> torch.amp.GradScaler:
    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=True,
        init_scale=FROZEN_AMP_SCALE,
        growth_interval=FROZEN_AMP_GROWTH_INTERVAL,
    )
    if float(scaler.get_scale()) != FROZEN_AMP_SCALE:
        raise RuntimeError("CPU GradScaler did not initialize at the frozen scale 128")
    return scaler


def _new_model_ema(model: nn.Module) -> ModelEMA:
    return ModelEMA(model)


def _synthetic_batch(imgsz: int) -> dict[str, Any]:
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    return {
        "img": torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32),
        "cls": torch.zeros((1,), dtype=torch.long),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "batch_idx": torch.zeros((1,), dtype=torch.long),
        "gt_groups": [1],
    }


def _loss_tensor(model: nn.Module, batch: dict[str, Any]) -> Tensor:
    result = model.loss(batch)
    loss = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise TypeError("FDR synthetic training loss must be a scalar tensor")
    return loss


def _gradient_groups(trainer: Any) -> dict[str, list[nn.Parameter]]:
    method = getattr(trainer, "gradient_parameter_groups", None)
    if callable(method):
        groups = method()
    else:
        groups = {
            "gradient_norm": [
                parameter
                for parameter in trainer.model.parameters()
                if parameter.requires_grad
            ]
        }
    if not groups or not any(groups.values()):
        raise RuntimeError("resume step has no trainable parameter groups")
    return groups


def execute_resume_step(
    context: LegacyResumeContext, *, imgsz: int
) -> dict[str, Any]:
    """Restore optimizer/scaler/EMA and execute one finite CPU train step."""

    trainer = context.trainer
    checkpoint = context.checkpoint
    if "epoch" not in checkpoint:
        raise ValueError("checkpoint is missing epoch")
    checkpoint_epoch = int(checkpoint["epoch"])
    expected_start_epoch = checkpoint_epoch + 1

    trainer.optimizer = trainer.build_optimizer(
        trainer.model,
        name=FROZEN_OPTIMIZER["name"],
        lr=FROZEN_OPTIMIZER["lr"],
        momentum=FROZEN_OPTIMIZER["momentum"],
        decay=FROZEN_OPTIMIZER["decay"],
        iterations=100_000,
    )
    if type(trainer.optimizer).__name__ != "MuSGD":
        raise TypeError(
            "frozen resume optimizer must be MuSGD; "
            f"got {type(trainer.optimizer).__name__}"
        )
    trainer.scaler = _new_cpu_scaler()
    # A truthy sentinel asks BaseTrainer.resume_training() to construct and
    # restore the real ModelEMA once. Creating a full EMA copy here as well
    # would briefly double peak memory for this 200 MB checkpoint.
    trainer.ema = True
    trainer.epochs = expected_start_epoch + 1
    trainer.args.close_mosaic = 0
    trainer.resume_training(checkpoint)
    if int(trainer.start_epoch) != expected_start_epoch:
        raise RuntimeError(
            "resume start epoch mismatch: "
            f"expected {expected_start_epoch}, got {trainer.start_epoch}"
        )

    optimizer_groups = len(trainer.optimizer.param_groups)
    optimizer_states = len(trainer.optimizer.state)
    scale_before = float(trainer.scaler.get_scale())
    ema_before = int(trainer.ema.updates)
    if scale_before != FROZEN_AMP_SCALE:
        raise RuntimeError(
            f"restored AMP scale must be 128, got {scale_before}"
        )

    # The restored optimizer, scaler and EMA now own the required state. Drop
    # the serialized model/EMA/optimizer copies before the synthetic forward to
    # keep this CPU verifier usable on ordinary workstations.
    for field in ("model", "ema", "optimizer", "scaler"):
        if field in checkpoint:
            checkpoint[field] = None
    gc.collect()

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    batch = _synthetic_batch(int(imgsz))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        loss = _loss_tensor(trainer.model, batch)
    finite_loss = bool(torch.isfinite(loss.detach()).item())
    if not finite_loss:
        raise RuntimeError("synthetic resume step produced non-finite loss")

    trainer.scaler.scale(loss).backward()
    trainer.scaler.unscale_(trainer.optimizer)
    gradient_count = 0
    finite_gradients = True
    tracked_parameter: nn.Parameter | None = None
    tracked_index = 0
    tracked_before: Tensor | None = None
    for parameter in trainer.model.parameters():
        gradient = parameter.grad
        if not parameter.requires_grad or gradient is None:
            continue
        gradient_count += 1
        finite_gradients &= bool(torch.isfinite(gradient).all().item())
        if tracked_parameter is None and bool(torch.count_nonzero(gradient).item()):
            tracked_index = int(gradient.detach().abs().reshape(-1).argmax().item())
            tracked_parameter = parameter
            tracked_before = parameter.detach().reshape(-1)[tracked_index].clone()
    if gradient_count == 0:
        raise RuntimeError("synthetic resume step produced no gradients")
    if not finite_gradients:
        raise RuntimeError("synthetic resume step produced non-finite gradients")
    if tracked_parameter is None or tracked_before is None:
        raise RuntimeError("synthetic resume step produced no non-zero gradients")

    for parameters in _gradient_groups(trainer).values():
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    scale_after = float(trainer.scaler.get_scale())
    tracked_after = tracked_parameter.detach().reshape(-1)[tracked_index]
    optimizer_step = not torch.equal(tracked_before, tracked_after)
    if not optimizer_step:
        raise RuntimeError("MuSGD did not update the tracked model parameter")
    if scale_after != FROZEN_AMP_SCALE:
        raise RuntimeError(
            f"fixed AMP scale changed during resume step: {scale_before} -> {scale_after}"
        )
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.ema.update(trainer.model)
    ema_after = int(trainer.ema.updates)
    if ema_after != ema_before + 1:
        raise RuntimeError(
            f"EMA update counter did not advance exactly once: {ema_before} -> {ema_after}"
        )

    loss_value = float(loss.detach().cpu())
    if not math.isfinite(loss_value):
        raise RuntimeError("serialized resume loss is non-finite")
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "resume_start_epoch": int(trainer.start_epoch),
        "optimizer_param_groups": optimizer_groups,
        "optimizer_state_entries": optimizer_states,
        "amp_scale_before_step": scale_before,
        "amp_scale_after_step": scale_after,
        "ema_updates_before_step": ema_before,
        "ema_updates_after_step": ema_after,
        "synthetic_resume_step_input": [1, 3, int(imgsz), int(imgsz)],
        "cpu_autocast_dtype": "bfloat16",
        "synthetic_resume_step_loss": loss_value,
        "finite_loss": finite_loss,
        "finite_gradients": finite_gradients,
        "gradient_tensor_count": gradient_count,
        "optimizer_step": optimizer_step,
        "resume_step_verified": True,
    }


def verify_legacy_resume_step(
    *,
    checkpoint: str | Path,
    output: str | Path,
    nc: int = 10,
    imgsz: int = 128,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output).resolve()
    context = prepare_legacy_resume(checkpoint=checkpoint_path, nc=int(nc))
    step = execute_resume_step(context, imgsz=int(imgsz))
    report = {
        "checkpoint": str(checkpoint_path),
        "sha256": checkpoint_sha256(checkpoint_path),
        "saved_yaml_head": context.saved_yaml_head,
        "normalized_yaml_head": context.normalized_yaml_head,
        "state_tensor_count": context.state_tensor_count,
        **step,
    }
    write_json_atomic(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify legacy FDR YAML normalization and one complete CPU resume step."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nc", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_legacy_resume_step(
        checkpoint=args.checkpoint,
        output=args.output,
        nc=args.nc,
        imgsz=args.imgsz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

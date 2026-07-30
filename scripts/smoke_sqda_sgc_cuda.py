from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_sqda_sgc import (
    BASELINE_SHA256,
    MATCHED_AMP_GROWTH_INTERVAL,
    MATCHED_AMP_SCALE,
    SQDASGCDetectionModel,
    build_sqda_optimizer,
    freeze_stock_model,
    load_mature_baseline,
)


REPRESENTATIVE_GRADIENTS = (
    "point_offset_heads.0.weight",
    "value_projector.0.weight",
    "point_query.weight",
    "edge_query.weight",
    "reliability_projection.weight",
    "gate.0.weight",
    "fusion.weight",
    "context_logit",
    "layer_scale_logit",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one fixed-AMP CUDA backward step and audit the SQDA-SGC freeze contract."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(f"cuda:{int(args.device)}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = SQDASGCDetectionModel("rtdetr-l.yaml", nc=10, verbose=False)
    load_mature_baseline(
        model,
        args.checkpoint,
        expected_sha256=BASELINE_SHA256,
    )
    model = model.to(device).train()
    freeze_stock_model(model)
    stock_buffers = {
        name: value.detach().cpu().clone()
        for name, value in model.model.named_buffers()
    }
    optimizer = build_sqda_optimizer(model)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=MATCHED_AMP_SCALE,
        growth_interval=MATCHED_AMP_GROWTH_INTERVAL,
    )
    batch = {
        "img": torch.rand(1, 3, 640, 640, device=device),
        "batch_idx": torch.tensor([0.0], device=device),
        "cls": torch.tensor([[0.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device),
    }
    with torch.autocast("cuda", enabled=True):
        loss, loss_items = model(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("CUDA smoke loss is non-finite")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.sqda_sgc.parameters(),
        max_norm=0.1,
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
        raise RuntimeError("CUDA smoke gradient norm is invalid")
    parameters = dict(model.sqda_sgc.named_parameters())
    gradient_nonzero = {}
    for name in REPRESENTATIVE_GRADIENTS:
        gradient = parameters[name].grad
        valid = bool(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient)
        )
        gradient_nonzero[name] = valid
        if not valid:
            raise RuntimeError(f"CUDA smoke branch has an invalid gradient: {name}")
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() != MATCHED_AMP_SCALE:
        raise RuntimeError(f"fixed AMP scale drifted to {scaler.get_scale()}")
    if any(parameter.grad is not None for parameter in model.model.parameters()):
        raise RuntimeError("a frozen stock parameter received a gradient")
    changed_buffers = [
        name
        for name, value in model.model.named_buffers()
        if not torch.equal(value.detach().cpu(), stock_buffers[name])
    ]
    if changed_buffers:
        raise RuntimeError(f"frozen stock buffers changed: {changed_buffers[:5]}")

    report = {
        "passed": True,
        "loss": float(loss.detach().cpu()),
        "loss_items": [float(value) for value in loss_items.detach().cpu()],
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "gradient_clip": 0.1,
        "fixed_amp_scale": float(scaler.get_scale()),
        "fixed_amp_growth_interval": MATCHED_AMP_GROWTH_INTERVAL,
        "optimizer": "module-only AdamW",
        "optimizer_groups": len(optimizer.param_groups),
        "stock_parameters_with_grad": 0,
        "stock_buffers_changed": [],
        "representative_gradients_nonzero": gradient_nonzero,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

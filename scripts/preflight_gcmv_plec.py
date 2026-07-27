from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

from src.gcmv_data import GCMVRTDETRDataset
from src.rtdetr_gcmv_plec import (
    GCMVPLECDetectionModel,
    batchnorm_buffer_fingerprint,
    load_plec_initial_state,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"
PLEC_GRADIENT_FAMILIES = (
    "view_embedding",
    "phase_mlp",
    "metadata_mlp",
    "phase_reducer",
    "spatial_mixer",
    "pointwise",
    "overlap_head",
    "output_norm",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real CUDA forward/backward through GCMV-EI."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        leaves: list[torch.Tensor] = []
        for key in sorted(value):
            leaves.extend(_tensor_leaves(value[key]))
        return leaves
    if isinstance(value, (tuple, list)):
        leaves = []
        for item in value:
            leaves.extend(_tensor_leaves(item))
        return leaves
    return []


def tensor_tree_equal(first: Any, second: Any) -> bool:
    first_tensors = _tensor_leaves(first)
    second_tensors = _tensor_leaves(second)
    return len(first_tensors) == len(second_tensors) and all(
        torch.equal(left, right)
        for left, right in zip(first_tensors, second_tensors)
    )


def require_nonzero_gradient_families(
    module: torch.nn.Module,
    *,
    prefixes: Iterable[str],
) -> dict[str, float]:
    norms: dict[str, float] = {}
    named = tuple(module.named_parameters())
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in named
            if name.startswith(prefix)
        ]
        if not gradients:
            raise RuntimeError(f"missing gradient family={prefix}")
        if any(gradient is None for gradient in gradients):
            raise RuntimeError(f"detached gradient family={prefix}")
        norm = sum(
            float(gradient.detach().float().abs().sum().item())
            for gradient in gradients
            if gradient is not None
        )
        if not torch.isfinite(torch.tensor(norm)) or norm <= 0:
            raise RuntimeError(f"zero/nonfinite gradient family={prefix}")
        norms[prefix] = norm
    return norms


def require_detached_tensors(
    tensors: Iterable[torch.Tensor],
    *,
    label: str,
) -> None:
    for tensor in tensors:
        if tensor.requires_grad or tensor.grad_fn is not None or tensor.grad is not None:
            raise RuntimeError(f"{label} unexpectedly owns an autograd path")


def _dataset(args: argparse.Namespace, data: dict) -> GCMVRTDETRDataset:
    hyp = get_cfg(
        overrides={
            "imgsz": 640,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "mask_ratio": 4,
            "overlap_mask": True,
            "bgr": 0.0,
            "mosaic": 1.0,
            "close_mosaic": 10,
            "mixup": 0.0,
            "scale": 0.5,
            "translate": 0.1,
            "degrees": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "cutmix": 0.0,
            "copy_paste": 0.0,
        }
    )
    dataset = GCMVRTDETRDataset(
        img_path=data["train"],
        imgsz=640,
        local_imgsz=1088,
        batch_size=args.batch,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        prefix="preflight: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    dataset.close_mosaic(hyp)
    return dataset


def _load_model(
    *,
    model_path: str,
    initial_state_path: str,
    data: dict,
    device: torch.device,
) -> GCMVPLECDetectionModel:
    model = GCMVPLECDetectionModel(
        model_path,
        nc=int(data["nc"]),
        ch=int(data["channels"]),
        verbose=False,
    )
    artifact = torch.load(
        initial_state_path, map_location="cpu", weights_only=False
    )
    if not isinstance(artifact, dict):
        raise RuntimeError("initial state must contain a mapping")
    load_plec_initial_state(model, artifact, seed=0)
    return model.to(device)


def _prepare_batch(batch: dict, device: torch.device) -> dict:
    prepared = dict(batch)
    prepared["img"] = (
        batch["img"].to(device, non_blocking=True).float() / 255
    )
    prepared["local_views"] = (
        batch["local_views"].to(device, non_blocking=True).float() / 255
    )
    prepared["source_shape"] = batch["source_shape"].to(
        device, non_blocking=True
    )
    for key in ("source_to_global", "global_to_source"):
        prepared[key] = batch[key].to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )
    return prepared


def run_preflight(args: argparse.Namespace) -> dict:
    if args.batch != 8 or args.workers != 8:
        raise ValueError("GCMV-EI preflight requires batch=8 and workers=8")
    if not torch.cuda.is_available():
        raise RuntimeError("GCMV-EI preflight requires CUDA")
    device = torch.device(f"cuda:{args.device}")
    data = check_det_dataset(args.data, autodownload=False)
    dataset = _dataset(args, data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    batch = _prepare_batch(next(iter(loader)), device)
    model = _load_model(
        model_path=args.model,
        initial_state_path=args.initial_state,
        data=data,
        device=device,
    )

    model.eval()
    model.gcmv_injector.peg.rho.data.zero_()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        model.gcmv_enabled = False
        stock = model.predict(batch["img"])
        model.gcmv_enabled = True
        paired = model.predict(
            batch["img"],
            local_views=batch["local_views"],
            source_shapes=batch["source_shape"],
            global_to_source=batch["global_to_source"],
        )
    identity = tensor_tree_equal(stock, paired)
    if not identity:
        raise RuntimeError("gamma-zero GCMV output is not exact stock output")
    del stock, paired
    torch.cuda.empty_cache()

    model.train()
    model.audit_local_batchnorm = True
    model.capture_local_feature_gradients = True
    model.gcmv_injector.peg.rho.data.fill_(1.0)
    local_views = batch["local_views"].detach()
    batch["local_views"] = local_views
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.float16):
        loss, loss_items = model.loss(batch)
    if not bool(torch.isfinite(loss.float()).all()):
        raise RuntimeError("GCMV-EI preflight produced nonfinite loss")
    bn_after_forward = batchnorm_buffer_fingerprint(model)
    loss.backward()
    bn_after_backward = batchnorm_buffer_fingerprint(model)

    if not model.last_local_bn_preserved or bn_after_forward != bn_after_backward:
        raise RuntimeError("local feature passes mutated BatchNorm buffers")
    plec_gradients = require_nonzero_gradient_families(
        model.plec, prefixes=PLEC_GRADIENT_FAMILIES
    )
    injector_gradients = require_nonzero_gradient_families(
        model.gcmv_injector, prefixes=("gglf", "peg")
    )
    require_detached_tensors([local_views], label="local view pixels")
    if model.last_local_p3 is None:
        raise RuntimeError("local P3 tensors were not retained for audit")
    require_detached_tensors(model.last_local_p3, label="local P3")

    report = {
        "schema_version": "gcmv-ei-preflight/v2",
        "batch": int(args.batch),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "stock_identity_at_gamma_zero": identity,
        "audit_rho": 1.0,
        "audit_gamma": float(
            torch.tanh(model.gcmv_injector.peg.rho).detach().item()
        ),
        "loss": float(loss.detach().float().item()),
        "loss_items": [
            float(value)
            for value in loss_items.detach().float().cpu().reshape(-1)
        ],
        "local_batchnorm_preserved": True,
        "plec_gradient_l1": plec_gradients,
        "injector_gradient_l1": injector_gradients,
        "local_view_autograd_detached": True,
        "local_p3_autograd_detached": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
        / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }
    if report["peak_reserved_gib"] >= 23.0:
        raise RuntimeError(
            "GCMV-EI preflight exceeded the 23 GiB reserved-memory gate"
        )
    model.gcmv_injector.peg.rho.data.zero_()
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_preflight(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

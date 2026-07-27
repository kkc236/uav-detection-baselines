from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics.data.utils import check_det_dataset

from scripts.preflight_gcmv_plec import (
    PLEC_GRADIENT_FAMILIES,
    _dataset,
    _prepare_batch,
    require_nonzero_gradient_families,
    tensor_tree_equal,
)
from src.gcmv_warmstart import (
    PLEC_EXTRA_PREFIXES,
    load_baseline_checkpoint,
    open_residual_scalar,
)
from src.rtdetr_gcmv_plec import GCMVPLECDetectionModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight mature-baseline calibration and fine-tuning."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser


def _gradient_l1(
    model: torch.nn.Module,
    *,
    module_parameters: bool,
) -> float:
    total = 0.0
    found = False
    for name, parameter in model.named_parameters():
        selected = name.startswith(PLEC_EXTRA_PREFIXES)
        if selected != module_parameters:
            continue
        if parameter.grad is not None:
            found = True
            total += float(parameter.grad.detach().float().abs().sum().item())
    if not found or not torch.isfinite(torch.tensor(total)) or total <= 0.0:
        role = "module" if module_parameters else "detector"
        raise RuntimeError(f"{role} gradients are absent or non-finite")
    return total


def run_preflight(args: argparse.Namespace) -> dict:
    if args.batch != 8 or args.workers != 8:
        raise ValueError("warm-start preflight requires batch8/workers8")
    if not torch.cuda.is_available():
        raise RuntimeError("warm-start preflight requires CUDA")
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
    model = GCMVPLECDetectionModel(
        args.model,
        nc=int(data["nc"]),
        ch=int(data["channels"]),
        verbose=False,
    )
    baseline = load_baseline_checkpoint(model, args.baseline)
    model.to(device)

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
        raise RuntimeError("gamma-zero output is not exact baseline output")
    del stock, paired
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    model.calibration_only = True
    model.gcmv_injector.peg.rho.data.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith(PLEC_EXTRA_PREFIXES)
            and name != "gcmv_injector.peg.rho"
        )
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        calibration_loss, _ = model.loss(batch)
    (calibration_loss * 128.0).backward()
    if any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith(PLEC_EXTRA_PREFIXES)
    ):
        raise RuntimeError("calibration leaked gradients into detector")
    if model.gcmv_injector.peg.rho.grad is not None:
        raise RuntimeError("calibration leaked gradients into PEG rho")
    calibration_plec = require_nonzero_gradient_families(
        model.plec,
        prefixes=PLEC_GRADIENT_FAMILIES,
    )
    calibration_heads = require_nonzero_gradient_families(
        model.gcmv_injector,
        prefixes=("gglf.tiny_head", "peg.gate_head"),
    )
    calibration_module_gradient = _gradient_l1(
        model, module_parameters=True
    )

    model.calibration_only = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    open_residual_scalar(model, gamma=0.02)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        method_loss, _ = model.loss(batch)
    (method_loss * 128.0).backward()
    detector_gradient = _gradient_l1(model, module_parameters=False)
    module_gradient = _gradient_l1(model, module_parameters=True)
    rho_gradient = model.gcmv_injector.peg.rho.grad
    if (
        rho_gradient is None
        or not torch.isfinite(rho_gradient).all()
        or float(rho_gradient.detach().abs().item()) <= 0.0
    ):
        raise RuntimeError("method PEG rho gradient is absent or non-finite")

    report = {
        "schema_version": "gcmv-ei-warmstart-preflight/v1",
        "baseline": baseline,
        "batch": int(args.batch),
        "gpu": torch.cuda.get_device_name(device),
        "stock_identity_at_gamma_zero": identity,
        "calibration": {
            "loss": float(calibration_loss.detach().float().item()),
            "gamma": 0.0,
            "detector_gradients_absent": True,
            "rho_gradient_absent": True,
            "module_gradient_l1": calibration_module_gradient,
            "plec_gradient_l1": calibration_plec,
            "head_gradient_l1": calibration_heads,
        },
        "method": {
            "loss": float(method_loss.detach().float().item()),
            "gamma": float(
                torch.tanh(model.gcmv_injector.peg.rho)
                .detach()
                .float()
                .item()
            ),
            "detector_gradient_l1": detector_gradient,
            "module_gradient_l1": module_gradient,
            "rho_gradient_l1": float(
                rho_gradient.detach().float().abs().item()
            ),
        },
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3
        ),
    }
    if report["peak_reserved_gib"] >= 23.0:
        raise RuntimeError("preflight exceeded 23 GiB reserved-memory gate")
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_preflight(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

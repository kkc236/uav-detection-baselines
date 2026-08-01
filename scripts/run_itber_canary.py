"""Run I-TBER v1.1 Gate 0 against immutable detector and data authorities."""

from __future__ import annotations

import argparse
import copy
import io
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.itber_protocol import (  # noqa: E402
    ProtocolViolation,
    assert_detector_frozen,
    module_state_sha256,
    validate_authorities,
    write_immutable_report,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    current_environment,
    dataset_signature,
    file_sha256,
    select_hashed_subset,
    subset_signature,
    ultralytics_source_paths,
)
from src.rtdetr_itber import FrozenITBERAdapter  # noqa: E402


def _loss_tensors(losses: Any) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in vars(losses).items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }


def run_private_step_canary(
    adapter: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    """Prove zero identity, one private update, and detector invariance."""
    assert_detector_frozen(adapter.detector)
    private_parameters = [
        parameter for parameter in adapter.refiner.parameters() if parameter.requires_grad
    ]
    if not private_parameters:
        raise ValueError("I-TBER refiner has no trainable private parameters")
    device = private_parameters[0].device
    amp_enabled = device.type == "cuda" if use_amp is None else bool(use_amp)
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP Gate 0 requires CUDA private parameters")

    initial_output = adapter.forward_evidence(batch["img"])
    zero_identity = bool(
        torch.equal(initial_output.stock_boxes, initial_output.refined_boxes)
    )
    detector_before = module_state_sha256(adapter.detector)
    private_before = module_state_sha256(adapter.refiner)
    optimizer = torch.optim.AdamW(
        private_parameters,
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=128.0,
        growth_interval=2**31 - 1,
    )
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        losses = adapter.training_step(batch)
    named_losses = _loss_tensors(losses)
    finite_private_loss = bool(named_losses) and all(
        bool(torch.isfinite(value.detach()).all()) for value in named_losses.values()
    )
    scaler.scale(losses.total).backward()
    scaler.unscale_(optimizer)
    private_gradient_present = any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in private_parameters
    )
    detector_gradient_absent = all(
        parameter.grad is None for parameter in adapter.detector.parameters()
    )
    torch.nn.utils.clip_grad_norm_(private_parameters, max_norm=10.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    detector_state_unchanged = module_state_sha256(adapter.detector) == detector_before
    private_state_changed = module_state_sha256(adapter.refiner) != private_before
    buffer = io.BytesIO()
    torch.save(adapter.refiner.state_dict(), buffer)
    buffer.seek(0)
    roundtrip = copy.deepcopy(adapter.refiner)
    roundtrip.load_state_dict(
        torch.load(buffer, map_location=device, weights_only=True), strict=True
    )
    checkpoint_roundtrip = module_state_sha256(roundtrip) == module_state_sha256(
        adapter.refiner
    )
    checks = {
        "zero_init_identity": zero_identity,
        "finite_private_loss": finite_private_loss,
        "private_gradient_present": private_gradient_present,
        "detector_gradient_absent": detector_gradient_absent,
        "detector_state_unchanged": detector_state_unchanged,
        "private_state_changed": private_state_changed,
        "checkpoint_roundtrip": checkpoint_roundtrip,
    }
    failed = {
        f"canary.{name}": {"expected": True, "actual": value}
        for name, value in checks.items()
        if not value
    }
    if failed:
        raise ProtocolViolation(failed)
    return {
        "status": "passed",
        "checks": checks,
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in named_losses.items()
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-4,
            "clip_grad_norm": 10.0,
            "amp": amp_enabled,
            "amp_scale": 128.0 if amp_enabled else None,
        },
    }


def _assert_tensor_tree_equal(actual: Any, expected: Any, path: str = "output") -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if not torch.equal(actual, expected):
            raise ProtocolViolation(
                {f"canary.stock_wrapper_equality.{path}": {"expected": True, "actual": False}}
            )
        return
    if isinstance(actual, (tuple, list)) and isinstance(expected, type(actual)):
        if len(actual) != len(expected):
            raise ProtocolViolation(
                {f"canary.stock_wrapper_equality.{path}.length": {"expected": len(expected), "actual": len(actual)}}
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_tensor_tree_equal(actual_item, expected_item, f"{path}.{index}")
        return
    if actual != expected:
        raise ProtocolViolation(
            {f"canary.stock_wrapper_equality.{path}": {"expected": repr(expected), "actual": repr(actual)}}
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report: dict[str, Any] = {"status": "engineering_invalid"}
    try:
        train_images = sorted((args.dataset_root / "images" / "train").glob("**/*"))
        train_images = [path for path in train_images if path.is_file()]
        selected = select_hashed_subset(
            train_images, root=args.dataset_root, fraction=0.10
        )
        source_hashes = {
            name: file_sha256(path) for name, path in ultralytics_source_paths().items()
        }
        authority = validate_authorities(
            baseline_sha256=file_sha256(args.baseline_checkpoint),
            dataset_sha256=str(dataset_signature(args.dataset_root)["sha256"]),
            subset_sha256=subset_signature(selected, root=args.dataset_root),
            category_sha256=category_mapping_sha256(CATEGORY_NAMES),
            source_sha256=source_hashes,
            environment=current_environment(),
        )

        from ultralytics import RTDETR

        device = torch.device(f"cuda:{args.device}")
        detector = RTDETR(str(args.baseline_checkpoint)).model.to(device).eval()
        image = torch.zeros(1, 3, 640, 640, device=device)
        with torch.no_grad():
            stock_output = detector.predict(image)
        adapter = FrozenITBERAdapter.from_detector(
            detector,
            private_seed=10_000,
            probe="p3",
            image_size=640,
            rho=0.05,
        ).to(device)
        with torch.no_grad():
            wrapped_output = detector.predict(image)
        _assert_tensor_tree_equal(wrapped_output, stock_output)
        batch = {
            "img": image,
            "cls": torch.tensor([[0.0]], device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device),
            "batch_idx": torch.tensor([0.0], device=device),
        }
        canary = run_private_step_canary(adapter, batch, use_amp=True)
        report = {
            "status": "passed",
            "authority": authority,
            "stock_wrapper_equality": True,
            "canary": canary,
        }
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        if isinstance(error, ProtocolViolation):
            report["violations"] = error.violations
    write_immutable_report(args.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

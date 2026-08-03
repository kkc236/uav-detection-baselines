"""Run IBER-BE Gate-0 against immutable detector and data authorities."""

from __future__ import annotations

import argparse
import copy
import io
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_iber_stock import (  # noqa: E402
    AMENDED_GATE_STATUS,
    EXPECTED_CATEGORY_SHA256,
    current_execution_environment,
)
from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PRIVATE_OPTIMIZER,
    PRIVATE_SEED,
    RUNTIME_AMENDMENT_SHA256,
    execution_environment,
    file_sha256,
    module_state_sha256,
    write_immutable_report,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    select_hashed_subset,
    subset_signature,
)
from src.rtdetr_iber import FrozenIBERAdapter  # noqa: E402


class CanaryViolation(ValueError):
    """One or more Gate-0 engineering invariants failed."""

    def __init__(self, violations: Mapping[str, Any]) -> None:
        self.violations = dict(violations)
        super().__init__("IBER-BE canary violation: " + ", ".join(sorted(violations)))


def _assert_detector_frozen(detector: torch.nn.Module) -> None:
    violations: dict[str, Any] = {}
    trainable = [name for name, value in detector.named_parameters() if value.requires_grad]
    if trainable:
        violations["detector.requires_grad"] = trainable[:10]
    training = [name for name, module in detector.named_modules() if module.training]
    if training:
        violations["detector.training"] = training[:10]
    if violations:
        raise CanaryViolation(violations)


def _loss_tensors(losses: Any) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in vars(losses).items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }


def _finite_nonzero_gradient(parameters: Sequence[torch.nn.Parameter]) -> bool:
    gradients = [value.grad for value in parameters if value.grad is not None]
    return bool(gradients) and all(
        bool(torch.isfinite(value).all()) for value in gradients
    ) and any(bool(value.abs().max() > 0) for value in gradients)


def _canary_image(device: torch.device, *, size: int = 640) -> torch.Tensor:
    """Return deterministic non-flat RGB evidence for both boundary samplers."""
    if size < 2:
        raise ValueError("canary image size must be at least two pixels")
    x = torch.linspace(0.0, 1.0, size, device=device, dtype=torch.float32).view(
        1, 1, 1, size
    )
    y = torch.linspace(0.0, 1.0, size, device=device, dtype=torch.float32).view(
        1, 1, size, 1
    )
    horizontal = x.expand(1, 1, size, size)
    vertical = y.expand(1, 1, size, size)
    diagonal = 0.5 * (horizontal + vertical)
    return torch.cat((horizontal, vertical, diagonal), dim=1).contiguous()


def _clone_matches(
    matches: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    if matches is None:
        return None
    return [(source.detach().clone(), target.detach().clone()) for source, target in matches]


def _same_matches(
    first: list[tuple[torch.Tensor, torch.Tensor]] | None,
    second: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> bool:
    if first is None:
        return True
    if second is None or len(first) != len(second):
        return False
    return all(
        torch.equal(a_source.cpu(), b_source.cpu())
        and torch.equal(a_target.cpu(), b_target.cpu())
        for (a_source, a_target), (b_source, b_target) in zip(first, second)
    )


def run_private_step_canary(
    adapter: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    """Prove zero identity, both boundary gradients, and detector invariance."""
    _assert_detector_frozen(adapter.detector)
    private_parameters = [
        parameter for parameter in adapter.refiner.parameters() if parameter.requires_grad
    ]
    if not private_parameters:
        raise CanaryViolation({"refiner.parameters": "none_trainable"})
    device = private_parameters[0].device
    amp_enabled = device.type == "cuda" if use_amp is None else bool(use_amp)
    if amp_enabled and device.type != "cuda":
        raise CanaryViolation({"amp.device": device.type})

    initial_output = adapter.forward_evidence(batch["img"])
    zero_identity = bool(
        torch.equal(initial_output.stock_boxes, initial_output.refined_boxes)
    )
    decoder = getattr(getattr(adapter.detector, "model", [None])[-1], "decoder", None)
    normal_query_isolation = True
    if decoder is not None and hasattr(decoder, "normal_query_count"):
        normal_query_isolation = (
            int(decoder.normal_query_count) == 300
            and int(initial_output.stock_boxes.shape[1]) == 300
        )
    detector_before = module_state_sha256(adapter.detector)
    private_before = module_state_sha256(adapter.refiner)
    optimizer = torch.optim.AdamW(
        private_parameters,
        lr=float(PRIVATE_OPTIMIZER["lr"]),
        betas=tuple(PRIVATE_OPTIMIZER["betas"]),
        weight_decay=float(PRIVATE_OPTIMIZER["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=128.0,
        growth_interval=2**31 - 1,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        first_losses = adapter.training_step(batch)
    # Compare the exact AMP training path before and after the private update.
    # A separate FP32 preflight matcher can legitimately choose a different
    # Hungarian assignment near ties and does not test private-head isolation.
    expected_matches = _clone_matches(
        getattr(adapter, "last_match_indices", None)
    )
    scaler.scale(first_losses.total).backward()
    scaler.unscale_(optimizer)
    private_gradient_present = _finite_nonzero_gradient(private_parameters)
    detector_gradient_absent = all(
        parameter.grad is None for parameter in adapter.detector.parameters()
    )
    torch.nn.utils.clip_grad_norm_(
        private_parameters, max_norm=float(PRIVATE_OPTIMIZER["clip"])
    )
    scaler.step(optimizer)
    scaler.update()

    # Zero-initialized output heads block upstream evidence gradients on step 1.
    # Step 2 proves that both enabled B3 evidence paths receive useful gradients.
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        losses = adapter.training_step(batch)
    named_losses = _loss_tensors(losses)
    finite_private_loss = bool(named_losses) and all(
        bool(torch.isfinite(value.detach()).all()) for value in named_losses.values()
    )
    scaler.scale(losses.total).backward()
    scaler.unscale_(optimizer)
    named_parameters = dict(adapter.refiner.named_parameters())
    f3_parameters = [
        value
        for name, value in named_parameters.items()
        if name.startswith(("f3_projection.", "f3_encoder."))
    ]
    rgb_parameters = [
        value for name, value in named_parameters.items() if name.startswith("rgb_encoder.")
    ]
    f3_gradient_present = bool(f3_parameters) and _finite_nonzero_gradient(f3_parameters)
    rgb_gradient_present = bool(rgb_parameters) and _finite_nonzero_gradient(rgb_parameters)
    detector_gradient_absent &= all(
        parameter.grad is None for parameter in adapter.detector.parameters()
    )
    optimizer.zero_grad(set_to_none=True)

    actual_matches = getattr(adapter, "last_match_indices", None)
    matcher_indices_same = _same_matches(expected_matches, actual_matches)
    if actual_matches is not None:
        normal_query_isolation &= all(
            not len(source) or int(source.max()) < 300 for source, _ in actual_matches
        )
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
    updated = adapter.forward_evidence(batch["img"])
    stock_unchanged = torch.equal(initial_output.stock_boxes, updated.stock_boxes)
    checkpoint_mode_switching = stock_unchanged and not torch.equal(
        updated.stock_boxes, updated.refined_boxes
    )

    original_refiner = adapter.refiner
    adapter.refiner = roundtrip
    try:
        restored = adapter.forward_evidence(batch["img"])
    finally:
        adapter.refiner = original_refiner
    checkpoint_roundtrip &= torch.equal(restored.refined_boxes, updated.refined_boxes)

    checks = {
        "zero_init_identity": zero_identity,
        "finite_private_loss": finite_private_loss,
        "private_gradient_present": private_gradient_present,
        "f3_gradient_present": f3_gradient_present,
        "rgb_gradient_present": rgb_gradient_present,
        "detector_gradient_absent": detector_gradient_absent,
        "detector_state_unchanged": detector_state_unchanged,
        "private_state_changed": private_state_changed,
        "normal_query_isolation": normal_query_isolation,
        "matcher_indices_same": matcher_indices_same,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "checkpoint_mode_switching": checkpoint_mode_switching,
    }
    failed = {
        f"canary.{name}": {"expected": True, "actual": value}
        for name, value in checks.items()
        if not value
    }
    if failed:
        raise CanaryViolation(failed)
    return {
        "status": "passed",
        "checks": checks,
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in named_losses.items()
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(PRIVATE_OPTIMIZER["lr"]),
            "betas": list(PRIVATE_OPTIMIZER["betas"]),
            "weight_decay": float(PRIVATE_OPTIMIZER["weight_decay"]),
            "clip_grad_norm": float(PRIVATE_OPTIMIZER["clip"]),
            "amp": amp_enabled,
            "amp_scale": 128.0 if amp_enabled else None,
        },
    }


def _assert_tensor_tree_equal(actual: Any, expected: Any, path: str = "output") -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if not torch.equal(actual, expected):
            raise CanaryViolation({f"stock_wrapper_equality.{path}": False})
        return
    if isinstance(actual, (tuple, list)) and isinstance(expected, type(actual)):
        if len(actual) != len(expected):
            raise CanaryViolation({f"stock_wrapper_equality.{path}.length": False})
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_tensor_tree_equal(actual_item, expected_item, f"{path}.{index}")
        return
    if actual != expected:
        raise CanaryViolation({f"stock_wrapper_equality.{path}": False})


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report: dict[str, Any] = {
        "design_version": DESIGN_VERSION,
        "status": "engineering_invalid",
    }
    try:
        train_images = sorted(
            path
            for path in (args.dataset_root / "images" / "train").glob("**/*")
            if path.is_file()
        )
        selected = select_hashed_subset(
            train_images, root=args.dataset_root, fraction=0.10
        )
        actual_environment = current_execution_environment()
        authority = {
            "baseline_sha256": file_sha256(args.baseline_checkpoint),
            "dataset_sha256": str(dataset_signature(args.dataset_root)["sha256"]),
            "subset_sha256": subset_signature(selected, root=args.dataset_root),
            "category_sha256": category_mapping_sha256(CATEGORY_NAMES),
            "source_commit": _source_commit(),
            "execution_environment": actual_environment,
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        }
        expected = {
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "subset_sha256": EXPECTED_SUBSET_SHA256,
            "category_sha256": EXPECTED_CATEGORY_SHA256,
            "execution_environment": execution_environment(),
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        }
        drift = {
            name: {"expected": value, "actual": authority.get(name)}
            for name, value in expected.items()
            if authority.get(name) != value
        }
        if drift:
            raise CanaryViolation(drift)

        from ultralytics import RTDETR

        device = torch.device(f"cuda:{args.device}")
        detector = RTDETR(str(args.baseline_checkpoint)).model.to(device).eval()
        detector.requires_grad_(False)
        image = _canary_image(device)
        with torch.no_grad():
            stock_output = detector.predict(image)
        with FrozenIBERAdapter.from_detector(
            detector,
            private_seed=PRIVATE_SEED,
            probe="b3",
            image_size=640,
            rho=0.05,
        ).to(device) as adapter:
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
            "design_version": DESIGN_VERSION,
            "status": AMENDED_GATE_STATUS,
            "authority": authority,
            "stock_wrapper_equality": True,
            "canary": canary,
        }
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        if isinstance(error, CanaryViolation):
            report["violations"] = error.violations
    write_immutable_report(args.output, report)
    return 0 if report["status"] == AMENDED_GATE_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())

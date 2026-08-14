"""Static and deterministic gates before the retrospective B-arm Smoke2."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bpdd_loss import BPDDDetectionLoss  # noqa: E402
from src.fdr_bpdd_bridge_protocol import (  # noqa: E402
    BRIDGE_VARIANT,
    load_bridge_authority,
)
from src.fdr_bpdd_ra_glgm_protocol import BPDD_SOURCE_BLOB  # noqa: E402
from src.ra_experiment_protocol import BASELINE_PARAMETERS, file_sha256  # noqa: E402
from src.rtdetr_fdr_bpdd_bridge import FDRBPDDBridgeDetectionModel  # noqa: E402
from src.rtdetr_ra_glgm import (  # noqa: E402
    RAGLGMControlDetectionModel,
    _load_pair_state,
)


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 128, 128, device=device),
        "cls": torch.tensor([[1.0], [-1.0]], device=device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]], device=device
        ),
        "batch_idx": torch.tensor([0.0, 0.0], device=device),
    }


def run_preflight(*, authority_path: Path, device_name: str) -> dict:
    authority = load_bridge_authority(authority_path, repository_root=ROOT)
    blob = subprocess.run(
        ["git", "-C", str(ROOT), "hash-object", "src/bpdd_loss.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != BPDD_SOURCE_BLOB:
        raise ValueError("preflight BPDD source blob differs from D's locked algorithm")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("preflight requested CUDA but CUDA is unavailable")
    baseline = RAGLGMControlDetectionModel(nc=10, verbose=False).to(device)
    bridge = FDRBPDDBridgeDetectionModel(nc=10, verbose=False).to(device)
    initial_path = Path(authority["initial_state"]["path"])
    _load_pair_state(baseline, initial_path, variant="baseline")
    _load_pair_state(bridge, initial_path, variant="baseline")

    parameters = sum(parameter.numel() for parameter in bridge.parameters())
    if parameters != BASELINE_PARAMETERS:
        raise ValueError("preflight B parameter count differs from A")
    if set(baseline.state_dict()) != set(bridge.state_dict()):
        raise ValueError("preflight BPDD changed A model state keys")
    if any(
        not torch.equal(value, bridge.state_dict()[name])
        for name, value in baseline.state_dict().items()
    ):
        raise ValueError("preflight B/A initial tensor bytes differ")
    if any("bpdd" in name.lower() for name in bridge.state_dict()):
        raise ValueError("preflight BPDD unexpectedly added checkpoint state")

    image = torch.rand(1, 3, 128, 128, device=device)
    baseline.eval()
    bridge.eval()
    with torch.no_grad():
        baseline_output = baseline.predict(image)[0]
        bridge_output = bridge.predict(image)[0]
    if not torch.equal(baseline_output, bridge_output):
        raise ValueError("preflight BPDD changed A's evaluation graph")

    bridge.train()
    loss, displayed = bridge.loss(_batch(device))
    loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(displayed).all()):
        raise FloatingPointError("preflight B loss is non-finite")
    if not isinstance(bridge.criterion, BPDDDetectionLoss):
        raise TypeError("preflight B criterion is not BPDD-enabled")
    if (
        bridge.criterion.stock_match_calls != 7
        or bridge.criterion.fgl_extra_match_calls != 0
    ):
        raise ValueError("preflight matcher call contract differs")
    if bridge.criterion.last_normal_decoder_assignment is None:
        raise ValueError("preflight final ordinary assignment is absent")

    bpdd = float(bridge.last_fdr_losses["loss_bpdd"])
    eligible_edges = int(bridge.last_bpdd_statistics["eligible_edges"])
    backbone_gradient = next(
        parameter.grad
        for name, parameter in bridge.named_parameters()
        if name.startswith("model.0.") and parameter.grad is not None
    )
    gradients = {
        "backbone": float(backbone_gradient.detach().float().norm()),
        "fdr_distribution": float(
            bridge.model[-1]
            .dec_bbox_head[0]
            .layers[-1]
            .weight.grad.detach()
            .float()
            .norm()
        ),
    }
    if not math.isfinite(bpdd) or bpdd < 0.0 or eligible_edges <= 0:
        raise ValueError("preflight BPDD graph has no finite eligible-edge signal")
    if any(not math.isfinite(value) or value <= 0.0 for value in gradients.values()):
        raise ValueError("preflight one required gradient is absent")
    return {
        "format_version": 1,
        "status": "passed",
        "variant": BRIDGE_VARIANT,
        "authority_sha256": file_sha256(authority_path),
        "run_id": authority["run_identities"]["smoke"]["run_id"],
        "bpdd_source_blob": blob,
        "model_parameters": parameters,
        "bpdd_parameters": 0,
        "initial_state_bit_exact_to_A": True,
        "eval_bit_exact_to_A": True,
        "stock_match_calls": 7,
        "fgl_extra_match_calls": 0,
        "loss_bpdd": bpdd,
        "bpdd_eligible_edges": eligible_edges,
        "bpdd_initial_dormancy_expected": bpdd == 0.0,
        "gradients": gradients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(
        authority_path=args.authority.resolve(), device_name=args.device
    )
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to replace preflight report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

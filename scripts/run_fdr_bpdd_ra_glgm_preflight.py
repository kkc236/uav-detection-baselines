"""Static and deterministic model gates before combination Smoke2."""

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
from src.fdr_bpdd_ra_glgm_protocol import (  # noqa: E402
    BPDD_SOURCE_BLOB,
    COMBO_PARAMETERS,
    load_combo_authority,
)
from src.ra_experiment_protocol import file_sha256  # noqa: E402
from src.rtdetr_fdr_bpdd_ra_glgm import FDRBPDDRAGLGMDetectionModel  # noqa: E402
from src.rtdetr_ra_glgm import RAGLGMDetectionModel  # noqa: E402


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
    authority = load_combo_authority(authority_path, repository_root=ROOT)
    blob = subprocess.run(
        ["git", "-C", str(ROOT), "hash-object", "src/bpdd_loss.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != BPDD_SOURCE_BLOB:
        raise ValueError("preflight BPDD source blob differs from locked algorithm")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("preflight requested CUDA but CUDA is unavailable")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ra = RAGLGMDetectionModel(nc=10, verbose=False).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        combo = FDRBPDDRAGLGMDetectionModel(nc=10, verbose=False).to(device)
    if sum(parameter.numel() for parameter in combo.parameters()) != COMBO_PARAMETERS:
        raise ValueError("preflight combo parameter count differs")
    if set(ra.state_dict()) != set(combo.state_dict()):
        raise ValueError("preflight BPDD changed model state keys")
    if any(not torch.equal(value, combo.state_dict()[name]) for name, value in ra.state_dict().items()):
        raise ValueError("preflight combo/RA initial tensor bytes differ")
    if not torch.count_nonzero(combo.ra_glgm.alpha).eq(0):
        raise ValueError("preflight RA alpha is not identity-initialized")
    image = torch.rand(1, 3, 128, 128, device=device)
    ra.eval()
    combo.eval()
    with torch.no_grad():
        ra_output = ra.predict(image)[0]
        combo_output = combo.predict(image)[0]
    if not torch.equal(ra_output, combo_output):
        raise ValueError("preflight BPDD changed the evaluation graph")
    combo.train()
    loss, displayed = combo.loss(_batch(device))
    loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(displayed).all()):
        raise FloatingPointError("preflight combo loss is non-finite")
    if not isinstance(combo.criterion, BPDDDetectionLoss):
        raise TypeError("preflight combo criterion is not BPDD-enabled")
    if combo.criterion.stock_match_calls != 7 or combo.criterion.fgl_extra_match_calls != 0:
        raise ValueError("preflight matcher call contract differs")
    if combo.criterion.last_normal_decoder_assignment is None:
        raise ValueError("preflight final ordinary assignment is absent")
    bpdd = float(combo.last_fdr_losses["loss_bpdd"])
    eligible_edges = int(combo.last_bpdd_statistics["eligible_edges"])
    gradients = {
        "ra_alpha": float(combo.ra_glgm.alpha.grad.detach().float().norm()),
        "ra_support": float(combo.ra_glgm.support_head.weight.grad.detach().float().norm()),
        "fdr_distribution": float(
            combo.model[-1].dec_bbox_head[0].layers[-1].weight.grad.detach().float().norm()
        ),
    }
    if not math.isfinite(bpdd) or bpdd < 0.0 or eligible_edges <= 0:
        raise ValueError("preflight BPDD graph has no finite eligible-edge signal")
    if any(not math.isfinite(value) or value <= 0.0 for value in gradients.values()):
        raise ValueError("preflight one required gradient is absent")
    return {
        "format_version": 1,
        "status": "passed",
        "authority_sha256": file_sha256(authority_path),
        "run_id": authority["run_identities"]["smoke"]["run_id"],
        "bpdd_source_blob": blob,
        "model_parameters": COMBO_PARAMETERS,
        "bpdd_parameters": 0,
        "ra_identity_alpha_zero": True,
        "eval_bit_exact_to_ra": True,
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
    result = run_preflight(authority_path=args.authority.resolve(), device_name=args.device)
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

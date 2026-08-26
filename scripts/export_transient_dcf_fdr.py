"""Export an eligible Transient DCF checkpoint with a true Clean FDR shape."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CLEAN_FDR_CONFIG = ROOT / "configs" / "rtdetr-l-clean-fdr.yaml"
sys.path.insert(0, str(ROOT))

from src.transient_dcf_export import (  # noqa: E402
    assert_exact_output_structure,
    detach_distribution_feedback,
    load_eligible_schedule_evidence,
    require_zero_feedback_scale,
    strip_distribution_feedback_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an eligible Transient DCF checkpoint as Clean FDR."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--schedule-evidence", type=Path, required=True)
    parser.add_argument("--paper-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-size", type=int, default=64)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_model(payload: Mapping[str, Any]) -> tuple[str, nn.Module]:
    for key in ("ema", "model"):
        value = payload.get(key)
        if isinstance(value, nn.Module):
            return key, value
    raise ValueError("checkpoint contains neither an EMA nor live model module")


def _verify_fp32_outputs(source: nn.Module, exported: nn.Module, size: int) -> None:
    if size <= 0 or size % 32 != 0:
        raise ValueError("verify size must be a positive multiple of 32")
    source = deepcopy(source).cpu().float().eval()
    exported = deepcopy(exported).cpu().float().eval()
    sample = torch.zeros((1, 3, size, size), dtype=torch.float32)
    with torch.inference_mode():
        before = source(sample)
        after = exported(sample)
    assert_exact_output_structure(before, after)


def export_checkpoint(
    *,
    checkpoint_path: Path,
    schedule_evidence_path: Path,
    paper_epoch: int,
    output_path: Path,
    manifest_path: Path,
    verify_size: int,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("export output or manifest already exists")
    schedule_row = load_eligible_schedule_evidence(
        schedule_evidence_path, paper_epoch
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    model_key, source_model = _checkpoint_model(payload)
    require_zero_feedback_scale(source_model)
    clean_state, removed_state = strip_distribution_feedback_state(
        source_model.state_dict()
    )
    exported_model = deepcopy(source_model)
    removed_parameters = detach_distribution_feedback(exported_model)
    if set(exported_model.state_dict()) != set(clean_state):
        raise RuntimeError("Clean export state differs by more than declared DCF keys")
    _verify_fp32_outputs(source_model, exported_model, verify_size)

    exported_payload = dict(payload)
    exported_payload[model_key] = exported_model
    if isinstance(exported_payload.get("train_args"), Mapping):
        train_args = dict(exported_payload["train_args"])
        train_args["model"] = str(CLEAN_FDR_CONFIG.resolve())
        exported_payload["train_args"] = train_args
    exported_payload["transient_dcf_export"] = {
        "paper_epoch": int(paper_epoch),
        "clean_config": str(CLEAN_FDR_CONFIG.resolve()),
        "removed_state_keys": sorted(removed_state),
        "removed_parameters": removed_parameters,
        "fp32_exact_output_verified": True,
        "verify_size": verify_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported_payload, output_path)
    manifest = {
        "format_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "schedule_evidence": dict(schedule_row),
        "paper_epoch": int(paper_epoch),
        "removed_state_keys": sorted(removed_state),
        "removed_parameters": removed_parameters,
        "fp32_exact_output_verified": True,
        "verify_size": verify_size,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    manifest = export_checkpoint(
        checkpoint_path=args.checkpoint,
        schedule_evidence_path=args.schedule_evidence,
        paper_epoch=args.paper_epoch,
        output_path=args.output,
        manifest_path=manifest_path,
        verify_size=args.verify_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

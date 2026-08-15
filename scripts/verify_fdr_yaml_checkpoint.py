"""Strictly verify checkpoint compatibility with the declarative FDR model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_head import FDRRTDETRDecoder  # noqa: E402
from src.rtdetr_fdr import FDRRTDETRDetectionModel  # noqa: E402


def checkpoint_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _primary_output(prediction: Any) -> torch.Tensor:
    if isinstance(prediction, torch.Tensor):
        return prediction
    if isinstance(prediction, (tuple, list)) and prediction:
        output = prediction[0]
        if isinstance(output, torch.Tensor):
            return output
    raise TypeError("FDR eval inference did not return a tensor output")


def _load_checkpoint_state(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], str]:
    artifact = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(artifact, dict):
        for source_field in ("ema", "model"):
            source_model = artifact.get(source_field)
            if source_model is not None:
                break
        else:
            raise ValueError(
                "checkpoint dict contains neither a non-null 'ema' nor 'model'"
            )
    else:
        source_model = artifact
        source_field = "direct"
    return source_model.float().state_dict(), source_field


def _verify_config(
    *,
    cfg_path: Path,
    source_state: dict[str, Any],
    nc: int,
    imgsz: int,
) -> dict[str, Any]:
    model = FDRRTDETRDetectionModel(
        cfg_path,
        ch=3,
        nc=nc,
        verbose=False,
    )
    head = model.model[-1]
    if not isinstance(head, FDRRTDETRDecoder):
        raise TypeError(
            "declarative FDR model must end with FDRRTDETRDecoder; "
            f"got {type(head).__name__} for {cfg_path}"
        )
    incompatible = model.load_state_dict(source_state, strict=True)

    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32))
    primary_output = _primary_output(prediction)
    finite_output = bool(torch.isfinite(primary_output).all().item())
    if not finite_output:
        raise RuntimeError(
            f"FDR eval inference produced non-finite output for {cfg_path}"
        )

    return {
        "cfg": str(cfg_path),
        "strict_load": True,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
        "finite_output": finite_output,
        "output_shape": list(primary_output.shape),
        "head_type": type(head).__name__,
    }


def verify_checkpoint(
    *,
    cfg: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    nc: int = 10,
    imgsz: int = 128,
) -> dict[str, Any]:
    cfg_path = Path(cfg).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output).resolve()
    source_state, source_field = _load_checkpoint_state(checkpoint_path)
    config_report = _verify_config(
        cfg_path=cfg_path,
        source_state=source_state,
        nc=nc,
        imgsz=imgsz,
    )

    report = {
        **config_report,
        "checkpoint": str(checkpoint_path),
        "sha256": checkpoint_sha256(checkpoint_path),
        "source_field": source_field,
        "state_tensor_count": sum(
            isinstance(value, torch.Tensor) for value in source_state.values()
        ),
    }
    write_json_atomic(output_path, report)
    return report


def verify_checkpoint_configs(
    *,
    cfgs: Sequence[str | Path],
    checkpoint: str | Path,
    output: str | Path,
    nc: int = 10,
    imgsz: int = 128,
) -> dict[str, Any]:
    cfg_paths = [Path(cfg).resolve() for cfg in cfgs]
    if not cfg_paths:
        raise ValueError("at least one declarative FDR config is required")
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output).resolve()
    source_state, source_field = _load_checkpoint_state(checkpoint_path)

    config_reports = [
        _verify_config(
            cfg_path=cfg_path,
            source_state=source_state,
            nc=nc,
            imgsz=imgsz,
        )
        for cfg_path in cfg_paths
    ]
    report = {
        "checkpoint": str(checkpoint_path),
        "sha256": checkpoint_sha256(checkpoint_path),
        "source_field": source_field,
        "strict_load": True,
        "all_configs_verified": True,
        "config_count": len(config_reports),
        "state_tensor_count": sum(
            isinstance(value, torch.Tensor) for value in source_state.values()
        ),
        "configs": config_reports,
    }
    write_json_atomic(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly verify an existing checkpoint against declarative FDR YAML."
    )
    configs = parser.add_mutually_exclusive_group(required=True)
    configs.add_argument(
        "--cfg",
        type=Path,
        help="single declarative FDR YAML (backward-compatible mode)",
    )
    configs.add_argument(
        "--cfgs",
        type=Path,
        nargs="+",
        help="declarative FDR YAMLs to verify against the same checkpoint",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nc", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cfgs is not None:
        report = verify_checkpoint_configs(
            cfgs=args.cfgs,
            checkpoint=args.checkpoint,
            output=args.output,
            nc=args.nc,
            imgsz=args.imgsz,
        )
    else:
        report = verify_checkpoint(
            cfg=args.cfg,
            checkpoint=args.checkpoint,
            output=args.output,
            nc=args.nc,
            imgsz=args.imgsz,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

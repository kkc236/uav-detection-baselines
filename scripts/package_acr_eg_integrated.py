"""Build one checkpoint containing mature RT-DETR and registered ACR-EG."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_gcqf_g0 import MODULE_ARTIFACT_SCHEMA
from src.acr_eg_integration import (
    ACREGIntegratedRTDETR,
    build_integrated_artifact,
    load_acr_eg_config,
)


MATURE_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package mature RT-DETR and ACR-EG as one network checkpoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--module-checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def package(args: argparse.Namespace) -> Path:
    baseline = args.baseline_checkpoint.resolve()
    module_path = args.module_checkpoint.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    for path in (baseline, module_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    baseline_sha = _sha256_file(baseline)
    if baseline_sha != MATURE_BASELINE_SHA256:
        raise ValueError("mature baseline SHA-256 drift")
    module_artifact = torch.load(
        module_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(module_artifact, dict)
        or module_artifact.get("schema_version") != MODULE_ARTIFACT_SCHEMA
        or not isinstance(module_artifact.get("module_state"), dict)
    ):
        raise ValueError("ACR-EG module checkpoint schema drift")

    from ultralytics import RTDETR

    loaded = RTDETR(str(baseline))
    detector = getattr(loaded, "model", None)
    if not isinstance(detector, nn.Module):
        raise RuntimeError("RT-DETR checkpoint did not expose an nn.Module")
    config = load_acr_eg_config(config_path)
    wrapper = ACREGIntegratedRTDETR(detector, config)
    wrapper.acr_eg.load_state_dict(
        module_artifact["module_state"],
        strict=True,
    )
    artifact = build_integrated_artifact(
        wrapper,
        baseline_sha256=baseline_sha,
        module_sha256=_sha256_file(module_path),
        source_commit=args.source_commit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(
        f"ACR_EG_INTEGRATED_CHECKPOINT {output} "
        f"sha256={_sha256_file(output)}",
        flush=True,
    )
    return output


def main() -> None:
    print(package(build_parser().parse_args()))


if __name__ == "__main__":
    main()

"""Build one byte-audited scratch state shared by FDR and FDR+RA-GLGM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ra_experiment_protocol import (  # noqa: E402
    BASELINE_PARAMETERS,
    RA_EXPERIMENT_PROTOCOL_SHA256,
)
from src.ra_glgm_protocol import (  # noqa: E402
    RA_GLGM_PRIVATE_PARAMETERS,
    build_ra_glgm_initial_state,
    validate_ra_glgm_initial_state,
)
from src.rtdetr_ra_glgm import (  # noqa: E402
    RAGLGMControlDetectionModel,
    RAGLGMDetectionModel,
)


def build_initial_state(output: str | Path) -> dict:
    """Create the immutable seed0 artifact without consuming caller RNG."""

    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace initial state: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(0)
        control = RAGLGMControlDetectionModel(nc=10, verbose=False)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(0)
        method = RAGLGMDetectionModel(nc=10, verbose=False)
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    method_parameters = sum(parameter.numel() for parameter in method.parameters())
    if control_parameters != BASELINE_PARAMETERS:
        raise ValueError(
            f"FDR baseline parameter drift: expected={BASELINE_PARAMETERS}, actual={control_parameters}"
        )
    if method_parameters - control_parameters != RA_GLGM_PRIVATE_PARAMETERS:
        raise ValueError("RA private parameter delta differs from frozen authority")
    artifact = build_ra_glgm_initial_state(
        control.state_dict(),
        method.state_dict(),
        metadata={
            "design": "ra-glgm-on-fdr-v1.2-continuous-scale-modulation",
            "seed": 0,
            "initialization": "fresh_scratch",
            "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
            "control_parameters": control_parameters,
            "method_parameters": method_parameters,
        },
    )
    validate_ra_glgm_initial_state(artifact)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale initial-state temporary exists: {temporary}")
    torch.save(artifact, temporary)
    os.replace(temporary, destination)
    if os.name != "nt":
        destination.chmod(0o444)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_initial_state(args.output)
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))
    print(json.dumps(artifact["fingerprints"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Frozen authority for the first integrated GCMV-EI screen."""

from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import platform
from pathlib import Path

import torch

from src.ascv_loc_protocol import state_fingerprint


EXPECTED_DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)
EXPECTED_SUBSET_SHA256 = (
    "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
)
EXPECTED_SUBSET_FILE_SHA256 = (
    "4BDEE4F03CC903422ADBBF4BD3511027628000DB578DEFC07DFE6E45F1E7CB60"
)
EXPECTED_DATA_YAML_SHA256 = (
    "D8CD1CD89012DA4AC31788E40F3B5E48C07E0D171CF55492F1892E1A47CDD0A3"
)
EXPECTED_SUBSET_COUNT = 647
EXPECTED_INITIAL_STATE_SHA256 = {
    0: "C1D93F83EE8BB90CC8A41B313B446E68E91945E53C7CCB597D5434FC3580304A",
    1: "E6C986F53C4FB7076BA52948E959FFBE71F16EE4762E16D4553827C3A46EC465",
    2: "EBA9851A3BAF98DE77228443702F944058C5581A72C00ADA1C452B83BCB598C4",
}
EXPECTED_OPTIMIZER_ATTEMPTS = 145
EXPECTED_ENVIRONMENT = {
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "ultralytics": "8.4.90",
    "cuda": "12.1",
    "gpu": "NVIDIA GeForce RTX 4090",
}


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_plec_initial_state_artifact(
    artifact: dict,
    *,
    seed: int,
) -> None:
    if not isinstance(artifact, dict):
        raise TypeError("GCMV initial-state artifact must be a mapping")
    if artifact.get("metadata", {}).get("seed") != seed:
        raise ValueError("GCMV initial-state seed mismatch")
    common = artifact.get("common_state")
    expected = artifact.get("fingerprints", {}).get("common")
    if not isinstance(common, dict) or state_fingerprint(common) != expected:
        raise ValueError("GCMV initial-state common fingerprint mismatch")


def current_environment() -> dict[str, str]:
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "cuda": str(torch.version.cuda),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "unavailable"
        ),
    }


def validate_runtime_environment() -> dict[str, str]:
    observed = current_environment()
    if observed != EXPECTED_ENVIRONMENT:
        raise ValueError(
            f"GCMV runtime environment drift: {observed}"
        )
    return observed

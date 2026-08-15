from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.prepare_lpr_protocol import create_initial_state_artifact, prepare_data_files, validate_data_authority
from src.lpr_protocol import dataset_signature, load_initial_state, subset_signature
from src.rtdetr_lpr import LPRRTDETRDetectionModel
from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[1]


def _tiny_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "VisDrone"
    for split, count in (("train", 20), ("val", 2)):
        for index in range(count):
            image = root / "images" / split / f"image-{index:03}.jpg"
            label = root / "labels" / split / f"image-{index:03}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"{split}-{index}".encode())
            label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    return root


def test_prepare_data_files_uses_one_locked_hashed_subset(tmp_path) -> None:
    dataset_root = _tiny_dataset(tmp_path)
    output = tmp_path / "protocol"

    result = prepare_data_files(dataset_root, output, fraction=0.10)

    subset_paths = [Path(line) for line in result["subset_path"].read_text(encoding="utf-8").splitlines()]
    screen = yaml.safe_load(result["screen_data_path"].read_text(encoding="utf-8"))
    formal = yaml.safe_load(result["formal_data_path"].read_text(encoding="utf-8"))
    assert len(subset_paths) == 2
    assert result["subset"]["sha256"] == subset_signature(subset_paths, root=dataset_root)
    assert Path(screen["train"]) == result["subset_path"]
    assert Path(formal["train"]) == dataset_root / "images" / "train"
    assert screen["names"][0] == "pedestrian"


def test_data_authority_rejects_wrong_semantic_hash(tmp_path) -> None:
    dataset_root = _tiny_dataset(tmp_path)

    with pytest.raises(ValueError, match="dataset"):
        validate_data_authority(dataset_signature(dataset_root), {"count": 647, "sha256": "BAD"})


def test_created_initial_state_loads_exact_public_and_private_keys() -> None:
    artifact = create_initial_state_artifact(seed=0, nc=10, channels=3)
    control = RTDETRDetectionModel("rtdetr-l.yaml", nc=10, ch=3, verbose=False)
    method = LPRRTDETRDetectionModel("rtdetr-l.yaml", nc=10, ch=3, verbose=False, lpr_seed=10_000)

    load_initial_state(control, artifact, variant="control")
    load_initial_state(method, artifact, variant="lpr")

    assert artifact["metadata"]["seed"] == 0
    assert artifact["lpr_state"]
    for name, value in control.state_dict().items():
        assert value.equal(method.state_dict()[name])


def test_prepare_protocol_script_runs_as_direct_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_lpr_protocol.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset-root" in result.stdout

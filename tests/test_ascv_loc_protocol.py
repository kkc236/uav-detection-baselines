from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.prepare_ascv_loc_protocol import prepare_protocol, sha256_file


def _make_dataset(root: Path, count: int = 21) -> Path:
    train = root / "images" / "train"
    train.mkdir(parents=True)
    for index in range(count):
        (train / f"{index:06d}.jpg").write_bytes(f"image-{index}".encode())
    source_yaml = root / "VisDrone.yaml"
    source_yaml.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "names": {0: "car"}}))
    return source_yaml


def test_protocol_uses_exact_hash_sorted_ceiling_ten_percent(tmp_path: Path) -> None:
    dataset = tmp_path / "VisDrone"
    source_yaml = _make_dataset(dataset)
    checkpoint = tmp_path / "mature.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "protocol"

    first = prepare_protocol(
        checkpoint=checkpoint,
        dataset_root=dataset,
        dataset_yaml=source_yaml,
        output_dir=output,
        repo_root=Path(__file__).resolve().parents[1],
    )
    selected_first = (output / "train_10pct_hash.txt").read_text().splitlines()
    second = prepare_protocol(
        checkpoint=checkpoint,
        dataset_root=dataset,
        dataset_yaml=source_yaml,
        output_dir=output,
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert len(selected_first) == 3
    assert first["subset"]["count"] == 3
    assert first["subset"]["train_list_sha256"] == second["subset"]["train_list_sha256"]
    assert first["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert first["full_train"]["count"] == 21
    assert json.loads((output / "protocol_manifest.json").read_text()) == second


def test_train_only_yaml_cannot_resolve_real_val_or_test_dev(tmp_path: Path) -> None:
    dataset = tmp_path / "VisDrone"
    source_yaml = _make_dataset(dataset, count=10)
    checkpoint = tmp_path / "mature.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "protocol"

    prepare_protocol(
        checkpoint=checkpoint,
        dataset_root=dataset,
        dataset_yaml=source_yaml,
        output_dir=output,
        repo_root=Path(__file__).resolve().parents[1],
    )
    train_only = yaml.safe_load((output / "train_only.yaml").read_text())

    assert train_only["train"] == train_only["val"]
    assert "test" not in train_only
    assert "test-dev" not in (output / "train_only.yaml").read_text().lower()


def test_test_dev_paths_fail_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "test-dev"
    source_yaml = _make_dataset(dataset)
    checkpoint = tmp_path / "mature.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="test-dev path is forbidden"):
        prepare_protocol(
            checkpoint=checkpoint,
            dataset_root=dataset,
            dataset_yaml=source_yaml,
            output_dir=tmp_path / "protocol",
            repo_root=Path(__file__).resolve().parents[1],
        )

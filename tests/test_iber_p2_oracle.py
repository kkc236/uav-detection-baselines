from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from src.iber_p2_oracle import (
    ORACLE_EPOCHS,
    P2OracleCacheViolation,
    P2_NORMAL_OFFSETS_PX,
    P2_TANGENT_FRACTIONS,
    correction_direction_targets,
    decide_p2_viability,
    enable_p2_oracle_determinism,
    load_p2_oracle_cache,
    sample_p2_edge_profiles,
    train_p2_oracles,
    write_p2_oracle_cache,
)


def test_p2_sampling_contract_and_shape_are_frozen() -> None:
    assert P2_NORMAL_OFFSETS_PX == (-12, -8, -4, 0, 4, 8, 12)
    assert P2_TANGENT_FRACTIONS == (0.1, 0.3, 0.5, 0.7, 0.9)
    feature = torch.zeros((1, 3, 160, 160), dtype=torch.float32)
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]], dtype=torch.float32)

    profiles = sample_p2_edge_profiles(feature, boxes)

    assert profiles.shape == (1, 1, 4, 7, 3)
    assert torch.isfinite(profiles).all()


def test_p2_profiles_follow_left_right_top_bottom_normal_coordinates() -> None:
    coordinate = (torch.arange(160, dtype=torch.float32) + 0.5) / 160
    x = coordinate.view(1, 1, 1, 160).expand(1, 1, 160, 160)
    y = coordinate.view(1, 1, 160, 1).expand(1, 1, 160, 160)
    feature = torch.cat((x, y), dim=1)
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]], dtype=torch.float32)

    profiles = sample_p2_edge_profiles(feature, boxes, image_size=640)[0, 0]
    offsets = torch.tensor(P2_NORMAL_OFFSETS_PX, dtype=torch.float32) / 640

    torch.testing.assert_close(profiles[0, :, 0], 0.375 + offsets, atol=1e-6, rtol=0)
    torch.testing.assert_close(profiles[1, :, 0], 0.625 + offsets, atol=1e-6, rtol=0)
    torch.testing.assert_close(profiles[2, :, 1], 0.375 + offsets, atol=1e-6, rtol=0)
    torch.testing.assert_close(profiles[3, :, 1], 0.625 + offsets, atol=1e-6, rtol=0)
    torch.testing.assert_close(profiles[:2, :, 1], torch.full((2, 7), 0.5), atol=1e-6, rtol=0)
    torch.testing.assert_close(profiles[2:, :, 0], torch.full((2, 7), 0.5), atol=1e-6, rtol=0)


def test_p2_sampler_is_finite_for_border_and_empty_boxes() -> None:
    feature = torch.randn((1, 2, 160, 160), generator=torch.Generator().manual_seed(3))
    border = torch.tensor([[[0.0, 0.0, 0.001, 0.001]]], dtype=torch.float32)
    sampled = sample_p2_edge_profiles(feature, border)
    empty = sample_p2_edge_profiles(feature, border[:, :0])

    assert sampled.shape == (1, 1, 4, 7, 2)
    assert torch.isfinite(sampled).all()
    assert empty.shape == (1, 0, 4, 7, 2)


def test_correction_direction_reuses_gate_nonzero_sign_and_target_minus_stock() -> None:
    stock = torch.tensor([[0.2, 0.3, 0.5, 0.7]], dtype=torch.float32)
    target = stock + torch.tensor([[2 / 640, -3 / 640, 0.5 / 640, 4 / 640]])

    labels, valid = correction_direction_targets(stock, target, image_size=640)

    torch.testing.assert_close(labels, torch.tensor([[1.0, 0.0, 1.0, 1.0]]))
    assert valid.tolist() == [[True, True, True, True]]


def _authority() -> dict[str, str]:
    return {
        "baseline_sha256": "A" * 64,
        "dataset_sha256": "B" * 64,
        "subset_sha256": "C" * 64,
        "runtime_amendment_sha256": "D" * 64,
        "source_commit": "e" * 40,
        "schema_sha256": "F" * 64,
    }


def _record(image_id: str, offset: float = 0.0) -> dict[str, object]:
    generator = torch.Generator().manual_seed(int(offset * 1000) + 9)
    count = 4
    profiles = torch.randn((count, 4, 7, 2), generator=generator).half()
    labels = torch.tensor(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]],
        dtype=torch.float32,
    )
    return {
        "image_id": image_id,
        "profiles": profiles,
        "hidden": torch.randn((count, 3), generator=generator).half(),
        "geometry": torch.randn((count, 2), generator=generator).float(),
        "labels": labels,
        "valid": torch.ones((count, 4), dtype=torch.bool),
        "buckets": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }


def test_p2_oracle_cache_is_immutable_hashed_and_authority_bound(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    manifest = write_p2_oracle_cache(
        root,
        train=[_record("images/train/a.jpg")],
        val=[_record("images/val/b.jpg", 0.1)],
        authority=_authority(),
    )
    loaded = load_p2_oracle_cache(root, authority=_authority())

    assert manifest["complete"] is True
    assert manifest["split_counts"] == {"train": 1, "val": 1}
    assert all(len(item["sha256"]) == 64 and item["sha256"].isupper() for item in manifest["artifacts"])
    assert loaded["train"][0]["image_id"] == "images/train/a.jpg"
    torch.testing.assert_close(loaded["val"][0]["profiles"], _record("images/val/b.jpg", 0.1)["profiles"])

    with pytest.raises(FileExistsError):
        write_p2_oracle_cache(root, train=[], val=[], authority=_authority())
    changed = _authority()
    changed["source_commit"] = "0" * 40
    with pytest.raises(P2OracleCacheViolation, match="source_commit"):
        load_p2_oracle_cache(root, authority=changed)


def test_p2_oracle_cache_rejects_cross_split_images_and_corruption(tmp_path: Path) -> None:
    with pytest.raises(P2OracleCacheViolation, match="overlap"):
        write_p2_oracle_cache(
            tmp_path / "overlap",
            train=[_record("same.jpg")],
            val=[_record("same.jpg", 0.1)],
            authority=_authority(),
        )

    root = tmp_path / "corrupt"
    manifest = write_p2_oracle_cache(
        root,
        train=[_record("train.jpg")],
        val=[_record("val.jpg", 0.1)],
        authority=_authority(),
    )
    artifact = root / manifest["artifacts"][0]["path"]
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(P2OracleCacheViolation, match="bytes|sha256"):
        load_p2_oracle_cache(root, authority=_authority())


@pytest.mark.parametrize(
    ("tiny", "small", "status"),
    [
        (0.624866, 0.634066, "passed"),
        (0.624865, 0.900000, "scientific_failed"),
        (0.900000, 0.634065, "scientific_failed"),
    ],
)
def test_p2_viability_uses_exact_frozen_thresholds(tiny: float, small: float, status: str) -> None:
    decision = decide_p2_viability(
        {
            "selection": "final_epoch_only",
            "evaluated_epoch": ORACLE_EPOCHS,
            "context": {
                "validation": {
                    "tiny_direction_accuracy": tiny,
                    "small_direction_accuracy": small,
                }
            },
        }
    )

    assert decision["status"] == status
    assert decision["thresholds"] == {
        "tiny_direction_accuracy": 0.624866,
        "small_direction_accuracy": 0.634066,
    }


def test_p2_oracle_training_is_deterministic_and_final_epoch_only() -> None:
    cache = {
        "train": tuple(_record(f"train-{index}.jpg", index / 100) for index in range(3)),
        "val": tuple(_record(f"val-{index}.jpg", (index + 10) / 100) for index in range(2)),
    }

    first = train_p2_oracles(cache, device=torch.device("cpu"))
    second = train_p2_oracles(cache, device=torch.device("cpu"))

    assert first == second
    assert first["selection"] == "final_epoch_only"
    assert first["evaluated_epoch"] == ORACLE_EPOCHS == 20
    assert len(first["p2_only"]["history"]) == ORACLE_EPOCHS
    assert len(first["context_only"]["history"]) == ORACLE_EPOCHS
    assert len(first["context"]["history"]) == ORACLE_EPOCHS
    assert set(first["majority_baseline"]) >= {
        "tiny_direction_accuracy",
        "small_direction_accuracy",
    }
    assert set(first["context"]["validation"]) >= {
        "tiny_direction_accuracy",
        "small_direction_accuracy",
        "valid_edges",
    }


def test_p2_oracle_enables_cuda_safe_determinism() -> None:
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

        enable_p2_oracle_determinism()

        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.benchmark = previous_benchmark
        if previous_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas


def _load_cli_module():
    path = Path("scripts/run_iber_p2_oracle.py")
    spec = importlib.util.spec_from_file_location("run_iber_p2_oracle_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p2_oracle_cli_exposes_only_paths_and_device() -> None:
    module = _load_cli_module()
    args = module._parse_args(
        [
            "--baseline-checkpoint", "baseline.pt",
            "--dataset-root", "dataset",
            "--cache-root", "cache",
            "--report-root", "report",
        ]
    )

    assert args.baseline_checkpoint == Path("baseline.pt")
    assert args.dataset_root == Path("dataset")
    assert args.cache_root == Path("cache")
    assert args.report_root == Path("report")
    assert args.device == "0"
    assert module.P2_LAYER_INDEX == 1
    for forbidden in ("--epochs", "--seed", "--threshold", "--layer", "--offsets"):
        with pytest.raises(SystemExit):
            module._parse_args(
                [
                    "--baseline-checkpoint", "baseline.pt",
                    "--dataset-root", "dataset",
                    "--cache-root", "cache",
                    "--report-root", "report",
                    forbidden, "3",
                ]
            )


def test_p2_oracle_cli_source_locks_layer_matching_hashes_and_detector_isolation() -> None:
    path = Path("scripts/run_iber_p2_oracle.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source)

    assert "P2_LAYER_INDEX = 1" in source
    assert "detector.model[P2_LAYER_INDEX].register_forward_hook" in source
    assert "with FrozenIBERAdapter.from_detector" in source
    assert ".criterion.matcher(" in source
    assert "sample_p2_edge_profiles(" in source
    assert "EXPECTED_BASELINE_SHA256" in source
    assert "EXPECTED_DATASET_SHA256" in source
    assert "EXPECTED_SUBSET_SHA256" in source
    assert "select_hashed_subset" in source
    assert "parameter.grad is not None" in source
    assert "weights_only=False" not in source
    assert "best_epoch" not in source

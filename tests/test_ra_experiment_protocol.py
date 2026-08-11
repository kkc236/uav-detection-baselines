from __future__ import annotations

import json
from pathlib import Path

import scripts.train_rtdetr_ra_glgm as ra_train
from types import SimpleNamespace

import pytest
import src.ra_experiment_protocol as protocol_module

from src.fdr_protocol import canonical_json_bytes
from src.ra_experiment_protocol import (
    BASELINE_PARAMETERS,
    MAX_PEAK_VRAM_MIB,
    RA_EXPERIMENT_PROTOCOL,
    RA_EXPERIMENT_PROTOCOL_SHA256,
    ignore_sidecar_signature,
    current_source_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_clean_source_identity_rejects_any_porcelain_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protocol_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="?? src/foreign_runtime.py\n"),
    )
    with pytest.raises(ValueError, match="must be clean"):
        current_source_identity(ROOT, require_clean=True)


def test_frozen_ra_protocol_preserves_fdr_line_and_screen_scheduler() -> None:
    protocol = RA_EXPERIMENT_PROTOCOL
    assert protocol["baseline"] == "Ultralytics RT-DETR-L + FDR"
    assert protocol["pairing"] == {
        "single_physical_gpu": True,
        "sequential_arms": True,
        "ddp": False,
        "scratch": True,
        "shared_public_initialization": "byte-identical",
        "private_seed": 20_000,
    }
    assert protocol["training"]["screen_schedule_epochs"] == 50
    assert protocol["training"]["screen_cutoff_epoch"] == 30
    assert protocol["training"]["screen10_cutoff_epoch"] == 10
    assert protocol["training"]["explore50_schedule_epochs"] == 50
    assert protocol["training"]["explore50_cutoff_epoch"] == 50
    assert protocol["training"]["full100_schedule_epochs"] == 100
    assert protocol["training"]["full100_cutoff_epoch"] == 100
    assert protocol["training"]["full100_batch"] == 16
    assert protocol["training"]["full100_nbs"] == 64
    assert protocol["training"]["formal_schedule_epochs"] == 100
    assert protocol["training"]["save_period"] == 1
    assert protocol["dataset"]["screen_train_images"] == 647
    assert protocol["dataset"]["ignore_sidecar"]["files"] == {
        "train": 6471,
        "val": 548,
    }
    assert protocol["dataset"]["ignore_sidecar"]["boxes"] == {
        "train": 10_343,
        "val": 1_410,
    }
    assert protocol["dataset"]["ignore_sidecar"]["raw_score_zero_rows"] == {
        "train": 10_345,
        "val": 1_410,
    }
    assert protocol["dataset"]["ignore_sidecar"][
        "invalid_zero_area_rows_excluded"
    ] == {"train": 2, "val": 0}
    module = protocol["module"]
    assert module["private_parameters"] == 813_396
    assert module["input"] == {
        "source": "FDR decoder P3 only",
        "shape": "[B,256,H,W]",
        "private_branch_input": "x.detach()",
    }
    assert module["router"]["input"] == "shared reduced feature"
    assert module["router"]["initialization"] == "zeros"
    assert module["scale_gate"] == {
        "operator": "1x1 Conv",
        "channels": "192->3",
        "classes": ["tiny", "small", "regular"],
        "thresholds_on_640": [256.0, 1024.0],
        "activation": "per-position softmax",
        "initialization": "zero weight and zero bias",
        "factors": [1.25, 1.0, 0.75],
        "initial_gate": 1.0,
        "inference_inputs": ["fused_private_feature_U"],
        "forbidden_inference_inputs": ["ground_truth", "IoU", "Hungarian_assignment"],
    }
    assert "q_tiny" in module["output_equation"]
    assert module["residual_difficulty"]["prediction_source"] == (
        "final ordinary decoder Query only"
    )
    assert module["residual_difficulty"]["assignment"] == (
        "reuse existing Hungarian assignment; no second matcher"
    )
    assert module["auxiliary_focal"] == {
        "objective": "soft binary focal BCE",
        "alpha": 0.25,
        "gamma": 2.0,
        "reduction": "valid-pixel mean",
        "weight": 0.05,
    }
    assert module["ignore_boxes"]["class_id"] == -1
    assert module["ignore_boxes"]["overlapping_positive_gaussian_pixels"] == "valid"
    assert protocol["module"]["peak_vram_mib_limit"] == MAX_PEAK_VRAM_MIB
    assert protocol["screen_gate"]["epoch30_map_delta_min"] == 0.005
    assert protocol["screen_gate"]["class_ap_wins_min"] == 7
    assert protocol["screen10_gate"]["decision_role"].startswith("rejection-only")
    assert protocol["screen10_gate"]["scale_gate_mean_abs_deviation_tail3_min"] == 0.01
    assert protocol["screen10_gate"]["scale_gate_std_tail3_min"] == 0.01
    assert protocol["dataset"]["selection_set"]["official_val_used"] is False
    assert protocol["dataset"]["screen30_selection_set"]["official_val_used"] is False
    assert protocol["advancement"]["formal_validation"].startswith("official val")
    assert protocol["advancement"]["formal_initialization"].startswith("fresh paired scratch")
    assert protocol["exploration"] == {
        "explore50_role": "post-hoc trajectory evidence only; never confirmatory",
        "fresh_paired_scratch": True,
        "validation": "frozen train-derived selection_set; official val remains isolated",
        "report_every_epochs": 5,
        "no_advancement_gate": True,
    }
    assert protocol["evaluation"]["explore50_evaluated_epochs"] == list(range(5, 51, 5))
    assert protocol["evaluation"]["full100_evaluated_epochs"] == list(range(5, 101, 5))
    assert protocol["full100"]["train_images"] == 6471
    assert protocol["full100"]["validation_images"] == 548
    assert protocol["full100"]["fresh_paired_scratch"] is True
    assert protocol["publication"]["publish_pt"] is False
    assert BASELINE_PARAMETERS == 33_156_614


def test_checked_in_preregistration_is_exact_protocol_payload() -> None:
    path = ROOT / "research" / "ra_glgm" / "RA_GLGM_V11_PREREGISTRATION.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == RA_EXPERIMENT_PROTOCOL
    import hashlib

    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper() == RA_EXPERIMENT_PROTOCOL_SHA256


def test_smoke_and_screen_stages_never_validate_on_official_val(
    tmp_path: Path, monkeypatch,
) -> None:
    dataset = (tmp_path / "VisDrone").resolve()
    dataset.mkdir()
    authority_root = tmp_path / "authority"
    screen10 = tmp_path / "screen10.txt"
    screen30 = tmp_path / "screen30.txt"
    screen10.write_text("screen10.jpg\n", encoding="utf-8")
    screen30.write_text("screen30.jpg\n", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"path": str(dataset), "train": "train", "val": "images/val"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ra_train, "ignore_sidecar_signature", lambda _root: {})
    monkeypatch.setattr(
        ra_train.fdr_train,
        "prepare_data_yaml",
        lambda *_args, **_kwargs: source,
    )
    manifest = {
        "dataset_authority": {
            "root": str(dataset),
            "positive": {"sha256": RA_EXPERIMENT_PROTOCOL["dataset"]["sha256"]},
            "ignore": {},
            "selection_set": {
                "path": str(screen10.resolve()),
                "sha256": ra_train.file_sha256(screen10),
            },
            "screen30_selection_set": {
                "path": str(screen30.resolve()),
                "sha256": ra_train.file_sha256(screen30),
            },
        }
    }

    for stage, expected in (
        ("smoke", screen10),
        ("screen10", screen10),
        ("screen", screen30),
        ("explore50", screen10),
    ):
        generated = ra_train.prepare_data(dataset, stage, authority_root, manifest)
        assert json.loads(generated.read_text(encoding="utf-8"))["val"] == str(
            expected.resolve()
        )


def test_full100_uses_full_data_and_batch16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = (tmp_path / "VisDrone").resolve()
    dataset.mkdir()
    authority_root = tmp_path / "authority"
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {"path": str(dataset), "train": "images/train", "val": "images/val"}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ra_train, "ignore_sidecar_signature", lambda _root: {})
    monkeypatch.setattr(
        ra_train.fdr_train, "prepare_data_yaml", lambda *_args, **_kwargs: source
    )
    manifest = {
        "dataset_authority": {
            "root": str(dataset),
            "positive": {"sha256": RA_EXPERIMENT_PROTOCOL["dataset"]["sha256"]},
            "ignore": {},
        }
    }
    generated = ra_train.prepare_data(
        dataset, "full100", authority_root, manifest
    )
    assert generated == source
    settings = ra_train.build_settings(
        SimpleNamespace(
            variant="baseline",
            stage="full100",
            output_root=tmp_path / "runs",
            name="full100-test",
            resume=None,
        ),
        generated,
    )
    assert settings["epochs"] == 100
    assert settings["batch"] == 16
    assert settings["nbs"] == 64
    assert settings["workers"] == 8


def test_ignore_sidecar_signature_freezes_split_and_empty_file_distribution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "VisDrone"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels_ignore" / split).mkdir(parents=True)
    (root / "images" / "train" / "a.jpg").write_bytes(b"image")
    (root / "images" / "train" / "b.jpg").write_bytes(b"image")
    (root / "images" / "val" / "c.jpg").write_bytes(b"image")
    (root / "labels_ignore" / "train" / "a.txt").write_text(
        "0.5 0.5 0.1 0.1\n", encoding="ascii"
    )
    (root / "labels_ignore" / "train" / "b.txt").write_text("", encoding="ascii")
    (root / "labels_ignore" / "val" / "c.txt").write_text(
        "0.2 0.3 0.2 0.1\n0.7 0.8 0.1 0.1\n", encoding="ascii"
    )

    signature = ignore_sidecar_signature(root)

    assert signature["files"] == 3
    assert signature["boxes"] == 3
    assert signature["nonempty_files"] == 2
    assert signature["empty_files"] == 1
    assert signature["splits"]["train"] == {
        "files": 2,
        "boxes": 1,
        "nonempty_files": 1,
        "empty_files": 1,
    }
    assert len(signature["sha256"]) == 64


def test_ignore_sidecar_signature_rejects_missing_per_image_file(tmp_path: Path) -> None:
    root = tmp_path / "VisDrone"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels_ignore" / split).mkdir(parents=True)
    (root / "images" / "train" / "missing.jpg").write_bytes(b"image")

    import pytest

    with pytest.raises(ValueError, match="sidecar/image mismatch"):
        ignore_sidecar_signature(root)

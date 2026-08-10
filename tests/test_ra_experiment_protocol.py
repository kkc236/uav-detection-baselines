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
    validate_scale_prior_authority,
)
from src.ra_glgm_loss import SCALE_LOG_AREA_KNOTS, SCALE_PRIOR_AUDIT_SHA256


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


def test_frozen_ra_v12_protocol_preserves_fdr_line_and_full_screen_scheduler() -> None:
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
    assert protocol["training"]["explore50_schedule_epochs"] == 50
    assert protocol["training"]["explore50_cutoff_epoch"] == 50
    assert protocol["training"]["formal_schedule_epochs"] == 100
    assert protocol["training"]["save_period"] == 1
    assert protocol["dataset"]["screen_train_images"] == 6471
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
    assert module["private_parameters"] == 813_018
    assert module["input"] == {
        "source": "FDR decoder P3 only",
        "shape": "[B,256,H,W]",
        "private_branch_input": "x.detach()",
    }
    assert module["router"]["input"] == "shared reduced feature"
    assert module["router"]["initialization"] == "zeros"
    assert module["scale_gate"]["channels"] == "192->1"
    assert module["scale_gate"]["activation"] == "per-position sigmoid"
    assert module["scale_gate"]["initial_value"] == 0.5
    assert module["scale_gate"]["groups"] == 8
    assert module["scale_gate"]["forbidden_inference_inputs"] == [
        "ground_truth",
        "IoU",
        "Hungarian_assignment",
    ]
    assert module["output_equation"] == "X + 0.5*tanh(alpha)*O*tanh(Wo(U))"
    assert len(module["scale_prior"]["log_area_knots"]) == 21
    assert "natural-image reference calibration" in module["scale_prior"]["role"]
    assert "post-augmentation" in module["auxiliary_scale"]["target"]
    assert module["scale_prior"]["audit_sha256"] == (
        "598487AD96F59D1E4B01DE8AA026D4C9D90251785BFE9D98016CE8A5785A2454"
    )
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
    assert protocol["screen_gate"]["tail5_map_delta_min"] == 0.002
    assert protocol["screen_gate"]["tail5_ap_tiny_delta_min"] == 0.0015
    assert protocol["screen_gate"]["tail5_ap_small_delta_min"] == 0.0015
    assert protocol["screen_gate"]["class_ap_non_decreasing_min"] == 7
    assert protocol["screen_gate"]["scale_pearson_min"] == 0.40
    assert protocol["screen_gate"]["scale_spearman_min"] == 0.40
    assert protocol["screen_gate"]["scale_slope_rms_min"] == 0.0001
    assert (
        protocol["screen_gate"]["scale_modulation_route_delta_mean_min"]
        == 0.0001
    )
    assert protocol["advancement"]["screen30_requires_screen10_gate"] is False
    assert protocol["dataset"]["selection_set"]["official_val_used"] is False
    assert protocol["dataset"]["screen30"]["official_val_used_for_exploratory_gate"] is True
    assert protocol["advancement"]["screen30_validation"].startswith("official val")
    assert protocol["advancement"]["formal_evidence_status"].startswith(
        "selection-conditioned exploratory"
    )
    assert protocol["advancement"]["formal_initialization"].startswith("fresh paired scratch")
    assert protocol["exploration"] == {
        "explore50_role": "post-hoc trajectory evidence only; never confirmatory",
        "fresh_paired_scratch": True,
        "validation": "frozen train-derived selection_set; official val remains isolated",
        "report_every_epochs": 5,
        "no_advancement_gate": True,
    }
    assert protocol["evaluation"]["explore50_evaluated_epochs"] == list(range(5, 51, 5))
    assert protocol["publication"]["publish_pt"] is False
    assert BASELINE_PARAMETERS == 33_156_614


def test_v12_protocol_hash_is_canonical_and_versioned() -> None:
    import hashlib

    assert RA_EXPERIMENT_PROTOCOL["design"].startswith("ra-glgm-on-fdr-v1.2-")
    assert hashlib.sha256(
        canonical_json_bytes(RA_EXPERIMENT_PROTOCOL)
    ).hexdigest().upper() == RA_EXPERIMENT_PROTOCOL_SHA256


def test_full_precision_scale_prior_is_executable_and_matches_runtime_constants() -> None:
    payload = validate_scale_prior_authority(ROOT)

    assert payload["content_sha256"] == SCALE_PRIOR_AUDIT_SHA256
    assert tuple(payload["log_area_knots"]) == SCALE_LOG_AREA_KNOTS
    assert payload["log_area_knots"] == RA_EXPERIMENT_PROTOCOL["module"][
        "scale_prior"
    ]["log_area_knots"]


def test_screen_uses_official_val_while_smoke_and_explore_use_diagnostic_selection(
    tmp_path: Path, monkeypatch,
) -> None:
    dataset = (tmp_path / "VisDrone").resolve()
    dataset.mkdir()
    authority_root = tmp_path / "authority"
    screen10 = tmp_path / "screen10.txt"
    screen10.write_text("screen10.jpg\n", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"path": str(dataset), "train": "train", "val": "images/val"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ra_train, "ignore_sidecar_signature", lambda _root: {})
    prepared_stages: list[str] = []
    monkeypatch.setattr(
        ra_train.fdr_train,
        "prepare_data_yaml",
        lambda _root, stage, _authority: prepared_stages.append(stage) or source,
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
        }
    }

    for stage, expected in (
        ("smoke", screen10),
        ("screen", "images/val"),
        ("explore50", screen10),
    ):
        generated = ra_train.prepare_data(dataset, stage, authority_root, manifest)
        actual = json.loads(generated.read_text(encoding="utf-8"))["val"]
        if stage == "screen":
            assert actual == expected
        else:
            assert actual == str(expected.resolve())
    assert prepared_stages == ["screen", "formal", "screen"]


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

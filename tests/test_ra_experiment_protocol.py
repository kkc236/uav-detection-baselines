from __future__ import annotations

import json
from pathlib import Path

from src.fdr_protocol import canonical_json_bytes
from src.ra_experiment_protocol import (
    BASELINE_PARAMETERS,
    MAX_PEAK_VRAM_MIB,
    RA_EXPERIMENT_PROTOCOL,
    RA_EXPERIMENT_PROTOCOL_SHA256,
    ignore_sidecar_signature,
)


ROOT = Path(__file__).resolve().parents[1]


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
    assert module["private_parameters"] == 812_817
    assert module["input"] == {
        "source": "FDR decoder P3 only",
        "shape": "[B,256,H,W]",
        "private_branch_input": "x.detach()",
    }
    assert module["router"]["input"] == "shared reduced feature"
    assert module["router"]["initialization"] == "zeros"
    assert module["output_equation"] == "X + 0.5*tanh(alpha)*O*tanh(Wo(U))"
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
    assert protocol["advancement"]["formal_initialization"].startswith("fresh paired scratch")
    assert protocol["publication"]["publish_pt"] is False
    assert BASELINE_PARAMETERS == 33_156_614


def test_checked_in_preregistration_is_exact_protocol_payload() -> None:
    path = ROOT / "research" / "ra_glgm" / "RA_GLGM_PREREGISTRATION.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == RA_EXPERIMENT_PROTOCOL
    import hashlib

    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper() == RA_EXPERIMENT_PROTOCOL_SHA256


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

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from scripts.train_rtdetr_gcmv_plec import (
    build_run_manifest,
    build_parser,
    build_settings,
    trainer_class,
    validate_training_completion,
    validate_screen_inputs,
)
from src.rtdetr_gcmv_plec import (
    GCMVPLECControlTrainer,
    GCMVPLECTrainer,
    load_plec_initial_state,
)
from src.ascv_loc_protocol import state_fingerprint
import src.gcmv_plec_protocol as plec_protocol


def test_training_settings_are_bounded_and_explicit(tmp_path):
    args = build_parser().parse_args(
        [
            "--initial-state",
            "initial-state-seed0.pt",
            "--data",
            "visdrone.yaml",
            "--project",
            str(tmp_path),
            "--name",
            "plec-screen",
            "--source-commit",
            "a" * 40,
        ]
    )

    settings = build_settings(args)

    assert settings["model"].endswith("rtdetr-l-gcmv-plec.yaml")
    assert settings["pretrained"] is False
    assert settings["data"] == "visdrone.yaml"
    assert settings["epochs"] == 10
    assert settings["fraction"] == 1.0
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["imgsz"] == 640
    assert settings["warmup_epochs"] == 3.0
    assert settings["nbs"] == 64
    assert settings["val"] is False
    assert settings["mosaic"] == 1.0
    assert settings["close_mosaic"] == 10
    assert settings["scale"] == 0.5
    assert settings["translate"] == 0.1
    assert settings["fliplr"] == 0.5
    assert settings["mixup"] == 0.0
    assert settings["cutmix"] == 0.0
    assert settings["copy_paste"] == 0.0


def test_control_flag_selects_the_matched_stock_arm():
    method = build_parser().parse_args(
        [
            "--initial-state",
            "initial-state-seed0.pt",
            "--data",
            "visdrone.yaml",
            "--project",
            "runs",
            "--name",
            "method",
            "--source-commit",
            "a" * 40,
        ]
    )
    control = build_parser().parse_args(
        [
            "--initial-state",
            "initial-state-seed0.pt",
            "--data",
            "visdrone.yaml",
            "--project",
            "runs",
            "--name",
            "control",
            "--source-commit",
            "a" * 40,
            "--control",
        ]
    )

    assert trainer_class(method) is GCMVPLECTrainer
    assert trainer_class(control) is GCMVPLECControlTrainer


def test_control_preprocessing_drops_local_inputs_and_normalizes_global():
    trainer = object.__new__(GCMVPLECControlTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.0)
    batch = {
        "img": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "local_views": torch.zeros((1, 4, 3, 2, 2), dtype=torch.uint8),
        "source_shape": torch.tensor([[2, 2]]),
        "source_to_global": torch.eye(3).unsqueeze(0),
        "global_to_source": torch.eye(3).unsqueeze(0),
    }

    processed = trainer.preprocess_batch(batch)

    assert set(processed) == {"img"}
    assert processed["img"].dtype == torch.float32
    assert torch.equal(processed["img"], torch.ones_like(processed["img"]))


class FakePLECModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stock = torch.nn.Linear(2, 2)
        self.plec = torch.nn.Linear(2, 2)
        self.gcmv_injector = torch.nn.Linear(2, 2)


def test_initial_state_loads_only_stock_keys_and_preserves_plec_init():
    model = FakePLECModel()
    before_plec = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.startswith(("plec.", "gcmv_injector."))
    }
    common = {
        name: torch.full_like(value, 3)
        for name, value in model.state_dict().items()
        if name.startswith("stock.")
    }
    artifact = {
        "metadata": {"seed": 0},
        "common_state": common,
        "fingerprints": {"common": state_fingerprint(common)},
    }

    load_plec_initial_state(model, artifact, seed=0)

    for name, value in model.state_dict().items():
        if name.startswith("stock."):
            assert torch.equal(value, common[name])
        else:
            assert torch.equal(value, before_plec[name])


def test_seed0_protocol_constants_match_the_frozen_authority():
    assert plec_protocol.EXPECTED_DATASET_SHA256 == (
        "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
    )
    assert plec_protocol.EXPECTED_SUBSET_SHA256 == (
        "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
    )
    assert plec_protocol.EXPECTED_SUBSET_COUNT == 647
    assert plec_protocol.EXPECTED_INITIAL_STATE_SHA256[0] == (
        "C1D93F83EE8BB90CC8A41B313B446E68E91945E53C7CCB597D5434FC3580304A"
    )
    assert plec_protocol.EXPECTED_OPTIMIZER_ATTEMPTS == 145


def test_initial_state_validator_rejects_seed_drift():
    common = {"weight": torch.ones(1)}
    artifact = {
        "metadata": {"seed": 1},
        "common_state": common,
        "fingerprints": {"common": state_fingerprint(common)},
    }

    with pytest.raises(ValueError, match="seed"):
        plec_protocol.validate_plec_initial_state_artifact(
            artifact,
            seed=0,
        )


def test_screen_input_validation_rejects_wrong_initial_state_hash(tmp_path):
    initial = tmp_path / "initial-state-seed0.pt"
    data = tmp_path / "VisDrone-d2-10pct.yaml"
    initial.write_bytes(b"wrong")
    data.write_text("path: /missing\ntrain: /missing/list.txt\n", encoding="utf-8")
    args = SimpleNamespace(
        initial_state=str(initial),
        data=str(data),
        seed=0,
        device="0",
    )

    with pytest.raises(ValueError, match="initial-state checksum"):
        validate_screen_inputs(args, check_environment=False)


def test_run_manifest_records_protocol_and_completed_runtime():
    args = SimpleNamespace(
        control=False,
        seed=0,
        source_commit="a" * 40,
    )
    trainer = SimpleNamespace(
        plec_optimizer_attempts=plec_protocol.EXPECTED_OPTIMIZER_ATTEMPTS,
        plec_amp_scale_min=128.0,
        plec_amp_scale_max=128.0,
    )
    inputs = {
        "initial_state_sha256": "initial",
        "data_yaml_sha256": "yaml",
        "subset_file_sha256": "list",
        "subset": {"count": 647, "sha256": "semantic"},
        "environment": {"gpu": "RTX 4090"},
    }

    manifest = build_run_manifest(
        args=args,
        inputs=inputs,
        trainer=trainer,
        status="completed",
    )

    assert manifest["schema_version"] == 1
    assert manifest["arm"] == "method"
    assert manifest["status"] == "completed"
    assert manifest["source_commit"] == "a" * 40
    assert manifest["runtime"]["optimizer_attempts"] == 145
    assert manifest["runtime"]["amp_scale_min"] == 128.0
    assert manifest["runtime"]["amp_scale_max"] == 128.0
    assert manifest["protocol"]["subset"]["count"] == 647


def test_training_completion_rejects_optimizer_or_amp_drift():
    trainer = SimpleNamespace(
        plec_optimizer_attempts=144,
        plec_amp_scale_min=128.0,
        plec_amp_scale_max=128.0,
    )
    with pytest.raises(RuntimeError, match="optimizer attempts"):
        validate_training_completion(trainer)

    trainer.plec_optimizer_attempts = 145
    trainer.plec_amp_scale_max = 256.0
    with pytest.raises(RuntimeError, match="AMP scale"):
        validate_training_completion(trainer)

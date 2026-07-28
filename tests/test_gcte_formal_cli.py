from __future__ import annotations

from types import SimpleNamespace

from scripts.train_gcte_formal import build_parser, build_settings


def test_formal_parser_defaults_to_frozen_seed0_protocol() -> None:
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert args.epochs == 100
    assert args.imgsz == 640
    assert args.batch == 8
    assert args.workers == 8
    assert args.device == "0"
    assert args.seed == 0
    assert settings["pretrained"] is False
    assert settings["amp"] is True
    assert settings["deterministic"] is True
    assert settings["optimizer"] == "MuSGD"
    assert settings["lr0"] == 0.01
    assert settings["lrf"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["weight_decay"] == 0.0005
    assert settings["warmup_epochs"] == 3.0
    assert settings["nbs"] == 64
    assert settings["workers"] == 8
    assert settings["max_det"] == 300
    assert settings["nms"] is False
    assert settings["amp_scale"] == 128.0


def test_formal_settings_keep_frozen_augmentation() -> None:
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert settings["mosaic"] == 1.0
    assert settings["close_mosaic"] == 10
    assert settings["mixup"] == 0.0
    assert settings["scale"] == 0.5
    assert settings["translate"] == 0.1
    assert settings["degrees"] == 0.0
    assert settings["shear"] == 0.0
    assert settings["perspective"] == 0.0
    assert settings["flipud"] == 0.0
    assert settings["fliplr"] == 0.5
    assert settings["hsv_h"] == 0.015
    assert settings["hsv_s"] == 0.7
    assert settings["hsv_v"] == 0.4
    assert settings["cutmix"] == 0.0
    assert settings["copy_paste"] == 0.0


def test_formal_entry_uses_new_output_and_explicit_protocol_manifest(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--project",
            str(tmp_path),
            "--name",
            "acr-eg-formal-100",
            "--data",
            "/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml",
        ]
    )
    settings = build_settings(args)

    assert settings["project"] == str(tmp_path.resolve())
    assert settings["name"] == "acr-eg-formal-100"
    assert settings["data"].endswith("source-VisDrone-full.yaml")
    assert settings["exist_ok"] is False
    assert settings["resume"] is False
    assert settings["cache"] is False


def test_formal_entry_accepts_a_downloaded_checkpoint_for_resume(tmp_path) -> None:
    checkpoint = tmp_path / "epoch-002-last.pt"
    args = build_parser().parse_args(["--resume", str(checkpoint)])
    settings = build_settings(args)

    assert settings["resume"] == str(checkpoint.resolve())


def test_formal_entry_accepts_yaml_and_mature_baseline_checkpoint(tmp_path) -> None:
    config = tmp_path / "rtdetr-l-gcte.yaml"
    config.write_text(
        "model: rtdetr-l.yaml\n"
        "gcte:\n"
        "  enabled: true\n"
        "  forward_integration: true\n"
        "  query_dim: 256\n"
        "  num_classes: 10\n"
        "  num_heads: 8\n"
        "  num_views: 4\n"
        "  residual_eta: 0.2\n"
        "  residual_enabled: true\n"
        "  acr_eg_off: false\n"
        "  gcte_off: false\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "matched-baseline-best-epoch-0100.pt"
    args = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--baseline-checkpoint",
            str(baseline),
        ]
    )
    settings = build_settings(args)

    assert settings["gcte_config"] == str(config.resolve())
    assert settings["model"] == str(config.resolve())
    assert settings["baseline_checkpoint"] == str(baseline.resolve())
    assert settings["gcte_forward_integration"] is True
    assert settings["baseline_sha256"] == (
        "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
    )


def test_resume_reapplies_new_project_name_and_frozen_runtime_overrides(
    monkeypatch, tmp_path
) -> None:
    from src.gcte_formal_trainer import GCTEFormalTrainer
    from src.rtdetr_acr_eg import ACREGFormalTrainer

    def fake_check_resume(self, _overrides) -> None:
        self.resume = True
        self.args = SimpleNamespace(
            project="/old/project",
            name="old-name",
            imgsz=320,
            batch=1,
            workers=1,
            device="1",
            close_mosaic=0,
            save_period=10,
            cache=True,
            val=True,
            plots=True,
            epochs=100,
            seed=0,
            deterministic=True,
            optimizer="MuSGD",
        )

    monkeypatch.setattr(GCTEFormalTrainer, "check_resume", fake_check_resume)
    trainer = object.__new__(ACREGFormalTrainer)
    overrides = {
        "project": str(tmp_path / "new-project"),
        "name": "resume-epoch009-to-100",
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "close_mosaic": 10,
        "save_period": 1,
        "cache": False,
        "val": False,
        "plots": False,
    }

    ACREGFormalTrainer.check_resume(trainer, overrides)

    for key, value in overrides.items():
        assert getattr(trainer.args, key) == value

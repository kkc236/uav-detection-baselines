from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path("scripts/train_rtdetr_fdr.py")


def _load_module():
    assert SCRIPT.is_file(), "FDR training CLI has not been implemented"
    spec = importlib.util.spec_from_file_location("train_rtdetr_fdr", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **changes):
    values = {
        "variant": "fdr",
        "stage": "screen",
        "protocol_manifest": tmp_path / "protocol.json",
        "initial_state": tmp_path / "initial-state.pt",
        "dataset_root": tmp_path / "VisDrone",
        "output_root": tmp_path / "runs",
        "resume": None,
        "publication_queue": None,
        "name": None,
        "dry_run": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _manifest(module, state: Path) -> dict:
    digest = hashlib.sha256(state.read_bytes()).hexdigest().upper()
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    identities = {
        f"{variant}_{stage}": module.build_run_identity(
            source, stage=stage, variant=variant, seed=0
        )
        for variant in ("control", "fdr")
        for stage in ("screen", "formal")
    }
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": module.public_state_sha256(source),
        "protocol": module.FDR_PROTOCOL,
        "protocol_sha256": module.FDR_PROTOCOL_SHA256,
        "initial_state": {"path": str(state.resolve()), "sha256": digest},
        "run_identities": identities,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        module.canonical_json_bytes(manifest)
    ).hexdigest().upper()
    return manifest


def test_cli_exposes_only_arm_stage_paths_resume_queue_and_dry_run() -> None:
    assert SCRIPT.is_file(), "FDR training CLI has not been implemented"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for allowed in (
        "--variant",
        "--stage",
        "--protocol-manifest",
        "--initial-state",
        "--dataset-root",
        "--output-root",
        "--resume",
        "--publication-queue",
        "--name",
        "--dry-run",
    ):
        assert allowed in result.stdout
    for forbidden in (
        "--epochs",
        "--seed",
        "--batch",
        "--workers",
        "--imgsz",
        "--device",
        "--lr0",
        "--optimizer",
        "--mosaic",
        "--amp-scale",
    ):
        assert forbidden not in result.stdout


@pytest.mark.parametrize(
    ("stage", "expected_epochs"), (("screen", 50), ("formal", 100))
)
def test_settings_are_frozen_and_screen_keeps_50_epoch_schedule(
    tmp_path: Path, stage: str, expected_epochs: int
) -> None:
    module = _load_module()
    args = _args(tmp_path, variant="control", stage=stage)
    data_yaml = tmp_path / f"{stage}.yaml"
    settings = module.build_settings(args, data_yaml)

    assert settings == {
        "model": "rtdetr-l.yaml",
        "data": str(data_yaml.resolve()),
        "epochs": expected_epochs,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "pretrained": False,
        "cache": False,
        "amp": True,
        "deterministic": True,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "plots": True,
        "val": True,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "seed": 0,
        "project": str(args.output_root.resolve()),
        "name": f"{stage}-seed0-control-fdr-v1",
        "exist_ok": False,
    }


def test_resume_is_the_only_runtime_training_override(tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "run" / "weights" / "last.pt"
    args = _args(tmp_path, resume=checkpoint)
    settings = module.build_settings(args, tmp_path / "data.yaml")
    assert settings["resume"] == str(checkpoint.resolve())
    assert settings["epochs"] == 50
    assert settings["seed"] == 0


def test_initial_state_is_bound_to_manifest_and_strictly_validated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"paired-state")
    manifest = _manifest(module, state)
    seen: list[object] = []
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(module, "validate_fdr_initial_state", seen.append)

    record = module.validate_initial_state_file(state, manifest)
    assert record["sha256"] == hashlib.sha256(b"paired-state").hexdigest().upper()
    assert seen == [{"ok": True}]

    manifest["initial_state"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        module.validate_initial_state_file(state, manifest)


def test_authority_rejects_manifest_or_checked_out_source_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"state")
    manifest = _manifest(module, state)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = module.load_authority(path)
    monkeypatch.setattr(
        module,
        "current_source_identity",
        lambda _root=module.ROOT: dict(manifest["source"]),
    )
    module.validate_source_authority(loaded)

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["source"]["git_commit"] = "c" * 40
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256"):
        module.load_authority(path)

    monkeypatch.setattr(
        module,
        "current_source_identity",
        lambda _root=module.ROOT: {
            **manifest["source"],
            "tree_sha256": "D" * 64,
        },
    )
    with pytest.raises(ValueError, match="checked-out source"):
        module.validate_source_authority(loaded)


def test_arm_factory_uses_paired_trainers_and_shared_initial_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    calls: list[tuple[str, dict]] = []

    class Control:
        def __init__(self, **kwargs):
            calls.append(("control", kwargs))

    class FDR:
        def __init__(self, **kwargs):
            calls.append(("fdr", kwargs))

    monkeypatch.setattr(module, "_load_trainer_types", lambda: (Control, FDR))
    state = tmp_path / "initial.pt"
    settings = {"epochs": 50}

    module.create_trainer("control", settings, state)
    module.create_trainer("fdr", settings, state)

    assert calls == [
        (
            "control",
            {"overrides": settings, "initial_state_path": state.resolve()},
        ),
        (
            "fdr",
            {
                "overrides": settings,
                "initial_state_path": state.resolve(),
                "experiment_seed": 0,
            },
        ),
    ]


def test_resume_requires_matching_frozen_run_identity(tmp_path: Path) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"state")
    manifest = _manifest(module, state)
    expected = manifest["run_identities"]["fdr_screen"]
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run / "fdr-run.json").write_text(
        json.dumps({"run_identity": expected}), encoding="utf-8"
    )

    assert module.validate_resume_checkpoint(checkpoint, expected) == checkpoint.resolve()

    changed = {**expected, "variant": "control"}
    (run / "fdr-run.json").write_text(
        json.dumps({"run_identity": changed}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resume authority"):
        module.validate_resume_checkpoint(checkpoint, expected)


def _fake_trainer(run: Path, *, epoch: int = 0):
    model = SimpleNamespace(
        last_fdr_losses={
            "loss_fgl": torch.tensor(0.15),
            "loss_fgl_aux": torch.tensor(0.05),
            "loss_bbox_pre": torch.tensor(0.25),
            "loss_giou_pre": torch.tensor(0.35),
        }
    )
    return SimpleNamespace(
        epoch=epoch,
        save_dir=run,
        args=SimpleNamespace(epochs=50),
        model=model,
        tloss=torch.tensor([1.0, 2.0, 3.0]),
        metrics={
            "metrics/precision(B)": 0.10,
            "metrics/recall(B)": 0.20,
            "metrics/mAP50(B)": 0.30,
            "metrics/mAP50-95(B)": 0.40,
        },
        validator=SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(map75=0.25))),
        last_gradient_norms={"gradient_norm": 4.0, "fdr_gradient_norm": 5.0},
        stop=False,
    )


def test_each_epoch_writes_idempotent_jsonl_and_csv_evidence(tmp_path: Path) -> None:
    module = _load_module()
    run = tmp_path / "run"
    trainer = _fake_trainer(run)
    context = {
        "variant": "fdr",
        "stage": "screen",
        "run_identity": {"run_id": "fdr-screen-seed0"},
    }

    first = module.write_epoch_evidence(trainer, context)
    second = module.write_epoch_evidence(trainer, context)

    assert first == second
    rows = [
        json.loads(line)
        for line in (run / "fdr-epochs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["completed_epoch"] == 1
    assert rows[0]["map"] == pytest.approx(0.40)
    assert rows[0]["map75"] == pytest.approx(0.25)
    assert rows[0]["loss_fgl"] == pytest.approx(0.15)
    assert rows[0]["fdr_gradient_norm"] == pytest.approx(5.0)
    with (run / "fdr-epochs.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 1
    assert csv_rows[0]["completed_epoch"] == "1"
    assert float(csv_rows[0]["map"]) == pytest.approx(0.40)


def test_control_epoch_has_null_fdr_fields(tmp_path: Path) -> None:
    module = _load_module()
    trainer = _fake_trainer(tmp_path / "control")
    record = module.write_epoch_evidence(
        trainer,
        {
            "variant": "control",
            "stage": "screen",
            "run_identity": {"run_id": "control-screen-seed0"},
        },
    )
    assert record["loss_fgl"] is None
    assert record["fdr_gradient_norm"] is None


def test_epoch_publication_queue_is_local_idempotent_and_resume_safe(tmp_path: Path) -> None:
    module = _load_module()
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "epoch0.pt"
    checkpoint.write_bytes(b"epoch-one")
    trainer = _fake_trainer(run)
    context = {
        "variant": "fdr",
        "stage": "screen",
        "run_identity": {"run_id": "fdr-screen-seed0"},
        "publication_queue": tmp_path / "publication-queue.jsonl",
    }
    module.write_epoch_evidence(trainer, context)

    first = module.queue_epoch_publication(trainer, context)
    second = module.queue_epoch_publication(trainer, context)

    assert first == second
    queue = context["publication_queue"]
    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["completed_epoch"] == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["checkpoint"] == str(checkpoint.resolve())
    assert rows[0]["checkpoint_sha256"] == hashlib.sha256(b"epoch-one").hexdigest().upper()
    assert str((run / "fdr-epochs.jsonl").resolve()) in rows[0]["artifacts"]
    assert str((run / "fdr-epochs.csv").resolve()) in rows[0]["artifacts"]


def test_screen_stops_at_protocol_cutoff_only_after_evidence_is_queued(tmp_path: Path) -> None:
    module = _load_module()
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    context = {
        "variant": "fdr",
        "stage": "screen",
        "run_identity": {"run_id": "fdr-screen-seed0"},
        "publication_queue": tmp_path / "queue.jsonl",
    }
    for epoch in range(29):
        module.write_epoch_evidence(_fake_trainer(run, epoch=epoch), context)
    (weights / "epoch29.pt").write_bytes(b"epoch-thirty")
    trainer = _fake_trainer(run, epoch=29)

    module.finalize_epoch(trainer, context)

    assert trainer.stop is True
    rows = [json.loads(line) for line in context["publication_queue"].read_text().splitlines()]
    assert rows[-1]["completed_epoch"] == 30


def test_dry_run_validates_without_constructing_a_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"state")
    manifest = _manifest(module, state)
    args = _args(tmp_path, dry_run=True)
    args.protocol_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    args.dataset_root.mkdir()
    data_yaml = tmp_path / "screen.yaml"
    data_yaml.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "validate_initial_state_file", lambda *args: manifest["initial_state"])
    monkeypatch.setattr(module, "validate_source_authority", lambda *args: manifest["source"])
    monkeypatch.setattr(module, "prepare_data_yaml", lambda *args: data_yaml)
    monkeypatch.setattr(
        module,
        "create_trainer",
        lambda *args, **kwargs: pytest.fail("dry-run must not construct trainer"),
    )

    result = module.execute(args)

    assert result["status"] == "dry-run-passed"
    assert result["settings"]["epochs"] == 50
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run-passed"

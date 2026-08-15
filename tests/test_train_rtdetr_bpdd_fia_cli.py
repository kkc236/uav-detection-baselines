from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_rtdetr_fdr as fdr_cli
from src.state_hash import state_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_rtdetr_bpdd_fia.py"
EXPECTED_INITIAL_STATE_SHA256 = (
    "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"
)


def _load_module():
    assert SCRIPT.is_file(), "BPDD FIA training CLI has not been implemented"
    spec = importlib.util.spec_from_file_location("train_rtdetr_bpdd_fia", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **changes):
    values = {
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


def _source() -> dict[str, str]:
    return {"git_commit": "a" * 40, "tree_sha256": "A" * 64}


def _rehash_manifest(module, manifest: dict) -> dict:
    unhashed = deepcopy(manifest)
    unhashed.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        module.canonical_json_bytes(unhashed)
    ).hexdigest().upper()
    return manifest


def _manifest(module, state: Path) -> dict:
    source = _source()
    identity = module.build_run_identity(
        source, stage="formal", variant="fdr_bpdd_fia", seed=0
    )
    return _rehash_manifest(
        module,
        {
            "format_version": 1,
            "source": source,
            "source_sha256": module.public_state_sha256(source),
            "protocol": module.BPDD_FIA_PROTOCOL,
            "protocol_sha256": module.BPDD_FIA_PROTOCOL_SHA256,
            "initial_state": {
                "path": str(state.resolve()),
                "sha256": EXPECTED_INITIAL_STATE_SHA256,
            },
            "run_identities": {"fdr_bpdd_fia_formal": identity},
        },
    )


def _fake_trainer(run: Path, *, epoch: int = 0):
    fia = SimpleNamespace(residual_scale=torch.nn.Parameter(torch.tensor(0.125)))
    model = SimpleNamespace(
        fia=fia,
        last_fdr_losses={
            "loss_fgl": torch.tensor(0.15),
            "loss_fgl_aux": torch.tensor(0.05),
            "loss_bbox_pre": torch.tensor(0.25),
            "loss_giou_pre": torch.tensor(0.35),
            "loss_bpdd": torch.tensor(0.45),
        },
        last_bpdd_statistics={
            "active_edge_ratio": torch.tensor(0.25),
            "mean_reliability": torch.tensor(0.75),
            "mean_teacher_improvement": torch.tensor(0.125),
            "mixture_beats_final_ratio": torch.tensor(0.50),
            "mean_mixture_advantage_over_final": torch.tensor(0.025),
        },
    )
    ema_model = torch.nn.Linear(2, 2)
    return SimpleNamespace(
        epoch=epoch,
        save_dir=run,
        args=SimpleNamespace(epochs=100),
        model=model,
        ema=SimpleNamespace(ema=ema_model),
        tloss=torch.tensor([1.0, 2.0, 3.0]),
        metrics={
            "metrics/precision(B)": 0.10,
            "metrics/recall(B)": 0.20,
            "metrics/mAP50(B)": 0.30,
            "metrics/mAP50-95(B)": 0.40,
        },
        validator=SimpleNamespace(
            metrics=SimpleNamespace(box=SimpleNamespace(map75=0.25))
        ),
        last_gradient_norms={
            "gradient_norm": 4.0,
            "fdr_gradient_norm": 5.0,
            "fia_gradient_norm": 6.0,
        },
    )


def test_cli_is_formal_only_and_exposes_no_scientific_knobs() -> None:
    assert SCRIPT.is_file(), "BPDD FIA training CLI has not been implemented"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for allowed in (
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
        "--variant",
        "--stage",
        "--epochs",
        "--seed",
        "--batch",
        "--workers",
        "--imgsz",
        "--optimizer",
        "--bpdd-weight",
        "--fia-seed",
    ):
        assert forbidden not in result.stdout


def test_formal100_settings_are_exact_frozen_fdr_settings(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    data_yaml = tmp_path / "formal.yaml"
    settings = module.build_settings(args, data_yaml)

    expected = {
        **fdr_cli.FROZEN_SETTINGS,
        "model": str(
            (ROOT / "configs" / "rtdetr-l-fdr-bpdd-fia.yaml").resolve()
        ),
        "data": str(data_yaml.resolve()),
        "epochs": 100,
        "seed": 0,
        "project": str(args.output_root.resolve()),
        "name": "formal-seed0-fdr_bpdd_fia-v1",
        "exist_ok": False,
    }
    assert settings == expected


def test_trainer_factory_uses_combined_trainer_and_frozen_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    calls: list[dict] = []

    class Trainer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "_load_trainer_type", lambda: Trainer)
    state = tmp_path / "initial-state.pt"
    settings = {"epochs": 100}

    module.create_trainer(settings, state)
    assert calls == [
        {
            "overrides": settings,
            "initial_state_path": state.resolve(),
            "experiment_seed": 0,
        }
    ]


def test_manifest_binds_only_combined_formal_identity(tmp_path: Path) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    manifest = _manifest(module, state)
    path = tmp_path / "protocol.json"
    path.write_bytes(module.canonical_json_bytes(manifest) + b"\n")

    loaded = module.load_authority(path)
    assert set(loaded["run_identities"]) == {"fdr_bpdd_fia_formal"}
    assert loaded["initial_state"]["sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert loaded["protocol"] == module.BPDD_FIA_PROTOCOL

    changed = deepcopy(manifest)
    changed["run_identities"]["extra"] = changed["run_identities"][
        "fdr_bpdd_fia_formal"
    ]
    _rehash_manifest(module, changed)
    path.write_bytes(module.canonical_json_bytes(changed) + b"\n")
    with pytest.raises(ValueError, match="run identities"):
        module.load_authority(path)


def test_epoch_evidence_is_create_only_and_contains_all_module_signals(
    tmp_path: Path,
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    trainer = _fake_trainer(run)
    weights = run / "weights"
    weights.mkdir(parents=True)
    torch.save({"epoch": 0, "ema": torch.nn.Linear(2, 2).half()}, weights / "epoch0.pt")
    context = {
        "run_identity": {"run_id": "combined-formal-seed0"},
        "publication_queue": None,
    }

    first = module.write_epoch_evidence(trainer, context)
    repeated = module.write_epoch_evidence(trainer, context)
    assert repeated == first
    assert first["variant"] == "fdr_bpdd_fia"
    assert first["stage"] == "formal"
    assert first["loss_bpdd"] == pytest.approx(0.45)
    assert first["bpdd_active_edge_ratio"] == pytest.approx(0.25)
    assert first["bpdd_mean_reliability"] == pytest.approx(0.75)
    assert first["fia_gradient_norm"] == pytest.approx(6.0)
    assert first["fia_residual_scale"] == pytest.approx(0.125)
    assert first["gradients_finite"] is True

    trainer.model.fia.residual_scale.data.fill_(0.25)
    with pytest.raises(ValueError, match="changed BPDD FIA evidence"):
        module.write_epoch_evidence(trainer, context)

    with (run / "fdr-epochs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["fia_residual_scale"]) == pytest.approx(0.125)


def test_finalize_epoch_records_checkpoint_and_ema_hashes_and_queues_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "epoch0.pt"
    saved_ema = torch.nn.Linear(2, 2).half()
    torch.save({"epoch": 0, "ema": saved_ema}, checkpoint)
    (run / "optimizer-evidence.jsonl").write_text(
        '{"optimizer_attempt":1,"amp_scale_before":128.0,"amp_scale_after":128.0}\n',
        encoding="utf-8",
    )
    (run / "bpdd-fia-run.json").write_text("{}\n", encoding="utf-8")
    trainer = _fake_trainer(run)
    context = {
        "run_identity": {"run_id": "combined-formal-seed0"},
        "publication_queue": tmp_path / "queue.jsonl",
    }

    result = module.finalize_epoch(trainer, context)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    ema_sha = state_sha256(saved_ema.state_dict())
    assert result["evidence"]["checkpoint_sha256"] == checkpoint_sha
    assert result["evidence"]["ema_state_sha256"] == ema_sha
    assert result["publication"]["checkpoint_sha256"] == checkpoint_sha
    assert result["publication"]["ema_state_sha256"] == ema_sha
    assert {Path(path).name for path in result["publication"]["artifacts"]} == {
        "fdr-epochs.jsonl",
        "fdr-epochs.csv",
        "bpdd-fia-run.json",
        "optimizer-evidence.jsonl",
    }


def test_resume_requires_exact_identity_and_registered_checkpoint_hash(
    tmp_path: Path,
) -> None:
    module = _load_module()
    identity = module.build_run_identity(
        _source(), stage="formal", variant="fdr_bpdd_fia", seed=0
    )
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "epoch8.pt"
    checkpoint.write_bytes(b"epoch-nine")
    checkpoint_sha = hashlib.sha256(b"epoch-nine").hexdigest().upper()
    queue = run / "publication-queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "run_id": identity["run_id"],
                "completed_epoch": 9,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "bpdd-fia-run.json").write_text(
        json.dumps(
            {
                "run_identity": identity,
                "publication_queue": str(queue.resolve()),
            }
        ),
        encoding="utf-8",
    )

    assert module.validate_resume_checkpoint(checkpoint, identity) == checkpoint.resolve()
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        module.validate_resume_checkpoint(checkpoint, identity)

    checkpoint.write_bytes(b"epoch-nine")
    wrong = {**identity, "run_id": "wrong-run"}
    with pytest.raises(ValueError, match="run_id"):
        module.validate_resume_checkpoint(checkpoint, wrong)


def test_dry_run_validates_without_constructing_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"state")
    manifest = _manifest(module, state)
    args = _args(tmp_path, dry_run=True)
    args.protocol_manifest.write_bytes(module.canonical_json_bytes(manifest) + b"\n")
    args.dataset_root.mkdir()
    data_yaml = tmp_path / "formal.yaml"
    data_yaml.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "validate_source_authority", lambda *args: manifest["source"])
    monkeypatch.setattr(module, "validate_initial_state_file", lambda *args: manifest["initial_state"])
    monkeypatch.setattr(module, "prepare_data_yaml", lambda *args: data_yaml)
    monkeypatch.setattr(
        module,
        "create_trainer",
        lambda *args, **kwargs: pytest.fail("dry-run must not construct trainer"),
    )

    result = module.execute(args)
    assert result["status"] == "dry-run-passed"
    assert result["settings"]["epochs"] == 100
    assert result["variant"] == "fdr_bpdd_fia"
    assert json.loads(capsys.readouterr().out)["stage"] == "formal"

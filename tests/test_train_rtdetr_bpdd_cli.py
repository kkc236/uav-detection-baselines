from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_rtdetr_fdr as fdr_cli


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_rtdetr_bpdd.py"
EXPECTED_FDR_PROTOCOL_SHA256 = (
    "2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302"
)
EXPECTED_FDR_SOURCE_COMMIT = "d97e1eb7f98414752a1c1f38287697db3f2a0679"
EXPECTED_INITIAL_STATE_SHA256 = (
    "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D"
)


def _load_module():
    assert SCRIPT.is_file(), "BPDD training CLI has not been implemented"
    spec = importlib.util.spec_from_file_location("train_rtdetr_bpdd", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **changes):
    values = {
        "variant": "fdr_bpdd",
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


def _source() -> dict[str, str]:
    return {
        "git_commit": "a" * 40,
        "tree_sha256": "A" * 64,
    }


def _manifest(module, state: Path) -> dict:
    source = _source()
    identities = {
        f"{variant}_{stage}": module.build_run_identity(
            source, stage=stage, variant=variant, seed=0
        )
        for variant in ("fdr", "fdr_bpdd")
        for stage in ("screen", "formal")
    }
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": module.public_state_sha256(source),
        "protocol": module.BPDD_PROTOCOL,
        "protocol_sha256": module.BPDD_PROTOCOL_SHA256,
        "initial_state": {
            "path": str(state.resolve()),
            "sha256": EXPECTED_INITIAL_STATE_SHA256,
        },
        "run_identities": identities,
    }
    return _rehash_manifest(module, manifest)


def _rehash_manifest(module, manifest: dict) -> dict:
    unhashed = deepcopy(manifest)
    unhashed.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        module.canonical_json_bytes(unhashed)
    ).hexdigest().upper()
    return manifest


def _write_manifest(module, path: Path, manifest: dict) -> None:
    path.write_bytes(module.canonical_json_bytes(manifest) + b"\n")


def test_cli_exposes_only_frozen_arm_stage_authority_and_runtime_paths() -> None:
    assert SCRIPT.is_file(), "BPDD training CLI has not been implemented"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
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
    for arm in ("fdr", "fdr_bpdd"):
        assert arm in result.stdout
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
        "--bpdd-weight",
        "--bpdd-temperature",
        "--bpdd-margin",
        "--bpdd-eps",
    ):
        assert forbidden not in result.stdout


@pytest.mark.parametrize(
    ("stage", "expected_epochs", "expected_cutoff"),
    [("screen", 50, 30), ("formal", 100, None)],
)
def test_settings_match_fdr_and_arms_differ_only_by_model_name_or_variant(
    tmp_path: Path,
    stage: str,
    expected_epochs: int,
    expected_cutoff: int | None,
) -> None:
    module = _load_module()
    data_yaml = tmp_path / f"{stage}.yaml"
    fdr = module.build_settings(
        _args(tmp_path, variant="fdr", stage=stage), data_yaml
    )
    bpdd = module.build_settings(
        _args(tmp_path, variant="fdr_bpdd", stage=stage), data_yaml
    )
    reference = fdr_cli.build_settings(
        _args(tmp_path, variant="fdr", stage=stage), data_yaml
    )

    ignored = {"model", "name", "variant"}
    for actual in (fdr, bpdd):
        assert {key: value for key, value in actual.items() if key not in ignored} == {
            key: value for key, value in reference.items() if key not in ignored
        }
    differing = {
        key
        for key in fdr.keys() | bpdd.keys()
        if fdr.get(key) != bpdd.get(key)
    }
    assert {"model", "name"}.issubset(differing)
    assert differing <= ignored
    assert Path(fdr["model"]) == (
        module.ROOT / "configs" / "rtdetr-l-fdr.yaml"
    ).resolve()
    assert Path(bpdd["model"]) == (
        module.ROOT / "configs" / "rtdetr-l-fdr-bpdd.yaml"
    ).resolve()
    assert fdr["epochs"] == bpdd["epochs"] == expected_epochs
    assert fdr["seed"] == bpdd["seed"] == 0
    assert module.SCREEN_CUTOFF_EPOCH == 30
    assert module.FORMAL_EPOCHS == 100
    assert (
        module.SCREEN_CUTOFF_EPOCH if stage == "screen" else None
    ) == expected_cutoff


def test_cli_rejects_any_arm_other_than_fdr_or_fdr_bpdd(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="variant"):
        module.build_settings(_args(tmp_path, variant="control"), tmp_path / "data.yaml")


def test_manifest_binds_exact_source_hash_fdr_hash_and_initial_state(
    tmp_path: Path,
) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    manifest = _manifest(module, state)
    path = tmp_path / "protocol.json"
    _write_manifest(module, path, manifest)

    loaded = module.load_authority(path)

    assert loaded["source"]["git_commit"] == "a" * 40
    assert loaded["source_sha256"] == module.public_state_sha256(loaded["source"])
    assert loaded["protocol"]["fdr_authority"]["protocol_sha256"] == (
        EXPECTED_FDR_PROTOCOL_SHA256
    )
    assert loaded["initial_state"]["sha256"] == EXPECTED_INITIAL_STATE_SHA256

    bad_source_hash = deepcopy(manifest)
    bad_source_hash["source_sha256"] = "B" * 64
    _rehash_manifest(module, bad_source_hash)
    _write_manifest(module, path, bad_source_hash)
    with pytest.raises(ValueError, match="source SHA256"):
        module.load_authority(path)

    bad_state = deepcopy(manifest)
    bad_state["initial_state"]["sha256"] = "C" * 64
    _rehash_manifest(module, bad_state)
    _write_manifest(module, path, bad_state)
    with pytest.raises(ValueError, match="initial-state SHA256"):
        module.load_authority(path)


def test_checked_out_source_must_match_the_manifest_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    manifest = _manifest(module, tmp_path / "initial-state.pt")
    monkeypatch.setattr(module, "current_source_identity", lambda _root=module.ROOT: _source())

    assert module.validate_source_authority(manifest) == _source()

    monkeypatch.setattr(
        module,
        "current_source_identity",
        lambda _root=module.ROOT: {**_source(), "tree_sha256": "D" * 64},
    )
    with pytest.raises(ValueError, match="checked-out source"):
        module.validate_source_authority(manifest)


@pytest.mark.parametrize(
    ("actual_stage", "actual_variant", "mismatch"),
    [
        ("screen", "fdr", "variant"),
        ("formal", "fdr_bpdd", "stage"),
    ],
)
def test_resume_checkpoint_rejects_cross_variant_and_cross_stage(
    tmp_path: Path,
    actual_stage: str,
    actual_variant: str,
    mismatch: str,
) -> None:
    module = _load_module()
    expected = module.build_run_identity(
        _source(), stage="screen", variant="fdr_bpdd", seed=0
    )
    actual = module.build_run_identity(
        _source(), stage=actual_stage, variant=actual_variant, seed=0
    )
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run / "bpdd-run.json").write_text(
        json.dumps({"run_identity": actual}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=mismatch):
        module.validate_resume_checkpoint(checkpoint, expected)


def _fake_trainer(run: Path, *, variant: str, epoch: int = 0):
    bpdd_enabled = variant == "fdr_bpdd"
    model = SimpleNamespace(
        last_fdr_losses={
            "loss_fgl": torch.tensor(0.15),
            "loss_fgl_aux": torch.tensor(0.05),
            "loss_bbox_pre": torch.tensor(0.25),
            "loss_giou_pre": torch.tensor(0.35),
            **({"loss_bpdd": torch.tensor(0.45)} if bpdd_enabled else {}),
        },
        last_bpdd_statistics=(
            {
                "active_edge_ratio": torch.tensor(0.25),
                "mean_reliability": torch.tensor(0.75),
                "mean_teacher_improvement": torch.tensor(0.125),
            }
            if bpdd_enabled
            else {}
        ),
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
        last_gradient_norms={
            "gradient_norm": 4.0,
            "fdr_gradient_norm": 5.0,
            "gradients_finite": True,
        },
        stop=False,
    )


def test_every_epoch_writes_bpdd_activity_and_finite_gradient_evidence(
    tmp_path: Path,
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    trainer = _fake_trainer(run, variant="fdr_bpdd")
    context = {
        "variant": "fdr_bpdd",
        "stage": "screen",
        "run_identity": {"run_id": "fdr_bpdd-screen-seed0"},
    }

    records = []
    for epoch in (0, 1):
        trainer.epoch = epoch
        records.append(module.write_epoch_evidence(trainer, context))

    assert [record["completed_epoch"] for record in records] == [1, 2]
    required = {
        "loss_bpdd",
        "bpdd_active_edge_ratio",
        "bpdd_mean_reliability",
        "bpdd_mean_teacher_improvement",
        "gradient_norm",
        "fdr_gradient_norm",
        "gradients_finite",
        "cuda_peak_mib",
    }
    for record in records:
        assert required <= record.keys()
        assert record["loss_bpdd"] == pytest.approx(0.45)
        assert record["bpdd_active_edge_ratio"] == pytest.approx(0.25)
        assert record["bpdd_mean_reliability"] == pytest.approx(0.75)
        assert record["bpdd_mean_teacher_improvement"] == pytest.approx(0.125)
        assert record["gradients_finite"] is True
        assert math.isfinite(record["gradient_norm"])
        assert math.isfinite(record["fdr_gradient_norm"])
        assert math.isfinite(record["cuda_peak_mib"])

    jsonl_rows = [
        json.loads(line)
        for line in (run / "bpdd-epochs.jsonl").read_text("utf-8").splitlines()
    ]
    assert jsonl_rows == records
    with (run / "bpdd-epochs.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2
    assert required <= csv_rows[0].keys()


def test_fdr_arm_keeps_the_same_bpdd_evidence_schema_with_null_values(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = _fake_trainer(tmp_path / "run", variant="fdr")
    record = module.write_epoch_evidence(
        trainer,
        {
            "variant": "fdr",
            "stage": "screen",
            "run_identity": {"run_id": "fdr-screen-seed0"},
        },
    )

    for field in (
        "loss_bpdd",
        "bpdd_active_edge_ratio",
        "bpdd_mean_reliability",
        "bpdd_mean_teacher_improvement",
    ):
        assert field in record
        assert record[field] is None
    assert record["gradients_finite"] is True


def test_every_epoch_queues_exact_checkpoint_hash_and_bpdd_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    trainer = _fake_trainer(run, variant="fdr_bpdd")
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "epoch0.pt"
    checkpoint.write_bytes(b"epoch-zero")
    context = {
        "variant": "fdr_bpdd",
        "stage": "screen",
        "run_identity": {"run_id": "fdr_bpdd-screen-seed0"},
        "publication_queue": None,
    }

    queued = module.queue_epoch_publication(trainer, context)
    repeated = module.queue_epoch_publication(trainer, context)

    assert repeated == queued
    assert queued["completed_epoch"] == 1
    assert queued["checkpoint"] == str(checkpoint.resolve())
    assert queued["checkpoint_sha256"] == hashlib.sha256(b"epoch-zero").hexdigest().upper()
    assert {Path(path).name for path in queued["artifacts"]} == {
        "bpdd-epochs.jsonl",
        "bpdd-epochs.csv",
        "bpdd-run.json",
    }


def test_formal100_is_fresh_and_cannot_inherit_a_screen30_checkpoint(
    tmp_path: Path,
) -> None:
    module = _load_module()
    data_yaml = tmp_path / "formal.yaml"
    formal_settings = module.build_settings(
        _args(tmp_path, stage="formal", variant="fdr_bpdd", resume=None), data_yaml
    )

    assert formal_settings["epochs"] == 100
    assert "resume" not in formal_settings

    screen_identity = module.build_run_identity(
        _source(), stage="screen", variant="fdr_bpdd", seed=0
    )
    formal_identity = module.build_run_identity(
        _source(), stage="formal", variant="fdr_bpdd", seed=0
    )
    run = tmp_path / "screen-run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "last.pt"
    checkpoint.write_bytes(b"screen")
    (run / "bpdd-run.json").write_text(
        json.dumps({"run_identity": screen_identity}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="stage"):
        module.validate_resume_checkpoint(checkpoint, formal_identity)

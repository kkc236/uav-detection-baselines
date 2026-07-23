import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics.utils import YAML

import scripts.train_rtdetr_ebc_qp as launcher
from src.ebc_qp_config import SOURCE_SHA256
from src.ebc_qp_protocol import dataset_signature, state_fingerprint, subset_signature


def test_e1_pair_validation_recomputes_manifest_dataset_subset_and_state(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    for relative in ("images/train/a.jpg", "labels/train/a.txt", "images/val/b.jpg", "labels/val/b.txt"):
        path = dataset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    subset_path = tmp_path / "subset.txt"
    train_image = (dataset_root / "images/train/a.jpg").resolve()
    subset_path.write_text(f"{train_image}\n", encoding="utf-8")
    data_path = tmp_path / "data.yaml"
    names = {0: "object"}
    YAML.save(
        data_path,
        {"path": str(dataset_root.resolve()), "train": str(subset_path.resolve()), "val": "images/val", "names": names},
    )

    experiment_payload = {"frozen": "e1"}
    experiment_signature = launcher._json_sha256(experiment_payload)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(
        json.dumps({"payload": experiment_payload, "experiment_signature": experiment_signature}),
        encoding="utf-8",
    )
    common = {"weight": torch.tensor([1.0])}
    innovation = {"p2_adapter.weight": torch.tensor([2.0])}
    artifact = {
        "common_state": common,
        "innovation_state": innovation,
        "metadata": {"seed": 0, "experiment_signature": experiment_signature},
        "fingerprints": {"common": state_fingerprint(common), "innovation": state_fingerprint(innovation)},
    }
    initial_path = tmp_path / "initial.pt"
    torch.save(artifact, initial_path)
    subset_record = {
        "path": str(subset_path.resolve()),
        "count": 1,
        "fraction": 0.1,
        "sha256": subset_signature([train_image], root=dataset_root),
    }
    manifest = {
        "format_version": 1,
        "seed": 0,
        "experiment_signature": experiment_signature,
        "dataset": dataset_signature(dataset_root),
        "category_mapping_sha256": launcher._json_sha256(names),
        "subset": subset_record,
        "data": {"path": str(data_path.resolve()), "sha256": launcher._file_sha256(data_path)},
        "initial_state": {"path": str(initial_path.resolve()), "sha256": launcher._file_sha256(initial_path)},
        "source_sha256": SOURCE_SHA256,
        "git_commit": launcher._git_commit(),
        "environment": launcher._current_environment(),
    }
    manifest["signature"] = launcher._json_sha256(manifest)
    manifest_path = tmp_path / "protocol.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        stage="e1",
        seed=0,
        data=str(data_path),
        initial_state=initial_path,
        protocol_manifest=manifest_path,
        frozen_manifest=frozen_path,
    )

    launcher._validate_pair_artifacts(args)

    subset_path.write_text(f"{train_image}\n{train_image}\n", encoding="utf-8")
    with pytest.raises((SystemExit, ValueError), match="subset"):
        launcher._validate_pair_artifacts(args)


def test_e1_launch_rejects_an_existing_exact_run_directory(tmp_path: Path):
    run_dir = tmp_path / "e1-control-seed0"
    run_dir.mkdir()

    with pytest.raises(SystemExit, match="already exists"):
        launcher._assert_e1_launch_environment({"project": str(tmp_path), "name": run_dir.name})


def test_e1_run_manifest_closes_optimizer_results_and_checkpoint_evidence(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "e1-control-seed0"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (run_dir / "args.yaml").write_text("epochs: 10\n", encoding="utf-8")
    metric_header = (
        "epoch,metrics/mAP50-95(B),metrics/mAP50(B),metrics/AP-tiny,metrics/Recall-tiny,"
        "metrics/AP-r<8,metrics/AP-8<=r<=16\n"
    )
    (run_dir / "results.csv").write_text(
        metric_header + "".join(f"{epoch},{0.01 * epoch},0.1,0.0,0.0,0.0,0.0\n" for epoch in range(10)),
        encoding="utf-8",
    )
    evidence = {
        "optimizer_attempt": 1,
        "amp_step_skipped": False,
        "amp_scale_before": 128.0,
        "amp_scale_after": 128.0,
        "nonfinite_fields": [],
        "runtime_violation": None,
        "shallow_applied_ratio": 0.0,
        "p2_entry_count": None,
        "ordinary_query_count": None,
    }
    records = [{**evidence, "optimizer_attempt": attempt} for attempt in range(1, 146)]
    (run_dir / "optimizer-evidence.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    for name in ("last.pt", "best.pt"):
        (weights / name).write_bytes(name.encode())
    for epoch in (7, 8, 9):
        torch.save(
            {
                "epoch": epoch,
                "ema": {"weight": torch.tensor([float(epoch)])},
                "optimizer": {"state": {0: {"step": epoch}}},
                "scaler": {"scale": 128.0},
                "updates": epoch + 1,
            },
            weights / f"epoch{epoch}.pt",
        )

    protocol = {
        "signature": "protocol",
        "experiment_signature": "experiment",
        "git_commit": "commit",
        "dataset": {"sha256": "dataset"},
        "subset": {"sha256": "subset"},
        "data": {"sha256": "data"},
        "initial_state": {"sha256": "initial"},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    initial_path = tmp_path / "initial.pt"
    initial_path.write_bytes(b"initial")
    monkeypatch.setattr(launcher, "_assert_tracked_worktree_clean", lambda: None)
    monkeypatch.setattr(launcher, "_git_commit", lambda: "commit")
    args = SimpleNamespace(
        arm="control",
        seed=0,
        controlled_amp_scale=128.0,
        protocol_manifest=protocol_path,
        initial_state=initial_path,
    )
    settings = {"project": str(tmp_path / "runs"), "name": "e1-control-seed0"}
    trainer = SimpleNamespace(save_dir=run_dir)

    result = launcher.write_e1_run_manifest(args, settings, trainer)

    assert result["results"]["epochs"] == 10
    assert result["controlled_amp"]["init_scale"] == 128.0
    assert result["controlled_amp"]["skipped_attempts"] == 0
    assert (run_dir / "e1-run-manifest.json").is_file()


def _write_minimal_e1_optimizer_evidence(run_dir: Path, records: list[dict]) -> tuple[SimpleNamespace, dict, object]:
    run_dir.mkdir(parents=True)
    (run_dir / "optimizer-evidence.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    args = SimpleNamespace(arm="control", controlled_amp_scale=128.0)
    settings = {"project": str(run_dir.parent), "name": run_dir.name}
    trainer = SimpleNamespace(save_dir=run_dir)
    return args, settings, trainer


def _valid_e1_optimizer_records() -> list[dict]:
    return [
        {
            "optimizer_attempt": attempt,
            "amp_step_skipped": False,
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "nonfinite_fields": [],
            "runtime_violation": None,
            "shallow_applied_ratio": 0.0,
            "p2_entry_count": None,
            "ordinary_query_count": None,
        }
        for attempt in range(1, 146)
    ]


@pytest.mark.parametrize("attempt_count", [144, 146])
def test_e1_run_manifest_requires_exact_145_optimizer_attempts(tmp_path: Path, attempt_count: int):
    records = _valid_e1_optimizer_records()[:attempt_count]
    if attempt_count == 146:
        records.append({**records[-1], "optimizer_attempt": 146})
    args, settings, trainer = _write_minimal_e1_optimizer_evidence(tmp_path / "run", records)

    with pytest.raises(RuntimeError, match="exactly 145 optimizer attempts"):
        launcher.write_e1_run_manifest(args, settings, trainer)


@pytest.mark.parametrize(("before", "after"), [(256.0, 128.0), (128.0, 256.0)])
def test_e1_run_manifest_rejects_amp_scale_drift_before_or_after(
    tmp_path: Path,
    before: float,
    after: float,
):
    records = _valid_e1_optimizer_records()
    records[72].update(amp_scale_before=before, amp_scale_after=after)
    args, settings, trainer = _write_minimal_e1_optimizer_evidence(tmp_path / "run", records)

    with pytest.raises(RuntimeError, match="changed AMP scale"):
        launcher.write_e1_run_manifest(args, settings, trainer)


def test_e1_tail_checkpoint_validation_rejects_extra_or_non_resumable_files(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = {
        "epoch": 7,
        "ema": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {"step": 1}}},
        "scaler": {"scale": 128.0},
        "updates": 1,
    }
    for epoch in (7, 8, 9):
        torch.save({**checkpoint, "epoch": epoch}, weights / f"epoch{epoch}.pt")

    launcher._validate_e1_tail_checkpoints(weights, arm="control")

    torch.save({**checkpoint, "epoch": 6}, weights / "epoch6.pt")
    with pytest.raises(RuntimeError, match="exactly epoch7/8/9"):
        launcher._validate_e1_tail_checkpoints(weights, arm="control")
    (weights / "epoch6.pt").unlink()

    torch.save({"epoch": 8, "ema": {}}, weights / "epoch8.pt")
    with pytest.raises(RuntimeError, match="resume state"):
        launcher._validate_e1_tail_checkpoints(weights, arm="control")


def test_e1_tsgr_tail_checkpoint_requires_ebc_metadata(tmp_path: Path):
    checkpoint = {
        "ema": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {"step": 1}}},
        "scaler": {"scale": 128.0},
        "updates": 1,
    }
    for epoch in (7, 8, 9):
        torch.save({**checkpoint, "epoch": epoch}, tmp_path / f"epoch{epoch}.pt")

    with pytest.raises(RuntimeError, match="EBC-QP metadata"):
        launcher._validate_e1_tail_checkpoints(tmp_path, arm="tsgr-p2")

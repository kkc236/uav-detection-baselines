from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_bpdd_publication_queue.py"


def _load_module():
    assert SCRIPT.is_file(), "BPDD publication queue sync CLI has not been implemented"
    spec = importlib.util.spec_from_file_location("sync_bpdd_publication_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save_checkpoint(path: Path, completed_epoch: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": completed_epoch - 1,
            "optimizer": {"state": {}, "param_groups": []},
            "ema": {"weights": torch.ones(1)},
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _queue_row(run: Path, completed_epoch: int) -> dict:
    checkpoint = run / "weights" / f"epoch{completed_epoch - 1}.pt"
    sha256 = _save_checkpoint(checkpoint, completed_epoch)
    return {
        "run_id": "fdr-bpdd-screen-seed0",
        "variant": "fdr_bpdd",
        "stage": "screen",
        "completed_epoch": completed_epoch,
        "status": "pending",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256,
        "artifacts": [str((run / "bpdd-epochs.jsonl").resolve())],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(tmp_path: Path, run: Path, queue: Path, **changes) -> argparse.Namespace:
    token_file = tmp_path / "github-token"
    token_file.write_text("TOP-SECRET-TOKEN", encoding="utf-8")
    values = {
        "queue": queue,
        "token_file": token_file,
        "repo": "owner/repository",
        "tag": "bpdd-live",
        "branch": "main",
        "asset_prefix": "bpdd-screen",
        "run_dir": run,
        "interval": 1,
        "local_retain": 3,
        "once": True,
        "status_file": tmp_path / "status.json",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_cli_exposes_exact_queue_release_and_runtime_controls() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        [
            "--queue",
            "queue.jsonl",
            "--token-file",
            "token",
            "--repo",
            "owner/repository",
            "--tag",
            "bpdd-live",
            "--branch",
            "evidence",
            "--asset-prefix",
            "bpdd",
            "--run-dir",
            "run",
            "--interval",
            "17",
            "--local-retain",
            "5",
            "--once",
            "--status-file",
            "status.json",
        ]
    )

    assert args.queue == Path("queue.jsonl")
    assert args.token_file == Path("token")
    assert args.repo == "owner/repository"
    assert args.tag == "bpdd-live"
    assert args.branch == "evidence"
    assert args.asset_prefix == "bpdd"
    assert args.run_dir == Path("run")
    assert args.interval == 17
    assert args.local_retain == 5
    assert args.once is True
    assert args.status_file == Path("status.json")


def test_cli_defaults_to_three_local_epoch_checkpoints() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        [
            "--queue",
            "queue.jsonl",
            "--token-file",
            "token",
            "--repo",
            "owner/repository",
            "--tag",
            "bpdd-live",
            "--asset-prefix",
            "bpdd",
            "--run-dir",
            "run",
        ]
    )

    assert args.local_retain == 3


@pytest.mark.parametrize("local_retain", (0, -1))
def test_resolve_args_rejects_non_positive_local_retain(
    tmp_path: Path, local_retain: int
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    args = _args(tmp_path, run, queue, local_retain=local_retain)

    with pytest.raises(ValueError, match="local retain"):
        module._resolve_args(args)


def test_queue_validation_binds_path_sha_and_checkpoint_epoch(tmp_path: Path) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    rows = [_queue_row(run, epoch) for epoch in (1, 2)]
    _write_jsonl(queue, rows)

    entries = module.load_validated_queue(queue, run)

    assert [entry.completed_epoch for entry in entries] == [1, 2]
    assert [entry.checkpoint for entry in entries] == [
        Path(row["checkpoint"]) for row in rows
    ]
    assert [entry.checkpoint_sha256 for entry in entries] == [
        row["checkpoint_sha256"] for row in rows
    ]


@pytest.mark.parametrize("corruption", ("gap", "sha", "epoch", "outside"))
def test_queue_corruption_fails_before_publication(
    tmp_path: Path, corruption: str
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    rows = [_queue_row(run, epoch) for epoch in (1, 2)]
    if corruption == "gap":
        rows[1]["completed_epoch"] = 3
    elif corruption == "sha":
        rows[1]["checkpoint_sha256"] = "A" * 64
    elif corruption == "epoch":
        rows[1]["completed_epoch"] = 3
        rows[1]["checkpoint"] = rows[0]["checkpoint"]
        rows[1]["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]
        rows.insert(1, _queue_row(run, 2))
    else:
        outside = tmp_path / "outside.pt"
        rows[0]["checkpoint_sha256"] = _save_checkpoint(outside, 1)
        rows[0]["checkpoint"] = str(outside.resolve())
    _write_jsonl(queue, rows)

    with pytest.raises((ValueError, FileNotFoundError), match="epoch|SHA256|run directory"):
        module.load_validated_queue(queue, run)


def test_sync_publishes_every_pending_epoch_in_order_and_appends_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    rows = [_queue_row(run, epoch) for epoch in (1, 2, 3)]
    _write_jsonl(queue, rows)
    args = _args(tmp_path, run, queue)
    calls: list[int] = []

    monkeypatch.setattr(module, "validate_token_file", lambda _path: "TOP-SECRET-TOKEN")
    monkeypatch.setattr(module, "github_session", lambda _token: object())
    monkeypatch.setattr(module, "verify_ledger_assets", lambda *_args, **_kwargs: None)

    def fake_publish(_session, entry, _args):
        calls.append(entry.completed_epoch)
        return {
            "format_version": 1,
            "run_id": entry.run_id,
            "variant": entry.variant,
            "stage": entry.stage,
            "completed_epoch": entry.completed_epoch,
            "queue_record_sha256": entry.queue_record_sha256,
            "checkpoint": {
                "asset_id": 100 + entry.completed_epoch,
                "asset_name": f"bpdd-screen-epoch-{entry.completed_epoch:04d}.pt",
                "bytes": entry.checkpoint.stat().st_size,
                "sha256": entry.checkpoint_sha256.lower(),
            },
            "manifest": {
                "asset_id": 200 + entry.completed_epoch,
                "asset_name": f"bpdd-screen-epoch-{entry.completed_epoch:04d}.json",
            },
            "release_url": "https://example.invalid/releases/bpdd-live",
            "verified": True,
        }

    monkeypatch.setattr(module, "publish_entry", fake_publish)

    published = module.sync_once(args)

    assert calls == [1, 2, 3]
    assert [row["completed_epoch"] for row in published] == [1, 2, 3]
    ledger_path = run / "publication-ledger.jsonl"
    ledger_rows = [json.loads(line) for line in ledger_path.read_text("utf-8").splitlines()]
    assert [row["completed_epoch"] for row in ledger_rows] == [1, 2, 3]
    assert all(row["verified"] is True for row in ledger_rows)
    assert json.loads(args.status_file.read_text("utf-8"))["completed_epoch"] == 3


def test_reentry_verifies_existing_assets_without_reupload_or_ledger_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    _write_jsonl(queue, [_queue_row(run, 1)])
    args = _args(tmp_path, run, queue)
    monkeypatch.setattr(module, "validate_token_file", lambda _path: "TOP-SECRET-TOKEN")
    monkeypatch.setattr(module, "github_session", lambda _token: object())
    monkeypatch.setattr(module, "verify_ledger_assets", lambda *_args, **_kwargs: None)

    def fake_publish(_session, entry, _args):
        return {
            "format_version": 1,
            "run_id": entry.run_id,
            "variant": entry.variant,
            "stage": entry.stage,
            "completed_epoch": 1,
            "queue_record_sha256": entry.queue_record_sha256,
            "checkpoint": {
                "asset_id": 101,
                "asset_name": "bpdd-screen-epoch-0001.pt",
                "bytes": entry.checkpoint.stat().st_size,
                "sha256": entry.checkpoint_sha256.lower(),
            },
            "manifest": {
                "asset_id": 201,
                "asset_name": "bpdd-screen-epoch-0001.json",
            },
            "release_url": "https://example.invalid/releases/bpdd-live",
            "verified": True,
        }

    monkeypatch.setattr(module, "publish_entry", fake_publish)
    assert len(module.sync_once(args)) == 1
    ledger = run / "publication-ledger.jsonl"
    original_bytes = ledger.read_bytes()
    verified: list[int] = []
    monkeypatch.setattr(
        module,
        "verify_ledger_assets",
        lambda _session, records, _args: verified.extend(
            record["completed_epoch"] for record in records
        ),
    )
    monkeypatch.setattr(
        module,
        "publish_entry",
        lambda *_args, **_kwargs: pytest.fail("an uploaded epoch must not be reuploaded"),
    )

    assert module.sync_once(args) == []
    assert verified == [1]
    assert ledger.read_bytes() == original_bytes


def test_reentry_accepts_missing_checkpoints_only_in_verified_ledger_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    rows = [_queue_row(run, epoch) for epoch in (1, 2)]
    _write_jsonl(queue, rows)
    args = _args(tmp_path, run, queue, local_retain=1)
    monkeypatch.setattr(module, "validate_token_file", lambda _path: "TOP-SECRET-TOKEN")
    monkeypatch.setattr(module, "github_session", lambda _token: object())
    monkeypatch.setattr(module, "verify_ledger_assets", lambda *_args, **_kwargs: None)

    def fake_publish(_session, entry, _args):
        return {
            "format_version": 1,
            "run_id": entry.run_id,
            "variant": entry.variant,
            "stage": entry.stage,
            "completed_epoch": entry.completed_epoch,
            "queue_record_sha256": entry.queue_record_sha256,
            "checkpoint": {
                "asset_id": 100 + entry.completed_epoch,
                "asset_name": f"bpdd-screen-epoch-{entry.completed_epoch:04d}.pt",
                "bytes": entry.checkpoint.stat().st_size,
                "sha256": entry.checkpoint_sha256.lower(),
            },
            "manifest": {
                "asset_id": 200 + entry.completed_epoch,
                "asset_name": f"bpdd-screen-epoch-{entry.completed_epoch:04d}.json",
            },
            "release_url": "https://example.invalid/releases/bpdd-live",
            "verified": True,
        }

    monkeypatch.setattr(module, "publish_entry", fake_publish)
    assert [record["completed_epoch"] for record in module.sync_once(args)] == [1, 2]
    assert not Path(rows[0]["checkpoint"]).exists()
    assert Path(rows[1]["checkpoint"]).is_file()

    monkeypatch.setattr(
        module,
        "publish_entry",
        lambda *_args, **_kwargs: pytest.fail("verified epochs must not be reuploaded"),
    )
    assert module.sync_once(args) == []

    pending = _queue_row(run, 3)
    _write_jsonl(queue, [*rows, pending])
    Path(pending["checkpoint"]).unlink()
    with pytest.raises(FileNotFoundError, match="epoch 3"):
        module.sync_once(args)


def test_pruning_retains_latest_verified_epochs_and_never_unpublished(
    tmp_path: Path
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    rows = [_queue_row(run, epoch) for epoch in range(1, 6)]
    _write_jsonl(queue, rows)
    entries = module.load_validated_queue(queue, run)

    removed = module.prune_verified_local_checkpoints(
        entries,
        verified_through_epoch=4,
        local_retain=2,
    )

    assert removed == [Path(rows[0]["checkpoint"]), Path(rows[1]["checkpoint"])]
    assert not Path(rows[0]["checkpoint"]).exists()
    assert not Path(rows[1]["checkpoint"]).exists()
    assert Path(rows[2]["checkpoint"]).is_file()
    assert Path(rows[3]["checkpoint"]).is_file()
    assert Path(rows[4]["checkpoint"]).is_file(), "unpublished epoch must not be pruned"


def test_pruning_never_removes_last_or_best_checkpoint(tmp_path: Path) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    rows = [_queue_row(run, epoch) for epoch in (1, 2, 3)]
    protected = []
    for row, name in zip(rows[:2], ("last.pt", "best.pt")):
        old = Path(row["checkpoint"])
        replacement = old.with_name(name)
        old.replace(replacement)
        row["checkpoint"] = str(replacement.resolve())
        row["checkpoint_sha256"] = hashlib.sha256(replacement.read_bytes()).hexdigest().upper()
        protected.append(replacement)
    _write_jsonl(queue, rows)
    entries = module.load_validated_queue(queue, run)

    removed = module.prune_verified_local_checkpoints(
        entries,
        verified_through_epoch=3,
        local_retain=1,
    )

    assert removed == []
    assert all(path.is_file() for path in protected)


def test_changed_queue_or_ledger_history_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    row = _queue_row(run, 1)
    _write_jsonl(queue, [row])
    entry = module.load_validated_queue(queue, run)[0]
    ledger = run / "publication-ledger.jsonl"
    bad = {
        "format_version": 1,
        "run_id": entry.run_id,
        "variant": entry.variant,
        "stage": entry.stage,
        "completed_epoch": 1,
        "queue_record_sha256": "F" * 64,
        "checkpoint": {
            "asset_id": 1,
            "asset_name": "bpdd-screen-epoch-0001.pt",
            "bytes": entry.checkpoint.stat().st_size,
            "sha256": entry.checkpoint_sha256.lower(),
        },
        "manifest": {"asset_id": 2, "asset_name": "bpdd-screen-epoch-0001.json"},
        "release_url": "https://example.invalid/release",
        "verified": True,
    }
    _write_jsonl(ledger, [bad])

    with pytest.raises(ValueError, match="ledger|queue"):
        module.load_validated_ledger(ledger, [entry])


def test_publish_entry_uses_existing_release_api_for_exact_checkpoint_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    _write_jsonl(queue, [_queue_row(run, 1)])
    entry = module.load_validated_queue(queue, run)[0]
    args = _args(tmp_path, run, queue)
    release = {
        "url": "https://api.example.invalid/release/1",
        "html_url": "https://example.invalid/release/1",
        "upload_url": "https://uploads.example.invalid/1{?name}",
        "assets": [],
    }
    publish_calls: list[dict] = []
    upload_calls: list[tuple[str, bytes]] = []

    def fake_publish(_session, **kwargs):
        publish_calls.append(kwargs)
        return {
            "completed_epoch": 1,
            "release_url": release["html_url"],
            "checkpoint": {
                "asset_id": 101,
                "asset_name": "bpdd-screen-epoch-0001.pt",
                "bytes": entry.checkpoint.stat().st_size,
                "sha256": entry.checkpoint_sha256.lower(),
            },
        }

    monkeypatch.setattr(module, "publish_checkpoint", fake_publish)
    monkeypatch.setattr(module, "get_or_create_release", lambda *_args, **_kwargs: release)

    def fake_upload(_session, *, release, path, asset_name):
        del release
        upload_calls.append((asset_name, path.read_bytes()))
        return {"id": 201, "name": asset_name, "size": path.stat().st_size}

    monkeypatch.setattr(module, "upload_asset", fake_upload)
    monkeypatch.setattr(module, "verify_publication_assets", lambda *_args, **_kwargs: None)

    record = module.publish_entry(object(), entry, args)

    assert len(publish_calls) == 1
    assert Path(publish_calls[0]["checkpoint"]) == entry.checkpoint
    assert publish_calls[0]["asset_prefix"] == "bpdd-screen"
    assert publish_calls[0]["retain"] >= 100
    assert upload_calls[0][0] == "bpdd-screen-epoch-0001.json"
    manifest = json.loads(upload_calls[0][1])
    assert manifest["completed_epoch"] == 1
    assert manifest["checkpoint"]["sha256"] == entry.checkpoint_sha256.lower()
    assert manifest["queue_record_sha256"] == entry.queue_record_sha256
    assert record["manifest"]["asset_name"] == "bpdd-screen-epoch-0001.json"
    assert record["verified"] is True


def test_retry_status_and_stderr_never_expose_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    run = tmp_path / "run"
    queue = run / "publication-queue.jsonl"
    _write_jsonl(queue, [_queue_row(run, 1)])
    args = _args(tmp_path, run, queue, once=False)
    secret = args.token_file.read_text(encoding="utf-8")

    def fail(_args):
        raise RuntimeError(f"remote rejected Bearer {secret}")

    monkeypatch.setattr(module, "sync_once", fail)

    class StopLoop(RuntimeError):
        pass

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(StopLoop()))

    with pytest.raises(StopLoop):
        module.run_continuously(args)

    status_text = args.status_file.read_text(encoding="utf-8")
    output = capsys.readouterr()
    assert secret not in status_text
    assert secret not in output.out
    assert secret not in output.err
    assert "[REDACTED]" in status_text

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def git(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


def _create_results_origin(tmp_path: Path) -> tuple[Path, Path]:
    seed = tmp_path / "seed"
    origin = tmp_path / "origin.git"
    seed.mkdir()
    git("init", "-b", "main", cwd=seed)
    git("config", "user.name", "test", cwd=seed)
    git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "README.md").write_text("seed", encoding="utf-8")
    git("add", "README.md", cwd=seed)
    git("commit", "-m", "seed", cwd=seed)
    git("branch", "training-results", cwd=seed)
    git("clone", "--bare", str(seed), str(origin), cwd=tmp_path)
    return seed, origin


def test_equal_remote_oid_skips_fetch_and_rebase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = importlib.import_module("scripts.sync_experiment_checkpoint")
    _, origin = _create_results_origin(tmp_path)
    checkout = tmp_path / "checkout"
    token_file = tmp_path / "github_token"
    token_file.write_text("test-token", encoding="utf-8")
    sync.ensure_results_checkout(
        checkout,
        repo_url=str(origin),
        branch="training-results",
        token_file=token_file,
    )
    original_run = sync._run
    calls: list[list[str]] = []

    def tracked(command, **kwargs):
        calls.append(list(command))
        return original_run(command, **kwargs)

    monkeypatch.setattr(sync, "_run", tracked)
    sync.ensure_results_checkout(
        checkout,
        repo_url=str(origin),
        branch="training-results",
        token_file=token_file,
    )

    assert any(command[1] == "ls-remote" for command in calls)
    assert any(command[1:3] == ["rev-parse", "HEAD"] for command in calls)
    assert not any(command[1] == "fetch" for command in calls)
    assert not any(command[1] == "rebase" for command in calls)


def test_different_remote_oid_fetches_and_rebases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = importlib.import_module("scripts.sync_experiment_checkpoint")
    _, origin = _create_results_origin(tmp_path)
    checkout = tmp_path / "checkout"
    writer = tmp_path / "writer"
    token_file = tmp_path / "github_token"
    token_file.write_text("test-token", encoding="utf-8")
    sync.ensure_results_checkout(
        checkout,
        repo_url=str(origin),
        branch="training-results",
        token_file=token_file,
    )
    git("clone", str(origin), str(writer), cwd=tmp_path)
    git("switch", "training-results", cwd=writer)
    git("config", "user.name", "test", cwd=writer)
    git("config", "user.email", "test@example.com", cwd=writer)
    (writer / "remote.txt").write_text("new remote commit", encoding="utf-8")
    git("add", "remote.txt", cwd=writer)
    git("commit", "-m", "advance remote", cwd=writer)
    git("push", "origin", "training-results", cwd=writer)
    original_run = sync._run
    calls: list[list[str]] = []

    def tracked(command, **kwargs):
        calls.append(list(command))
        return original_run(command, **kwargs)

    monkeypatch.setattr(sync, "_run", tracked)
    sync.ensure_results_checkout(
        checkout,
        repo_url=str(origin),
        branch="training-results",
        token_file=token_file,
    )

    assert any(command[1] == "fetch" for command in calls)
    assert any(command[1] == "rebase" for command in calls)
    assert git("rev-parse", "HEAD", cwd=checkout) == git(
        f"--git-dir={origin}", "rev-parse", "training-results", cwd=tmp_path
    )


def test_git_network_calls_use_one_finite_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = importlib.import_module("scripts.sync_experiment_checkpoint")
    results_repo = tmp_path / "results"
    (results_repo / ".git").mkdir(parents=True)
    token_file = tmp_path / "github_token"
    token_file.write_text("test-token", encoding="utf-8")
    calls: list[tuple[list[str], object]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs.get("timeout")))
        if command[1:3] == ["branch", "--list"]:
            stdout = "training-results\n"
        elif command[1] == "ls-remote":
            stdout = f"{'a' * 40}\trefs/heads/training-results\n"
        elif command[1:3] == ["rev-parse", "HEAD"]:
            stdout = f"{'b' * 40}\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(sync, "_run", fake_run)
    monkeypatch.setattr(sync, "_git_environment", lambda *_args: {})
    sync.ensure_results_checkout(
        results_repo,
        repo_url="https://example.invalid/results.git",
        branch="training-results",
        token_file=token_file,
    )
    result_directory = results_repo / "results" / "run"
    result_directory.mkdir(parents=True)
    sync.commit_and_push_results(
        results_repo,
        result_directory=result_directory,
        completed_epoch=1,
        branch="training-results",
        environment={},
    )

    network_calls = [
        (command, timeout)
        for command, timeout in calls
        if command[1] in {"ls-remote", "fetch", "push"}
    ]
    assert {command[1] for command, _ in network_calls} == {
        "ls-remote",
        "fetch",
        "push",
    }
    assert network_calls
    assert all(
        isinstance(timeout, (int, float)) and 0 < timeout <= 300
        for _, timeout in network_calls
    )


@pytest.mark.parametrize("module_name", ("scripts.sync_experiment_checkpoint",))
def test_concurrent_result_writers_preserve_both_experiments(tmp_path: Path, module_name: str):
    sync = importlib.import_module(module_name)
    seed = tmp_path / "seed"
    origin = tmp_path / "origin.git"
    first_checkout = tmp_path / "first"
    second_checkout = tmp_path / "second"
    token_file = tmp_path / "github_token"
    seed.mkdir()
    token_file.write_text("test-token", encoding="utf-8")

    git("init", "-b", "main", cwd=seed)
    git("config", "user.name", "test", cwd=seed)
    git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "README.md").write_text("seed", encoding="utf-8")
    git("add", "README.md", cwd=seed)
    git("commit", "-m", "seed", cwd=seed)
    git("branch", "training-results", cwd=seed)
    git("clone", "--bare", str(seed), str(origin), cwd=tmp_path)

    first_environment = sync.ensure_results_checkout(
        first_checkout,
        repo_url=str(origin),
        branch="training-results",
        token_file=token_file,
    )
    second_environment = sync.ensure_results_checkout(
        second_checkout,
        repo_url=str(origin),
        branch="training-results",
        token_file=token_file,
    )

    first_result = first_checkout / "results" / "run-a"
    first_result.mkdir(parents=True)
    (first_result / "latest.json").write_text('{"completed_epoch": 3}', encoding="utf-8")
    sync.commit_and_push_results(
        first_checkout,
        result_directory=first_result,
        completed_epoch=3,
        branch="training-results",
        environment=first_environment,
    )

    second_result = second_checkout / "results" / "run-b"
    second_result.mkdir(parents=True)
    (second_result / "latest.json").write_text('{"completed_epoch": 7}', encoding="utf-8")
    sync.commit_and_push_results(
        second_checkout,
        result_directory=second_result,
        completed_epoch=7,
        branch="training-results",
        environment=second_environment,
    )

    first = git(
        f"--git-dir={origin}",
        "show",
        "training-results:results/run-a/latest.json",
        cwd=tmp_path,
    )
    second = git(
        f"--git-dir={origin}",
        "show",
        "training-results:results/run-b/latest.json",
        cwd=tmp_path,
    )
    assert json.loads(first)["completed_epoch"] == 3
    assert json.loads(second)["completed_epoch"] == 7


def test_continuous_sync_retries_when_checkpoint_tree_advances_during_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = importlib.import_module("scripts.sync_experiment_checkpoint")
    before = (("epoch32.pt", 100, 10), ("last.pt", 100, 10))
    after = (
        ("epoch32.pt", 100, 10),
        ("epoch33.pt", 101, 20),
        ("last.pt", 101, 20),
    )
    fingerprints = iter((before, after, after, after))
    published_epochs = iter((33, 34))
    calls: list[int] = []

    monkeypatch.setattr(
        sync,
        "checkpoint_tree_fingerprint",
        lambda _run_dir: next(fingerprints),
    )

    def fake_sync_once(_args):
        completed_epoch = next(published_epochs)
        calls.append(completed_epoch)
        return {
            "completed_epoch": completed_epoch,
            "release_url": "https://example.invalid/release",
        }

    monkeypatch.setattr(sync, "sync_once", fake_sync_once)

    class StopLoop(RuntimeError):
        pass

    sleeps = 0

    def stop_after_second_iteration(_interval):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise StopLoop

    monkeypatch.setattr(sync.time, "sleep", stop_after_second_iteration)
    args = SimpleNamespace(run_dir=tmp_path, interval=30, status_file=tmp_path / "status.json")

    with pytest.raises(StopLoop):
        sync.run_continuously(args)

    assert calls == [33, 34]

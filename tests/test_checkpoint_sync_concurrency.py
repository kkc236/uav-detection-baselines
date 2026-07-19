from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest


def git(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


@pytest.mark.parametrize("module_name", ("scripts.sync_btdse_checkpoint",))
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

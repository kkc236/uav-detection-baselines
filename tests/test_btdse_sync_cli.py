import json
import subprocess
from pathlib import Path

from scripts.sync_btdse_checkpoint import (
    LIGHTWEIGHT_ARTIFACTS,
    build_parser,
    collect_lightweight_artifacts,
    commit_and_push_results,
    ensure_results_checkout,
    write_json_atomic,
)


def test_sync_cli_has_protective_defaults(tmp_path: Path):
    run_dir = tmp_path / "run"
    args = build_parser().parse_args(["--run-dir", str(run_dir), "--token-file", "secret"])

    assert args.repo == "kkc236/uav-detection-baselines"
    assert args.tag == "btdse-v2.5-s-4090-live"
    assert args.retain == 1
    assert args.interval == 60
    assert args.results_branch == "training-results"
    assert args.once is False


def test_only_lightweight_artifacts_and_manifest_are_collected(tmp_path: Path):
    run_dir = tmp_path / "run"
    destination = tmp_path / "results"
    run_dir.mkdir()
    for name in LIGHTWEIGHT_ARTIFACTS:
        (run_dir / name).write_text(name, encoding="utf-8")
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "last.pt").write_bytes(b"large checkpoint")

    copied = collect_lightweight_artifacts(
        run_dir,
        destination,
        {"completed_epoch": 7, "checkpoint": {"sha256": "abc"}},
    )

    assert {path.name for path in copied} == {*LIGHTWEIGHT_ARTIFACTS, "latest.json"}
    assert not (destination / "weights").exists()
    assert json.loads((destination / "latest.json").read_text(encoding="utf-8"))["completed_epoch"] == 7


def test_ioqc_sa_diagnostics_and_adaptive_state_are_lightweight_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    destination = tmp_path / "results"
    run_dir.mkdir()
    expected = {"ioqc_sa_diagnostics.jsonl", "batch_history.jsonl", "adaptive_state.json"}
    for name in expected:
        (run_dir / name).write_text(name, encoding="utf-8")

    copied = collect_lightweight_artifacts(run_dir, destination, {"completed_epoch": 3})

    assert expected <= {path.name for path in copied}


def test_atomic_json_writer_replaces_complete_document(tmp_path: Path):
    path = tmp_path / "status.json"

    write_json_atomic(path, {"state": "uploading"})
    write_json_atomic(path, {"state": "published", "completed_epoch": 8})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "state": "published",
        "completed_epoch": 8,
    }
    assert not path.with_suffix(".json.tmp").exists()


def test_results_are_committed_to_isolated_training_branch(tmp_path: Path):
    seed = tmp_path / "seed"
    origin = tmp_path / "origin.git"
    checkout = tmp_path / "results-checkout"
    token_file = tmp_path / "github_token"
    seed.mkdir()
    token_file.write_text("test-token", encoding="utf-8")

    def git(*arguments: str, cwd: Path = seed) -> str:
        return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.com")
    (seed / "README.md").write_text("seed", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "seed")
    git("clone", "--bare", str(seed), str(origin), cwd=tmp_path)

    environment = ensure_results_checkout(
        checkout,
        repo_url=str(origin),
        branch="training-results",
        token_file=token_file,
    )
    result_directory = checkout / "results" / "run"
    result_directory.mkdir(parents=True)
    (result_directory / "latest.json").write_text('{"completed_epoch": 2}', encoding="utf-8")

    commit_and_push_results(
        checkout,
        result_directory=result_directory,
        completed_epoch=2,
        branch="training-results",
        environment=environment,
    )

    content = git(
        f"--git-dir={origin}",
        "show",
        "training-results:results/run/latest.json",
        cwd=tmp_path,
    )
    assert json.loads(content)["completed_epoch"] == 2

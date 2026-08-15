from pathlib import Path

from src.result_publisher import (
    PublicationStatus,
    collect_result_artifacts,
    latest_completed_epoch,
    repo_slug_from_remote,
    sha256_file,
    verified_release_assets,
)


def test_latest_completed_epoch_reads_last_results_row(tmp_path: Path):
    results = tmp_path / "results.csv"
    results.write_text("epoch,time\n98,100\n99,110\n100,120\n", encoding="utf-8")

    assert latest_completed_epoch(results) == 100


def test_missing_or_empty_results_are_incomplete(tmp_path: Path):
    assert latest_completed_epoch(tmp_path / "missing.csv") == 0
    empty = tmp_path / "empty.csv"
    empty.write_text("epoch,time\n", encoding="utf-8")
    assert latest_completed_epoch(empty) == 0


def test_sha256_file_returns_reproducible_digest(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"uav-baseline")

    assert sha256_file(artifact) == "e00bbfcbd00df38083ad680a0c9ff6a888b22698b0ed1ce7fc04bb43b928b46b"


def test_release_assets_require_matching_names_and_sizes():
    expected = {"best.pt": 123, "results.tar.gz": 456}

    assert verified_release_assets(expected, [{"name": "best.pt", "size": 123}]) is False
    assert (
        verified_release_assets(
            expected,
            [{"name": "best.pt", "size": 123}, {"name": "results.tar.gz", "size": 456}],
        )
        is True
    )


def test_shutdown_gate_requires_epoch_push_and_release_verification():
    status = PublicationStatus(completed_epoch=100, git_push_verified=True, release_verified=False)
    assert status.shutdown_allowed is False

    status.release_verified = True
    assert status.shutdown_allowed is True

    status.completed_epoch = 99
    assert status.shutdown_allowed is False


def test_collect_result_artifacts_excludes_large_weights(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "results.csv").write_text("epoch\n100\n", encoding="utf-8")
    (run_dir / "args.yaml").write_text("batch: 16\n", encoding="utf-8")
    (run_dir / "results.png").write_bytes(b"png")
    (run_dir / "weights" / "best.pt").write_bytes(b"large")
    destination = tmp_path / "published"

    copied = collect_result_artifacts(run_dir, destination)

    assert {path.name for path in copied} == {"results.csv", "args.yaml", "results.png"}
    assert not (destination / "weights").exists()


def test_repo_slug_parses_https_and_ssh_remotes():
    assert repo_slug_from_remote("https://github.com/kkc236/uav-detection-baselines.git") == (
        "kkc236/uav-detection-baselines"
    )
    assert repo_slug_from_remote("git@github.com:kkc236/uav-detection-baselines.git") == (
        "kkc236/uav-detection-baselines"
    )

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from deploy.iber.verify_bundle import BundleViolation, verify_bundle


SCRIPTS = (
    Path("deploy/iber/verify_host.sh"),
    Path("deploy/iber/build_wheelhouse.sh"),
    Path("deploy/iber/bootstrap_ubuntu.sh"),
)
BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
)
DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)
SUBSET_SHA256 = (
    "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
)
SOURCE_COMMIT = "a" * 40


def _write_manifest(root: Path, relative: str, *, source_commit: str = SOURCE_COMMIT) -> Path:
    artifact = root / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"iber-be\n")
    payload = {
        "format_version": 1,
        "design_version": "iber-be-v1.0",
        "source_commit": source_commit,
        "baseline_sha256": BASELINE_SHA256,
        "dataset": {"sha256": DATASET_SHA256},
        "subset": {"sha256": SUBSET_SHA256},
        "execution_environment": {
            "python": "3.10.12",
            "torch": "2.5.1+cu121",
            "torchvision": "0.20.1+cu121",
            "ultralytics": "8.4.90",
            "cuda": "12.1",
            "gpu": "NVIDIA GeForce RTX 4090",
            "reported_memory_mib": 49140,
            "driver": "570.133.07",
        },
        "files": [
            {
                "path": relative,
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest().upper(),
            }
        ],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_scripts_exist_use_strict_bash_and_embed_no_credentials() -> None:
    for path in SCRIPTS:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert "$HOME" not in content
        assert "~/" not in content
        assert "github_pat_" not in content
        assert "a1314520" not in content


def test_host_verifier_is_read_only_and_locks_amended_runtime() -> None:
    content = SCRIPTS[0].read_text(encoding="utf-8")
    for required in (
        "NVIDIA GeForce RTX 4090",
        "expected_gpu_memory_mib=49140",
        'baseline_reference_driver="550.142"',
        'expected_driver="570.133.07"',
        "passed_with_runtime_amendment",
        "Python 3.10.12",
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "ultralytics==8.4.90",
        "CUDA 12.1",
        "df -Pk /data",
        "MemTotal",
        "github.com",
        "mirrors.aliyun.com",
        "download.pytorch.org",
    ):
        assert required in content
    for mutation in ("apt-get", "mkdir", "touch", "chmod", "install -d", "rm -"):
        assert mutation not in content


def test_wheelhouse_is_mirror_first_and_exactly_version_pinned() -> None:
    content = SCRIPTS[1].read_text(encoding="utf-8")
    for required in (
        "/data/uav/staging/iber-be-v1-wheelhouse",
        "mirrors.aliyun.com/pypi/simple",
        "download.pytorch.org/whl/cu121",
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "ultralytics==8.4.90",
        "ultralytics-thop==2.0.18",
        "python3.10",
        "sha256sum",
        "foreground",
    ):
        assert required in content
    assert content.index("mirrors.aliyun.com/pypi/simple") < content.index(
        "download.pytorch.org/whl/cu121"
    )
    assert "thop==0.1.1.post2209072238" not in content


def test_bootstrap_uses_immutable_source_and_run_roots_and_mode_600_secrets() -> None:
    content = SCRIPTS[2].read_text(encoding="utf-8")
    for required in (
        "/data/uav/source/uav-detection-baselines-",
        "/data/uav/venvs/iber-be-v1",
        "/data/uav/cache/iber-be-v1-",
        "/data/uav/runs/iber-be-v1/",
        "/data/uav/results/iber-be-v1-",
        "/data/uav/logs/iber-be-v1-",
        "/data/uav/deploy/iber-be-v1/markers",
        "/data/uav/HANDOFFS/secrets/github_token",
        "/data/uav/config/iber-be-v1/publication-screen.json",
        "/data/uav/config/iber-be-v1/git-http.env",
        "stat -c %a",
        '"600"',
        "chmod 600",
        "source_commit",
        "source_short_sha",
        "checkout --detach",
        "rev-parse HEAD",
        "status --porcelain",
        "python3.10 -m venv",
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "ultralytics==8.4.90",
        "ultralytics-thop==2.0.18",
        "YOLO_CONFIG_DIR",
        "publication_config_mode",
        "git -c http.version=HTTP/1.1 clone",
        "git -c http.version=HTTP/1.1 -C",
        "GIT_CONFIG_COUNT=1",
        "GIT_CONFIG_KEY_0=http.version",
        "GIT_CONFIG_VALUE_0=HTTP/1.1",
    ):
        assert required in content
    assert not re.search(r"/data/uav/(?:runs|results|cache)/itber(?:/|[-_.])", content, re.I)


def test_bootstrap_rechecks_completed_marker_and_uses_only_public_remote() -> None:
    content = SCRIPTS[2].read_text(encoding="utf-8")
    assert 'source_remote="https://github.com/kkc236/uav-detection-baselines.git"' in content
    assert '${2:-' not in content
    marker_guard = content.split('if [[ -f "$marker_path" ]]', maxsplit=1)[1].split(
        "fi", maxsplit=1
    )[0]
    assert "exit 0" not in marker_guard
    assert 'bootstrap_required=false' in marker_guard


def test_artifact_manifest_template_locks_all_scientific_authorities() -> None:
    path = Path("deploy/iber/artifact-manifest.template.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["design_version"] == "iber-be-v1.0"
    assert payload["source_commit"] == "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT_SHA"
    assert payload["baseline_sha256"] == BASELINE_SHA256
    assert payload["dataset"] == {
        "name": "VisDrone2019-DET",
        "sha256": DATASET_SHA256,
        "train_images": 6471,
        "val_images": 548,
        "classes": 10,
    }
    assert payload["subset"] == {"train_images": 647, "sha256": SUBSET_SHA256}
    assert payload["execution_environment"] == {
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "ultralytics": "8.4.90",
        "cuda": "12.1",
        "gpu": "NVIDIA GeForce RTX 4090",
        "reported_memory_mib": 49140,
        "driver": "570.133.07",
    }


def test_publication_template_is_iber_only_credential_free_and_gate_locked() -> None:
    path = Path("deploy/iber/publication-screen.template.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["design_version"] == "iber-be-v1.0"
    assert payload["stage"] == "screen"
    assert payload["probe"] == "b3"
    assert payload["seed"] == 0
    assert payload["expected_private_epochs"] == 30
    assert payload["source_commit"] == "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT_SHA"
    assert payload["results_branch"] == "iber-be-v1-results"
    assert payload["tag"] == "iber-be-v1-rtdetr-l-live"
    assert payload["asset_prefix"] == "iber-be-v1.0-screen-seed0-b3"
    assert payload["run_name"].startswith("iber-be-v1.0-screen-seed0-b3-")
    assert payload["token_file"] == "/data/uav/HANDOFFS/secrets/github_token"
    assert payload["results_repo"].startswith("/data/uav/results/iber-be-v1-")
    assert payload["gate1_decision"] == (
        "/data/uav/runs/iber-be-v1/"
        "SOURCE_SHORT_SHA-seed0-amended/probe/gate1-decision.json"
    )
    content = path.read_text(encoding="utf-8")
    assert "github_pat_" not in content
    assert "a1314520" not in content
    assert not re.search(r"(?:^|[-_/])itber(?:$|[-_./])", content, re.I)


def test_server_guide_pipeline_command_contains_every_required_path() -> None:
    guide = Path("docs/IBER_BE_SERVER_GUIDE.md").read_text(encoding="utf-8")
    for option in (
        "--baseline-checkpoint",
        "--dataset-root",
        "--run-root",
        "--cache-root",
        "--publication-config",
        "--device 0",
    ):
        assert option in guide


def test_bundle_verifier_accepts_exact_authorities_and_source_commit(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "artifacts/report.json")

    report = verify_bundle(tmp_path, manifest, expected_source_commit=SOURCE_COMMIT)

    assert report["status"] == "passed"
    assert report["design_version"] == "iber-be-v1.0"
    assert report["source_commit"] == SOURCE_COMMIT
    assert report["file_count"] == 1


@pytest.mark.parametrize(
    ("relative", "source_commit"),
    (
        ("results/itber-v1.1/report.json", SOURCE_COMMIT),
        ("results/I-TBER-v1.1/report.json", SOURCE_COMMIT),
        ("artifacts/report.json", "b" * 40),
        ("../escape.json", SOURCE_COMMIT),
    ),
)
def test_bundle_verifier_rejects_old_identity_source_drift_and_traversal(
    tmp_path: Path,
    relative: str,
    source_commit: str,
) -> None:
    if relative.startswith("../"):
        outside = tmp_path.parent / "escape.json"
        outside.write_bytes(b"iber-be\n")
        payload = {
            "format_version": 1,
            "design_version": "iber-be-v1.0",
            "source_commit": SOURCE_COMMIT,
            "baseline_sha256": BASELINE_SHA256,
            "dataset": {"sha256": DATASET_SHA256},
            "subset": {"sha256": SUBSET_SHA256},
            "files": [
                {
                    "path": relative,
                    "bytes": outside.stat().st_size,
                    "sha256": hashlib.sha256(outside.read_bytes()).hexdigest().upper(),
                }
            ],
        }
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    else:
        manifest = _write_manifest(tmp_path, relative, source_commit=source_commit)

    with pytest.raises(BundleViolation):
        verify_bundle(tmp_path, manifest, expected_source_commit=SOURCE_COMMIT)

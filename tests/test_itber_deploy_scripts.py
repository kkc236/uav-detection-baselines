from __future__ import annotations

from pathlib import Path

import pytest


SCRIPTS = (
    Path("deploy/itber/verify_host.sh"),
    Path("deploy/itber/build_wheelhouse.sh"),
    Path("deploy/itber/bootstrap_ubuntu.sh"),
)


def test_scripts_exist_and_use_strict_bash() -> None:
    for path in SCRIPTS:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert "$HOME" not in content
        assert "~/" not in content
        assert "github_pat_" not in content
        assert "a1314520" not in content


def test_host_verifier_is_read_only_and_reports_hard_requirements() -> None:
    content = SCRIPTS[0].read_text(encoding="utf-8")
    for required in (
        "nvidia-smi",
        "NVIDIA GeForce RTX 4090",
        "550.142",
        "python3",
        "git --version",
        "df -Pk /data",
        "MemTotal",
        "github.com",
        "pypi.org",
    ):
        assert required in content
    for mutation in ("apt-get", "mkdir", "touch", "chmod", "rm -"):
        assert mutation not in content


def test_wheelhouse_uses_mirror_first_and_official_cuda_index() -> None:
    content = SCRIPTS[1].read_text(encoding="utf-8")
    assert "/data/uav/staging/itber-v1.1-wheelhouse" in content
    assert "mirrors.aliyun.com/pypi/simple" in content
    assert "download.pytorch.org/whl/cu121" in content
    assert "requirements-itber.lock" in content
    assert "sha256sum" in content
    assert "nohup" in content or "foreground" in content


def test_bootstrap_has_idempotent_lock_marker_and_secret_policy() -> None:
    content = SCRIPTS[2].read_text(encoding="utf-8")
    for required in (
        "/data/uav/venvs/itber-v1.1",
        "/data/uav/deploy/markers",
        "requirements-itber.lock",
        "sha256sum",
        "python3.10 -m venv",
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "download.pytorch.org/whl/cu121",
        "mirrors.aliyun.com/pypi/simple",
        "/data/uav/HANDOFFS/secrets/github_token",
        "stat -c %a",
        '"600"',
        "NVIDIA GeForce RTX 4090",
        "550.142",
        "available_kib",
        "YOLO_CONFIG_DIR",
    ):
        assert required in content


@pytest.mark.parametrize("path", SCRIPTS)
def test_scripts_use_only_absolute_uav_roots(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    assert "/data/uav" in content or path.name == "verify_host.sh"

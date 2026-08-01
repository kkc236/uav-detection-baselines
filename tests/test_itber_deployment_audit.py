from __future__ import annotations

from pathlib import Path

from scripts.audit_itber_deployment import audit_local_readiness


def test_repository_is_locally_ready_without_claiming_remote_verification() -> None:
    report = audit_local_readiness(Path.cwd())

    assert report["status"] == "ready_waiting_for_server"
    assert report["local"]["ready"] is True
    assert report["remote"] == {
        "endpoint": "unresolved",
        "host_key": "unresolved",
        "os": "unresolved",
        "gpu": "unresolved",
        "driver": "unresolved",
        "disk": "unresolved",
        "network": "unresolved",
    }
    assert report["remote_verified"] is False
    for required in (
        "scripts/run_itber_pipeline.py",
        "scripts/train_itber.py",
        "scripts/evaluate_itber.py",
        "scripts/publish_itber_epoch.py",
        "scripts/restore_itber_checkpoint.py",
        "scripts/evaluate_itber_stock.py",
        "deploy/itber/publication-screen.template.json",
        "deploy/itber/publication-formal.template.json",
    ):
        assert required in report["local"]["required_files"]


def test_empty_tree_is_not_ready(tmp_path) -> None:
    report = audit_local_readiness(tmp_path)

    assert report["status"] == "not_ready"
    assert report["local"]["ready"] is False
    assert report["local"]["missing"]
    assert report["remote_verified"] is False


def test_guide_contains_safe_transfer_gate0_publication_and_recovery_contracts() -> None:
    guide = Path("docs/ITBER_BARE_SERVER_GUIDE.md").read_text(encoding="utf-8")
    for marker in (
        "SSH host key",
        "/data/uav",
        "rsync",
        "SHA256",
        "chmod 600",
        "run_itber_canary.py",
        "P0-P3",
        "每个 epoch",
        "restore_itber_checkpoint.py",
        "run_itber_pipeline.py",
        "publication-screen.json",
        "publication-formal.json",
        "BASELINE_TRAINING_CONTRACT_SHA256",
        "550.142",
        "570.133.07",
        "passed_with_runtime_amendment",
        "stock-authority.json",
        "不得使用历史 550.142 环境的 baseline 指标",
        "至少 80 GiB",
        "用户明确提供并授权",
    ):
        assert marker in guide
    assert "github_pat_" not in guide
    assert "a1314520" not in guide

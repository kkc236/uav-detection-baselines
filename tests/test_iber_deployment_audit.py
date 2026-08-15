from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_iber_deployment import REQUIRED_FILES, audit_local_readiness


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


def _complete_tree(root: Path) -> None:
    for relative in REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("iber-be-v1.0\n", encoding="utf-8")

    manifest = {
        "format_version": 1,
        "design_version": "iber-be-v1.0",
        "source_commit": "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT_SHA",
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
            "reported_memory_mib": 24564,
            "driver": "550.142",
        },
        "files": [],
    }
    (root / "deploy/iber/artifact-manifest.template.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    publication = {
        "format_version": 1,
        "design_version": "iber-be-v1.0",
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "expected_private_epochs": 30,
        "source_commit": "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT_SHA",
        "results_branch": "iber-be-v1-results",
        "tag": "iber-be-v1-rtdetr-l-live",
        "asset_prefix": "iber-be-v1.0-screen-seed0-b3",
        "run_name": "iber-be-v1.0-screen-seed0-b3-SOURCE_SHORT_SHA",
        "token_file": "/data/uav/HANDOFFS/secrets/github_token",
        "results_repo": "/data/uav/results/iber-be-v1-SOURCE_SHORT_SHA",
        "gate1_decision": (
            "/data/uav/runs/iber-be-v1/SOURCE_SHORT_SHA-seed0-amended/"
            "gate1-decision.json"
        ),
        "retain": 3,
    }
    (root / "deploy/iber/publication-screen.template.json").write_text(
        json.dumps(publication), encoding="utf-8"
    )
    guide = "\n".join(
        (
            "SSH host key",
            "mirror-first",
            "immutable source checkout",
            "chmod 600",
            "passed_with_runtime_amendment",
            "run_iber_canary.py",
            "run_iber_pipeline.py",
            "pipeline-state.json",
            "nvidia-smi",
            "publish_iber_epoch.py",
            "restore_iber_checkpoint.py",
            "engineering_invalid repair",
            "scientific_failed stop",
            "Gate-1 must not be bypassed",
            "禁止绕过 Gate-1",
            "I-TBER paths and results must never be reused",
            "http.version=HTTP/1.1",
            "git-http.env",
            "GIT_CONFIG_COUNT",
            "兼容状态，不表示存在硬件差异",
            "`-seed0-amended` 是旧路径标签",
            "iber-be-v1.0-baseline-aligned-runtime-2026-08-02",
        )
    )
    (root / "docs/IBER_BE_SERVER_GUIDE.md").write_text(guide, encoding="utf-8")


def test_complete_local_tree_is_ready_without_claiming_remote_state(tmp_path: Path) -> None:
    _complete_tree(tmp_path)

    report = audit_local_readiness(tmp_path, source_commit=SOURCE_COMMIT)

    assert report["status"] == "ready_waiting_for_server"
    assert report["local"]["ready"] is True
    assert report["local"]["source_commit"] == SOURCE_COMMIT
    assert report["local"]["baseline_sha256"] == BASELINE_SHA256
    assert report["local"]["dataset_sha256"] == DATASET_SHA256
    assert report["local"]["subset_sha256"] == SUBSET_SHA256
    assert report["remote"] == {
        "endpoint": "unresolved",
        "host_key": "unresolved",
        "os": "unresolved",
        "gpu": "unresolved",
        "driver": "unresolved",
        "reported_memory_mib": "unresolved",
        "disk": "unresolved",
        "network": "unresolved",
    }
    assert report["remote_verified"] is False
    for required in (
        "scripts/run_iber_canary.py",
        "scripts/evaluate_iber_stock.py",
        "scripts/cache_iber_evidence.py",
        "scripts/run_iber_probe.py",
        "scripts/train_iber.py",
        "scripts/evaluate_iber.py",
        "scripts/publish_iber_epoch.py",
        "scripts/restore_iber_checkpoint.py",
        "scripts/benchmark_iber.py",
        "scripts/run_iber_pipeline.py",
        "deploy/iber/publication-screen.template.json",
    ):
        assert required in report["local"]["required_files"]


def test_empty_tree_and_invalid_source_commit_are_not_ready(tmp_path: Path) -> None:
    report = audit_local_readiness(tmp_path, source_commit="not-a-commit")

    assert report["status"] == "not_ready"
    assert report["local"]["ready"] is False
    assert report["local"]["missing"]
    assert "source_commit" in report["local"]["invalid"]
    assert report["remote_verified"] is False


def test_audit_rejects_i_tber_results_identity(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    publication_path = tmp_path / "deploy/iber/publication-screen.template.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["results_branch"] = "itber-v1.1-results"
    publication["results_repo"] = "/data/uav/results/itber-v1.1"
    publication_path.write_text(json.dumps(publication), encoding="utf-8")

    report = audit_local_readiness(tmp_path, source_commit=SOURCE_COMMIT)

    assert report["status"] == "not_ready"
    assert "publication_identity" in report["local"]["invalid"]


def test_audit_rejects_guide_without_git_smart_http_fallback(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    guide_path = tmp_path / "docs/IBER_BE_SERVER_GUIDE.md"
    guide = guide_path.read_text(encoding="utf-8").replace(
        "http.version=HTTP/1.1", "missing-smart-http-contract"
    )
    guide_path.write_text(guide, encoding="utf-8")

    report = audit_local_readiness(tmp_path, source_commit=SOURCE_COMMIT)

    assert report["status"] == "not_ready"
    assert "http.version=HTTP/1.1" in report["local"]["invalid"]["guide_markers"]


def test_repository_guide_documents_monitoring_recovery_and_gate1_prohibition() -> None:
    guide = Path("docs/IBER_BE_SERVER_GUIDE.md").read_text(encoding="utf-8")
    for marker in (
        "SSH host key",
        "mirror-first",
        "immutable source checkout",
        "chmod 600",
        "passed_with_runtime_amendment",
        "run_iber_canary.py",
        "run_iber_pipeline.py",
        "pipeline-state.json",
        "nvidia-smi",
        "publish_iber_epoch.py",
        "restore_iber_checkpoint.py",
        "engineering_invalid",
        "scientific_failed",
        "Gate-1",
        "禁止绕过 Gate-1",
        "http.version=HTTP/1.1",
        "git-http.env",
        "GIT_CONFIG_COUNT",
        "兼容状态，不表示存在硬件差异",
        "`-seed0-amended` 是旧路径标签",
    ):
        assert marker in guide
    assert "github_pat_" not in guide
    assert "a1314520" not in guide

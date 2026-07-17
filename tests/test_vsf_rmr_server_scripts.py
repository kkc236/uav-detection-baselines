from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_setup_uses_persistent_storage_and_supported_torch_wheels():
    content = script("setup_vsf_rmr_server.sh")

    assert "STORAGE_ROOT" in content
    assert "nvidia-smi" in content
    assert "VisDrone" in content
    assert "torch==2.5.1" in content
    assert "torch==2.7.1" in content
    assert "https://download.pytorch.org/whl/cu128" in content
    assert "*5090*" in content
    assert "chmod 600" in content
    assert "github_pat_" not in content


def test_run_script_separates_variants_and_protects_final_shutdown():
    content = script("run_vsf_rmr_server.sh")

    assert 'VARIANT="${VARIANT:-vsf-rmr}"' in content
    assert 'SOURCE_BRANCH="${SOURCE_BRANCH:-codex/vsf-rmr}"' in content
    assert "vsf-rmr-matched-baseline-live" in content
    assert "vsf-rmr-rtdetr-l-live" in content
    assert "vsf-matched-baseline-last" in content
    assert "vsf-rmr-last" in content
    assert "supervise_vsf_rmr.py" in content
    assert "sync_experiment_checkpoint.py" in content
    assert "--retain 3" in content
    assert 'AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"' in content
    assert 'ENABLE_GITHUB_SYNC="${ENABLE_GITHUB_SYNC:-1}"' in content
    assert "published=1" in content
    assert "shutdown -h now" in content


def test_run_script_keeps_mutable_data_outside_git_checkout():
    content = script("run_vsf_rmr_server.sh")

    assert 'PROJECT_DIR="${PROJECT_DIR:-$STORAGE_ROOT/runs/vsf-rmr}"' in content
    assert 'LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs/$VARIANT}"' in content
    assert 'TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"' in content
    assert 'RESULTS_REPO="${RESULTS_REPO:-$STORAGE_ROOT/results-checkout}"' in content
    assert 'STATE_FILE="$STORAGE_ROOT/state/${STATE_KEY}_adaptive_state.json"' in content


def test_shell_scripts_use_strict_mode_and_have_no_embedded_secret():
    for name in ("setup_vsf_rmr_server.sh", "run_vsf_rmr_server.sh"):
        content = script(name)
        assert "set -euo pipefail" in content
        assert "GxpHy" not in content
        assert "fyZZ" not in content


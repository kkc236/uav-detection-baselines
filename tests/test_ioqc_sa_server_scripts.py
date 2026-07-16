from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_setup_script_is_gpu_generic_and_uses_persistent_storage():
    content = text("setup_ioqc_sa_server.sh")

    assert "STORAGE_ROOT" in content
    assert "nvidia-smi" in content
    assert "detect_gpu_profile" in content
    assert "VisDrone" in content
    assert "torch==2.5.1" in content
    assert "torch==2.7.1" in content
    assert "torchvision==0.22.1" in content
    assert "https://download.pytorch.org/whl/cu128" in content
    assert "*5090*" in content
    assert "ultralytics==8.4.90" not in content  # requirements.txt owns the Ultralytics pin
    assert "Expected an RTX 4090" not in content
    assert "chmod 600" in content


def test_run_script_locks_starts_watcher_and_verifies_final_publication():
    content = text("run_ioqc_sa_server.sh")

    assert "flock -n" in content
    assert "supervise_ioqc_sa.py" in content
    assert "sync_btdse_checkpoint.py" in content
    assert "ioqc-sa-rtdetr-l-live" in content
    assert 'SOURCE_BRANCH="${SOURCE_BRANCH:-codex/ioqc-sa}"' in content
    assert "--retain 3" in content
    assert "--asset-prefix ioqc-sa-last" in content
    assert "--release-name" in content
    assert "--once" in content
    assert 'AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"' in content
    assert 'ENABLE_GITHUB_SYNC="${ENABLE_GITHUB_SYNC:-1}"' in content
    assert 'SAVE_PERIOD="${SAVE_PERIOD:-1}"' in content
    assert 'OPTIMIZER="${OPTIMIZER:-AdamW}"' in content
    assert 'MIN_FREE_GIB="${MIN_FREE_GIB:-8}"' in content
    assert 'BATCH_LEVELS="${BATCH_LEVELS:-}"' in content
    assert 'nvidia-smi --query-gpu=index' in content
    assert '--save-period "$SAVE_PERIOD"' in content
    assert '--optimizer "$OPTIMIZER"' in content
    assert '--min-free-gib "$MIN_FREE_GIB"' in content
    assert 'supervisor_arguments+=(--batch-levels "$BATCH_LEVELS")' in content
    assert '[[ "$ENABLE_GITHUB_SYNC" == "1" ]]' in content
    assert '[[ "$AUTO_SHUTDOWN" == "1" ]]' in content
    assert "shutdown -h now" in content
    assert "ioqc_sa_launcher.pid" in content
    assert 'kill -TERM "$supervisor_pid"' in content


def test_run_script_keeps_runs_logs_and_secrets_outside_checkout():
    content = text("run_ioqc_sa_server.sh")

    assert 'PROJECT_DIR="${PROJECT_DIR:-$STORAGE_ROOT/runs/ioqc-sa}"' in content
    assert 'LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"' in content
    assert 'TOKEN_FILE="${TOKEN_FILE:-$STORAGE_ROOT/secrets/github_token}"' in content
    assert 'RESULTS_REPO="${RESULTS_REPO:-$STORAGE_ROOT/results-checkout}"' in content
    assert 'STATE_FILE="$STORAGE_ROOT/state/ioqc_sa_adaptive_state.json"' in content


def test_blockdata_launcher_uses_persistent_disk_and_preserves_existing_run():
    content = text("start_ioqc_sa_blockdata.sh")

    assert 'WORK_ROOT="${WORK_ROOT:-/root/blockdata/ioqc-sa}"' in content
    assert 'REPO_DIR="${REPO_DIR:-$WORK_ROOT/repo}"' in content
    assert 'STORAGE_ROOT="${STORAGE_ROOT:-$WORK_ROOT/storage}"' in content
    assert 'OPTIMIZER="${OPTIMIZER:-AdamW}"' in content
    assert 'MIN_FREE_GIB="${MIN_FREE_GIB:-8}"' in content
    assert 'ioqc_sa_launcher.pid' in content
    assert 'kill -0 "$existing_pid"' in content
    assert 'nohup env' in content
    assert 'scripts/run_ioqc_sa_server.sh' in content

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_server_runner_uses_fixed_protocol_and_separate_artifacts():
    content = (ROOT / "scripts" / "run_matched_baseline_server.sh").read_text(encoding="utf-8")

    assert 'BATCH="${BATCH:-8}"' in content
    assert "BATCH_LEVELS" not in content
    assert "INITIAL_BATCH" not in content
    assert "--batch \"$BATCH\"" not in content
    assert "scripts/train_rtdetr_matched_baseline.py" in content
    assert "--resume \"$resume_checkpoint\"" in content
    assert 'TAG="${TAG:-rtdetr-l-btdse-matched-baseline-live}"' in content
    assert 'ASSET_PREFIX="${ASSET_PREFIX:-matched-baseline-last}"' in content
    assert "scripts/sync_experiment_checkpoint.py" in content
    assert "CUDA out of memory" in content
    assert "NONFINITE_LOSS" in content
    assert "reducing batch" not in content.lower()
    assert "amp false" not in content.lower()


def test_server_runner_forwards_stop_to_the_training_process_group():
    content = (ROOT / "scripts" / "run_matched_baseline_server.sh").read_text(encoding="utf-8")

    assert 'train_pid=""' in content
    assert 'setsid "${command[@]}"' in content
    assert 'kill -TERM -- "-$train_pid"' in content
    assert "wait \"$train_pid\"" in content


def test_setup_script_creates_persistent_baseline_storage():
    content = (ROOT / "scripts" / "setup_matched_baseline_server.sh").read_text(encoding="utf-8")

    assert '"$STORAGE_ROOT/runs/matched-baseline"' in content
    assert '"$STORAGE_ROOT/datasets/VisDrone/images/train"' in content
    assert "torch==2.7.1" in content
    assert "torch==2.5.1" in content
    assert "scripts/prepare_visdrone.py" in content


def test_setup_uses_tsinghua_for_every_supported_package_source():
    content = (ROOT / "scripts" / "setup_matched_baseline_server.sh").read_text(encoding="utf-8")

    assert 'PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"' in content
    assert 'PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"' in content
    assert 'pip install --index-url "$PYPI_INDEX_URL"' in content
    assert 'pip install --index-url "$PYPI_INDEX_URL" -r "$REPO_DIR/requirements.txt"' in content
    assert "https://download.pytorch.org/whl/cu121" in content
    assert "https://download.pytorch.org/whl/cu128" in content


def test_server_guide_uses_tsinghua_apt_and_pypi_with_documented_exceptions():
    content = (ROOT / "docs" / "BASELINE.md").read_text(encoding="utf-8")

    assert "https://mirrors.tuna.tsinghua.edu.cn/ubuntu" in content
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in content
    assert "PyTorch CUDA轮子" in content
    assert "GitHub仓库" in content
    assert "VisDrone数据压缩包" in content


def test_server_guide_is_complete_and_uses_real_scripts():
    content = (ROOT / "docs" / "BASELINE.md").read_text(encoding="utf-8")

    required = (
        "codex/matched-baseline",
        "setup_matched_baseline_server.sh",
        "run_matched_baseline_server.sh",
        "github_token",
        "matched_baseline_training.log",
        "matched_baseline_github_sync.json",
        "rtdetr-l-btdse-matched-baseline-live",
        "results.csv",
        "last.pt",
        "真正断点恢复",
    )
    for value in required:
        assert value in content

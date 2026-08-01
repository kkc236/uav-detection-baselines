from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from scripts.run_lpr_g_paired import (
    SCREEN_ORDER,
    build_arm_command,
    normalize_python_executable,
    next_stage,
    run_model_canary,
    set_stock_model_class_count,
)


ROOT = Path(__file__).resolve().parents[1]


def test_only_seed0_control_then_lprg_is_scheduled() -> None:
    assert SCREEN_ORDER == ((0, "control"), (0, "lprg"))


def test_python_executable_path_is_not_symlink_resolved(
    tmp_path: Path, monkeypatch
) -> None:
    relative = Path("venvs/rtdetr-lpr-py310/bin/python")

    def reject_resolve(self: Path, *args, **kwargs) -> Path:
        raise AssertionError("the venv launcher must not be symlink-resolved")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "resolve", reject_resolve)

    assert normalize_python_executable(relative) == tmp_path / relative


def test_scientific_entrypoints_configure_cublas_before_torch_import() -> None:
    entrypoints = (
        "run_lpr_g_paired.py",
        "prepare_lpr_g_protocol.py",
        "train_rtdetr_lpr_g.py",
        "evaluate_lpr_g.py",
        "benchmark_lpr_g.py",
    )

    for name in entrypoints:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        config_offset = source.index('CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"')
        torch_offset = source.index("import torch")
        assert config_offset < torch_offset, name


def test_stock_canary_class_count_comes_from_locked_model_yaml() -> None:
    class StockModel:
        yaml = {"nc": 10}

    model = StockModel()
    set_stock_model_class_count(model)

    assert model.nc == 10


def test_model_canary_keeps_forward_cache_usable_for_backward() -> None:
    source = inspect.getsource(run_model_canary)

    assert "torch.use_deterministic_algorithms(True, warn_only=True)" in source
    assert "torch.no_grad()" in source
    assert "torch.inference_mode()" not in source


def test_arm_command_exposes_no_scientific_override(tmp_path: Path) -> None:
    command = build_arm_command(
        python=Path("/data/uav/venvs/rtdetr-lpr-py310/bin/python"),
        protocol=tmp_path / "protocol-seed0.json",
        initial_state=tmp_path / "initial-state-seed0.pt",
        project=tmp_path / "runs",
        stage="screen",
        variant="lprg",
    )

    assert command[1] == "scripts/train_rtdetr_lpr_g.py"
    assert command[command.index("--seed") + 1] == "0"
    assert not {"--batch", "--workers", "--optimizer", "--mosaic", "--epochs"}.intersection(
        command
    )


def test_formal_launch_requires_passed_screen() -> None:
    assert next_stage({"status": "passed"}, through_formal=True) == "formal"
    assert next_stage({"status": "scientific_failed"}, through_formal=True) == "stop"
    assert next_stage({"status": "engineering_invalid"}, through_formal=True) == "repair"


def test_supervisor_cli_exposes_only_operational_authority() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_lpr_g_paired.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset-root" in result.stdout
    assert "--through-formal" in result.stdout
    assert "--epochs" not in result.stdout
    assert "--batch" not in result.stdout
    assert "--optimizer" not in result.stdout

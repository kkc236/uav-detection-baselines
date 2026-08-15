from pathlib import Path

from scripts.run_lpr_paired_screen import SCREEN_ORDER, build_arm_command, build_pairs_manifest


def test_screen_order_is_frozen_to_reduce_order_bias() -> None:
    assert SCREEN_ORDER == (
        (0, "control"),
        (0, "lpr"),
        (1, "lpr"),
        (1, "control"),
        (2, "control"),
        (2, "lpr"),
    )


def test_arm_command_exposes_no_scientific_overrides(tmp_path: Path) -> None:
    command = build_arm_command(
        python=Path("/venv/bin/python"),
        protocol_dir=tmp_path / "protocol",
        project=tmp_path / "runs",
        seed=1,
        variant="lpr",
    )

    assert command[:2] == [str(Path("/venv/bin/python")), "scripts/train_rtdetr_lpr.py"]
    assert command[command.index("--variant") + 1] == "lpr"
    assert command[command.index("--stage") + 1] == "screen"
    assert command[command.index("--seed") + 1] == "1"
    assert not {
        "--batch",
        "--workers",
        "--optimizer",
        "--mosaic",
        "--amp",
        "--imgsz",
    }.intersection(command)


def test_pairs_manifest_names_all_three_control_lpr_pairs(tmp_path: Path) -> None:
    manifest = build_pairs_manifest(tmp_path)

    assert set(manifest["pairs"]) == {"0", "1", "2"}
    assert manifest["pairs"]["0"]["control"].endswith("screen-seed0-control-lpr-v1")
    assert manifest["pairs"]["2"]["lpr"].endswith("screen-seed2-lpr-lpr-v1")

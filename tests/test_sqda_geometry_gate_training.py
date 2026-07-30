from __future__ import annotations

import pytest

from scripts.train_rtdetr_sqda_geometry_gate import RUN_NAMES, build_parser, build_settings


@pytest.mark.parametrize(("gate", "epochs"), [("g1", 3), ("g2", 10)])
def test_geometry_gate_settings_are_fixed_and_require_inherited_adapter(
    tmp_path,
    gate: str,
    epochs: int,
) -> None:
    args = build_parser().parse_args(
        [
            "--gate",
            gate,
            "--checkpoint",
            str(tmp_path / "baseline.pt"),
            "--adapter-checkpoint",
            str(tmp_path / "g2.pt"),
            "--data",
            str(tmp_path / "VisDrone.yaml"),
            "--project",
            str(tmp_path / "runs"),
        ]
    )

    settings = build_settings(args)

    assert settings["epochs"] == epochs
    assert settings["name"] == RUN_NAMES[gate]
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["seed"] == 0
    assert settings["deterministic"] is True
    assert settings["amp"] is True
    assert settings["nms"] is False
    assert settings["max_det"] == 300
    assert settings["optimizer"] == "AdamW"
    assert settings["lr0"] == pytest.approx(1e-4)
    assert settings["weight_decay"] == pytest.approx(1e-4)
    assert settings["freeze"] == list(range(29))


def test_geometry_gate_cli_excludes_training_protocol_mutations() -> None:
    options = {action.dest for action in build_parser()._actions}
    assert not {
        "epochs",
        "seed",
        "imgsz",
        "batch",
        "optimizer",
        "lr0",
        "amp",
        "max_det",
    }.intersection(options)

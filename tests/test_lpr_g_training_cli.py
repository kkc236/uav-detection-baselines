from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.train_rtdetr_lpr_g import (
    FORMAL_EPOCHS,
    FROZEN_PROTOCOL,
    SCREEN_CUTOFF_EPOCHS,
    SCREEN_SCHEDULE_EPOCHS,
    build_parser,
    build_settings,
    cutoff_after_verified_publication,
    validate_resume_authority,
    write_lpr_g_diagnostics,
)
from src.lpr_protocol import EXPECTED_ENVIRONMENT


def _manifest(tmp_path: Path) -> dict:
    state = tmp_path / "initial-state-seed0.pt"
    state.touch()
    return {
        "format_version": 2,
        "seed": 0,
        "data": {
            "screen": {"path": str(tmp_path / "screen.yaml")},
            "formal": {"path": str(tmp_path / "formal.yaml")},
        },
        "initial_state": {"path": str(state)},
    }


def _args(tmp_path: Path, *extra: str):
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "protocol-seed0.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--variant", "lprg",
            "--stage", "screen",
            "--seed", "0",
            "--protocol-manifest", str(manifest_path),
            "--initial-state", manifest["initial_state"]["path"],
            "--project", str(tmp_path / "runs"),
            "--token-file", str(tmp_path / "github_token"),
            "--tag", "lpr-g-v2-live",
            "--asset-prefix", "screen-seed0-lprg",
            *extra,
        ]
    )
    return args, manifest


def test_screen_and_formal_settings_are_frozen() -> None:
    manifest = {"data": {"screen": {"path": "screen.yaml"}, "formal": {"path": "full.yaml"}}}
    args = SimpleNamespace(
        stage="screen",
        seed=0,
        variant="lprg",
        project=Path("runs"),
        name=None,
        resume=None,
        preflight=False,
    )

    screen = build_settings(args, manifest)
    args.stage = "formal"
    formal = build_settings(args, manifest)

    assert screen["epochs"] == 50
    assert formal["epochs"] == 100
    assert screen["data"] == "screen.yaml"
    assert formal["data"] == "full.yaml"
    assert FROZEN_PROTOCOL["batch"] == 8
    assert FROZEN_PROTOCOL["workers"] == 8
    assert FROZEN_PROTOCOL["optimizer"] == "MuSGD"
    assert FROZEN_PROTOCOL["save_period"] == 1
    assert FROZEN_PROTOCOL["mosaic"] == 1.0


def test_screen_keeps_50_epoch_schedule_but_stops_at_verified_30() -> None:
    assert SCREEN_SCHEDULE_EPOCHS == 50
    assert SCREEN_CUTOFF_EPOCHS == 30
    assert FORMAL_EPOCHS == 100

    trainer = SimpleNamespace(
        epoch=29,
        stop=False,
        lpr_g_publication_record={"completed_epoch": 30, "verified": True},
    )
    args = SimpleNamespace(stage="screen", preflight=False)

    assert cutoff_after_verified_publication(trainer, args=args) is True
    assert trainer.stop is True


@pytest.mark.parametrize(
    "record",
    (
        {"completed_epoch": 30, "verified": False},
        {"completed_epoch": 29, "verified": True},
    ),
)
def test_screen_cutoff_rejects_unverified_or_wrong_epoch_publication(record) -> None:
    trainer = SimpleNamespace(
        epoch=29,
        stop=False,
        lpr_g_publication_record=record,
    )

    with pytest.raises(RuntimeError, match="verified epoch 30"):
        cutoff_after_verified_publication(
            trainer,
            args=SimpleNamespace(stage="screen", preflight=False),
        )


def test_formal_and_preflight_do_not_use_screen_cutoff() -> None:
    trainer = SimpleNamespace(epoch=29, stop=False)

    assert cutoff_after_verified_publication(
        trainer,
        args=SimpleNamespace(stage="formal", preflight=False),
    ) is False
    assert cutoff_after_verified_publication(
        trainer,
        args=SimpleNamespace(stage="screen", preflight=True),
    ) is False


def test_epoch_save_callbacks_publish_before_cutoff() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_rtdetr_lpr_g.py"
    ).read_text(encoding="utf-8")

    diagnostics = source.index("lambda current: write_lpr_g_diagnostics")
    audit = source.index('"on_model_save", write_common_state_audit')
    publication = source.index("lambda current: publish_current_epoch")
    cutoff = source.index("lambda current: cutoff_after_verified_publication")

    assert diagnostics < audit < publication < cutoff


def test_nonzero_seed_is_rejected_before_manifest_access() -> None:
    args = SimpleNamespace(stage="screen", seed=1)
    with pytest.raises(ValueError, match="seed0"):
        build_settings(args, {})


def test_resume_rejects_cross_stage_checkpoint(tmp_path: Path) -> None:
    args, manifest = _args(tmp_path)
    run = tmp_path / "resume-run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "last.pt"
    torch.save({"epoch": 3, "optimizer": {}, "ema": {}}, checkpoint)
    runtime = {
        "protocol": FROZEN_PROTOCOL,
        "authority": manifest,
        "environment": dict(EXPECTED_ENVIRONMENT),
        "variant": "lprg",
        "stage": "screen",
        "seed": 0,
        "epochs": 50,
        "initial_state": str(args.initial_state.resolve()),
    }
    (run / "lpr_g_protocol.json").write_text(json.dumps(runtime), encoding="utf-8")
    args.resume = checkpoint
    args.stage = "formal"

    with pytest.raises(ValueError, match="stage"):
        validate_resume_authority(args, manifest, dict(EXPECTED_ENVIRONMENT))


def test_diagnostics_write_ap75_private_distributions_and_null_control(tmp_path: Path) -> None:
    refiner = SimpleNamespace(
        last_quality=torch.tensor([[[0.1], [0.9]]]),
        last_gate=torch.tensor([[[0.2], [0.01]]]),
        last_residual=torch.tensor([[[0.1, -0.1, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0]]]),
    )
    method_model = SimpleNamespace(
        model=[SimpleNamespace(decoder=SimpleNamespace(lpr_g_refiner=refiner))],
        last_lpr_g_losses={"loss_bbox_refine": torch.tensor(0.3), "loss_giou_refine": torch.tensor(0.4)},
    )
    trainer = SimpleNamespace(
        model=method_model,
        epoch=0,
        args=SimpleNamespace(epochs=50),
        save_dir=tmp_path,
        tloss=torch.tensor([1.0, 2.0, 3.0]),
        validator=SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(map75=0.05))),
        last_gradient_norms={"gradient_norm": 4.0, "lpr_g_gradient_norm": 5.0},
    )

    write_lpr_g_diagnostics(trainer, variant="lprg")

    record = json.loads((tmp_path / "lpr_g_diagnostics.jsonl").read_text(encoding="utf-8"))
    assert record["epoch"] == 1
    assert record["map75"] == pytest.approx(0.05)
    assert record["gate_p95"] > 0
    assert record["residual_rms"] > 0
    assert record["lpr_g_gradient_norm"] == 5.0

    control_dir = tmp_path / "control"
    trainer.save_dir = control_dir
    write_lpr_g_diagnostics(trainer, variant="control")
    control = json.loads((control_dir / "lpr_g_diagnostics.jsonl").read_text(encoding="utf-8"))
    assert control["gate_p95"] is None
    assert control["loss_bbox_refine"] is None


def test_scientific_cli_requires_github_publication_authority() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--variant", "lprg",
                "--stage", "screen",
                "--seed", "0",
                "--protocol-manifest", "protocol.json",
                "--initial-state", "initial.pt",
            ]
        )

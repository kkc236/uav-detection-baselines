from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn
import yaml

from scripts import train_dcf_fdr as launcher
from scripts import train_persistent_gradient_dcf_fdr as persistent_gradient
from scripts import train_transient_dcf_fdr as transient
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS
from src.fdr_head import (
    DistributionConditionedFeedback,
    FDRDeformableTransformerDecoder,
)


def test_dcf_launcher_binds_one_frozen_formal_method(tmp_path: Path) -> None:
    settings = launcher.build_settings(
        data_yaml=tmp_path / "data.yaml",
        output_root=tmp_path / "runs",
    )
    assert launcher.DCF_FDR_CONFIG.name == "rtdetr-l-dcf-fdr.yaml"
    assert settings["model"] == str(launcher.DCF_FDR_CONFIG.resolve())
    assert settings["name"] == "formal-seed0-dcf-fdr-v1"
    assert settings["epochs"] == FORMAL_EPOCHS
    assert settings["seed"] == 0
    for key, value in FROZEN_SETTINGS.items():
        if key not in {"model", "epochs", "seed", "save_period"}:
            assert settings[key] == value

    clean = launcher.build_settings(
        data_yaml=tmp_path / "data.yaml",
        output_root=tmp_path / "runs",
        arm="clean",
    )
    assert clean["model"] == str(launcher.CLEAN_FDR_CONFIG.resolve())
    assert clean["name"] == "formal-seed0-clean-fdr-v1"


def test_dcf_launch_record_has_distinct_method_identity(tmp_path: Path) -> None:
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"state")
    record = launcher.build_launch_record(
        source_identity={"commit": "abc"},
        initial_state_path=initial,
        dataset={"sha256": "dataset"},
        settings={"seed": 0, "model": str(launcher.DCF_FDR_CONFIG)},
    )
    assert record["method"] == "dcf_fdr"
    assert record["config"]["path"] == str(launcher.DCF_FDR_CONFIG.resolve())


class _Layer(nn.Module):
    pass


class _StockDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(6)])
        self.hidden_dim = 16
        self.num_layers = 6
        self.eval_idx = 5


def _feedback_model() -> nn.Module:
    model = nn.Module()
    model.decoder = FDRDeformableTransformerDecoder.from_stock(
        _StockDecoder(),
        pre_bbox_head=nn.Linear(16, 4),
        distribution_feedback=DistributionConditionedFeedback(
            16, private_seed=10_001
        ),
    )
    return model


def _fake_transient_trainer(*, paper_epoch: int, best_fitness: float):
    model = _feedback_model()
    return SimpleNamespace(
        epoch=paper_epoch - 1,
        epochs=FORMAL_EPOCHS,
        model=model,
        ema=SimpleNamespace(ema=deepcopy(model)),
        best_fitness=best_fitness,
    )


def test_transient_launcher_binds_frozen_formal100_identity(tmp_path: Path) -> None:
    settings = transient.build_settings(
        data_yaml=tmp_path / "data.yaml", output_root=tmp_path / "runs"
    )
    assert Path(settings["model"]).name == "rtdetr-l-transient-dcf-fdr.yaml"
    assert settings["epochs"] == 100
    assert settings["seed"] == 0
    assert "resume" not in settings
    assert transient.build_schedule_record()["full_through_ratio"] == "2/3"
    assert transient.build_schedule_record()["off_from_ratio"] == "3/4"
    assert "resume" not in transient.build_parser().format_help()


def test_epoch75_resets_best_once_and_writes_eligible_evidence(
    tmp_path: Path,
) -> None:
    trainer = _fake_transient_trainer(paper_epoch=75, best_fitness=0.9)
    evidence = tmp_path / "transient-dcf-schedule.jsonl"

    transient.configure_transient_epoch(trainer, evidence)

    assert trainer.best_fitness is None
    assert trainer.transient_tail_best_reset is True
    with pytest.raises(ValueError, match="duplicate paper epoch"):
        transient.configure_transient_epoch(trainer, evidence)
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert rows[-1]["checkpoint_eligible"] is True
    assert rows[-1]["live_scale"] == rows[-1]["ema_scale"] == 0.0


def test_pre75_epoch_does_not_reset_best_and_records_frozen_decay(
    tmp_path: Path,
) -> None:
    trainer = _fake_transient_trainer(paper_epoch=67, best_fitness=0.9)
    evidence = tmp_path / "transient-dcf-schedule.jsonl"

    transient.configure_transient_epoch(trainer, evidence)

    assert trainer.best_fitness == 0.9
    assert not hasattr(trainer, "transient_tail_best_reset")
    row = json.loads(evidence.read_text().strip())
    assert row["paper_epoch"] == 67
    assert row["frozen"] is True
    assert 0.0 < row["scale"] < 1.0
    assert row["checkpoint_eligible"] is False


def test_persistent_gradient_launcher_binds_all_on_formal100(tmp_path: Path) -> None:
    settings = persistent_gradient.build_settings(
        data_yaml=tmp_path / "data.yaml", output_root=tmp_path / "runs"
    )
    assert Path(settings["model"]).name == (
        "rtdetr-l-persistent-gradient-dcf-fdr.yaml"
    )
    assert settings["epochs"] == 100
    assert settings["seed"] == 0
    assert settings["save_period"] == -1
    assert "resume" not in settings
    assert "resume" not in persistent_gradient.build_parser().format_help()


def test_persistent_gradient_config_matches_transient_epoch1_to66() -> None:
    root = Path(__file__).resolve().parents[1]
    persistent_cfg = yaml.safe_load(
        (
            root / "configs/rtdetr-l-persistent-gradient-dcf-fdr.yaml"
        ).read_text()
    )
    transient_cfg = yaml.safe_load(
        (root / "configs/rtdetr-l-transient-dcf-fdr.yaml").read_text()
    )
    assert persistent_cfg == transient_cfg


def test_persistent_authority_excludes_every_transient_behavior() -> None:
    authority = persistent_gradient.build_method_record()
    assert authority == {
        "kind": "persistent_gradient_dcf_v1",
        "scale": "1.0_all_epochs",
        "trainable": "all_epochs",
        "checkpoint_eligible_from_epoch": 1,
        "resume_policy": "restart_from_epoch_0",
    }
    source = Path(persistent_gradient.__file__).read_text(encoding="utf-8")
    assert "configure_transient_epoch" not in source
    assert "freeze_distribution_feedback" not in source
    assert "best_fitness = None" not in source

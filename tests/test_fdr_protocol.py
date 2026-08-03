from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from src.fdr_protocol import (
    FDR_PROTOCOL,
    FDR_PROTOCOL_SHA256,
    PRIVATE_SEED,
    build_fdr_initial_state,
    build_run_identity,
    canonical_json_bytes,
    copy_public_pre_head,
    initialize_private_module,
    load_fdr_initial_state,
    partition_state_dicts,
    public_state_sha256,
    validate_optimizer_coverage,
    validate_resume_authority,
    write_create_only_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class _Control(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.pre_bbox_head = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))


class _Method(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.pre_bbox_head = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.distribution_finals = nn.ModuleList([nn.Linear(4, 132) for _ in range(6)])


def _paired_modules() -> tuple[_Control, _Method]:
    torch.manual_seed(0)
    control = _Control()
    torch.manual_seed(123)
    method = _Method()
    method.backbone.load_state_dict(control.backbone.state_dict())
    copy_public_pre_head(control.pre_bbox_head, method.pre_bbox_head)
    initialize_private_module(
        method.distribution_finals,
        private_seed=PRIVATE_SEED,
        zero_final_layers=tuple(method.distribution_finals),
    )
    return control, method


def test_frozen_protocol_contains_complete_baseline_and_fdr_contract() -> None:
    assert FDR_PROTOCOL["dfine_commit"] == "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
    assert FDR_PROTOCOL["reg_max"] == 32
    assert FDR_PROTOCOL["reg_scale"] == 4.0
    assert FDR_PROTOCOL["up"] == 0.5
    assert FDR_PROTOCOL["loss_weights"] == {
        "vfl": 1.0,
        "bbox": 5.0,
        "giou": 2.0,
        "fgl": 0.15,
    }
    assert FDR_PROTOCOL["excluded"] == [
        "DDF",
        "GO-LSD",
        "teacher",
        "LQE",
        "target_gating",
    ]
    assert FDR_PROTOCOL["environment"] == {
        "model": "Ultralytics RT-DETR-L",
        "ultralytics": "8.4.90",
        "gpu": "NVIDIA GeForce RTX 4090",
        "driver": "550.142",
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda": "12.1",
    }
    assert FDR_PROTOCOL["dataset"] == {
        "name": "VisDrone",
        "train_images": 6471,
        "val_images": 548,
        "classes": 10,
        "sha256": "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB",
        "screen_train_images": 647,
        "screen_sha256": "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0",
    }
    assert FDR_PROTOCOL["training"] == {
        "pretrained": False,
        "screen_epochs": 30,
        "formal_epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": 0,
        "amp": True,
        "amp_scale": 128.0,
        "seeds": [0],
        "deterministic": True,
        "cache": False,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 64,
        "cos_lr": False,
        "queries": 300,
        "max_det": 300,
        "nms": False,
    }
    assert FDR_PROTOCOL["augmentation"] == {
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    }
    assert len(FDR_PROTOCOL_SHA256) == 64
    assert FDR_PROTOCOL_SHA256 == public_state_sha256(FDR_PROTOCOL)


def test_partition_and_initial_state_preserve_public_bytes() -> None:
    control, method = _paired_modules()
    public, private = partition_state_dicts(
        control.state_dict(),
        method.state_dict(),
        private_prefixes=("distribution_finals.",),
    )
    assert set(public) == set(control.state_dict())
    assert len(private) == 12
    assert all(name.startswith("distribution_finals.") for name in private)
    assert public_state_sha256(public) == public_state_sha256(control.state_dict())

    artifact = build_fdr_initial_state(
        control.state_dict(),
        method.state_dict(),
        private_prefixes=("distribution_finals.",),
        metadata={"seed": 0},
    )
    assert artifact["fingerprints"]["public"] == public_state_sha256(control.state_dict())
    assert artifact["fingerprints"]["private"] == public_state_sha256(private)

    fresh_control = _Control()
    fresh_method = _Method()
    load_fdr_initial_state(fresh_control, artifact, variant="control")
    load_fdr_initial_state(fresh_method, artifact, variant="fdr")
    assert public_state_sha256(fresh_control.state_dict()) == artifact["fingerprints"]["public"]
    assert public_state_sha256(fresh_method.state_dict()) == public_state_sha256(method.state_dict())


def test_partition_rejects_public_value_or_private_name_drift() -> None:
    control, method = _paired_modules()
    bad = deepcopy(method.state_dict())
    bad["backbone.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="public tensor differs"):
        partition_state_dicts(
            control.state_dict(), bad, private_prefixes=("distribution_finals.",)
        )

    bad["backbone.weight"] = control.state_dict()["backbone.weight"].clone()
    bad["unapproved.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unapproved private"):
        partition_state_dicts(
            control.state_dict(), bad, private_prefixes=("distribution_finals.",)
        )


def test_private_rng_fork_preserves_public_rng_and_zeros_six_finals() -> None:
    module = nn.ModuleList([nn.Linear(4, 132) for _ in range(6)])
    torch.manual_seed(77)
    before = torch.random.get_rng_state().clone()
    initialize_private_module(
        module,
        private_seed=PRIVATE_SEED,
        zero_final_layers=tuple(module),
    )
    after = torch.random.get_rng_state()

    assert torch.equal(before, after)
    assert len(module) == 6
    for layer in module:
        assert torch.count_nonzero(layer.weight) == 0
        assert torch.count_nonzero(layer.bias) == 0


def test_pre_head_copy_is_exact_and_rejects_shape_drift() -> None:
    source = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    target = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    copied_hash = copy_public_pre_head(source, target)
    assert copied_hash == public_state_sha256(source.state_dict())
    assert copied_hash == public_state_sha256(target.state_dict())

    with pytest.raises(ValueError, match="pre-head state keys"):
        copy_public_pre_head(source, nn.Linear(4, 4))


def test_optimizer_coverage_requires_every_trainable_parameter_exactly_once() -> None:
    _, method = _paired_modules()
    complete = torch.optim.SGD(method.parameters(), lr=0.01)
    assert validate_optimizer_coverage(method, complete)["parameter_count"] == sum(
        parameter.numel() for parameter in method.parameters()
    )

    incomplete = torch.optim.SGD(method.backbone.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="optimizer coverage"):
        validate_optimizer_coverage(method, incomplete)

    parameters = list(method.parameters())
    duplicate = type("_OptimizerFixture", (), {
        "param_groups": [{"params": parameters}, {"params": [parameters[0]]}]
    })()
    with pytest.raises(ValueError, match="duplicate"):
        validate_optimizer_coverage(method, duplicate)


def test_manifest_is_canonical_create_only_and_identity_bound(tmp_path: Path) -> None:
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    identity = build_run_identity(source, stage="screen", variant="fdr", seed=0)
    assert identity["source_sha256"] == public_state_sha256(source)
    assert identity["protocol_sha256"] == FDR_PROTOCOL_SHA256
    assert identity["run_id"].startswith("fdr-screen-seed0-")

    payload = {"identity": identity, "protocol": FDR_PROTOCOL}
    destination = tmp_path / "protocol.json"
    write_create_only_manifest(destination, payload)
    assert destination.read_bytes() == canonical_json_bytes(payload) + b"\n"
    with pytest.raises(FileExistsError):
        write_create_only_manifest(destination, payload)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("source_sha256", "A" * 64),
        ("protocol_sha256", "B" * 64),
        ("run_id", "foreign-run"),
        ("stage", "formal"),
        ("variant", "control"),
        ("seed", 1),
    ],
)
def test_resume_authority_rejects_every_identity_mismatch(field: str, replacement: object) -> None:
    expected = build_run_identity(
        {"git_commit": "a" * 40, "tree_sha256": "C" * 64},
        stage="screen",
        variant="fdr",
        seed=0,
    )
    checkpoint = dict(expected)
    checkpoint[field] = replacement
    with pytest.raises(ValueError, match=field):
        validate_resume_authority(checkpoint, expected)


def test_prepare_script_help_has_only_authority_inputs() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_fdr_protocol.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-commit" in result.stdout
    assert "--source-tree-sha256" in result.stdout
    assert "--initial-state" in result.stdout
    assert "--output" in result.stdout
    for forbidden in ("--reg-max", "--reg-scale", "--fgl-weight", "--seed"):
        assert forbidden not in result.stdout

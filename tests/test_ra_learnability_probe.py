from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import src.ra_learnability_probe as probe
from scripts.run_ra_learnability_probe import (
    load_mature_fdr_public_state,
    support_objective,
)
from src.ra_glgm_protocol import RA_GLGM_PRIVATE_PREFIX
from src.rtdetr_ra_glgm import RAGLGMDetectionModel
from src.ra_learnability_probe import (
    LEARNABILITY_GATE_SHA256,
    MATURE_FDR_CHECKPOINT_SHA256,
    deterministic_probe_split,
    evaluate_learnability_gate,
    freeze_for_support_probe,
    summarize_targets,
    validate_learnability_report,
)
from src.fdr_protocol import canonical_json_bytes


def _passing_evidence() -> dict:
    targets = {
        "batches": 10,
        "batches_with_targets": 10,
        "batches_with_targets_fraction": 1.0,
        "difficulty_count": 100,
        "difficulty_mean": 0.6,
        "difficulty_std": 0.15,
        "target_mean": 0.02,
        "positive_pixel_fraction": 0.08,
        "valid_fraction": 0.95,
    }
    return {
        "authority": {
            "protocol_sha256": probe.RA_EXPERIMENT_PROTOCOL_SHA256,
            "source": {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
            "dataset_authority": {"root": "/VisDrone"},
            "gpu_uuid": "GPU-fixed",
        },
        "mature_fdr_checkpoint": {"sha256": MATURE_FDR_CHECKPOINT_SHA256},
        "split": {
            "screen_count": 647,
            "screen_sha256": probe.SCREEN_SUBSET_SHA256,
            "train_count": 518,
            "dev_count": 129,
            "disjoint": True,
        },
        "freeze": {
            "trainable_names": [f"{RA_GLGM_PRIVATE_PREFIX}support_head.weight"],
            "public_sha256_before": "A" * 64,
            "public_sha256_after": "A" * 64,
        },
        "targets": {"train": targets, "dev": deepcopy(targets)},
        "losses": {
            "train_epochs": [0.10, 0.085, 0.075],
            "dev": [0.11, 0.105, 0.097, 0.09],
        },
    }


def test_learnability_gate_is_frozen_and_passing_evidence_authorizes_smoke2() -> None:
    report = evaluate_learnability_gate(_passing_evidence())

    assert len(LEARNABILITY_GATE_SHA256) == 64
    assert all(report["checks"].values())
    assert report["train_loss_relative_reduction"] == pytest.approx(0.25)
    assert report["dev_loss_relative_reduction"] == pytest.approx(1 - 0.09 / 0.11)
    assert report["passed"] is True
    assert report["smoke2_eligible"] is True
    assert "not detector accuracy" in report["scientific_scope"]


def test_checked_in_learnability_preregistration_binds_checkpoint_and_gate() -> None:
    path = Path(__file__).resolve().parents[1] / "research" / "ra_glgm" / "RA_LEARNABILITY_PREREGISTRATION.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["authority"] == (
        "exact RA protocol manifest source, dataset authority, and physical GPU UUID"
    )
    assert payload["checkpoint_sha256"] == MATURE_FDR_CHECKPOINT_SHA256
    assert payload["screen_sha256"] == probe.SCREEN_SUBSET_SHA256
    assert payload["gate"] == probe.LEARNABILITY_GATE
    assert payload["optimizer"] == probe.PROBE_OPTIMIZER
    assert payload["probe_seed"] == probe.PROBE_SEED


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda evidence: evidence["mature_fdr_checkpoint"].update(sha256="0" * 64),
            "mature_fdr_checkpoint_sha256",
        ),
        (
            lambda evidence: evidence["targets"]["dev"].update(difficulty_std=0.0),
            "dev_targets_non_degenerate",
        ),
        (
            lambda evidence: evidence["losses"].update(dev=[0.11, 0.109, 0.108, 0.107]),
            "holdout_loss_reduction_at_least_5_percent",
        ),
        (
            lambda evidence: evidence["freeze"].update(public_sha256_after="B" * 64),
            "public_fdr_state_unchanged",
        ),
        (
            lambda evidence: evidence["freeze"].update(
                trainable_names=["model.0.conv.weight"]
            ),
            "support_private_parameters_only",
        ),
    ],
)
def test_gate_fails_closed_for_authority_degeneracy_or_holdout_failure(
    mutation, failed_check: str
) -> None:
    evidence = _passing_evidence()
    mutation(evidence)

    report = evaluate_learnability_gate(evidence)

    assert report["checks"][failed_check] is False
    assert report["smoke2_eligible"] is False


def test_target_summary_uses_weighted_pixels_and_difficulty_variance() -> None:
    summary = summarize_targets(
        [
            {
                "target_sum": 2.0,
                "target_pixels": 100,
                "positive_pixels": 10,
                "valid_pixels": 90,
                "total_pixels": 100,
                "difficulty_count": 2,
                "difficulty_sum": 1.0,
                "difficulty_square_sum": 0.58,
            },
            {
                "target_sum": 1.0,
                "target_pixels": 50,
                "positive_pixels": 5,
                "valid_pixels": 40,
                "total_pixels": 50,
                "difficulty_count": 1,
                "difficulty_sum": 0.8,
                "difficulty_square_sum": 0.64,
            },
        ]
    )

    assert summary["target_mean"] == pytest.approx(3 / 150)
    assert summary["positive_pixel_fraction"] == pytest.approx(15 / 150)
    assert summary["valid_fraction"] == pytest.approx(130 / 150)
    assert summary["difficulty_mean"] == pytest.approx(0.6)
    assert summary["difficulty_std"] == pytest.approx((1.22 / 3 - 0.36) ** 0.5)
    assert summary["batches_with_targets_fraction"] == 1.0


def test_deterministic_split_is_disjoint_and_independent_of_input_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "VisDrone"
    images = root / "images" / "train"
    images.mkdir(parents=True)
    paths = []
    for index in range(647):
        path = images / f"{index:04d}.jpg"
        path.write_bytes(b"x")
        paths.append(path)
    monkeypatch.setattr(probe, "subset_signature", lambda values, root: probe.SCREEN_SUBSET_SHA256)

    train_a, dev_a, record_a = deterministic_probe_split(paths, root=root)
    train_b, dev_b, record_b = deterministic_probe_split(reversed(paths), root=root)

    assert train_a == train_b
    assert dev_a == dev_b
    assert record_a == record_b
    assert len(train_a) == 518 and len(dev_a) == 129
    assert not set(train_a) & set(dev_a)


class _PrivateRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.support_head = nn.Linear(2, 1)
        self.router = nn.Linear(2, 2)
        self.output_projection = nn.Linear(2, 2, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.public = nn.Linear(2, 2)
        self.model = nn.ModuleList([nn.Identity() for _ in range(29)])
        self.model[28] = nn.Module()
        self.model[28].ra_glgm = _PrivateRA()


def test_freeze_exposes_only_support_private_path() -> None:
    model = _ToyModel()

    record = freeze_for_support_probe(model)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == set(record["trainable_names"])
    assert trainable
    assert all(name.startswith(RA_GLGM_PRIVATE_PREFIX) for name in trainable)
    assert not any(name.endswith("alpha") for name in trainable)
    assert not any(name.endswith("output_projection.weight") for name in trainable)
    assert not model.public.weight.requires_grad


def test_checkpoint_loader_rejects_wrong_bytes_before_model_deserialization(
    tmp_path: Path
) -> None:
    checkpoint = tmp_path / "mature.pt"
    checkpoint.write_bytes(b"not-authorized")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_mature_fdr_public_state(_ToyModel(), checkpoint)  # type: ignore[arg-type]


def _write_bound_report(path: Path, evidence: dict) -> None:
    root = Path(__file__).resolve().parents[1]
    report = {
        "format_version": 1,
        "gate_sha256": probe.LEARNABILITY_GATE_SHA256,
        "evidence_sha256": hashlib.sha256(canonical_json_bytes(evidence)).hexdigest().upper(),
        "implementation": {
            "core": hashlib.sha256((root / "src" / "ra_learnability_probe.py").read_bytes()).hexdigest().upper(),
            "runner": hashlib.sha256((root / "scripts" / "run_ra_learnability_probe.py").read_bytes()).hexdigest().upper(),
        },
        "evidence": evidence,
        "gate": evaluate_learnability_gate(evidence),
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest().upper()
    path.write_text(json.dumps(report), encoding="utf-8")


def test_bound_report_validation_rejects_posthoc_evidence_tampering(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    _write_bound_report(path, _passing_evidence())
    assert validate_learnability_report(path)["gate"]["smoke2_eligible"] is True

    report = json.loads(path.read_text(encoding="utf-8"))
    report["evidence"]["losses"]["dev"][-1] = 999.0
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence SHA256"):
        validate_learnability_report(path)


def test_real_probe_objective_backpropagates_only_through_support_private_path() -> None:
    torch.manual_seed(0)
    model = RAGLGMDetectionModel(nc=3, verbose=False).train()
    freeze_for_support_probe(model)
    batch = {
        "img": torch.rand(1, 3, 128, 128),
        "cls": torch.tensor([[1.0], [-1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0.0, 0.0]),
    }

    loss, targets = support_objective(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert targets.difficulty.numel() == 1
    assert model.ra_glgm.support_head.weight.grad is not None
    assert model.ra_glgm.support_head.weight.grad.abs().sum() > 0
    assert model.ra_glgm.alpha.grad is None
    assert model.ra_glgm.output_projection.weight.grad is None
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(RA_GLGM_PRIVATE_PREFIX)
    )

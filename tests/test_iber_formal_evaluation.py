from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.evaluate_rtdetr_iber_formal as evaluation
from src.iber_formal_publication import FormalPublicationIdentity


def test_formal_validation_arguments_are_frozen() -> None:
    assert evaluation.FORMAL_VALIDATION_ARGS == {
        "model": "rtdetr-l.yaml",
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "max_det": 300,
        "nms": False,
        "cache": False,
        "plots": False,
        "save_json": False,
        "verbose": False,
        "task": "detect",
        "mode": "val",
        "split": "val",
        "rect": False,
    }


def test_epoch100_is_required_for_both_method_and_baseline(monkeypatch, tmp_path: Path) -> None:
    method = tmp_path / "method.pt"
    baseline = tmp_path / "baseline.pt"
    method.write_bytes(b"method")
    baseline.write_bytes(b"baseline")

    monkeypatch.setattr(
        evaluation,
        "checkpoint_record",
        lambda path: {
            "completed_epoch": 99 if Path(path) == method else 100,
            "bytes": Path(path).stat().st_size,
            "sha256": "1" * 64,
        },
    )
    with pytest.raises(ValueError, match="epoch 100"):
        evaluation.validate_epoch100_checkpoints(method, baseline)


def test_comparison_contains_finite_baseline_stock_and_refined_deltas() -> None:
    baseline = {"map": 0.10, "map50": 0.20, "map75": 0.08, "precision": 0.3, "recall": 0.4}
    stock = {"map": 0.11, "map50": 0.21, "map75": 0.09, "precision": 0.31, "recall": 0.41}
    refined = {"map": 0.13, "map50": 0.24, "map75": 0.12, "precision": 0.32, "recall": 0.43}

    report = evaluation.build_comparison(baseline, stock, refined)

    assert report["baseline"] == baseline
    assert report["method_stock"] == stock
    assert report["method_refined"] == refined
    assert report["delta"]["refined_vs_baseline"]["map"] == pytest.approx(0.03)
    assert report["delta"]["refined_vs_stock"]["map75"] == pytest.approx(0.03)


def test_comparison_rejects_nonfinite_or_incomplete_metrics() -> None:
    valid = {"map": 0.1, "map50": 0.2, "map75": 0.08, "precision": 0.3, "recall": 0.4}
    with pytest.raises(ValueError, match="metrics"):
        evaluation.build_comparison(valid, valid, {**valid, "map": float("nan")})
    with pytest.raises(ValueError, match="metrics"):
        evaluation.build_comparison(valid, valid, {"map": 0.1})


def test_immutable_report_refuses_changed_replacement(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    evaluation.write_immutable_report(path, {"status": "first"})
    evaluation.write_immutable_report(path, {"status": "first"})

    with pytest.raises(FileExistsError, match="replace"):
        evaluation.write_immutable_report(path, {"status": "changed"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "first"}


def test_epoch100_method_checkpoint_must_match_verified_publication_ledger(
    tmp_path: Path,
) -> None:
    identity = FormalPublicationIdentity(
        source_commit="1" * 40,
        protocol_sha256="2" * 64,
        initial_state_sha256="3" * 64,
    )
    ledger = tmp_path / "publication-ledger.jsonl"
    rows = []
    for epoch in range(1, 101):
        rows.append(
            {
                **identity.as_dict(),
                "completed_epoch": epoch,
                "checkpoint": {
                    "asset_name": f"formal-epoch-{epoch:04d}.pt",
                    "bytes": 100 + epoch,
                    "sha256": str(epoch % 10) * 64,
                },
                "result_commit_sha": "4" * 40,
                "verified": True,
            }
        )
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    runtime = {
        "source_commit": identity.source_commit,
        "protocol_sha256": identity.protocol_sha256.upper(),
        "initial_state_sha256": identity.initial_state_sha256.upper(),
    }
    method = {"completed_epoch": 100, "bytes": 200, "sha256": "0" * 64}

    record = evaluation.validate_method_publication(method, runtime, ledger)
    assert record["completed_epoch"] == 100

    with pytest.raises(ValueError, match="published epoch100"):
        evaluation.validate_method_publication(
            {**method, "sha256": "f" * 64}, runtime, ledger
        )


def test_evaluation_reconstructs_stock_and_refined_models_independently(
    monkeypatch, tmp_path: Path
) -> None:
    runtime_manifest = tmp_path / "iber_formal_protocol.json"
    runtime_manifest.write_text("{}\n", encoding="utf-8")
    ledger = tmp_path / "publication-ledger.jsonl"
    ledger.write_text("{}\n", encoding="utf-8")
    method_path = tmp_path / "method.pt"
    baseline_path = tmp_path / "baseline.pt"
    method_path.write_bytes(b"method")
    baseline_path.write_bytes(b"baseline")
    method_record = {"completed_epoch": 100, "bytes": 6, "sha256": "a" * 64}
    baseline_record = {
        "completed_epoch": 100,
        "bytes": 8,
        "sha256": evaluation.EXPECTED_BASELINE_SHA256,
    }
    method_models: list[SimpleNamespace] = []

    monkeypatch.setattr(
        evaluation,
        "validate_evaluation_authority",
        lambda _runtime: {"data": {"formal": {"path": "formal.yaml"}}},
    )
    monkeypatch.setattr(
        evaluation,
        "validate_epoch100_checkpoints",
        lambda *_args: (method_record, baseline_record),
    )
    monkeypatch.setattr(
        evaluation, "validate_method_publication", lambda *_args: {"verified": True}
    )
    monkeypatch.setattr(evaluation, "load_baseline", lambda _path: SimpleNamespace(kind="baseline"))

    def load_method(_path):
        model = SimpleNamespace(
            kind=f"method-{len(method_models)}",
            last_iber_output=SimpleNamespace(
                gates=torch.tensor([0.2]),
                residuals=torch.tensor([0.1]),
                f3_boundary_evidence=torch.tensor([0.3]),
                rgb_boundary_evidence=torch.tensor([0.4]),
            ),
        )
        method_models.append(model)
        return model

    def validate_model(model, *, data, mode=None):
        assert data == "formal.yaml"
        base = 0.1 + 0.01 * len(method_models)
        return {
            "map": base,
            "map50": base,
            "map75": base,
            "precision": base,
            "recall": base,
        }

    monkeypatch.setattr(evaluation, "load_method", load_method)
    monkeypatch.setattr(evaluation, "validate_model", validate_model)

    evaluation.evaluate_formal(
        method_path, baseline_path, runtime_manifest, ledger
    )

    assert len(method_models) == 2
    assert method_models[0] is not method_models[1]

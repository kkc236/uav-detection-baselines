from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import evaluate_fdr_bpdd_ra_glgm_comparison as evaluator
from src.fdr_protocol import canonical_json_bytes
from src.rtdetr_fdr import FDRRTDETRDetectionModel
from src.rtdetr_fdr_bpdd_ra_glgm import FDRBPDDRAGLGMDetectionModel


def _write_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for slot, kind in zip(
        evaluator.EXPECTED_SLOTS,
        ("fdr", "fdr", "ra", "ra"),
        strict=True,
    ):
        checkpoint = tmp_path / f"{slot}.pt"
        checkpoint.write_bytes(f"checkpoint-{slot}".encode())
        checkpoints.append(
            {
                "slot": slot,
                "label": f"arm-{slot}",
                "kind": kind,
                "path": str(checkpoint),
                "sha256": evaluator.file_sha256(checkpoint),
            }
        )
    data_yaml = tmp_path / "VisDrone.yaml"
    data_yaml.write_text("path: frozen\n", encoding="utf-8")
    payload: dict[str, object] = {
        "format_version": 1,
        "design": evaluator.DESIGN,
        "checkpoints": checkpoints,
        "dataset": {
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": evaluator.file_sha256(data_yaml),
            "root": str(tmp_path / "VisDrone"),
            "positive": {"authority": "positive"},
            "ignore": {"authority": "ignore"},
            "val_images": evaluator.EXPECTED_VAL_IMAGES,
            "val_objects": evaluator.EXPECTED_VAL_OBJECTS,
        },
        "evaluation": dict(evaluator.EVALUATION_PROTOCOL),
    }
    payload["manifest_sha256"] = evaluator.manifest_payload_sha256(payload)
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _rewrite_manifest(path: Path, payload: dict[str, object]) -> None:
    payload["manifest_sha256"] = evaluator.manifest_payload_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_hash_detects_mutation(tmp_path: Path) -> None:
    path, payload = _write_manifest(tmp_path)
    assert evaluator._read_manifest(path) == payload

    checkpoints = payload["checkpoints"]
    assert isinstance(checkpoints, list)
    checkpoints[0]["label"] = "mutated-after-signing"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        evaluator._read_manifest(path)


def test_entries_reject_checkpoint_sha_and_abcd_contract_drift(tmp_path: Path) -> None:
    path, payload = _write_manifest(tmp_path)
    checkpoints = payload["checkpoints"]
    assert isinstance(checkpoints, list)
    Path(checkpoints[0]["path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint A SHA256 mismatch"):
        evaluator._validate_entries(evaluator._read_manifest(path))

    path, payload = _write_manifest(tmp_path / "fresh")
    checkpoints = payload["checkpoints"]
    assert isinstance(checkpoints, list)
    checkpoints[2]["kind"] = "fdr"
    _rewrite_manifest(path, payload)
    with pytest.raises(ValueError, match="checkpoint C kind must be ra"):
        evaluator._validate_entries(evaluator._read_manifest(path))

    checkpoints[2]["kind"] = "ra"
    checkpoints[0], checkpoints[1] = checkpoints[1], checkpoints[0]
    _rewrite_manifest(path, payload)
    with pytest.raises(ValueError, match="ordered exactly A/B/C/D"):
        evaluator._validate_entries(evaluator._read_manifest(path))


class _ContractModel(nn.Module):
    def __init__(self, *, kind: str, include_bpdd: bool = False) -> None:
        super().__init__()
        self.payload = nn.Parameter(
            torch.empty(evaluator.EXPECTED_PARAMETERS[kind], device="meta")
        )
        if kind == "ra":
            self.model = nn.ModuleList(nn.Identity() for _ in range(29))
            wrapper = nn.Module()
            wrapper.add_module("ra_glgm", type("RAGLGM", (nn.Module,), {})())
            self.model[28] = wrapper
        if include_bpdd:
            self.add_module("criterion", type("BPDDCriterion", (nn.Module,), {})())


def test_deployment_contract_requires_exact_ra_path_and_excludes_bpdd() -> None:
    assert evaluator._validate_deployment_model(_ContractModel(kind="fdr"), kind="fdr") == (
        evaluator.EXPECTED_PARAMETERS["fdr"]
    )
    assert evaluator._validate_deployment_model(_ContractModel(kind="ra"), kind="ra") == (
        evaluator.EXPECTED_PARAMETERS["ra"]
    )
    with pytest.raises(ValueError, match="BPDD entered the deployment graph"):
        evaluator._validate_deployment_model(
            _ContractModel(kind="fdr", include_bpdd=True), kind="fdr"
        )
    with pytest.raises(ValueError, match="unique model.28.ra_glgm"):
        evaluator._validate_deployment_model(_ContractModel(kind="fdr"), kind="ra")


def test_historical_bpdd_pickle_module_loads_into_plain_fdr_graph(tmp_path: Path) -> None:
    module_name = "src.rtdetr_fdr_bpdd"
    previous = sys.modules.pop(module_name, None)
    historical_module = types.ModuleType(module_name)
    historical_class = type(
        "FDRBPDDDetectionModel",
        (FDRRTDETRDetectionModel,),
        {"__module__": module_name},
    )
    historical_module.FDRBPDDDetectionModel = historical_class
    sys.modules[module_name] = historical_module
    try:
        historical = historical_class(nc=10, verbose=False)
        checkpoint = tmp_path / "historical-bpdd.pt"
        torch.save({"ema": historical}, checkpoint)
        del historical
        del historical_class
        sys.modules.pop(module_name)

        loaded, source = evaluator._load_deployment_model(checkpoint, kind="fdr")
        assert source == "ema"
        assert type(loaded) is FDRRTDETRDetectionModel
        assert not any(type(module).__name__.startswith("BPDD") for module in loaded.modules())
        assert sum(parameter.numel() for parameter in loaded.parameters()) == 33_156_614
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


def test_combo_checkpoint_d_loads_strictly_as_ra_deployment_graph(tmp_path: Path) -> None:
    combo = FDRBPDDRAGLGMDetectionModel(nc=10, verbose=False)
    checkpoint = tmp_path / "combo.pt"
    torch.save({"ema": combo.state_dict()}, checkpoint)
    del combo

    loaded, source = evaluator._load_deployment_model(checkpoint, kind="ra")
    assert source == "ema"
    assert type(loaded).__name__ == "RAGLGMDetectionModel"
    assert [(name, type(module).__name__) for name, module in loaded.named_modules() if type(module).__name__ == "RAGLGM"] == [
        ("model.28.ra_glgm", "RAGLGM")
    ]
    assert not any(type(module).__name__.startswith("BPDD") for module in loaded.modules())
    assert sum(parameter.numel() for parameter in loaded.parameters()) == 33_970_010


def test_unified_evaluation_runs_each_slot_once_and_writes_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = _write_manifest(tmp_path)
    dataset_root = (tmp_path / "VisDrone").resolve()
    dataset = {
        "root": dataset_root,
        "names": [f"class-{index}" for index in range(10)],
        "images": [tmp_path / "sample.jpg"],
        "positive": {"authority": "positive"},
        "ignore": {"authority": "ignore"},
        "data_yaml_sha256": "A" * 64,
    }
    monkeypatch.setattr(
        evaluator,
        "_validate_dataset",
        lambda _payload: (tmp_path / "VisDrone.yaml", dataset),
    )
    monkeypatch.setattr(
        evaluator,
        "_coco_ground_truth",
        lambda *_args, **_kwargs: ({}, {"sample": 1}, {}, {}),
    )
    loaded: list[tuple[str, str]] = []
    evaluated: list[str] = []

    def fake_loader(path: Path, *, kind: str) -> tuple[nn.Module, str]:
        loaded.append((path.stem, kind))
        return _ContractModel(kind=kind), "ema"

    def fake_evaluator(model: nn.Module, **kwargs: object) -> dict[str, object]:
        slot = Path(str(kwargs["save_dir"])).name
        evaluated.append(slot)
        return {
            "precision": 0.1,
            "recall": 0.2,
            "map": 0.3,
            "map50": 0.4,
            "map75": 0.25,
            "ap_tiny": 0.05,
            "ap_small": 0.15,
            "class_ap": [0.3] * 10,
            "processed_images": evaluator.EXPECTED_VAL_IMAGES,
        }

    output = tmp_path / "results" / "comparison.jsonl"
    rows = evaluator.evaluate_comparison(
        manifest=manifest,
        output=output,
        work_dir=tmp_path / "work",
        model_loader=fake_loader,
        model_evaluator=fake_evaluator,
    )

    assert loaded == [("A", "fdr"), ("B", "fdr"), ("C", "ra"), ("D", "ra")]
    assert evaluated == list(evaluator.EXPECTED_SLOTS)
    assert [row["slot"] for row in rows] == list(evaluator.EXPECTED_SLOTS)
    assert all(row["bpdd_inference_module"] is False for row in rows)
    assert output.read_text(encoding="utf-8").count("\n") == 4
    previous = "0" * 64
    for row in rows:
        assert row["previous_evaluation_row_sha256"] == previous
        unhashed = dict(row)
        recorded = unhashed.pop("evaluation_row_sha256")
        assert recorded == hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
        previous = recorded

    with pytest.raises(FileExistsError, match="refusing to replace"):
        evaluator._write_create_only_jsonl(output, rows)

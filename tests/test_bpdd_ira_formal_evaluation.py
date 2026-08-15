from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from scripts import evaluate_bpdd_ira_formal as formal


CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


class TinyCheckpointModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))


class TinyCombinedModel(nn.Module):
    received_nc: list[int | None] = []
    instances: list["TinyCombinedModel"] = []

    def __init__(self, *args, nc: int | None = None, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.loaded_strict: bool | None = None
        type(self).received_nc.append(nc)
        type(self).instances.append(self)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.loaded_strict = strict
        return super().load_state_dict(state_dict, strict=strict)


@pytest.fixture(autouse=True)
def _reset_tiny_model() -> None:
    TinyCombinedModel.received_nc.clear()
    TinyCombinedModel.instances.clear()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _identity() -> dict[str, object]:
    return formal.build_run_identity(
        {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
        stage="formal",
        variant="fdr_bpdd_ira",
        seed=0,
    )


def _run_authority(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run = tmp_path / "formal-seed0-fdr_bpdd_ira-v1"
    weights = run / "weights"
    weights.mkdir(parents=True)
    data_yaml = tmp_path / "formal.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str((tmp_path / "VisDrone").resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": list(CLASS_NAMES),
                "nc": 10,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "protocol_sha256": formal.BPDD_IRA_PROTOCOL_SHA256,
        "source": {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
        "run_identity": _identity(),
        "initial_state": {
            "path": str((tmp_path / "initial-state.pt").resolve()),
            "sha256": formal.FDR_INITIAL_STATE_SHA256,
        },
        "data": str(data_yaml.resolve()),
        "publication_queue": str((run / "publication-queue.jsonl").resolve()),
    }
    (run / "bpdd-ira-run.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = weights / "epoch99.pt"
    checkpoint.write_bytes(b"combined-final")
    records = run / "fdr-epochs.jsonl"
    rows = [
        {
            "completed_epoch": epoch,
            "run_id": _identity()["run_id"],
            "variant": "fdr_bpdd_ira",
            "stage": "formal",
            "gradients_finite": True,
            "precision": 0.1,
            "recall": 0.2,
            "map50": 0.3,
            "map75": 0.2,
            "map": 0.15,
            "checkpoint_sha256": (
                _sha(checkpoint) if epoch == 100 else f"{epoch:064X}"
            ),
            "ema_state_sha256": f"{epoch + 100:064X}",
        }
        for epoch in range(1, 101)
    ]
    rows[-1]["ema_state_sha256"] = "E" * 64
    records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run, checkpoint, records, data_yaml


def _evaluation(
    path: Path,
    *,
    variant: str,
    map_value: float,
    strict: bool = True,
) -> Path:
    protocol_sha256 = {
        "fdr": formal.FDR_PROTOCOL_SHA256,
        "fdr_bpdd": formal.BPDD_PROTOCOL_SHA256,
    }.get(variant, "D" * 64)
    payload = {
        "format_version": 1,
        "evaluation_identity": {
            "run_id": f"historical-{variant}",
            "stage": "formal",
            "variant": variant,
            "seed": 0,
            "dataset_sha256": formal.BPDD_IRA_PROTOCOL["dataset"]["sha256"],
            "fdr_protocol_sha256": formal.FDR_PROTOCOL_SHA256,
            "protocol_sha256": protocol_sha256,
            "split": "val",
            "images": 548,
            "data": "/authority/formal.yaml",
        },
        "checkpoint": {
            "kind": "exact-final-ema",
            "completed_epoch": 100,
            "sha256": "C" * 64,
            "sha256_verified": True,
        },
        "metrics": {
            "precision": map_value + 0.20,
            "recall": map_value + 0.10,
            "f1": map_value + 0.15,
            "map50": map_value + 0.20,
            "map75": map_value + 0.05,
            "map": map_value,
        },
        "scales": {name: map_value for name in formal.SCALE_NAMES},
        "class_details": {
            name: {
                "id": index,
                "map50": map_value + 0.10,
                "map75": map_value + 0.02,
                "map": map_value,
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "evaluation_protocol": dict(formal.EVALUATION_PROTOCOL),
        "processed_images": 548,
        "prediction_passes": 1,
    }
    if not strict:
        payload = {
            "source": "legacy_screenshot_manual_transcription",
            "metrics": payload["metrics"],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _validation_payload(map_value: float = 0.31) -> dict[str, object]:
    return {
        "metrics": {
            "precision": 0.57,
            "recall": 0.50,
            "f1": 0.5327102804,
            "map50": 0.49,
            "map75": 0.30,
            "map": map_value,
        },
        "scales": {name: map_value for name in formal.SCALE_NAMES},
        "scale_details": {
            name: {"gt": 1, "map50": 0.4, "map75": 0.3, "map": map_value}
            for name in formal.SCALE_NAMES
        },
        "classes": {name: map_value for name in CLASS_NAMES},
        "class_details": {
            name: {"id": index, "map50": 0.4, "map75": 0.3, "map": map_value}
            for index, name in enumerate(CLASS_NAMES)
        },
        "processed_images": 548,
        "prediction_passes": 1,
    }


def test_cli_exposes_authorities_but_no_test_or_metric_overrides() -> None:
    parser = formal.build_parser()
    args = parser.parse_args(
        [
            "--run-dir",
            "run",
            "--checkpoint",
            "epoch99.pt",
            "--dataset-root",
            "VisDrone",
            "--fdr-evaluation",
            "fdr.json",
            "--bpdd-evaluation",
            "bpdd.json",
            "--ira-evaluation",
            "ira.json",
            "--output",
            "report.json",
        ]
    )
    assert args.device == "cuda:0"
    assert args.run_dir == Path("run")
    assert args.checkpoint == Path("epoch99.pt")
    assert args.dataset_root == Path("VisDrone")
    assert args.fdr_evaluation == Path("fdr.json")
    assert args.bpdd_evaluation == Path("bpdd.json")
    assert args.ira_evaluation == Path("ira.json")
    assert args.output == Path("report.json")
    assert not any(
        option in parser._option_string_actions
        for option in (
            "--data",
            "--split",
            "--test",
            "--imgsz",
            "--batch",
            "--conf",
            "--max-det",
            "--warmup",
            "--runs",
        )
    )


def test_protocols_are_frozen_for_one_val_pass_and_fp16_benchmark() -> None:
    assert formal.EVALUATION_PROTOCOL == {
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "conf": 0.001,
        "max_det": 300,
        "nms": False,
    }
    assert formal.BENCHMARK_PROTOCOL == {
        "imgsz": 640,
        "batch": 1,
        "half": True,
        "warmup": 50,
        "runs": 200,
    }


def test_epoch_records_require_exactly_one_immutable_row_for_epochs_1_to_100(
    tmp_path: Path,
) -> None:
    run, checkpoint, records, _ = _run_authority(tmp_path)

    summary = formal.validate_epoch_records(
        records,
        run_identity=_identity(),
        checkpoint=checkpoint,
    )

    assert summary["count"] == 100
    assert summary["completed_epochs"] == [1, 100]
    assert summary["sha256"] == _sha(records)
    assert summary["final_checkpoint_sha256"] == _sha(checkpoint)
    assert summary["all_gradients_finite"] is True
    assert summary["path"] == str(records.resolve())
    assert run.is_dir()


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "identity", "gradient", "sha"])
def test_epoch_records_fail_closed_on_incomplete_or_changed_evidence(
    tmp_path: Path, mutation: str
) -> None:
    _, checkpoint, records, _ = _run_authority(tmp_path)
    rows = [json.loads(line) for line in records.read_text("utf-8").splitlines()]
    if mutation == "gap":
        rows.pop(40)
    elif mutation == "duplicate":
        rows.insert(40, dict(rows[39]))
    elif mutation == "identity":
        rows[40]["run_id"] = "foreign"
    elif mutation == "gradient":
        rows[40]["gradients_finite"] = False
    else:
        rows[-1]["checkpoint_sha256"] = "0" * 64
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")

    with pytest.raises(ValueError, match="epoch|identity|gradient|SHA256|record"):
        formal.validate_epoch_records(
            records,
            run_identity=_identity(),
            checkpoint=checkpoint,
        )


def test_val_authority_rejects_test_yaml_and_requires_frozen_548_images(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "VisDrone"
    test_yaml = tmp_path / "test-data.yaml"
    test_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/test",
                "names": list(CLASS_NAMES),
                "nc": 10,
            }
        ),
        "utf-8",
    )
    signature = lambda _root: {  # noqa: E731
        "sha256": formal.BPDD_IRA_PROTOCOL["dataset"]["sha256"],
        "file_count": 14038,
    }

    with pytest.raises(ValueError, match="test|val"):
        formal.validate_val_authority(
            test_yaml,
            dataset_root=dataset_root,
            signature_fn=signature,
            image_count_fn=lambda _path: 548,
        )

    val_yaml = tmp_path / "formal.yaml"
    val_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "names": list(CLASS_NAMES),
                "nc": 10,
            }
        ),
        "utf-8",
    )
    authority = formal.validate_val_authority(
        val_yaml,
        dataset_root=dataset_root,
        signature_fn=signature,
        image_count_fn=lambda _path: 548,
    )
    assert authority["split"] == "val"
    assert authority["images"] == 548
    assert authority["dataset_sha256"] == formal.BPDD_IRA_PROTOCOL["dataset"]["sha256"]
    assert authority["yaml_sha256"] == _sha(val_yaml)

    with pytest.raises(ValueError, match="548"):
        formal.validate_val_authority(
            val_yaml,
            dataset_root=dataset_root,
            signature_fn=signature,
            image_count_fn=lambda _path: 547,
        )


def test_exact_final_loader_requires_epoch100_ema_and_strict_combined_graph(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "epoch99.pt"
    ema = TinyCheckpointModel(2.0)
    torch.save(
        {
            "epoch": 99,
            "ema": ema,
            "model": TinyCheckpointModel(9.0),
            "optimizer": {"state": {}, "param_groups": []},
        },
        checkpoint,
    )

    loaded = formal.load_exact_combined_checkpoint(
        checkpoint,
        expected_sha256=_sha(checkpoint),
        model_factory=TinyCombinedModel,
    )

    assert TinyCombinedModel.received_nc == [10]
    assert TinyCombinedModel.instances[0].loaded_strict is True
    torch.testing.assert_close(loaded.model.weight, torch.tensor([2.0]))
    assert loaded.metadata["kind"] == "exact-final-ema"
    assert loaded.metadata["completed_epoch"] == 100
    assert loaded.metadata["strict_fdr_bpdd_ira_graph"] is True
    assert loaded.metadata["ema_state_sha256"] == formal.state_sha256(ema.state_dict())


@pytest.mark.parametrize("filename,epoch,ema", [("last.pt", 99, True), ("epoch99.pt", 98, True), ("epoch99.pt", 99, False)])
def test_exact_final_loader_rejects_wrong_file_epoch_or_missing_ema(
    tmp_path: Path, filename: str, epoch: int, ema: bool
) -> None:
    checkpoint = tmp_path / filename
    torch.save(
        {
            "epoch": epoch,
            "ema": TinyCheckpointModel(2.0) if ema else None,
            "model": TinyCheckpointModel(9.0),
            "optimizer": {"state": {}, "param_groups": []},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="epoch99|epoch100|EMA"):
        formal.load_exact_combined_checkpoint(
            checkpoint,
            expected_sha256=_sha(checkpoint),
            model_factory=TinyCombinedModel,
        )


def test_latency_summary_reports_required_median_p95_and_fps() -> None:
    summary = formal.latency_summary([1.0, 2.0, 3.0, 4.0, 10.0])
    assert summary == {
        "median_ms": 3.0,
        "p95_ms": pytest.approx(8.8),
        "fps": pytest.approx(1000.0 / 3.0),
    }
    with pytest.raises(ValueError, match="finite|positive|empty"):
        formal.latency_summary([])


def test_preliminary_comparison_reports_core_scale_and_class_deltas(
    tmp_path: Path,
) -> None:
    current = _validation_payload(0.31)
    fdr = _evaluation(tmp_path / "fdr.json", variant="fdr", map_value=0.28)
    bpdd = _evaluation(tmp_path / "bpdd.json", variant="fdr_bpdd", map_value=0.30)
    ira = _evaluation(
        tmp_path / "ira.json", variant="fdr_ira", map_value=0.305, strict=False
    )

    comparison = formal.build_preliminary_comparisons(
        current,
        fdr_evaluation=fdr,
        bpdd_evaluation=bpdd,
        ira_evaluation=ira,
    )

    assert comparison["comparison_scope"] == "preliminary_cross_run"
    assert comparison["strict_paired"] is False
    assert [row["method"] for row in comparison["four_row_summary"]] == [
        "FDR",
        "FDR+BPDD",
        "FDR+IRA",
        "FDR+BPDD+IRA",
    ]
    assert [row["method"] for row in comparison["strict_delta_table"]] == [
        "FDR",
        "FDR+BPDD",
    ]
    assert comparison["four_row_summary"][0]["evidence_level"] == "strict_reference"
    assert comparison["four_row_summary"][1]["evidence_level"] == "strict_reference"
    assert (
        comparison["four_row_summary"][2]["evidence_level"]
        == "non_strict_historical_reference"
    )
    assert comparison["four_row_summary"][3]["evidence_level"] == "current_exact"
    assert [row["method"] for row in comparison["non_strict_historical_reference"]] == [
        "FDR+IRA"
    ]
    assert comparison["against_fdr"]["metrics_delta"]["map"] == pytest.approx(0.03)
    assert comparison["against_fdr_bpdd"]["metrics_delta"]["map"] == pytest.approx(0.01)
    assert comparison["against_fdr"]["scale_delta"]["tiny"] == pytest.approx(0.03)
    assert comparison["against_fdr_bpdd"]["class_delta"]["car"]["map75"] == pytest.approx(-0.02)
    assert comparison["against_fdr"]["authority_sha256"] == _sha(fdr)


def test_missing_historical_ira_authority_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    current = _validation_payload(0.31)
    fdr = _evaluation(tmp_path / "fdr.json", variant="fdr", map_value=0.28)
    bpdd = _evaluation(tmp_path / "bpdd.json", variant="fdr_bpdd", map_value=0.30)

    comparison = formal.build_preliminary_comparisons(
        current,
        fdr_evaluation=fdr,
        bpdd_evaluation=bpdd,
        ira_evaluation=None,
    )

    ira = comparison["four_row_summary"][2]
    assert ira == {
        "method": "FDR+IRA",
        "evidence_level": "unavailable",
        "strict_validation_failures": ["authority_unavailable"],
        "precision": None,
        "recall": None,
        "f1": None,
        "map50": None,
        "map75": None,
        "map": None,
    }
    assert comparison["unavailable_references"] == [ira]
    assert "against_fdr_ira" not in comparison


def test_cli_allows_missing_historical_ira_authority() -> None:
    args = formal.build_parser().parse_args(
        [
            "--run-dir",
            "run",
            "--checkpoint",
            "epoch99.pt",
            "--dataset-root",
            "VisDrone",
            "--fdr-evaluation",
            "fdr.json",
            "--bpdd-evaluation",
            "bpdd.json",
            "--output",
            "report.json",
        ]
    )

    assert args.ira_evaluation is None


@pytest.mark.parametrize("drift", ["split", "evaluator", "protocol", "images"])
def test_reference_is_never_strict_when_split_evaluator_or_protocol_drifts(
    tmp_path: Path, drift: str
) -> None:
    path = _evaluation(tmp_path / "reference.json", variant="fdr", map_value=0.28)
    payload = json.loads(path.read_text("utf-8"))
    if drift == "split":
        payload["evaluation_identity"]["split"] = "test"
    elif drift == "evaluator":
        payload["evaluation_protocol"]["max_det"] = 999
    elif drift == "protocol":
        payload["evaluation_identity"]["fdr_protocol_sha256"] = "0" * 64
    else:
        payload["processed_images"] = 1610
    path.write_text(json.dumps(payload), "utf-8")

    reference = formal.load_reference_authority(
        path,
        expected_variant="fdr",
        method="FDR",
    )

    assert reference["evidence_level"] == "non_strict_historical_reference"
    assert reference["strict"] is False
    assert reference["strict_validation_failures"]


def test_formal_evaluation_binds_all_authorities_and_emits_complete_report(
    tmp_path: Path,
) -> None:
    run, checkpoint, records, data_yaml = _run_authority(tmp_path)
    dataset_root = tmp_path / "VisDrone"
    fdr = _evaluation(tmp_path / "fdr.json", variant="fdr", map_value=0.28)
    bpdd = _evaluation(tmp_path / "bpdd.json", variant="fdr_bpdd", map_value=0.30)
    ira = _evaluation(
        tmp_path / "ira.json", variant="fdr_ira", map_value=0.305, strict=False
    )
    output = tmp_path / "combined-evaluation.json"
    validation_model = object()
    efficiency_model = object()
    loaded = SimpleNamespace(
        model=validation_model,
        metadata={
            "kind": "exact-final-ema",
            "completed_epoch": 100,
            "raw_epoch": 99,
            "sha256": _sha(checkpoint),
            "sha256_verified": True,
            "source_field": "ema",
            "ema_state_sha256": "E" * 64,
            "strict_fdr_bpdd_ira_graph": True,
        },
    )
    efficiency_loaded = SimpleNamespace(
        model=efficiency_model,
        metadata=dict(loaded.metadata),
    )
    calls: dict[str, object] = {}

    def fake_loader(path, *, expected_sha256):
        calls.setdefault("load", []).append((Path(path), expected_sha256))
        return loaded if len(calls["load"]) == 1 else efficiency_loaded

    def fake_val(model, *, data, save_dir):
        calls["val"] = (model, Path(data), Path(save_dir))
        return _validation_payload()

    def fake_efficiency(model, *, device):
        calls["efficiency"] = (model, device)
        return {
            "parameters": 33_500_000,
            "gflops": 105.2,
            "fp16": {
                "device": device,
                "median_ms": 5.0,
                "p95_ms": 5.5,
                "fps": 200.0,
                "peak_memory_mib": 250.0,
            },
        }

    def fake_val_authority(path, *, dataset_root):
        calls["authority"] = (Path(path), Path(dataset_root))
        return {
            "split": "val",
            "images": 548,
            "dataset_sha256": formal.BPDD_IRA_PROTOCOL["dataset"]["sha256"],
            "yaml": str(Path(path).resolve()),
            "yaml_sha256": _sha(Path(path)),
        }

    report = formal.evaluate_formal_checkpoint(
        run_dir=run,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        fdr_evaluation=fdr,
        bpdd_evaluation=bpdd,
        ira_evaluation=ira,
        output=output,
        device="cuda:0",
        checkpoint_loader=fake_loader,
        validation_runner=fake_val,
        efficiency_runner=fake_efficiency,
        val_authority_validator=fake_val_authority,
    )

    assert calls["load"] == [
        (checkpoint.resolve(), _sha(checkpoint)),
        (checkpoint.resolve(), _sha(checkpoint)),
    ]
    assert calls["val"] == (
        validation_model,
        data_yaml.resolve(),
        output.parent / "validator",
    )
    assert calls["efficiency"] == (efficiency_model, "cuda:0")
    assert calls["authority"] == (data_yaml.resolve(), dataset_root.resolve())
    assert report["evaluation_identity"] == {
        **_identity(),
        "data": str(data_yaml.resolve()),
        "dataset_sha256": formal.BPDD_IRA_PROTOCOL["dataset"]["sha256"],
        "split": "val",
        "images": 548,
    }
    assert report["epoch_records"]["count"] == 100
    assert report["epoch_records"]["sha256"] == _sha(records)
    assert report["checkpoint"]["strict_fdr_bpdd_ira_graph"] is True
    assert report["metrics"]["map"] == pytest.approx(0.31)
    assert len(report["class_details"]) == 10
    assert list(report["scales"]) == list(formal.SCALE_NAMES)
    assert report["efficiency"]["fp16"]["peak_memory_mib"] == 250.0
    assert report["comparisons"]["comparison_scope"] == "preliminary_cross_run"
    assert report["hashes"] == {
        "checkpoint_sha256": _sha(checkpoint),
        "ema_state_sha256": "E" * 64,
        "epoch_records_sha256": _sha(records),
        "data_yaml_sha256": _sha(data_yaml),
        "fdr_evaluation_sha256": _sha(fdr),
        "bpdd_evaluation_sha256": _sha(bpdd),
        "ira_evaluation_sha256": _sha(ira),
        "dataset_sha256": formal.BPDD_IRA_PROTOCOL["dataset"]["sha256"],
    }
    assert json.loads(output.read_text("utf-8")) == report
    with pytest.raises(FileExistsError):
        formal.evaluate_formal_checkpoint(
            run_dir=run,
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            fdr_evaluation=fdr,
            bpdd_evaluation=bpdd,
            ira_evaluation=ira,
            output=output,
            checkpoint_loader=fake_loader,
            validation_runner=fake_val,
            efficiency_runner=fake_efficiency,
            val_authority_validator=fake_val_authority,
        )


def test_run_manifest_rejects_test_data_and_wrong_combined_identity(tmp_path: Path) -> None:
    run, _, _, _ = _run_authority(tmp_path)
    manifest_path = run / "bpdd-ira-run.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["run_identity"]["variant"] = "fdr_bpdd"
    manifest_path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ValueError, match="identity|variant"):
        formal.validate_run_manifest(run)

    payload["run_identity"] = _identity()
    payload["data"] = str((tmp_path / "test-data.yaml").resolve())
    manifest_path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ValueError, match="test|val"):
        formal.validate_run_manifest(run)

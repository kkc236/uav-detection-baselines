from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path("scripts/run_rtdetr_quality_oracle.py")


def test_quality_oracle_runner_exists() -> None:
    assert SCRIPT.is_file(), "Task 3 runner has not been implemented"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_rtdetr_quality_oracle_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_surface_and_protocol_constants_are_frozen() -> None:
    module = _load_module()
    argv = [
        "--baseline-checkpoint",
        "baseline.pt",
        "--dataset-root",
        "dataset",
        "--cache-root",
        "cache",
        "--report-root",
        "report",
    ]

    args = module._parse_args(argv)

    assert args == Namespace(
        baseline_checkpoint=Path("baseline.pt"),
        dataset_root=Path("dataset"),
        cache_root=Path("cache"),
        report_root=Path("report"),
        device="0",
    )
    assert module.IMAGE_SIZE == 640
    assert module.BATCH_SIZE == 8
    assert module.WORKERS == 8
    assert module.CONFIDENCE == 0.001
    assert module.MAX_DET == 300
    assert module.NMS is False
    assert module.TRAIN_COUNT == 647
    assert module.DEV_COUNT == 129
    assert module.VAL_COUNT == 548
    assert module.ALPHA_GRID == (0.25, 0.5, 1.0, 2.0)
    for forbidden in (
        "--alpha",
        "--threshold",
        "--conf",
        "--workers",
        "--batch",
        "--split-salt",
        "--smoke-images",
        "--official-val-passes",
    ):
        with pytest.raises(SystemExit):
            module._parse_args([*argv, forbidden, "1"])


class _FakeHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.export = True

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        selected_scores, index = scores.flatten(1).topk(300)
        query_index = torch.div(index, 10, rounding_mode="floor")
        selected_boxes = boxes.gather(
            1, query_index.unsqueeze(-1).expand(-1, -1, 4)
        )
        classes = (index - query_index * 10).unsqueeze(-1).float()
        return torch.cat((selected_boxes, selected_scores.unsqueeze(-1), classes), -1)


class _FakeDetector(torch.nn.Module):
    def __init__(self, *, mutate: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.model = torch.nn.ModuleList([torch.nn.Identity(), _FakeHead()])
        self.mutate = mutate
        self.calls = 0

    def predict(self, images: torch.Tensor):
        assert torch.is_inference_mode_enabled()
        assert self.model[-1].export is False
        self.calls += 1
        if self.mutate:
            self.weight.add_(1)
        batch = images.shape[0]
        generator = torch.Generator().manual_seed(17)
        boxes = torch.rand((batch, 300, 4), generator=generator)
        logits = torch.randn((batch, 300, 10), generator=generator)
        stock = self.model[-1].postprocess(boxes, logits.sigmoid())
        auxiliary = (
            boxes.unsqueeze(0),
            logits.unsqueeze(0),
            torch.empty(0),
            torch.empty(0),
            None,
        )
        return stock, auxiliary


class _FakeValidator:
    def preprocess(self, batch):
        return batch


def _batch(count: int = 8) -> dict[str, object]:
    return {
        "img": torch.zeros((count, 3, 16, 16)),
        "batch_idx": torch.arange(count),
        "bboxes": torch.full((count, 4), 0.5),
        "cls": torch.arange(count).remainder(10).view(-1, 1),
        "im_file": [f"image-{index}.jpg" for index in range(count)],
    }


def test_auxiliary_tuple_is_reconstructed_exactly_without_gradients() -> None:
    module = _load_module()
    detector = _FakeDetector()

    stock, boxes, logits = module._extract_decoder_batch(
        detector, _batch()["img"], require_cuda_smoke_shape=True
    )

    assert stock.shape == (8, 300, 6)
    assert boxes.shape == (8, 300, 4)
    assert logits.shape == (8, 300, 10)
    assert detector.model[-1].export is False
    assert not boxes.requires_grad and not logits.requires_grad
    assert torch.isfinite(boxes).all() and torch.isfinite(logits).all()
    assert all(parameter.grad is None for parameter in detector.parameters())


def test_record_extraction_preserves_state_and_rejects_mutation() -> None:
    module = _load_module()
    detector = _FakeDetector()
    module._assert_cuda0_detector = lambda _detector: None
    module._assert_cuda0_tensor = lambda _tensor, *, label: None

    records = module._extract_records(
        detector,
        [_batch()],
        _FakeValidator(),
        device=torch.device("cuda:0"),
        expected_count=8,
        run_cuda_smoke=False,
    )

    assert len(records) == 8
    assert records[0]["boxes"].shape == (300, 4)
    assert records[0]["logits"].shape == (300, 10)
    assert records[0]["boxes"].device.type == "cpu"
    assert records[0]["target_classes"].dtype == torch.int64
    assert all(
        not value.requires_grad
        for record in records
        for value in record.values()
        if isinstance(value, torch.Tensor)
    )

    with pytest.raises(RuntimeError, match="state changed"):
        module._extract_records(
            _FakeDetector(mutate=True),
            [_batch()],
            _FakeValidator(),
            device=torch.device("cuda:0"),
            expected_count=8,
            run_cuda_smoke=False,
        )


def test_create_only_reports_are_canonical(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "report.json"

    module._write_canonical_json_create_only(path, {"z": 1, "a": [2]})

    assert path.read_bytes() == b'{"a":[2],"z":1}\n'
    with pytest.raises(FileExistsError):
        module._write_canonical_json_create_only(path, {"z": 1, "a": [2]})


def test_existing_reports_are_accepted_only_when_canonical_and_identical(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "report.json"
    payload = {"z": 1, "a": [2]}

    module._write_or_validate_canonical_json(path, payload)
    module._write_or_validate_canonical_json(path, payload)

    with pytest.raises(RuntimeError, match="immutable report differs"):
        module._write_or_validate_canonical_json(path, {"z": 2, "a": [2]})


def test_device_is_frozen_to_available_cuda_zero(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert module._device("0") == torch.device("cuda:0")
    for invalid in ("cpu", "1", "cuda:0", 0):
        with pytest.raises((TypeError, ValueError), match="only device 0"):
            module._device(invalid)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device 0 is unavailable"):
        module._device("0")


def test_detector_and_preprocessed_inputs_must_be_cuda_zero() -> None:
    module = _load_module()
    detector = _FakeDetector()

    with pytest.raises(RuntimeError, match="detector.*cuda:0"):
        module._assert_cuda0_detector(detector)
    with pytest.raises(RuntimeError, match="preprocessed input.*cuda:0"):
        module._assert_cuda0_tensor(torch.zeros(1), label="preprocessed input")


def test_stock_authority_requires_exact_gate_metrics_and_bounded_diagnostics() -> None:
    module = _load_module()
    exact = module._assert_stock_authority(dict(module.STOCK_AUTHORITY))
    assert exact["status"] == "passed_exact"
    assert exact["diagnostic_delta"] == {"precision": 0.0, "recall": 0.0}

    changed = dict(module.STOCK_AUTHORITY)
    changed["map"] += 1e-16

    with pytest.raises(RuntimeError, match="stock authority mismatch"):
        module._assert_stock_authority(changed)

    observed = dict(module.STOCK_AUTHORITY)
    observed["precision"] = 0.5119369292841953
    amended = module._assert_stock_authority(observed)
    assert amended["status"] == "passed_with_non_gate_float_amendment"
    assert 0.0 < amended["diagnostic_delta"]["precision"] < 1e-8
    assert amended["tolerance"] == 1e-8

    outside_tolerance = dict(module.STOCK_AUTHORITY)
    outside_tolerance["precision"] += 1.1e-8
    with pytest.raises(RuntimeError, match="stock authority mismatch"):
        module._assert_stock_authority(outside_tolerance)


def _records_for_paths(paths) -> list[dict[str, object]]:
    return [{"image_id": str(Path(path).resolve())} for path in paths]


def test_actual_loader_order_is_bound_separately_from_selection_order(
    tmp_path: Path,
) -> None:
    module = _load_module()
    root = tmp_path / "dataset"
    selected = tuple(root / "images" / "train" / f"dev-{index}.jpg" for index in range(129))
    actual = tuple(reversed(selected))

    binding = module._bind_record_identities(
        _records_for_paths(actual),
        expected_paths=selected,
        dataset_root=root,
        split_name="internal-dev",
    )

    assert binding["count"] == 129
    assert binding["actual_loader_order_sha256"] == module.ordered_path_sha256(
        actual, root=root
    )
    assert binding["actual_loader_image_paths"][0] == "images/train/dev-128.jpg"
    assert set(binding["actual_loader_image_paths"]) == {
        f"images/train/dev-{index}.jpg" for index in range(129)
    }

    with pytest.raises(RuntimeError, match="identity set mismatch"):
        module._bind_record_identities(
            _records_for_paths((*actual[:-1], root / "images" / "train" / "other.jpg")),
            expected_paths=selected,
            dataset_root=root,
            split_name="internal-dev",
        )


def test_official_validation_waits_for_immutable_alpha_and_runs_once(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    report_root = tmp_path / "reports"
    args = Namespace(
        baseline_checkpoint=tmp_path / "baseline.pt",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        report_root=report_root,
        device="0",
    )
    authority = {
        "baseline_sha256": "A" * 64,
        "dataset_sha256": module.EXPECTED_DATASET_SHA256,
        "subset_sha256": "C" * 64,
        "runtime_amendment_sha256": "D" * 64,
        "source_commit": "e" * 40,
        "schema_sha256": "F" * 64,
        "dev_sha256": module.EXPECTED_DEV_SHA256,
    }
    events: list[str] = []
    selected_paths = tuple(
        args.dataset_root / "images" / "train" / f"dev-{index}.jpg"
        for index in range(129)
    )
    actual_dev_paths = tuple(reversed(selected_paths))
    val_paths = tuple(
        args.dataset_root / "images" / "val" / f"val-{index}.jpg"
        for index in range(548)
    )
    dev_records = _records_for_paths(actual_dev_paths)
    val_records = _records_for_paths(val_paths)

    monkeypatch.setattr(
        module,
        "_build_pre_alpha_authority",
        lambda *_args, **_kwargs: authority,
        raising=False,
    )
    monkeypatch.setattr(module, "_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(module, "_load_detector", lambda *_args, **_kwargs: _FakeDetector())
    monkeypatch.setattr(module, "_assert_cuda0_detector", lambda _detector: None, raising=False)
    monkeypatch.setattr(module, "_assert_cuda0_tensor", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(
        module,
        "_prepare_internal_dev",
        lambda *_args, **_kwargs: selected_paths,
    )
    monkeypatch.setattr(
        module,
        "_expected_official_val_paths",
        lambda _root: val_paths,
        raising=False,
    )

    def dataset_signature(_root):
        events.append("dataset-signature")
        assert (report_root / "alpha-selection-report.json").is_file()
        return {"sha256": module.EXPECTED_DATASET_SHA256}

    monkeypatch.setattr(module, "dataset_signature", dataset_signature)

    def build_loader(*_args, split_name: str, **_kwargs):
        events.append(f"build:{split_name}")
        if split_name == "official-val":
            assert (report_root / "alpha-selection-report.json").is_file()
        return split_name, SimpleNamespace()

    monkeypatch.setattr(module, "_build_validation_loader", build_loader)

    def extract(_detector, loader, _validator, **_kwargs):
        events.append(f"extract:{loader}")
        return dev_records if loader == "internal-dev" else val_records

    monkeypatch.setattr(module, "_extract_records", extract)
    stock = dict(module.STOCK_AUTHORITY)
    candidates = {
        alpha: dict(stock, map=stock["map"] + alpha / 10_000, ap75=stock["ap75"] + alpha / 20_000)
        for alpha in module.ALPHA_GRID
    }
    selected = 2.0

    def evaluate(records, *, alphas):
        events.append(f"evaluate:{len(records)}:{tuple(alphas)}")
        if len(records) == 129:
            return {"stock": stock, "oracle": candidates}
        assert tuple(alphas) == (selected,)
        return {
            "stock": stock,
            "oracle": {selected: dict(stock, map=stock["map"] + 0.006, ap75=stock["ap75"] + 0.001)},
        }

    monkeypatch.setattr(module, "_evaluate_records", evaluate)
    monkeypatch.setattr(module, "_select_alpha", lambda _metrics: selected)
    def persist_cache(root, *_args, **_kwargs):
        Path(root).mkdir()
        (Path(root) / "manifest.json").write_text("{}\n", encoding="utf-8")
        return {"complete": True}

    monkeypatch.setattr(module, "_persist_cache", persist_cache)
    monkeypatch.setattr(module, "_execution_environment", lambda: {"gpu": "mock"})
    monkeypatch.setattr(module, "_file_sha256", lambda _path: "9" * 64)

    result = module._run(args)

    assert result == 0
    assert events.count("build:official-val") == 1
    assert events.count("extract:official-val") == 1
    assert events.count("dataset-signature") == 1
    assert events.index("evaluate:129:(0.25, 0.5, 1.0, 2.0)") < events.index("build:official-val")
    assert events.index("evaluate:129:(0.25, 0.5, 1.0, 2.0)") < events.index("dataset-signature")
    alpha_report = json.loads(
        (report_root / "alpha-selection-report.json").read_text()
    )
    assert alpha_report["selected_alpha"] == selected
    assert alpha_report["split"]["selection_order_image_paths"][0] == (
        "images/train/dev-0.jpg"
    )
    assert alpha_report["split"]["selection_order_sha256"] == module.EXPECTED_DEV_SHA256
    assert alpha_report["split"]["actual_loader_image_paths"][0] == (
        "images/train/dev-128.jpg"
    )
    assert alpha_report["split"]["actual_loader_order_sha256"] == module.ordered_path_sha256(
        actual_dev_paths, root=args.dataset_root
    )
    assert (report_root / "quality-oracle-report.json").is_file()
    assert (report_root / "quality-oracle-decision.json").is_file()
    inventory = json.loads(
        (report_root / "environment-hash-inventory.json").read_text()
    )
    assert inventory["inputs"]["cache_manifest"] == {
        "path": str((args.cache_root.resolve() / "manifest.json")),
        "sha256": "9" * 64,
    }


def test_resume_uses_external_cache_authority_and_skips_all_inference(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    report_root = tmp_path / "reports"
    args = Namespace(
        baseline_checkpoint=tmp_path / "baseline.pt",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        report_root=report_root,
        device="0",
    )
    authority = {
        "baseline_sha256": "A" * 64,
        "dataset_sha256": module.EXPECTED_DATASET_SHA256,
        "subset_sha256": "C" * 64,
        "runtime_amendment_sha256": "D" * 64,
        "source_commit": "e" * 40,
        "schema_sha256": "F" * 64,
        "dev_sha256": module.EXPECTED_DEV_SHA256,
    }
    selected_paths = tuple(
        args.dataset_root / "images" / "train" / f"dev-{index}.jpg"
        for index in range(129)
    )
    actual_dev_paths = tuple(reversed(selected_paths))
    val_paths = tuple(
        args.dataset_root / "images" / "val" / f"val-{index}.jpg"
        for index in range(548)
    )
    dev_records = _records_for_paths(actual_dev_paths)
    val_records = _records_for_paths(val_paths)
    selected = 2.0
    stock = dict(module.STOCK_AUTHORITY)
    candidates = {
        alpha: dict(stock, map=stock["map"] + alpha / 10_000, ap75=stock["ap75"] + alpha / 20_000)
        for alpha in module.ALPHA_GRID
    }

    monkeypatch.setattr(module, "_build_pre_alpha_authority", lambda *_args, **_kwargs: authority, raising=False)
    monkeypatch.setattr(module, "_prepare_internal_dev", lambda *_args, **_kwargs: selected_paths)
    monkeypatch.setattr(module, "_expected_official_val_paths", lambda _root: val_paths, raising=False)
    monkeypatch.setattr(module, "dataset_signature", lambda _root: {"sha256": module.EXPECTED_DATASET_SHA256})
    monkeypatch.setattr(module, "_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(module, "_assert_cuda0_detector", lambda _detector: None, raising=False)
    monkeypatch.setattr(module, "_assert_cuda0_tensor", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(module, "_load_detector", lambda *_args, **_kwargs: _FakeDetector())
    monkeypatch.setattr(
        module,
        "_build_validation_loader",
        lambda *_args, split_name, **_kwargs: (split_name, SimpleNamespace()),
    )
    monkeypatch.setattr(
        module,
        "_extract_records",
        lambda _detector, loader, _validator, **_kwargs: dev_records if loader == "internal-dev" else val_records,
    )

    def evaluate(records, *, alphas):
        if len(records) == 129:
            return {"stock": stock, "oracle": candidates}
        return {
            "stock": stock,
            "oracle": {selected: dict(stock, map=stock["map"] + 0.006, ap75=stock["ap75"] + 0.001)},
        }

    monkeypatch.setattr(module, "_evaluate_records", evaluate)
    monkeypatch.setattr(module, "_select_alpha", lambda _metrics: selected)
    monkeypatch.setattr(module, "_execution_environment", lambda: {"gpu": "mock"})
    monkeypatch.setattr(module, "_file_sha256", lambda _path: "9" * 64)

    def persist_cache(root, *_args, **_kwargs):
        Path(root).mkdir()
        (Path(root) / "manifest.json").write_text("{}\n", encoding="utf-8")
        return {"complete": True}

    monkeypatch.setattr(module, "_persist_cache", persist_cache)
    assert module._run(args) == 0
    cache_authority = report_root / "cache-manifest-authority.json"
    assert cache_authority.is_file()

    for name in (
        "quality-oracle-report.json",
        "quality-oracle-decision.json",
        "environment-hash-inventory.json",
    ):
        (report_root / name).unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resume must not run model inference")

    monkeypatch.setattr(module, "_load_detector", forbidden)
    monkeypatch.setattr(module, "_build_validation_loader", forbidden)
    monkeypatch.setattr(module, "_extract_records", forbidden)
    monkeypatch.setattr(module, "_persist_cache", forbidden)
    monkeypatch.setattr(
        module,
        "_load_cache",
        lambda *_args, **_kwargs: {"dev": tuple(dev_records), "val": tuple(val_records)},
        raising=False,
    )
    resumed_evaluations: list[int] = []

    def resume_evaluate(records, *, alphas):
        resumed_evaluations.append(len(records))
        return evaluate(records, alphas=alphas)

    monkeypatch.setattr(module, "_evaluate_records", resume_evaluate)

    assert module._run(args) == 0
    assert resumed_evaluations == [548]


def test_cache_without_external_manifest_authority_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    report_root = tmp_path / "reports"
    cache_root.mkdir()
    (cache_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    report_root.mkdir()

    with pytest.raises(RuntimeError, match="cache.*external authority"):
        module._validate_report_stage(report_root, cache_root)

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

    records = module._extract_records(
        detector,
        [_batch()],
        _FakeValidator(),
        device=torch.device("cpu"),
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
            device=torch.device("cpu"),
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


def test_stock_authority_requires_exact_frozen_metrics() -> None:
    module = _load_module()
    module._assert_stock_authority(dict(module.STOCK_AUTHORITY))
    changed = dict(module.STOCK_AUTHORITY)
    changed["map"] += 1e-16

    with pytest.raises(RuntimeError, match="stock authority mismatch"):
        module._assert_stock_authority(changed)


def _dummy_records(count: int, prefix: str) -> list[dict[str, object]]:
    return [{"image_id": f"{prefix}-{index}"} for index in range(count)]


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
        "dataset_sha256": "B" * 64,
        "subset_sha256": "C" * 64,
        "runtime_amendment_sha256": "D" * 64,
        "source_commit": "e" * 40,
        "schema_sha256": "F" * 64,
        "dev_sha256": module.EXPECTED_DEV_SHA256,
    }
    events: list[str] = []

    monkeypatch.setattr(module, "_build_authority", lambda *_args, **_kwargs: authority)
    monkeypatch.setattr(module, "_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(module, "_load_detector", lambda *_args, **_kwargs: _FakeDetector())
    monkeypatch.setattr(
        module,
        "_prepare_internal_dev",
        lambda *_args, **_kwargs: tuple(
            args.dataset_root / "images" / "train" / f"dev-{index}.jpg"
            for index in range(129)
        ),
    )

    def build_loader(*_args, split_name: str, **_kwargs):
        events.append(f"build:{split_name}")
        if split_name == "official-val":
            assert (report_root / "alpha-selection-report.json").is_file()
        return split_name, SimpleNamespace()

    monkeypatch.setattr(module, "_build_validation_loader", build_loader)

    def extract(_detector, loader, _validator, **_kwargs):
        events.append(f"extract:{loader}")
        return _dummy_records(129, "dev") if loader == "internal-dev" else _dummy_records(548, "val")

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
    monkeypatch.setattr(module, "_persist_cache", lambda *_args, **_kwargs: {"complete": True})
    monkeypatch.setattr(module, "_execution_environment", lambda: {"gpu": "mock"})
    monkeypatch.setattr(module, "_file_sha256", lambda _path: "9" * 64)

    result = module._run(args)

    assert result == 0
    assert events.count("build:official-val") == 1
    assert events.count("extract:official-val") == 1
    assert events.index("evaluate:129:(0.25, 0.5, 1.0, 2.0)") < events.index("build:official-val")
    alpha_report = json.loads(
        (report_root / "alpha-selection-report.json").read_text()
    )
    assert alpha_report["selected_alpha"] == selected
    assert alpha_report["split"]["ordered_image_paths"][0] == (
        "images/train/dev-0.jpg"
    )
    assert len(alpha_report["split"]["ordered_image_paths"]) == 129
    assert (report_root / "quality-oracle-report.json").is_file()
    assert (report_root / "quality-oracle-decision.json").is_file()
    inventory = json.loads(
        (report_root / "environment-hash-inventory.json").read_text()
    )
    assert inventory["inputs"]["cache_manifest"] == {
        "path": str((args.cache_root.resolve() / "manifest.json")),
        "sha256": "9" * 64,
    }

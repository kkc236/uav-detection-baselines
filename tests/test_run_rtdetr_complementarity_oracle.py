from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


SCRIPT = Path("scripts/run_rtdetr_complementarity_oracle.py")


def _load_module():
    assert SCRIPT.is_file(), "complementarity-oracle runner has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "run_rtdetr_complementarity_oracle_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_exists_and_cli_is_frozen() -> None:
    module = _load_module()
    argv = [
        "--fdr-checkpoint",
        "fdr.pt",
        "--frequencycm-checkpoint",
        "frequencycm.pt",
        "--dataset-root",
        "dataset",
        "--cache-root",
        "cache",
        "--report-root",
        "report",
    ]

    assert module._parse_args(argv) == Namespace(
        fdr_checkpoint=Path("fdr.pt"),
        frequencycm_checkpoint=Path("frequencycm.pt"),
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
    assert module.VAL_COUNT == 548
    assert module.NUM_CLASSES == 10
    for forbidden in (
        "--threshold",
        "--alpha",
        "--max-det",
        "--conf",
        "--batch",
        "--workers",
        "--nms",
    ):
        with pytest.raises(SystemExit):
            module._parse_args([*argv, forbidden, "1"])


def test_runner_is_bound_to_exact_checkpoint_authorities() -> None:
    module = _load_module()
    assert module.FDR_CHECKPOINT_SHA256 == (
        "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2"
    )
    assert module.FREQUENCYCM_CHECKPOINT_SHA256 == (
        "2BBCD6057FEFED5792F786A18E603F8FECA3EC426A6F68938F5F8ADA1603A141"
    )
    assert module.FREQUENCYCM_SOURCE_COMMIT == (
        "d3655b14c17a3c8ca14e1888517b6fde4e059766"
    )


def test_report_writer_is_create_only_and_labels_oracle_non_deployable(
    tmp_path: Path,
) -> None:
    module = _load_module()
    payload = {
        "decision": {"decision": "yellow"},
        "stock": {
            "fdr": {"map": 0.28966},
            "frequencycm": {"map": 0.28609},
        },
        "oracle": {"candidate_map_delta": 0.004},
        "coverage": {"tiny_small_recall50_delta": 0.012},
    }

    module._write_summary(tmp_path, payload)

    summary = json.loads((tmp_path / "oracle-summary.json").read_text("utf-8"))
    assert summary["interpretation"] == "non_deployable_design_selection_evidence"
    assert (tmp_path / "frequencycm-complementarity-report.md").is_file()
    assert (tmp_path / "SHA256SUMS.txt").is_file()
    with pytest.raises(FileExistsError):
        module._write_summary(tmp_path, payload)


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
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.model = torch.nn.ModuleList([torch.nn.Identity(), _FakeHead()])

    def predict(self, images: torch.Tensor):
        assert torch.is_inference_mode_enabled()
        assert self.model[-1].export is False
        generator = torch.Generator().manual_seed(71)
        boxes = torch.rand((images.shape[0], 300, 4), generator=generator)
        logits = torch.randn((images.shape[0], 300, 10), generator=generator)
        stock = self.model[-1].postprocess(boxes, logits.sigmoid())
        auxiliary = (
            boxes.unsqueeze(0),
            logits.unsqueeze(0),
            torch.empty(0),
            torch.empty(0),
            None,
        )
        return stock, auxiliary


def test_decoder_extraction_reconstructs_exact_stock_and_restores_export() -> None:
    module = _load_module()
    detector = _FakeDetector()
    images = torch.zeros((2, 3, 16, 16))

    stock, boxes, logits = module._extract_decoder_batch(detector, images)

    expected = detector.model[-1].postprocess(boxes, logits.sigmoid())
    assert torch.equal(stock, expected)
    assert boxes.shape == (2, 300, 4)
    assert logits.shape == (2, 300, 10)
    assert detector.model[-1].export is True
    assert not boxes.requires_grad and not logits.requires_grad
    assert all(parameter.grad is None for parameter in detector.parameters())


def test_checkpoint_hash_verification_is_exact_and_uppercase(tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"authority")
    expected = "8F76FD501BB68EF71F4E276BC28F29BCE1003B0C2C9D9478DE81B5BFC0CDE1E9"

    assert module._verify_checkpoint(checkpoint, expected) == expected
    with pytest.raises(RuntimeError, match="checkpoint SHA-256 mismatch"):
        module._verify_checkpoint(checkpoint, "0" * 64)


class _FakeValidator:
    @staticmethod
    def preprocess(batch):
        return batch


def _paired_batch() -> dict[str, object]:
    return {
        "img": torch.zeros((2, 3, 16, 16)),
        "batch_idx": torch.tensor([0, 0, 1]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1], [0.4, 0.4, 0.3, 0.3]]
        ),
        "cls": torch.tensor([[0], [1], [2]]),
        "im_file": ["000001.jpg", "000002.jpg"],
        "ori_shape": [(540, 960), (540, 960)],
    }


def test_paired_record_extraction_uses_one_preprocessed_batch() -> None:
    module = _load_module()
    fdr = _FakeDetector()
    frequencycm = _FakeDetector()

    records = module._extract_paired_records(
        fdr,
        frequencycm,
        [_paired_batch()],
        _FakeValidator(),
        expected_count=2,
    )

    assert [record["image_id"] for record in records] == [
        "000001.jpg",
        "000002.jpg",
    ]
    assert records[0]["original_shape"] == (540, 960)
    assert records[0]["fdr_boxes"].shape == (300, 4)
    assert records[0]["fdr_logits"].shape == (300, 10)
    assert records[0]["frequencycm_boxes"].shape == (300, 4)
    assert records[0]["frequencycm_logits"].shape == (300, 10)
    assert records[0]["target_boxes"].shape == (2, 4)
    assert records[0]["target_classes"].tolist() == [0, 1]
    assert all(
        value.device.type == "cpu" and not value.requires_grad
        for record in records
        for value in record.values()
        if isinstance(value, torch.Tensor)
    )


def test_paired_record_extraction_rejects_wrong_image_count() -> None:
    module = _load_module()
    with pytest.raises(RuntimeError, match="evidence count mismatch"):
        module._extract_paired_records(
            _FakeDetector(),
            _FakeDetector(),
            [_paired_batch()],
            _FakeValidator(),
            expected_count=3,
        )


def _oracle_record() -> dict[str, object]:
    fdr_boxes = torch.full((300, 4), 0.01, dtype=torch.float32)
    frequencycm_boxes = torch.full((300, 4), 0.01, dtype=torch.float32)
    fdr_boxes[0] = torch.tensor([0.25, 0.25, 0.04, 0.04])
    frequencycm_boxes[0] = torch.tensor([0.75, 0.75, 0.04, 0.04])
    fdr_logits = torch.full((300, 10), -20.0, dtype=torch.float32)
    frequencycm_logits = torch.full((300, 10), -20.0, dtype=torch.float32)
    fdr_logits[0, 0] = 10.0
    frequencycm_logits[0, 1] = 10.0
    return {
        "image_id": "000001.jpg",
        "original_shape": (640, 640),
        "fdr_boxes": fdr_boxes,
        "fdr_logits": fdr_logits,
        "frequencycm_boxes": frequencycm_boxes,
        "frequencycm_logits": frequencycm_logits,
        "target_boxes": torch.tensor(
            [[0.25, 0.25, 0.04, 0.04], [0.75, 0.75, 0.04, 0.04]],
            dtype=torch.float32,
        ),
        "target_classes": torch.tensor([0, 1], dtype=torch.long),
    }


def test_run_from_records_writes_every_frozen_output(tmp_path: Path) -> None:
    module = _load_module()

    report = module.run_from_records([_oracle_record()], tmp_path)

    expected = {
        "oracle-summary.json",
        "coverage-by-scale.csv",
        "coverage-by-class.csv",
        "missed-target-categories.csv",
        "oracle-arms.csv",
        "frequencycm-complementarity-report.md",
        "SHA256SUMS.txt",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert report["interpretation"] == "non_deployable_design_selection_evidence"
    assert report["reproduction"]["duplicate_fdr_neutral"] is True
    assert report["coverage"]["tiny_small_recall50_delta"] > 0
    assert report["oracle"]["candidate_map_delta"] >= 0
    for name in expected - {"SHA256SUMS.txt"}:
        assert name in (tmp_path / "SHA256SUMS.txt").read_text("ascii")


def test_run_from_records_is_create_only(tmp_path: Path) -> None:
    module = _load_module()
    module.run_from_records([_oracle_record()], tmp_path)

    with pytest.raises(FileExistsError):
        module.run_from_records([_oracle_record()], tmp_path)


def test_device_is_frozen_to_available_cuda_zero(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert module._device("0") == torch.device("cuda:0")
    with pytest.raises(ValueError, match="device 0"):
        module._device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        module._device("0")


def test_stock_reproduction_uses_frozen_endpoint_tolerance() -> None:
    module = _load_module()
    endpoint = {"precision": 0.5, "recall": 0.4, "ap50": 0.3, "map": 0.2}
    metrics = {
        "precision": 0.5004,
        "recall": 0.3996,
        "ap50": 0.3004,
        "map": 0.1996,
        "ap75": 0.1,
        "ap_tiny": 0.05,
        "ap_small": 0.15,
    }

    report = module._assert_stock_reproduction(metrics, endpoint, label="test")

    assert report["passed"] is True
    assert report["tolerance"] == 0.0005
    with pytest.raises(RuntimeError, match="stock reconstruction mismatch"):
        module._assert_stock_reproduction(
            {**metrics, "map": 0.1994}, endpoint, label="test"
        )


def test_target_scales_undo_square_letterbox_gain_for_original_pixels() -> None:
    module = _load_module()
    record = {
        "original_shape": (540, 960),
        # A 16x16 object becomes 16/960 in each normalized dimension after
        # aspect-preserving letterbox to the frozen square input.
        "target_boxes": torch.tensor(
            [[0.5, 0.5, 16 / 960, 16 / 960]], dtype=torch.float32
        ),
    }

    assert module._target_scales(record) == ("small",)

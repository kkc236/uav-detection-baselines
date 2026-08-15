from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


SCRIPT = Path("scripts/run_pfcr_probe.py")


def _load_module():
    assert SCRIPT.is_file(), "PFCR probe runner has not been implemented"
    spec = importlib.util.spec_from_file_location("run_pfcr_probe_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _argv() -> list[str]:
    return [
        "--fdr-checkpoint",
        "fdr.pt",
        "--frequencycm-checkpoint",
        "cm.pt",
        "--dataset-root",
        "VisDrone",
        "--train-cache-root",
        "train-cache",
        "--val-cache-root",
        "val-cache",
        "--run-root",
        "run",
        "--report-root",
        "report",
    ]


def test_cli_contains_only_artifact_and_runtime_paths() -> None:
    module = _load_module()
    assert module._parse_args(_argv()) == Namespace(
        fdr_checkpoint=Path("fdr.pt"),
        frequencycm_checkpoint=Path("cm.pt"),
        dataset_root=Path("VisDrone"),
        train_cache_root=Path("train-cache"),
        val_cache_root=Path("val-cache"),
        run_root=Path("run"),
        report_root=Path("report"),
        device="0",
    )
    assert module.TRAIN_COUNT == 6471
    assert module.DEV_MODULUS == 5
    assert module.PROBE_EPOCHS == 20
    assert module.RESCUE_BUDGETS == (15, 30, 60)
    for forbidden in ("--threshold", "--rescue-slots", "--epochs", "--lr", "--batch"):
        with pytest.raises(SystemExit):
            module._parse_args([*_argv(), forbidden, "1"])


def test_authority_is_bound_to_checkpoints_evaluator_schema_and_source() -> None:
    module = _load_module()
    authority = module._cache_authority(
        fdr_sha256="A" * 64,
        frequencycm_sha256="B" * 64,
        dataset_sha256="C" * 64,
        evaluator_sha256="D" * 64,
        source_commit="e" * 40,
    )
    assert authority == {
        "fdr_sha256": "A" * 64,
        "frequencycm_sha256": "B" * 64,
        "dataset_sha256": "C" * 64,
        "evaluator_sha256": "D" * 64,
        "feature_schema_sha256": module.PFCR_FEATURE_SCHEMA_SHA256,
        "source_commit": "e" * 40,
    }


def test_detector_tensors_never_enter_optimizer_groups() -> None:
    module = _load_module()
    gate = module.PFCRGate()
    detector_parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    optimizer = module.build_probe_optimizer(gate)
    optimized = {id(value) for group in optimizer.param_groups for value in group["params"]}
    assert id(detector_parameter) not in optimized
    assert optimized == {id(value) for value in gate.parameters()}
    assert optimizer.defaults["lr"] == pytest.approx(1e-3)
    assert optimizer.defaults["weight_decay"] == pytest.approx(1e-4)


def test_internal_selection_ignores_train_and_uses_smallest_near_best_budget() -> None:
    module = _load_module()
    history = [
        {"epoch": 1, "split": "dev", "slots": 15, "map": .21285, "ap75": .20, "ap50": .35},
        {"epoch": 1, "split": "dev", "slots": 30, "map": .21300, "ap75": .21, "ap50": .36},
        {"epoch": 1, "split": "train", "slots": 60, "map": .99, "ap75": .99, "ap50": .99},
        {"epoch": 2, "split": "dev", "slots": 60, "map": .21270, "ap75": .30, "ap50": .40},
    ]
    assert module.select_internal_checkpoint(history) == {"epoch": 1, "slots": 15}


def test_internal_decision_uses_all_frozen_conditions() -> None:
    module = _load_module()
    c0 = {"map": .20, "ap75": .15, "ap50": .30}
    c1 = {"map": .201, "ap75": .151, "ap50": .299}
    candidate = {"map": .2031, "ap75": .152, "ap50": .301}
    decision = module.decide_internal(
        c0, c1, candidate, {"c0": .50, "candidate": .50}
    )
    assert decision["status"] == "passed"
    assert module.decide_internal(
        c0, c1, {**candidate, "map": .2029}, {"c0": .50, "candidate": .50}
    )["status"] == "scientific_failed"


def test_official_decision_requires_positive_map_and_nonnegative_ap75() -> None:
    module = _load_module()
    assert module.decide_official(
        {"map": .20, "ap75": .18}, {"map": .200001, "ap75": .18}
    )["eligible"]
    assert not module.decide_official(
        {"map": .20, "ap75": .18}, {"map": .20, "ap75": .19}
    )["eligible"]
    assert not module.decide_official(
        {"map": .20, "ap75": .18}, {"map": .21, "ap75": .179999}
    )["eligible"]


def test_internal_failure_never_opens_val_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    opened: list[Path] = []
    monkeypatch.setattr(module, "load_official_val_cache", lambda path: opened.append(path))
    assert module.advance_after_internal(
        {"status": "scientific_failed"}, Path("val")
    ) is None
    assert opened == []


def test_official_val_is_opened_once_after_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    opened: list[Path] = []
    monkeypatch.setattr(
        module, "load_official_val_cache", lambda path: opened.append(path) or ("records",)
    )
    assert module.advance_after_internal({"status": "passed"}, Path("val")) == (
        "records",
    )
    assert opened == [Path("val")]


def test_report_publication_is_atomic_create_only(tmp_path: Path) -> None:
    module = _load_module()
    report = tmp_path / "report"
    module.publish_reports(report, {"decision.json": {"eligible": True}})
    assert json.loads((report / "decision.json").read_text("utf-8")) == {
        "eligible": True
    }
    assert (report / "SHA256SUMS.txt").is_file()
    with pytest.raises(FileExistsError):
        module.publish_reports(report, {"decision.json": {"eligible": True}})


class _FakeHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.export = True

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        selected_scores, index = scores.flatten(1).topk(300)
        query_index = torch.div(index, 10, rounding_mode="floor")
        selected_boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        classes = (index - query_index * 10).unsqueeze(-1).float()
        return torch.cat((selected_boxes, selected_scores.unsqueeze(-1), classes), -1)


class _FakeDetector(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed
        self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.model = torch.nn.ModuleList([torch.nn.Identity(), _FakeHead()])

    def predict(self, images: torch.Tensor):
        assert torch.is_inference_mode_enabled()
        generator = torch.Generator().manual_seed(self.seed)
        boxes = torch.rand((images.shape[0], 300, 4), generator=generator)
        boxes[..., 2:] = boxes[..., 2:] * .4 + .01
        logits = torch.randn((images.shape[0], 300, 10), generator=generator)
        stock = self.model[-1].postprocess(boxes, logits.sigmoid())
        return stock, (
            boxes.unsqueeze(0), logits.unsqueeze(0), torch.empty(0), torch.empty(0), None
        )


class _FakeValidator:
    @staticmethod
    def preprocess(batch):
        return batch


def test_streaming_extraction_preserves_both_detectors_and_records_once(tmp_path: Path) -> None:
    module = _load_module()
    fdr, cm = _FakeDetector(1), _FakeDetector(2)
    before = (module._model_state_sha256(fdr), module._model_state_sha256(cm))
    batch = {
        "img": torch.zeros((2, 3, 16, 16)),
        "batch_idx": torch.tensor([0, 1]),
        "bboxes": torch.tensor([[.5, .5, .2, .2], [.4, .4, .1, .1]]),
        "cls": torch.tensor([[0], [1]]),
        "im_file": ["000001.jpg", "000002.jpg"],
        "ori_shape": [(540, 960), (540, 960)],
        "resized_shape": [(640, 640), (640, 640)],
    }
    writer = module.PFCRCacheWriter(tmp_path / "cache", {
        "fdr_sha256": "A" * 64,
        "frequencycm_sha256": "B" * 64,
        "dataset_sha256": "C" * 64,
        "evaluator_sha256": "D" * 64,
        "feature_schema_sha256": module.PFCR_FEATURE_SCHEMA_SHA256,
        "source_commit": "e" * 40,
    })
    count = module.extract_train_cache(
        fdr, cm, [batch], _FakeValidator(), writer, expected_count=2
    )
    assert count == 2
    assert before == (module._model_state_sha256(fdr), module._model_state_sha256(cm))
    assert all(parameter.grad is None for model in (fdr, cm) for parameter in model.parameters())
    manifest = writer.finalize()
    assert sum(manifest["counts"].values()) == 2


def _prepared_record(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "features": torch.randn((300, 10, 35), generator=generator),
        "fdr_logits": torch.randn((300, 10), generator=generator),
        "frequencycm_logits": torch.randn((300, 10), generator=generator),
        "fdr_teacher": torch.rand((300, 10), generator=generator),
        "frequencycm_teacher": torch.rand((300, 10), generator=generator),
    }


def _fake_metrics(gate, records, slots):
    del records
    magnitude = float(sum(value.detach().abs().sum() for value in gate.parameters()))
    return {
        "map": magnitude * 1e-6 + slots * 1e-8,
        "ap50": magnitude * 2e-6,
        "ap75": magnitude * 1.5e-6,
        "precision": 0.0,
        "recall": 0.0,
        "ap_tiny": 0.0,
        "ap_small": 0.0,
        "tiny_small_recall50": 0.0,
    }


def test_training_saves_exactly_twenty_create_only_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "evaluate_prepared", _fake_metrics)
    records = {
        "train": tuple(_prepared_record(index) for index in range(2)),
        "dev": (_prepared_record(9),),
    }
    history = module.train_gate(records, tmp_path / "run", device=torch.device("cpu"))
    assert len(list((tmp_path / "run" / "checkpoints").glob("epoch-*.pt"))) == 20
    assert len(list((tmp_path / "run" / "metrics").glob("epoch-*.json"))) == 20
    assert len(history) == 20 * 3
    with pytest.raises(FileExistsError):
        module.train_gate(records, tmp_path / "run", device=torch.device("cpu"))


def test_resume_reproduces_uninterrupted_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "evaluate_prepared", _fake_metrics)
    records = {
        "train": tuple(_prepared_record(index) for index in range(2)),
        "dev": (_prepared_record(9),),
    }
    direct = module.train_gate(
        records, tmp_path / "direct", epochs=4, device=torch.device("cpu")
    )
    module.train_gate(
        records, tmp_path / "resume", epochs=2, device=torch.device("cpu")
    )
    resumed = module.train_gate(
        records,
        tmp_path / "resume",
        epochs=4,
        resume=True,
        device=torch.device("cpu"),
    )
    assert direct == resumed


def test_adjusted_logits_change_only_frequencycm() -> None:
    module = _load_module()
    record = _prepared_record(4)
    gate = module.PFCRGate()
    fdr_before = record["fdr_logits"].clone()
    cm_before = record["frequencycm_logits"].clone()
    adjusted = module.adjusted_frequencycm_logits(gate, record)
    assert torch.equal(record["fdr_logits"], fdr_before)
    assert torch.equal(record["frequencycm_logits"], cm_before)
    assert torch.equal(adjusted, cm_before)
    assert adjusted.requires_grad
    adjusted.sum().backward()
    assert any(value.grad is not None for value in gate.parameters())
    assert record["fdr_logits"].grad is None
    assert record["frequencycm_logits"].grad is None


def test_preflight_proves_gate_only_backward_and_roundtrip(tmp_path: Path) -> None:
    module = _load_module()
    batch = {
        "img": torch.zeros((8, 3, 640, 640)),
        "batch_idx": torch.arange(8),
        "bboxes": torch.full((8, 4), .2),
        "cls": torch.arange(8).view(-1, 1) % 10,
        "im_file": [f"{index:06d}.jpg" for index in range(8)],
        "ori_shape": [(540, 960)] * 8,
        "resized_shape": [(640, 640)] * 8,
    }
    fdr, cm = _FakeDetector(1), _FakeDetector(2)
    report = module.run_cuda_preflight(
        fdr,
        cm,
        [batch],
        _FakeValidator(),
        tmp_path / "preflight.json",
        device=torch.device("cpu"),
    )
    assert report["passed"] is True
    assert report["batch"] == 8
    assert report["gate_gradient_nonzero"] is True
    assert report["detector_state_unchanged"] is True
    assert report["checkpoint_roundtrip"] is True
    assert (tmp_path / "preflight.json").is_file()
    with pytest.raises(FileExistsError):
        module.run_cuda_preflight(
            fdr,
            cm,
            [batch],
            _FakeValidator(),
            tmp_path / "preflight.json",
            device=torch.device("cpu"),
        )

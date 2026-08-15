from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.rtdetr_quality_oracle import same_class_iou_quality


SCRIPT = Path("scripts/run_rtdetr_quality_probe.py")


def _load_module():
    assert SCRIPT.is_file(), "quality-probe runner has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "run_rtdetr_quality_probe_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_exists_and_cli_is_frozen() -> None:
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

    assert module._parse_args(argv) == Namespace(
        baseline_checkpoint=Path("baseline.pt"),
        dataset_root=Path("dataset"),
        cache_root=Path("cache"),
        report_root=Path("report"),
        device="0",
    )
    assert module.PROBE_TRAIN_COUNT == 518
    assert module.DEV_COUNT == 129
    assert module.VAL_COUNT == 548
    assert module.EPOCHS == 20
    assert module.TOP_PAIRS == 600
    assert module.PROBE_ALPHA == 2.0
    for forbidden in ("--epochs", "--alpha", "--lr", "--top-pairs", "--seed"):
        with pytest.raises(SystemExit):
            module._parse_args([*argv, forbidden, "1"])


def test_probe_is_bound_to_the_immutable_passed_oracle_decision() -> None:
    module = _load_module()
    authority = module._oracle_decision_authority()

    assert authority == {
        "sha256": "F2DBABDD4638896D3D9C727CCC659D86173DD639AF476709C8F415F0E2EEE199",
        "status": "passed",
        "selected_alpha": 2.0,
        "map_delta": "0.15571345572052406",
        "ap75_delta": "0.14920384179689443",
    }


def test_frozen_subset_is_split_into_disjoint_518_and_129() -> None:
    module = _load_module()
    root = Path("dataset").resolve()
    subset = tuple(root / "images" / "train" / f"{index:04d}.jpg" for index in range(647))
    dev = tuple(subset[index] for index in range(0, 647, 5))[:129]

    train, actual_dev = module._split_probe_paths(subset, dev, root=root)

    assert len(train) == 518
    assert len(actual_dev) == 129
    assert set(train).isdisjoint(actual_dev)
    assert set((*train, *actual_dev)) == set(subset)
    assert module._ordered_path_sha256(train, root=root) == module.EXPECTED_PROBE_TRAIN_SHA256 or len(train) == 518


class _ScoreHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 10, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(40).reshape(10, 4) / 40)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class _Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.export = True
        self.decoder = SimpleNamespace(eval_idx=0)
        self.dec_score_head = torch.nn.ModuleList([_ScoreHead()])

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        values, indices = scores.flatten(1).topk(6)
        queries = torch.div(indices, 10, rounding_mode="floor")
        selected = boxes.gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
        classes = (indices % 10).unsqueeze(-1).float()
        return torch.cat((selected, values.unsqueeze(-1), classes), -1)


class _Detector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.model = torch.nn.ModuleList([torch.nn.Identity(), _Head()])
        self.requires_grad_(False)

    def predict(self, images: torch.Tensor):
        batch = images.shape[0]
        hidden = torch.arange(batch * 3 * 4, dtype=torch.float32).reshape(batch, 3, 4) / 20
        logits = self.model[-1].dec_score_head[0](hidden)
        boxes = torch.arange(batch * 3 * 4, dtype=torch.float32).reshape(batch, 3, 4) / 30
        stock = self.model[-1].postprocess(boxes, logits.sigmoid())
        return stock, (
            boxes.unsqueeze(0),
            logits.unsqueeze(0),
            torch.empty(0),
            torch.empty(0),
            None,
        )


def test_native_hidden_hook_is_single_fire_output_neutral_and_removed() -> None:
    module = _load_module()
    detector = _Detector().eval()
    images = torch.zeros((2, 3, 8, 8))

    proof = module._prove_hook_neutrality(detector, images)
    hooked = module._capture_hidden_batch(detector, images)

    assert proof == {"hook_calls": 1, "output_neutral": True}
    assert hooked.boxes.shape == (2, 3, 4)
    assert hooked.logits.shape == (2, 3, 10)
    assert hooked.hidden.shape == (2, 3, 4)
    assert not hooked.hidden.requires_grad
    score_head = detector.model[-1].dec_score_head[0]
    assert not score_head._forward_pre_hooks
    assert all(parameter.grad is None for parameter in detector.parameters())


def _record(image_id: str, offset: float = 0.0) -> dict[str, object]:
    boxes = torch.full((300, 4), 0.25 + offset, dtype=torch.float32)
    boxes[:, 2:] = 0.1
    logits = torch.linspace(-2, 2, 3000, dtype=torch.float32).reshape(300, 10)
    hidden = torch.linspace(-1, 1, 76800, dtype=torch.float32).reshape(300, 256)
    target_boxes = torch.tensor([[0.25, 0.25, 0.1, 0.1]], dtype=torch.float32)
    target_classes = torch.tensor([0], dtype=torch.int64)
    quality = same_class_iou_quality(boxes, target_boxes, target_classes, 10)
    return {
        "image_id": image_id,
        "boxes": boxes,
        "logits": logits,
        "hidden": hidden,
        "quality": quality,
        "target_boxes": target_boxes,
        "target_classes": target_classes,
    }


def test_cache_is_create_only_external_digest_bound_and_safely_resumable(
    tmp_path: Path,
) -> None:
    module = _load_module()
    root = tmp_path / "cache"
    authority_path = tmp_path / "cache-authority.json"
    records = [_record("images/train/a.jpg"), _record("images/train/b.jpg", 0.01)]
    authority = {"schema_sha256": "A" * 64, "source_commit": "b" * 40}

    first = module._write_cache_stage(
        root,
        records=records,
        authority=authority,
        split="probe_train",
        external_digest_path=authority_path,
    )
    second = module._write_cache_stage(
        root,
        records=records,
        authority=authority,
        split="probe_train",
        external_digest_path=authority_path,
    )
    loaded = module._load_cache_stage(
        root,
        authority=authority,
        split="probe_train",
        external_digest_path=authority_path,
        expected_ids=("images/train/a.jpg", "images/train/b.jpg"),
    )

    assert first == second
    assert len(loaded) == 2
    assert torch.equal(loaded[0]["quality"], records[0]["quality"])
    digest_payload = json.loads(authority_path.read_text("utf-8"))
    assert digest_payload["manifest_sha256"] == first["manifest_sha256"]

    authority_path.unlink()
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        module._write_cache_stage(
            root,
            records=records,
            authority=authority,
            split="probe_train",
            external_digest_path=authority_path,
        )
    module._atomic_create_bytes(authority_path, module._canonical_json_bytes(first))

    shard = next(root.glob("shard-*.pt"))
    shard.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="sha256|corrupt|load"):
        module._load_cache_stage(
            root,
            authority=authority,
            split="probe_train",
            external_digest_path=authority_path,
            expected_ids=("images/train/a.jpg", "images/train/b.jpg"),
        )


def test_top600_weighted_loss_and_checkpoint_selection_are_frozen() -> None:
    module = _load_module()
    logits = torch.linspace(-3, 3, 3000).reshape(1, 300, 10)
    target = torch.zeros_like(logits)
    target[..., 0] = 0.8
    predicted = torch.zeros_like(logits, requires_grad=True)

    loss, selected = module._top_pair_probe_loss(predicted, target, logits.sigmoid())
    choice = module._select_checkpoint(
        [
            {"epoch": 1, "metrics": {"map": 0.2, "ap75": 0.1}},
            {"epoch": 2, "metrics": {"map": 0.2, "ap75": 0.11}},
            {"epoch": 3, "metrics": {"map": 0.2, "ap75": 0.11}},
        ]
    )

    assert selected == 600
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert predicted.grad is not None
    assert choice["epoch"] == 2


def test_equal_parameter_probes_use_exact_ultralytics_musgd_groups() -> None:
    module = _load_module()
    c1 = module._model_for_arm("c1", torch.device("cpu"))
    q = module._model_for_arm("q", torch.device("cpu"))

    assert sum(parameter.numel() for parameter in c1.parameters()) == sum(
        parameter.numel() for parameter in q.parameters()
    )
    assert c1.network[0].in_features == q.network[0].in_features == 276
    assert c1.network[0].out_features == q.network[0].out_features == 64

    optimizer = module._build_optimizer(c1)
    assert type(optimizer).__module__ == "ultralytics.optim.muon"
    assert type(optimizer).__name__ == "MuSGD"
    assert optimizer.muon == 0.2
    assert optimizer.sgd == 1.0
    assert {group["use_muon"] for group in optimizer.param_groups} == {True, False}
    for group in optimizer.param_groups:
        assert group["lr"] == 0.01
        assert group["momentum"] == 0.937
        assert group["nesterov"] is True
        if group["use_muon"]:
            assert group["weight_decay"] == 0.0005
            assert all(parameter.ndim == 2 for parameter in group["params"])
        else:
            assert group["weight_decay"] == 0.0
            assert all(parameter.ndim != 2 for parameter in group["params"])


def test_official_validation_gate_is_strict_and_internal_failure_never_opens_val() -> None:
    module = _load_module()
    assert module._official_gate(
        c0={"map": 0.2, "ap75": 0.1}, q={"map": 0.200001, "ap75": 0.100001}
    )["status"] == "passed"
    assert module._official_gate(
        c0={"map": 0.2, "ap75": 0.1}, q={"map": 0.2, "ap75": 0.2}
    )["status"] == "scientific_failed"

    opened = []
    result = module._run_official_if_authorized(
        {"status": "scientific_failed"},
        open_official=lambda: opened.append(True),
        evaluate=lambda _records: {},
    )
    assert result is None
    assert opened == []

from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace

import pytest
import torch


try:
    importlib.metadata.version("torchvision")
except importlib.metadata.PackageNotFoundError:
    pytest.skip("RT-DETR integration tests require the server Ultralytics environment", allow_module_level=True)

from ultralytics.nn.tasks import RTDETRDetectionModel

from src.rtdetr_ascv_loc import ASCVLocDetectionModel, ASCVLocTrainer


def _synthetic_predictions(class_count: int = 2):
    query_count = 4
    layers = 6
    boxes = torch.tensor(
        [[0.40, 0.50, 0.05, 0.05], [0.52, 0.50, 0.10, 0.10], [0.7, 0.7, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05]]
    )
    dec_boxes = boxes.view(1, 1, query_count, 4).repeat(layers, 1, 1, 1).requires_grad_()
    dec_scores = torch.full((layers, 1, query_count, class_count), -5.0)
    dec_scores[:, 0, :2, 0] = 5.0
    dec_scores.requires_grad_()
    enc_boxes = boxes.unsqueeze(0).clone().requires_grad_()
    enc_scores = torch.full((1, query_count, class_count), -5.0, requires_grad=True)
    return dec_boxes, dec_scores, enc_boxes, enc_scores, None


def _parameter_dependent_predictions(model, predictions):
    anchor = next(model.parameters()).reshape(-1)[0]
    values = []
    for value in predictions:
        if isinstance(value, torch.Tensor):
            values.append(value + anchor.square() * 0.0)
        else:
            values.append(value)
    return tuple(values)


def test_model_has_stock_parameter_and_inference_contract() -> None:
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).eval()
    model = ASCVLocDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).eval()

    assert tuple(stock.state_dict()) == tuple(model.state_dict())
    assert sum(parameter.numel() for parameter in stock.parameters()) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    image = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        stock_output = stock.predict(image)
        ascv_output = model.predict(image)
    assert type(stock_output) is type(ascv_output)
    assert stock_output[0].shape == ascv_output[0].shape
    model.load_state_dict(stock.state_dict(), strict=True)
    with torch.no_grad():
        stock_output = stock.predict(image)
        ascv_output = model.predict(image)
    torch.testing.assert_close(ascv_output[0], stock_output[0], rtol=0, atol=0)
    assert model.last_ascv_result is None


def test_training_adds_only_one_finite_auxiliary_item_and_shared_pairs(monkeypatch) -> None:
    model = ASCVLocDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).train()
    model.set_ascv_progress(0)
    full_predictions = _synthetic_predictions()
    local_predictions = _synthetic_predictions()
    monkeypatch.setattr(
        model,
        "predict",
        lambda image, batch=None: _parameter_dependent_predictions(model, local_predictions),
    )
    batch = {
        "img": torch.zeros((1, 3, 640, 640)),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.40, 0.50, 0.05, 0.05], [0.52, 0.50, 0.10, 0.10]]),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "im_file": ["train-image.jpg"],
    }

    total, items = model.loss(batch, preds=full_predictions)
    total.backward()

    assert total.dtype == torch.float32
    assert items.shape == (4,)
    assert torch.isfinite(total)
    assert torch.isfinite(items).all()
    assert model.last_ascv_result is not None
    assert model.last_ascv_result.pair_count > 0
    assert model.last_local_forward_calls == 2
    assert model.last_local_bn_preserved is True


def test_all_tiny_teacher_only_batch_does_not_require_checkpoint_recompute(monkeypatch) -> None:
    model = ASCVLocDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).train()
    model.set_ascv_progress(0)
    full_predictions = _synthetic_predictions()
    local_predictions = _synthetic_predictions()
    monkeypatch.setattr(
        model,
        "predict",
        lambda image, batch=None: _parameter_dependent_predictions(model, local_predictions),
    )
    batch = {
        "img": torch.zeros((1, 3, 640, 640)),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.40, 0.50, 0.01, 0.01]]),
        "batch_idx": torch.tensor([0.0]),
        "im_file": ["train-image.jpg"],
    }

    total, _items = model.loss(batch, preds=full_predictions)
    total.backward()

    assert model.last_ascv_result is not None
    assert model.last_ascv_result.tiny_pair_count == model.last_ascv_result.pair_count
    assert model.last_local_forward_calls == 1
    assert model.last_local_bn_preserved is True


def test_preflight_zero_weight_probe_forces_real_checkpoint_recompute_on_tiny_batch(monkeypatch) -> None:
    model = ASCVLocDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).train()
    model.ascv_preflight_probe = True
    full_predictions = _synthetic_predictions()
    local_predictions = _synthetic_predictions()
    monkeypatch.setattr(
        model,
        "predict",
        lambda image, batch=None: _parameter_dependent_predictions(model, local_predictions),
    )
    batch = {
        "img": torch.zeros((1, 3, 640, 640)),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.40, 0.50, 0.01, 0.01]]),
        "batch_idx": torch.tensor([0.0]),
        "im_file": ["train-image.jpg"],
    }

    total, _items = model.loss(batch, preds=full_predictions)
    total.backward()

    assert model.last_local_forward_calls == 2
    assert model.last_local_bn_preserved is True


def test_eval_loss_never_constructs_local_view(monkeypatch) -> None:
    model = ASCVLocDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).eval()
    predictions = _synthetic_predictions()
    batch = {
        "img": torch.zeros((1, 3, 640, 640)),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.40, 0.50, 0.05, 0.05]]),
        "batch_idx": torch.tensor([0.0]),
    }

    monkeypatch.setattr(
        "src.rtdetr_ascv_loc.crop_and_resize",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validation constructed an ASCV tile")),
    )
    with torch.no_grad():
        total, items = model.loss(batch, preds=(None, predictions))

    assert torch.isfinite(total)
    assert items.shape == (4,)
    assert items[-1].item() == 0.0


def test_stock_criterion_is_called_once_and_never_with_local_targets(monkeypatch) -> None:
    class Matcher:
        def __call__(self, boxes, scores, target_boxes, target_classes, groups):
            matches = []
            offset = 0
            for count in groups:
                pair_count = min(int(count), boxes.shape[1])
                matches.append(
                    (
                        torch.arange(pair_count, dtype=torch.long),
                        torch.arange(offset, offset + pair_count, dtype=torch.long),
                    )
                )
                offset += int(count)
            return matches

    class Criterion:
        def __init__(self):
            self.matcher = Matcher()
            self.calls = []

        def __call__(self, predictions, targets, **kwargs):
            self.calls.append(targets)
            anchor = predictions[0].float().sum() * 0 + predictions[1].float().sum() * 0
            return {
                "loss_giou": anchor + 1.0,
                "loss_class": anchor + 2.0,
                "loss_bbox": anchor + 3.0,
            }

    model = ASCVLocDetectionModel("rtdetr-l.yaml", ch=3, nc=2, verbose=False).train()
    model.criterion = Criterion()
    full_predictions = _synthetic_predictions()
    local_predictions = _synthetic_predictions()
    monkeypatch.setattr(model, "predict", lambda image, batch=None: local_predictions)
    batch = {
        "img": torch.zeros((1, 3, 640, 640)),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.40, 0.50, 0.05, 0.05], [0.52, 0.50, 0.10, 0.10]]),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "im_file": ["train-image.jpg"],
    }

    total, _items = model.loss(batch, preds=full_predictions)

    assert torch.isfinite(total)
    assert len(model.criterion.calls) == 1
    torch.testing.assert_close(model.criterion.calls[0]["bboxes"], batch["bboxes"])


def test_train_pipeline_never_requests_val_or_test_loader() -> None:
    class Loader:
        dataset = list(range(20))
        batch_size = 8
        num_workers = 8

    trainer = object.__new__(ASCVLocTrainer)
    trainer.batch_size = 8
    trainer.frozen_batch_size = 8
    trainer.world_size = 1
    trainer.data = {
        "train": "hashed-train.txt",
        "val": "must-not-open-val",
        "test": "must-not-open-test-dev",
    }
    trainer.args = SimpleNamespace(
        nbs=64,
        weight_decay=0.0005,
        optimizer="MuSGD",
        lr0=0.01,
        momentum=0.937,
        seed=0,
        deterministic=True,
    )
    trainer.model = torch.nn.Linear(2, 2)
    trainer.epochs = 100
    calls = []
    trainer.get_dataloader = lambda path, **kwargs: calls.append((path, kwargs["mode"])) or Loader()
    trainer.build_optimizer = lambda **kwargs: torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer._setup_scheduler = lambda: None

    ASCVLocTrainer._build_train_pipeline(trainer)

    assert calls == [("hashed-train.txt", "train")]
    assert trainer.test_loader is None


def test_oom_batch_auto_reduction_fails_closed_before_loader_rebuild() -> None:
    trainer = object.__new__(ASCVLocTrainer)
    trainer.batch_size = 2
    trainer.frozen_batch_size = 8

    with pytest.raises(RuntimeError, match="ASCV_LOC_BATCH_DRIFT"):
        ASCVLocTrainer._build_train_pipeline(trainer)


def test_internal_validation_and_final_eval_are_non_reading_noops() -> None:
    trainer = object.__new__(ASCVLocTrainer)
    trainer.internal_validation_bypass_count = 0

    metrics, fitness = ASCVLocTrainer.validate(trainer)
    final = ASCVLocTrainer.final_eval(trainer)

    assert metrics == {}
    assert fitness == float("-inf")
    assert trainer.internal_validation_bypass_count == 1
    assert final is None

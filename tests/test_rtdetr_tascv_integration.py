from __future__ import annotations

import importlib.metadata
import inspect
from types import SimpleNamespace

import pytest
import torch


try:
    importlib.metadata.version("torchvision")
except importlib.metadata.PackageNotFoundError:
    pytest.skip(
        "T-ASCV integration tests require the server Ultralytics environment",
        allow_module_level=True,
    )

from ultralytics.nn.tasks import RTDETRDetectionModel

from src.rtdetr_tascv import (
    CONTROL_LOSS_NAMES,
    LOSS_NAMES,
    MATCHED_AMP_SCALE,
    TASCVControlTrainer,
    TASCVDetectionModel,
    TASCVTrainer,
    load_matched_initial_state,
)
from src.ascv_loc_protocol import state_fingerprint
from src.tascv_stage import TASCVStage, stage_policy


def _predictions(
    boxes: torch.Tensor | None = None,
    *,
    class_count: int = 2,
):
    query_boxes = (
        boxes
        if boxes is not None
        else torch.tensor(
            [
                [0.50, 0.50, 0.04, 0.04],
                [0.50, 0.50, 0.10, 0.10],
                [0.70, 0.70, 0.05, 0.05],
                [0.90, 0.90, 0.05, 0.05],
            ]
        )
    )
    layers = 6
    queries = len(query_boxes)
    dec_boxes = (
        query_boxes.view(1, 1, queries, 4)
        .repeat(layers, 1, 1, 1)
        .requires_grad_()
    )
    dec_scores = torch.full(
        (layers, 1, queries, class_count),
        -5.0,
    )
    dec_scores[:, 0, :2, 0] = 5.0
    dec_scores.requires_grad_()
    enc_boxes = query_boxes.unsqueeze(0).clone().requires_grad_()
    enc_scores = torch.full(
        (1, queries, class_count),
        -5.0,
        requires_grad=True,
    )
    return dec_boxes, dec_scores, enc_boxes, enc_scores, None


def _parameter_dependent(model, predictions):
    anchor = next(model.parameters()).reshape(-1)[0]
    return tuple(
        value + anchor.square() * 0.0
        if isinstance(value, torch.Tensor)
        else value
        for value in predictions
    )


class _SequentialMatcher:
    def __call__(
        self,
        boxes,
        scores,
        target_boxes,
        target_classes,
        groups,
    ):
        matches = []
        offset = 0
        for count in groups:
            pair_count = min(int(count), boxes.shape[1])
            matches.append(
                (
                    torch.arange(pair_count, dtype=torch.long),
                    torch.arange(
                        offset,
                        offset + pair_count,
                        dtype=torch.long,
                    ),
                )
            )
            offset += int(count)
        return matches


class _ZeroStockCriterion:
    def __init__(self):
        self.matcher = _SequentialMatcher()
        self.calls = []

    def __call__(self, predictions, targets, **kwargs):
        self.calls.append(targets)
        anchor = (
            predictions[0].float().sum() * 0.0
            + predictions[1].float().sum() * 0.0
        )
        return {
            "loss_giou": anchor,
            "loss_class": anchor,
            "loss_bbox": anchor,
        }


def _batch():
    return {
        "img": torch.zeros((1, 3, 640, 640)),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor(
            [
                [0.50, 0.50, 0.02, 0.02],
                [0.50, 0.50, 0.10, 0.10],
            ]
        ),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "im_file": ["train-image.jpg"],
    }


def test_tascv_model_has_stock_parameter_and_inference_contract() -> None:
    stock = RTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=2, verbose=False
    ).eval()
    model = TASCVDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=2, verbose=False
    ).eval()

    assert LOSS_NAMES == (
        "giou_loss",
        "cls_loss",
        "l1_loss",
        "tascv_loss",
    )
    assert tuple(stock.state_dict()) == tuple(model.state_dict())
    assert sum(p.numel() for p in stock.parameters()) == sum(
        p.numel() for p in model.parameters()
    )
    model.load_state_dict(stock.state_dict(), strict=True)
    image = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        stock_output = stock.predict(image)
        treatment_output = model.predict(image)
    torch.testing.assert_close(
        treatment_output[0],
        stock_output[0],
        rtol=0,
        atol=0,
    )
    assert model.last_tascv_result is None


def test_tascv_module_cannot_import_stopped_bidirectional_logic() -> None:
    import src.rtdetr_tascv as module

    source = inspect.getsource(module)
    forbidden = (
        "compute_ascv_loc_loss",
        "ASCVLocLossResult",
        "full_to_local_xywh",
        "rtdetr_ascv_loc",
        "ascv_loc_diagnostics",
        "ascv_loc_stage",
    )
    assert not any(token in source for token in forbidden)


def test_eval_loss_never_constructs_local_view(monkeypatch) -> None:
    model = TASCVDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=2, verbose=False
    ).eval()
    monkeypatch.setattr(
        "src.rtdetr_tascv.crop_and_resize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("eval constructed a T-ASCV crop")
        ),
    )

    with torch.no_grad():
        total, items = model.loss(
            {
                "img": torch.zeros((1, 3, 640, 640)),
                "cls": torch.tensor([[0.0]]),
                "bboxes": torch.tensor(
                    [[0.5, 0.5, 0.02, 0.02]]
                ),
                "batch_idx": torch.tensor([0.0]),
            },
            preds=(None, _predictions()),
        )

    assert torch.isfinite(total)
    assert items.shape == (4,)
    assert items[-1].item() == 0.0


def test_stock_criterion_once_and_local_predict_has_no_batch(
    monkeypatch,
) -> None:
    model = TASCVDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=2, verbose=False
    ).train()
    model.criterion = _ZeroStockCriterion()
    local_predictions = _predictions()
    observed_batches = []

    def predict(image, batch=None):
        observed_batches.append(batch)
        return _parameter_dependent(model, local_predictions)

    monkeypatch.setattr(model, "predict", predict)
    monkeypatch.setattr(
        "src.rtdetr_tascv.select_target_anchored_crops",
        lambda **kwargs: torch.tensor([[128, 128, 512, 512]]),
    )

    total, items = model.loss(_batch(), preds=_predictions())

    assert torch.isfinite(total)
    assert items.shape == (4,)
    assert len(model.criterion.calls) == 1
    assert observed_batches == [None]


def test_mixed_batch_backpropagates_only_tiny_full_student(
    monkeypatch,
) -> None:
    model = TASCVDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=2, verbose=False
    ).train()
    model.criterion = _ZeroStockCriterion()
    full = _predictions()
    local = _predictions(
        torch.tensor(
            [
                [0.50, 0.50, 1 / 30, 1 / 30],
                [0.50, 0.50, 0.20, 0.20],
                [0.70, 0.70, 0.05, 0.05],
                [0.90, 0.90, 0.05, 0.05],
            ]
        )
    )
    monkeypatch.setattr(
        model,
        "predict",
        lambda image, batch=None: _parameter_dependent(model, local),
    )
    monkeypatch.setattr(
        "src.rtdetr_tascv.select_target_anchored_crops",
        lambda **kwargs: torch.tensor([[128, 128, 512, 512]]),
    )

    total, items = model.loss(_batch(), preds=full)
    total.backward()

    result = model.last_tascv_result
    assert result is not None
    assert result.matched_pair_count == 2
    assert result.auxiliary_tiny_pair_count == 1
    assert result.excluded_non_tiny_pair_count == 1
    assert total.dtype == torch.float32
    assert items.shape == (4,)
    full_gradient = full[0].grad[-1, 0]
    assert full_gradient[0, 2].item() > 0
    assert torch.equal(
        full_gradient[1],
        torch.zeros_like(full_gradient[1]),
    )
    assert local[0].grad is None or torch.equal(
        local[0].grad,
        torch.zeros_like(local[0].grad),
    )
    assert model.last_local_bn_preserved is True
    assert (
        model.last_tascv_diagnostics[
            "auxiliary_non_tiny_pair_count"
        ]
        == 0
    )


@pytest.mark.parametrize("preflight", (False, True))
def test_teacher_checkpoint_call_contract(monkeypatch, preflight) -> None:
    model = TASCVDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=2, verbose=False
    ).train()
    model.criterion = _ZeroStockCriterion()
    model.tascv_preflight_probe = preflight
    monkeypatch.setattr(
        model,
        "predict",
        lambda image, batch=None: _parameter_dependent(
            model,
            _predictions(),
        ),
    )
    monkeypatch.setattr(
        "src.rtdetr_tascv.select_target_anchored_crops",
        lambda **kwargs: torch.tensor([[128, 128, 512, 512]]),
    )

    total, _ = model.loss(_batch(), preds=_predictions())
    total.backward()

    assert model.last_local_forward_calls == (2 if preflight else 1)
    assert model.last_local_bn_preserved is True


def test_local_branch_bn_mutation_fails_closed(monkeypatch) -> None:
    model = TASCVDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=2, verbose=False
    ).train()
    model.criterion = _ZeroStockCriterion()
    local = _predictions()

    def mutate_bn(image, batch=None):
        batchnorm = next(
            child
            for child in model.modules()
            if isinstance(
                child,
                torch.nn.modules.batchnorm._BatchNorm,
            )
            and child.running_mean is not None
        )
        with torch.no_grad():
            batchnorm.running_mean.add_(1.0)
        return _parameter_dependent(model, local)

    monkeypatch.setattr(model, "predict", mutate_bn)
    monkeypatch.setattr(
        "src.rtdetr_tascv.select_target_anchored_crops",
        lambda **kwargs: torch.tensor([[128, 128, 512, 512]]),
    )

    with pytest.raises(
        RuntimeError,
        match="TASCV_LOCAL_BRANCH_MUTATED_BATCHNORM_BUFFERS",
    ):
        model.loss(_batch(), preds=_predictions())


def test_initial_state_loader_fails_closed_on_seed_fingerprint_and_keys() -> None:
    model = torch.nn.Linear(2, 2)
    common = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    valid = {
        "metadata": {"seed": 1},
        "common_state": common,
        "fingerprints": {"common": state_fingerprint(common)},
    }
    load_matched_initial_state(model, valid, 1)

    wrong_seed = {**valid, "metadata": {"seed": 0}}
    with pytest.raises(ValueError, match="SEED_MISMATCH"):
        load_matched_initial_state(model, wrong_seed, 1)
    wrong_fingerprint = {
        **valid,
        "fingerprints": {"common": "0" * 64},
    }
    with pytest.raises(ValueError, match="FINGERPRINT_MISMATCH"):
        load_matched_initial_state(model, wrong_fingerprint, 1)
    missing = {
        **valid,
        "common_state": {"weight": common["weight"]},
    }
    missing["fingerprints"] = {
        "common": state_fingerprint(missing["common_state"])
    }
    with pytest.raises(ValueError, match="KEYS_MISMATCH"):
        load_matched_initial_state(model, missing, 1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (None, "overrides"),
        ({"batch": 8.9, "seed": 0}, "batch"),
        ({"batch": "8", "seed": 0}, "batch"),
        ({"batch": True, "seed": 0}, "batch"),
        ({"batch": 8, "seed": 0.9}, "seed"),
        ({"batch": 8, "seed": "0"}, "seed"),
        ({"batch": 8, "seed": True}, "seed"),
        ({"batch": 8, "seed": 1}, "stage"),
    ),
)
def test_invalid_scientific_integer_configuration_fails_before_artifact_load(
    monkeypatch,
    overrides,
    message,
) -> None:
    load_calls = []
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=message):
        TASCVTrainer(
            stage=TASCVStage.PREFLIGHT_1,
            initial_state_path="must-not-be-loaded.pt",
            overrides=overrides,
        )

    assert load_calls == []


@pytest.mark.parametrize(("epoch", "valid"), ((0, True), (3, True), (-1, False), (1.2, False), ("1", False), (True, False)))
def test_tascv_epoch_progress_is_a_strict_nonnegative_integer(
    epoch,
    valid,
) -> None:
    model = object.__new__(TASCVDetectionModel)
    model.tascv_epoch = 0

    if valid:
        TASCVDetectionModel.set_tascv_progress(model, epoch)
        assert model.tascv_epoch == epoch
    else:
        with pytest.raises(ValueError, match="epoch"):
            TASCVDetectionModel.set_tascv_progress(model, epoch)


def test_non_musgd_optimizer_is_rejected_before_construction() -> None:
    trainer = object.__new__(TASCVTrainer)
    with pytest.raises(ValueError, match="TASCV_OPTIMIZER_DRIFT"):
        TASCVTrainer.build_optimizer(
            trainer,
            torch.nn.Linear(2, 2),
            name="AdamW",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_setup_train_installs_fixed_scale_128(monkeypatch) -> None:
    from ultralytics.models.rtdetr.train import RTDETRTrainer

    monkeypatch.setattr(
        RTDETRTrainer,
        "_setup_train",
        lambda self: None,
    )
    trainer = object.__new__(TASCVTrainer)
    trainer.args = SimpleNamespace(amp=True)
    trainer.device = torch.device("cuda:0")

    TASCVTrainer._setup_train(trainer)

    assert trainer.amp is True
    assert trainer.args.amp is True
    assert trainer.scaler.get_scale() == 128.0


def test_train_pipeline_never_requests_val_or_test_loader() -> None:
    class Loader:
        dataset = list(range(20))
        batch_size = 8
        num_workers = 8

    trainer = object.__new__(TASCVTrainer)
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
    trainer.get_dataloader = (
        lambda path, **kwargs: calls.append((path, kwargs["mode"]))
        or Loader()
    )
    trainer.build_optimizer = lambda **kwargs: torch.optim.SGD(
        trainer.model.parameters(), lr=0.01
    )
    trainer._setup_scheduler = lambda: None

    TASCVTrainer._build_train_pipeline(trainer)

    assert calls == [("hashed-train.txt", "train")]
    assert trainer.test_loader is None


def test_oom_batch_reduction_fails_before_loader_rebuild() -> None:
    trainer = object.__new__(TASCVTrainer)
    trainer.batch_size = 2
    trainer.frozen_batch_size = 8

    with pytest.raises(RuntimeError, match="TASCV_BATCH_DRIFT"):
        TASCVTrainer._build_train_pipeline(trainer)


def test_internal_validation_and_final_eval_are_non_reading_noops() -> None:
    trainer = object.__new__(TASCVTrainer)
    trainer.internal_validation_bypass_count = 0

    metrics, fitness = TASCVTrainer.validate(trainer)
    final = TASCVTrainer.final_eval(trainer)

    assert metrics == {}
    assert fitness == float("-inf")
    assert trainer.internal_validation_bypass_count == 1
    assert final is None


def _batch_of_size(batch_size: int) -> dict:
    return {
        "img": torch.zeros((batch_size, 3, 4, 4)),
        "cls": torch.zeros((batch_size, 1)),
        "bboxes": torch.zeros((batch_size, 4)),
        "batch_idx": torch.arange(batch_size, dtype=torch.float32),
        "im_file": [f"image-{index}.jpg" for index in range(batch_size)],
    }


def test_preprocess_batch_records_frozen_canaries_and_observed_sizes(
    monkeypatch,
) -> None:
    from ultralytics.models.rtdetr.train import RTDETRTrainer

    monkeypatch.setattr(
        RTDETRTrainer,
        "preprocess_batch",
        lambda self, batch: batch,
    )
    trainer = object.__new__(TASCVTrainer)
    trainer.tascv_observed_tensor_batch_sizes = set()
    trainer.tascv_batch_canaries = []
    trainer.tascv_preprocessed_batch_count = 0
    trainer.tascv_epoch_one_canary_recorded = False
    trainer.epoch = 0

    TASCVTrainer.preprocess_batch(trainer, _batch_of_size(8))
    TASCVTrainer.preprocess_batch(trainer, _batch_of_size(8))
    trainer.epoch = 1
    TASCVTrainer.preprocess_batch(trainer, _batch_of_size(7))
    TASCVTrainer.preprocess_batch(trainer, _batch_of_size(8))

    assert trainer.tascv_preprocessed_batch_count == 4
    assert trainer.tascv_observed_tensor_batch_sizes == {7, 8}
    assert [
        (record["epoch"], record["batch"])
        for record in trainer.tascv_batch_canaries
    ] == [(0, 1), (0, 2), (1, 3)]
    assert all(
        len(record["sha256"]) == 64
        for record in trainer.tascv_batch_canaries
    )


@pytest.mark.parametrize(
    ("after_scale", "error"),
    ((MATCHED_AMP_SCALE, None), (64.0, FloatingPointError)),
)
def test_optimizer_step_records_attempt_scale_range_and_fails_closed(
    monkeypatch,
    after_scale,
    error,
) -> None:
    from ultralytics.models.rtdetr.train import RTDETRTrainer

    super_calls = []
    monkeypatch.setattr(
        RTDETRTrainer,
        "optimizer_step",
        lambda self: super_calls.append("step"),
    )

    class Scaler:
        def __init__(self):
            self.calls = 0

        def get_scale(self):
            self.calls += 1
            return MATCHED_AMP_SCALE if self.calls == 1 else after_scale

    trainer = object.__new__(TASCVTrainer)
    trainer.scaler = Scaler()
    trainer.tascv_optimizer_attempts = 0
    trainer.tascv_amp_scale_min = MATCHED_AMP_SCALE
    trainer.tascv_amp_scale_max = MATCHED_AMP_SCALE

    if error is None:
        TASCVTrainer.optimizer_step(trainer)
    else:
        with pytest.raises(error, match="DRIFT_AFTER_STEP"):
            TASCVTrainer.optimizer_step(trainer)

    assert super_calls == ["step"]
    assert trainer.tascv_optimizer_attempts == 1
    assert trainer.tascv_amp_scale_min == min(
        MATCHED_AMP_SCALE,
        after_scale,
    )
    assert trainer.tascv_amp_scale_max == max(
        MATCHED_AMP_SCALE,
        after_scale,
    )


@pytest.mark.parametrize(
    ("stage", "limit"),
    (
        (TASCVStage.PREFLIGHT_1, 1),
        (TASCVStage.TINY_MECHANISM_500, 500),
    ),
)
def test_record_successful_batch_stops_exactly_at_frozen_limit(
    stage,
    limit,
) -> None:
    trainer = object.__new__(TASCVTrainer)
    trainer.tascv_policy = stage_policy(stage)
    trainer.tascv_successful_batches = 0
    trainer.stop = False

    for _ in range(limit - 1):
        TASCVTrainer.record_successful_batch(trainer)
        assert trainer.stop is False
    TASCVTrainer.record_successful_batch(trainer)

    assert trainer.tascv_successful_batches == limit
    assert trainer.stop is True


@pytest.mark.parametrize(
    ("trainer_class", "expected_names"),
    (
        (TASCVTrainer, LOSS_NAMES),
        (TASCVControlTrainer, CONTROL_LOSS_NAMES),
    ),
)
def test_both_trainer_mros_return_nonreading_validator_and_loss_schema(
    trainer_class,
    expected_names,
) -> None:
    trainer = object.__new__(trainer_class)

    validator = trainer_class.get_validator(trainer)

    assert trainer.loss_names == expected_names
    assert validator.metrics.keys == []
    with pytest.raises(
        RuntimeError,
        match="TASCV_INTERNAL_VALIDATION_FORBIDDEN",
    ):
        validator()


def test_control_model_is_stock_matched_and_rejects_supplied_weights(
    monkeypatch,
) -> None:
    import src.rtdetr_tascv as module

    class Stock(torch.nn.Module):
        def __init__(self, cfg, nc, ch, verbose):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.yaml = {"nc": nc}

    monkeypatch.setattr(module, "RTDETRDetectionModel", Stock)
    common = {"weight": torch.tensor([3.0])}
    trainer = object.__new__(TASCVControlTrainer)
    trainer.data = {"nc": 2, "channels": 3}
    trainer.initial_state = {
        "metadata": {"seed": 0},
        "common_state": common,
        "fingerprints": {"common": state_fingerprint(common)},
    }
    trainer.args = SimpleNamespace(seed=0)

    model = TASCVControlTrainer.get_model(
        trainer,
        cfg={"nc": 2},
        weights=None,
        verbose=False,
    )

    assert isinstance(model, Stock)
    assert model.weight.item() == 3.0
    assert not hasattr(model, "tascv_epoch")
    assert not hasattr(model, "last_local_forward_calls")
    with pytest.raises(ValueError, match="PRETRAINED_WEIGHTS_FORBIDDEN"):
        TASCVControlTrainer.get_model(
            trainer,
            cfg={"nc": 2},
            weights="forbidden.pt",
            verbose=False,
        )

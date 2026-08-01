from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager
from typing import Iterator

import pytest
import torch
from torch import nn
from ultralytics.nn.tasks import RTDETRDetectionModel

import src.rtdetr_iber as rtdetr_iber
from src.iber_head import IBEROutput
from src.iber_protocol import module_state_sha256
from src.itber_geometry import cxcywh_to_xyxy
from src.itber_loss import itber_private_loss
from src.rtdetr_iber import FrozenIBERAdapter, IBERRecordingDecoder


@contextmanager
def _one_torch_thread() -> Iterator[None]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _stock_head() -> nn.Module:
    return RTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False
    ).model[-1]


def _features() -> list[torch.Tensor]:
    return [torch.randn(1, 256, size, size) for size in (20, 10, 5)]


def _assert_nested_exact(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_exact(actual_item, expected_item)
        return
    assert actual == expected


def _assert_matches_equal(
    actual: list[tuple[torch.Tensor, torch.Tensor]],
    expected: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    assert len(actual) == len(expected)
    for (actual_source, actual_target), (expected_source, expected_target) in zip(
        actual, expected
    ):
        torch.testing.assert_close(actual_source, expected_source, rtol=0, atol=0)
        torch.testing.assert_close(actual_target, expected_target, rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True], ids=["eval", "training"])
def test_recording_decoder_preserves_stock_outputs_and_state_exactly(
    training: bool,
) -> None:
    torch.manual_seed(0)
    stock = _stock_head()
    wrapped_head = _stock_head()
    wrapped_head.load_state_dict(stock.state_dict())
    stock.train(training)
    wrapped_head.train(training)
    stock_decoder = wrapped_head.decoder
    expected_state = {
        key: value.detach().clone() for key, value in stock_decoder.state_dict().items()
    }

    wrapped_head.decoder = IBERRecordingDecoder.from_stock(stock_decoder)
    features = _features()
    with _one_torch_thread(), torch.no_grad():
        expected = stock(features)
        actual = wrapped_head(features)

    _assert_nested_exact(actual, expected)
    decoder = wrapped_head.decoder
    assert decoder.layers is stock_decoder.layers
    assert decoder.hidden_dim == stock_decoder.hidden_dim
    assert decoder.num_layers == stock_decoder.num_layers
    assert decoder.eval_idx == stock_decoder.eval_idx
    assert decoder.training is stock_decoder.training
    assert tuple(decoder.state_dict()) == tuple(expected_state)
    for key, expected_value in expected_state.items():
        torch.testing.assert_close(
            decoder.state_dict()[key], expected_value, rtol=0, atol=0
        )

    raw = actual if training else actual[1]
    decoded_boxes, decoded_scores = raw[:2]
    evidence_index = -1 if training else 0
    assert decoder.last_hidden is not None
    assert decoder.last_stock_scores is not None
    assert decoder.last_stock_boxes is not None
    assert decoder.last_hidden.shape == (1, 300, 256)
    assert decoder.last_stock_scores.shape == (1, 300, 10)
    assert decoder.last_stock_boxes.shape == (1, 300, 4)
    torch.testing.assert_close(
        decoder.last_stock_scores,
        decoded_scores[evidence_index, :, -300:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        decoder.last_stock_boxes,
        decoded_boxes[evidence_index, :, -300:],
        rtol=0,
        atol=0,
    )
    assert not decoder.last_hidden.requires_grad
    assert not decoder.last_stock_scores.requires_grad
    assert not decoder.last_stock_boxes.requires_grad


class _IncrementLayer(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, output: torch.Tensor, *_args: object) -> torch.Tensor:
        return output + self.increment


class _ExplodingLayer(nn.Module):
    def forward(self, _output: torch.Tensor, *_args: object) -> torch.Tensor:
        raise RuntimeError("decoder stopped")


def _direct_decoder(layers: nn.ModuleList, *, normal_query_count: int = 300) -> tuple:
    hidden_dim = 8
    decoder = IBERRecordingDecoder(
        layers,
        hidden_dim=hidden_dim,
        num_layers=len(layers),
        eval_idx=len(layers) - 1,
        normal_query_count=normal_query_count,
    ).eval()
    bbox_head = nn.ModuleList(
        [nn.Linear(hidden_dim, 4, bias=False) for _ in layers]
    )
    score_head = nn.ModuleList(
        [nn.Linear(hidden_dim, 10, bias=False) for _ in layers]
    )
    return decoder, bbox_head, score_head


def test_direct_decoder_excludes_prepended_queries_and_records_last_300() -> None:
    torch.manual_seed(1)
    layers = nn.ModuleList([_IncrementLayer(value) for value in (1.0, 2.0, 3.0)])
    decoder, bbox_head, score_head = _direct_decoder(layers)
    embed = torch.randn(1, 307, 8)
    refer_bbox = torch.randn(1, 307, 4)

    boxes, scores = decoder(
        embed,
        refer_bbox,
        torch.empty(1, 0, 8),
        [],
        bbox_head,
        score_head,
        nn.Identity(),
    )

    expected_hidden = ((embed + 1.0) + 2.0) + 3.0
    assert decoder.last_hidden is not None
    assert decoder.last_stock_scores is not None
    assert decoder.last_stock_boxes is not None
    assert decoder.last_hidden.shape == (1, 300, 8)
    torch.testing.assert_close(
        decoder.last_hidden, expected_hidden[:, -300:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        decoder.last_stock_scores, scores[0, :, -300:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        decoder.last_stock_boxes, boxes[0, :, -300:], rtol=0, atol=0
    )
    assert not torch.equal(decoder.last_hidden[:, 0], expected_hidden[:, 0])


@pytest.mark.parametrize(
    ("hidden_shape", "score_shape", "box_shape"),
    [
        ((2, 300, 8), (1, 300, 10), (2, 300, 4)),
        ((1, 301, 8), (1, 300, 10), (1, 301, 4)),
        ((1, 300, 8), (1, 300, 10), (1, 299, 4)),
    ],
    ids=["batch-mismatch", "score-query-mismatch", "box-query-mismatch"],
)
def test_record_rejects_inconsistent_batch_or_query_dimensions(
    hidden_shape: tuple[int, ...],
    score_shape: tuple[int, ...],
    box_shape: tuple[int, ...],
) -> None:
    decoder = IBERRecordingDecoder(
        nn.ModuleList(), hidden_dim=8, num_layers=0, eval_idx=0
    )

    with pytest.raises((RuntimeError, ValueError), match="batch|quer"):
        decoder._record(
            torch.randn(hidden_shape),
            torch.randn(score_shape),
            torch.randn(box_shape),
        )


def test_direct_decoder_rejects_runtime_with_fewer_than_300_queries() -> None:
    layers = nn.ModuleList([_IncrementLayer(1.0)])
    decoder, bbox_head, score_head = _direct_decoder(layers)

    with pytest.raises((RuntimeError, ValueError), match="300|normal quer"):
        decoder(
            torch.randn(1, 299, 8),
            torch.randn(1, 299, 4),
            torch.empty(1, 0, 8),
            [],
            bbox_head,
            score_head,
            nn.Identity(),
        )


def test_recording_decoder_rejects_nonpositive_counts_and_clears_stale_evidence() -> None:
    for query_count in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            IBERRecordingDecoder(
                nn.ModuleList(),
                hidden_dim=8,
                num_layers=0,
                eval_idx=0,
                normal_query_count=query_count,
            )

    decoder, bbox_head, score_head = _direct_decoder(
        nn.ModuleList([_ExplodingLayer()])
    )
    decoder.last_hidden = torch.ones(1, 1, 8)
    decoder.last_stock_scores = torch.ones(1, 1, 10)
    decoder.last_stock_boxes = torch.ones(1, 1, 4)
    with pytest.raises(RuntimeError, match="stopped"):
        decoder(
            torch.randn(1, 1, 8),
            torch.randn(1, 1, 4),
            torch.empty(1, 0, 8),
            [],
            bbox_head,
            score_head,
            nn.Identity(),
        )
    assert decoder.last_hidden is None
    assert decoder.last_stock_scores is None
    assert decoder.last_stock_boxes is None


@pytest.fixture(scope="module")
def real_adapter() -> FrozenIBERAdapter:
    detector = RTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False
    )
    return FrozenIBERAdapter.from_detector(
        detector,
        private_seed=10_000,
        image_size=160,
        normal_query_count=300,
    )


@pytest.fixture(scope="module")
def authority_detector() -> RTDETRDetectionModel:
    return RTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False
    )


@pytest.mark.parametrize("normal_query_count", [17, 301])
def test_from_detector_rejects_count_different_from_real_head_authority(
    authority_detector: RTDETRDetectionModel,
    normal_query_count: int,
) -> None:
    head = authority_detector.model[-1]
    assert head.num_queries == 300
    original_decoder = head.decoder
    created: FrozenIBERAdapter | None = None
    try:
        with pytest.raises(ValueError, match="300|normal quer|num_queries"):
            created = FrozenIBERAdapter.from_detector(
                authority_detector,
                private_seed=10_000,
                image_size=160,
                normal_query_count=normal_query_count,
            )
    finally:
        if created is not None:
            created._head_hook.remove()
        head.decoder = original_decoder
    assert head.decoder is original_decoder


def test_real_adapter_is_frozen_single_pass_and_zero_init_identity(
    real_adapter: FrozenIBERAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = real_adapter
    adapter.eval()
    adapter.refiner.zero_grad(set_to_none=True)
    detector = adapter.detector
    detector.zero_grad(set_to_none=True)
    before_sha = module_state_sha256(detector)
    predict_inputs: list[torch.Tensor] = []
    refiner_inputs: list[tuple[torch.Tensor, ...]] = []
    real_predict = detector.predict
    real_refiner_forward = adapter.refiner.forward

    def counted_predict(image: torch.Tensor, *args: object, **kwargs: object) -> object:
        predict_inputs.append(image)
        return real_predict(image, *args, **kwargs)

    def counted_refiner(
        hidden: torch.Tensor,
        stock_boxes: torch.Tensor,
        stock_scores: torch.Tensor,
        f3: torch.Tensor,
        image_rgb: torch.Tensor,
    ) -> IBEROutput:
        refiner_inputs.append((hidden, stock_boxes, stock_scores, f3, image_rgb))
        return real_refiner_forward(
            hidden, stock_boxes, stock_scores, f3, image_rgb
        )

    monkeypatch.setattr(detector, "predict", counted_predict)
    monkeypatch.setattr(adapter.refiner, "forward", counted_refiner)
    image = torch.rand(1, 3, 160, 160, requires_grad=True)

    with _one_torch_thread():
        output = adapter.forward_evidence(image)

    assert len(predict_inputs) == 1
    assert len(refiner_inputs) == 1
    assert predict_inputs[0] is refiner_inputs[0][4]
    assert predict_inputs[0].data_ptr() == image.data_ptr()
    assert not predict_inputs[0].requires_grad
    assert detector.training is False
    assert all(not parameter.requires_grad for parameter in detector.parameters())
    decoder = detector.model[-1].decoder
    assert isinstance(decoder, IBERRecordingDecoder)
    assert decoder.last_hidden is not None
    assert decoder.last_stock_scores is not None
    assert decoder.last_stock_boxes is not None
    assert decoder.last_hidden.shape == (1, 300, 256)
    assert decoder.last_stock_scores.shape == (1, 300, 10)
    assert decoder.last_stock_boxes.shape == (1, 300, 4)
    assert adapter._last_f3 is refiner_inputs[0][3]
    assert adapter._last_f3.shape[0] == 1
    assert adapter._last_f3.shape[1] == adapter.refiner.f3_projection.in_channels
    assert not adapter._last_f3.requires_grad
    assert output.stock_boxes.shape == (1, 300, 4)
    assert output.stock_scores.shape == (1, 300, 10)
    torch.testing.assert_close(output.stock_boxes, decoder.last_stock_boxes, rtol=0, atol=0)
    torch.testing.assert_close(output.stock_scores, decoder.last_stock_scores, rtol=0, atol=0)
    torch.testing.assert_close(output.refined_boxes, output.stock_boxes, rtol=0, atol=0)
    torch.testing.assert_close(
        output.boundary_off_boxes, output.stock_boxes, rtol=0, atol=0
    )
    assert module_state_sha256(detector) == before_sha

    losses = itber_private_loss(
        output,
        target_edges=cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, 0.2, 0.2]])),
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        rho=adapter.rho,
    )
    losses.total.backward()

    assert module_state_sha256(detector) == before_sha
    assert all(parameter.grad is None for parameter in detector.parameters())
    for name in (
        "base_gate_head",
        "boundary_gate_head",
        "base_residual_head",
        "boundary_residual_head",
    ):
        gradients = [parameter.grad for parameter in getattr(adapter.refiner, name).parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_training_step_matches_exact_stock_evidence_once(
    real_adapter: FrozenIBERAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = real_adapter.train()
    batch = {
        "img": torch.rand(2, 3, 160, 160),
        "cls": torch.tensor([[1.0], [3.0], [7.0]]),
        "bboxes": torch.tensor(
            [
                [0.50, 0.50, 0.20, 0.20],
                [0.25, 0.30, 0.10, 0.15],
                [0.70, 0.65, 0.18, 0.12],
            ]
        ),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    }
    with _one_torch_thread():
        expected_output = adapter.forward_evidence(batch["img"])
        target_boxes = batch["bboxes"].detach().to(
            device=batch["img"].device, dtype=expected_output.stock_boxes.dtype
        )
        target_classes = batch["cls"].detach().to(
            device=batch["img"].device, dtype=torch.long
        ).view(-1)
        groups = [2, 1]
        expected_matches = adapter.criterion.matcher(
            expected_output.stock_boxes.detach(),
            expected_output.stock_scores.detach(),
            target_boxes,
            target_classes,
            groups,
        )

    evidence_calls = 0
    matcher_inputs: list[tuple[object, ...]] = []
    real_forward_evidence = adapter.forward_evidence

    def counted_forward_evidence(image: torch.Tensor) -> IBEROutput:
        nonlocal evidence_calls
        evidence_calls += 1
        return real_forward_evidence(image)

    def capture_matcher_inputs(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        matcher_inputs.append(inputs)

    monkeypatch.setattr(adapter, "forward_evidence", counted_forward_evidence)
    handle = adapter.criterion.matcher.register_forward_pre_hook(capture_matcher_inputs)
    try:
        with _one_torch_thread():
            losses = adapter.training_step(batch)
    finally:
        handle.remove()

    assert torch.isfinite(losses.total)
    assert evidence_calls == 1
    assert len(matcher_inputs) == 1
    assert adapter.last_output is not None
    assert adapter.last_match_indices is not None
    passed_boxes, passed_scores, passed_targets, passed_classes, passed_groups = (
        matcher_inputs[0]
    )
    assert isinstance(passed_boxes, torch.Tensor)
    assert isinstance(passed_scores, torch.Tensor)
    assert passed_boxes.shape == (2, 300, 4)
    assert passed_scores.shape == (2, 300, 10)
    torch.testing.assert_close(
        passed_boxes, adapter.last_output.stock_boxes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        passed_scores, adapter.last_output.stock_scores, rtol=0, atol=0
    )
    decoder = adapter.detector.model[-1].decoder
    torch.testing.assert_close(passed_boxes, decoder.last_stock_boxes, rtol=0, atol=0)
    torch.testing.assert_close(passed_scores, decoder.last_stock_scores, rtol=0, atol=0)
    torch.testing.assert_close(passed_targets, target_boxes, rtol=0, atol=0)
    torch.testing.assert_close(passed_classes, target_classes, rtol=0, atol=0)
    assert passed_groups == groups
    _assert_matches_equal(adapter.last_match_indices, expected_matches)


def test_output_modes_select_only_boxes_and_keep_stock_scores_exact(
    real_adapter: FrozenIBERAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = real_adapter.eval()
    image = torch.rand(1, 3, 160, 160)
    with _one_torch_thread():
        output = adapter.forward_evidence(image)
    original_scores = output.stock_scores.detach().clone()
    evidence_calls = 0
    postprocess_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fixed_forward_evidence(_image: torch.Tensor) -> IBEROutput:
        nonlocal evidence_calls
        evidence_calls += 1
        return output

    def capture_postprocess(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        postprocess_inputs.append((boxes, scores))
        return boxes

    monkeypatch.setattr(adapter, "forward_evidence", fixed_forward_evidence)
    monkeypatch.setattr(adapter.detector.model[-1], "postprocess", capture_postprocess)
    for mode in ("stock", "refined", "boundary_off"):
        evidence_calls = 0
        postprocess_inputs.clear()
        adapter.set_output_mode(mode)

        selected = adapter.selected_boxes(output)
        result = adapter(image)

        assert selected is output.select_boxes(mode)
        assert result is selected
        assert evidence_calls == 1
        assert len(postprocess_inputs) == 1
        assert postprocess_inputs[0][0] is selected
        torch.testing.assert_close(
            postprocess_inputs[0][1], original_scores.sigmoid(), rtol=0, atol=0
        )
        torch.testing.assert_close(output.stock_scores, original_scores, rtol=0, atol=0)

    with pytest.raises(ValueError, match="mode"):
        adapter.set_output_mode("candidate")


def test_adapter_train_eval_toggles_only_private_refiner(
    real_adapter: FrozenIBERAdapter,
) -> None:
    adapter = real_adapter
    adapter.eval()
    assert adapter.training is False
    assert adapter.refiner.training is False
    assert adapter.detector.training is False

    adapter.train()
    assert adapter.training is True
    assert adapter.refiner.training is True
    assert adapter.detector.training is False
    assert all(not parameter.requires_grad for parameter in adapter.detector.parameters())


def test_existing_iber_wrapper_with_wrong_count_is_rejected(
    real_adapter: FrozenIBERAdapter,
) -> None:
    decoder = real_adapter.detector.model[-1].decoder
    decoder.normal_query_count = 299
    second: FrozenIBERAdapter | None = None
    try:
        with pytest.raises(ValueError, match="wrapped|300|normal quer"):
            second = FrozenIBERAdapter.from_detector(
                real_adapter.detector,
                private_seed=20_000,
                image_size=160,
                normal_query_count=300,
            )
    finally:
        if second is not None:
            second._head_hook.remove()
        decoder.normal_query_count = 300
    assert real_adapter.detector.model[-1].decoder is decoder


def test_existing_iber_wrapper_with_authoritative_count_is_left_intact(
    real_adapter: FrozenIBERAdapter,
) -> None:
    decoder = real_adapter.detector.model[-1].decoder
    second = FrozenIBERAdapter.from_detector(
        real_adapter.detector,
        private_seed=20_000,
        image_size=160,
        normal_query_count=300,
    )
    try:
        assert second.detector.model[-1].decoder is decoder
    finally:
        second._head_hook.remove()


def test_from_detector_rejects_head_query_count_outside_approved_authority(
    real_adapter: FrozenIBERAdapter,
) -> None:
    head = real_adapter.detector.model[-1]
    head.num_queries = 301
    second: FrozenIBERAdapter | None = None
    try:
        with pytest.raises(ValueError, match="approved|300|num_queries"):
            second = FrozenIBERAdapter.from_detector(
                real_adapter.detector,
                private_seed=20_001,
                image_size=160,
                normal_query_count=301,
            )
    finally:
        if second is not None:
            second._head_hook.remove()
        head.num_queries = 300


def test_incomplete_evidence_and_invalid_hook_inputs_fail_clearly(
    real_adapter: FrozenIBERAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = real_adapter
    with pytest.raises(RuntimeError, match="feature pyramid"):
        adapter._capture_head_input(adapter.detector.model[-1], ())
    with pytest.raises(RuntimeError, match="feature pyramid"):
        adapter._capture_head_input(
            adapter.detector.model[-1], (torch.zeros(1, 3, 4, 4),)
        )

    monkeypatch.setattr(adapter.detector, "predict", lambda _image: None)
    with pytest.raises(RuntimeError, match="evidence capture is incomplete"):
        adapter.forward_evidence(torch.rand(1, 3, 160, 160))


def test_rtdetr_iber_source_has_no_trajectory_or_itber_wrapper_identity() -> None:
    source = inspect.getsource(rtdetr_iber)
    lowered = source.lower()
    assert "trajectory" not in lowered
    assert "last_three_boxes" not in source

    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "src.rtdetr_itber" not in imported_modules
    assert "src.itber_head" not in imported_modules
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert identifiers.isdisjoint(
        {
            "ITBERRecordingDecoder",
            "FrozenITBERAdapter",
            "ITBERRefiner",
            "last_three_boxes",
            "box_l1",
            "box_l2",
        }
    )
    decoder = IBERRecordingDecoder(
        nn.ModuleList(), hidden_dim=8, num_layers=0, eval_idx=0
    )
    assert not hasattr(decoder, "last_three_boxes")
    assert not any("trajectory" in name for name, _ in decoder.named_modules())

from __future__ import annotations

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.rtdetr_itber import FrozenITBERAdapter, ITBERRecordingDecoder


def _stock_head():
    return RTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False
    ).model[-1]


def test_recording_decoder_preserves_stock_outputs_exactly() -> None:
    torch.manual_seed(0)
    stock = _stock_head().eval()
    wrapped = _stock_head().eval()
    wrapped.load_state_dict(stock.state_dict())
    wrapped.decoder = ITBERRecordingDecoder.from_stock(wrapped.decoder)
    features = [torch.randn(1, 256, size, size) for size in (20, 10, 5)]

    with torch.no_grad():
        expected_y, expected_raw = stock(features)
        actual_y, actual_raw = wrapped(features)

    torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=0)
    for actual, expected in zip(actual_raw[:4], expected_raw[:4]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_raw[4] == expected_raw[4]
    decoder = wrapped.decoder
    assert decoder.last_hidden is not None
    assert decoder.last_stock_scores is not None
    assert decoder.last_three_boxes is not None
    assert decoder.last_hidden.shape == (1, 300, 256)
    assert decoder.last_stock_scores.shape == (1, 300, 10)
    assert decoder.last_three_boxes.shape == (3, 1, 300, 4)


def test_frozen_adapter_has_no_detector_gradients() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        detector = RTDETRDetectionModel(
            "rtdetr-l.yaml", ch=3, nc=10, verbose=False
        )
        adapter = FrozenITBERAdapter.from_detector(
            detector, private_seed=10_000, image_size=160
        )
        adapter.train()
        batch = {
            "img": torch.rand(1, 3, 160, 160),
            "cls": torch.tensor([[1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "batch_idx": torch.tensor([0.0]),
        }

        losses = adapter.training_step(batch)
        losses.total.backward()

        assert adapter.detector.training is False
        assert all(not parameter.requires_grad for parameter in adapter.detector.parameters())
        assert all(parameter.grad is None for parameter in adapter.detector.parameters())
        assert any(parameter.grad is not None for parameter in adapter.refiner.parameters())
        assert adapter.last_match_indices is not None
        assert len(adapter.last_match_indices) == 1
    finally:
        torch.set_num_threads(previous_threads)


def test_adapter_mode_switch_only_changes_selected_private_boxes() -> None:
    detector = RTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False
    )
    adapter = FrozenITBERAdapter.from_detector(
        detector, private_seed=10_000, image_size=160
    )
    image = torch.rand(1, 3, 160, 160)

    output = adapter.forward_evidence(image)
    adapter.set_output_mode("stock")
    torch.testing.assert_close(adapter.selected_boxes(output), output.stock_boxes)
    adapter.set_output_mode("refined")
    torch.testing.assert_close(adapter.selected_boxes(output), output.refined_boxes)

    try:
        adapter.set_output_mode("candidate")
    except ValueError as error:
        assert "stock or refined" in str(error)
    else:
        raise AssertionError("unknown output mode was accepted")

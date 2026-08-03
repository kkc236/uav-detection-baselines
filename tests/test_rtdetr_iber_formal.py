from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.rtdetr_iber import IBERRecordingDecoder
from src.rtdetr_iber_formal import IBERFullRTDETRDetectionModel, IBERFullTrainer


@contextmanager
def _one_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 160, 160),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0.0]),
    }


def test_private_construction_preserves_seed0_public_initialization_and_rng() -> None:
    torch.manual_seed(0)
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    stock_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(0)
    method = IBERFullRTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False, private_seed=10_000
    )
    method_rng = torch.random.get_rng_state().clone()

    method_state = method.state_dict()
    assert isinstance(method.model[-1].decoder, IBERRecordingDecoder)
    assert any("iber_refiner." in name for name in method_state)
    for name, expected in stock.state_dict().items():
        torch.testing.assert_close(method_state[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(method_rng, stock_rng, rtol=0, atol=0)


def test_zero_initialization_preserves_stock_predictions_in_both_modes() -> None:
    torch.manual_seed(0)
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    torch.manual_seed(0)
    method = IBERFullRTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=10, verbose=False, private_seed=10_000
    ).eval()
    image = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(8))

    with _one_torch_thread(), torch.inference_mode():
        expected = stock.predict(image)[0]
        method.set_refinement_output("refined")
        refined = method.predict(image)[0]
        method.set_refinement_output("stock")
        stock_mode = method.predict(image)[0]

    torch.testing.assert_close(refined, expected, rtol=0, atol=0)
    torch.testing.assert_close(stock_mode, expected, rtol=0, atol=0)
    assert method.last_iber_output is not None
    assert method.last_iber_output.stock_scores.shape == (1, 300, 10)


def test_private_loss_is_isolated_from_every_public_parameter() -> None:
    model = IBERFullRTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=3, verbose=False
    ).train()

    with _one_torch_thread():
        model.loss(_batch())
        assert model.last_iber_loss_total is not None
        model.zero_grad(set_to_none=True)
        model.last_iber_loss_total.backward()

    private = [
        parameter
        for name, parameter in model.named_parameters()
        if "iber_refiner." in name
    ]
    public = [
        parameter
        for name, parameter in model.named_parameters()
        if "iber_refiner." not in name
    ]
    assert private
    assert any(parameter.grad is not None for parameter in private)
    assert all(parameter.grad is None for parameter in public)


def test_full_backward_preserves_stock_loss_items_and_public_gradients() -> None:
    with _one_torch_thread():
        torch.manual_seed(5)
        stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=3, verbose=False).train()
        stock.nc = stock.yaml["nc"]
        torch.manual_seed(5)
        method = IBERFullRTDETRDetectionModel(
            "rtdetr-l.yaml", ch=3, nc=3, verbose=False
        ).train()
        batch = _batch()

        torch.manual_seed(77)
        stock_total, stock_items = stock.loss(batch)
        torch.manual_seed(77)
        method_total, method_items = method.loss(batch)
        stock_total.backward()
        method_total.backward()

    torch.testing.assert_close(method_items, stock_items, rtol=0, atol=0)
    method_parameters = dict(method.named_parameters())
    for name, parameter in stock.named_parameters():
        method_gradient = method_parameters[name].grad
        if parameter.grad is None:
            assert method_gradient is None
        else:
            torch.testing.assert_close(method_gradient, parameter.grad, rtol=0, atol=0)


def test_trainer_partitions_one_musgd_optimizer_into_public_and_private_groups() -> None:
    model = IBERFullRTDETRDetectionModel(
        "rtdetr-l.yaml", ch=3, nc=3, verbose=False
    )
    trainer = SimpleNamespace(model=model)

    groups = IBERFullTrainer.gradient_parameter_groups(trainer)

    assert groups["gradient_norm"]
    assert groups["iber_gradient_norm"]
    private_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if "iber_refiner." in name and parameter.requires_grad
    }
    assert {id(parameter) for parameter in groups["iber_gradient_norm"]} == private_ids
    assert not ({id(parameter) for parameter in groups["gradient_norm"]} & private_ids)


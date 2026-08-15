from __future__ import annotations

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.rtdetr_lpr_g import LPRGRTDETRDetectionModel


def _batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 160, 160),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0.0]),
    }


def test_lpr_g_initial_eval_is_exact_stock_and_mode_switch_is_stable() -> None:
    torch.manual_seed(0)
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    torch.manual_seed(0)
    method = LPRGRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False).eval()
    image = torch.rand(1, 3, 160, 160)

    with torch.no_grad():
        stock_output = stock.predict(image)[0]
        method.set_refinement_output("refined")
        refined_output = method.predict(image)[0]
        method.set_refinement_output("stock")
        stock_mode_output = method.predict(image)[0]

    torch.testing.assert_close(refined_output, stock_output, rtol=0, atol=0)
    torch.testing.assert_close(stock_mode_output, stock_output, rtol=0, atol=0)


def test_private_loss_has_no_gradient_to_common_parameters() -> None:
    model = LPRGRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=3, verbose=False).train()

    model.loss(_batch())
    model.last_lpr_g_loss_total.backward()

    private = [parameter for name, parameter in model.named_parameters() if "lpr_g_refiner." in name]
    common = [parameter for name, parameter in model.named_parameters() if "lpr_g_refiner." not in name]
    assert private
    assert any(parameter.grad is not None for parameter in private)
    assert all(parameter.grad is None for parameter in common)


def test_full_method_backward_preserves_stock_loss_items_and_common_gradients() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(5)
        stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=3, verbose=False).train()
        stock.nc = stock.yaml["nc"]
        torch.manual_seed(5)
        method = LPRGRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=3, verbose=False).train()
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
    finally:
        torch.set_num_threads(previous_threads)

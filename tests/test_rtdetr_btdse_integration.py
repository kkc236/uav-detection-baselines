from pathlib import Path

import torch

from src.btd_se import BTDSE
from src.rtdetr_btdse import BTDSEDetectionModel, filter_detection_batch


MODEL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-btdse.yaml"


def test_filter_detection_batch_removes_ignore_instances_only():
    batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "cls": torch.tensor([[2.0], [-1.0], [4.0]]),
        "bboxes": torch.tensor(
            [[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]
        ),
        "batch_idx": torch.tensor([0.0, 0.0, 0.0]),
    }

    filtered = filter_detection_batch(batch)

    torch.testing.assert_close(filtered["cls"], torch.tensor([[2.0], [4.0]]))
    torch.testing.assert_close(
        filtered["bboxes"], torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]])
    )
    assert filtered["img"] is batch["img"]
    assert len(batch["cls"]) == 3


def test_custom_yaml_places_single_btdse_before_p3_repc3():
    model = BTDSEDetectionModel(MODEL_CONFIG, ch=3, nc=10, verbose=False)
    modules = [module for module in model.model if isinstance(module, BTDSE)]

    assert len(modules) == 1
    assert modules[0].i == 21
    assert model.model[20].type.endswith("Concat")
    assert model.model[22].type.endswith("RepC3")
    assert model.model[-1].f == [22, 25, 28]
    assert model.nc == 10


def test_custom_model_runs_inference_and_populates_auxiliary_maps():
    model = BTDSEDetectionModel(MODEL_CONFIG, ch=3, nc=10, verbose=False).eval()
    image = torch.rand(1, 3, 160, 160)

    with torch.no_grad():
        output = model.predict(image)

    module = next(module for module in model.model if isinstance(module, BTDSE))
    assert output is not None
    assert module.last_background_reliability.shape == (1, 1, 20, 20)
    assert module.last_saliency.shape == (1, 1, 20, 20)

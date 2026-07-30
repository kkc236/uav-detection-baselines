from argparse import Namespace

import cv2
import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.models.rtdetr.val import RTDETRDataset

from src.acr_eg_ultralytics_native import (
    NativePairedRTDETRDataset,
    build_native_result,
    extract_requested_metrics,
    predict_native_batch,
    require_paired_path,
    run_native_arm,
    validate_native_protocol,
)


BASELINE_SHA256 = "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
METHOD_SHA256 = "66E0B8D27706CDA594BE657B20BFD01CAA536D90B7EA0A05EDC2FEEC11C6E2B4"
DATASET_SIGNATURE = "A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A"


class FakeBoxMetrics:
    mp = 0.31
    mr = 0.42
    map50 = 0.53
    map75 = 0.29
    map = 0.24


class FakeDetMetrics:
    box = FakeBoxMetrics()


def frozen_args(**overrides) -> Namespace:
    values = {
        "device": "0",
        "batch": 1,
        "workers": 0,
        "imgsz": 640,
        "conf": 0.001,
        "max_det": 300,
        "expected_records": 548,
        "expected_epoch": 99,
        "expected_baseline_sha256": BASELINE_SHA256,
        "expected_checkpoint_sha256": METHOD_SHA256,
        "dataset_signature": DATASET_SIGNATURE,
        "amp": True,
        "smoke": False,
        "limit": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_extract_requested_metrics_uses_ultralytics_box_properties() -> None:
    assert extract_requested_metrics(FakeDetMetrics()) == {
        "Precision": 0.31,
        "Recall": 0.42,
        "AP50": 0.53,
        "AP75": 0.29,
        "mAP50-95": 0.24,
    }


def test_native_paired_dataset_preserves_stock_global_sample(tmp_path) -> None:
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    y, x = np.mgrid[:480, :800]
    image = np.stack(
        (
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ),
        axis=2,
    )
    assert cv2.imwrite(str(image_dir / "sample.jpg"), image)
    (label_dir / "sample.txt").write_text(
        "3 0.5 0.5 0.1 0.1\n",
        encoding="utf-8",
    )
    data = {
        "names": {index: str(index) for index in range(10)},
        "nc": 10,
        "channels": 3,
    }
    hyp = get_cfg(
        overrides={
            "imgsz": 640,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "bgr": 0.0,
        }
    )
    common = {
        "img_path": str(image_dir),
        "imgsz": 640,
        "batch_size": 1,
        "augment": False,
        "hyp": hyp,
        "rect": False,
        "cache": None,
        "prefix": "native-test: ",
        "data": data,
    }
    stock = RTDETRDataset(**common)[0]
    paired = NativePairedRTDETRDataset(**common)[0]
    assert torch.equal(paired["img"], stock["img"])
    assert torch.equal(paired["bboxes"], stock["bboxes"])
    assert torch.equal(paired["cls"], stock["cls"])
    assert paired["local_views"].shape == (4, 3, 640, 640)
    assert paired["source_shape"].tolist() == [480, 800]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_records", 547),
        ("expected_epoch", 98),
        ("imgsz", 608),
        ("batch", 2),
        ("amp", False),
        ("expected_checkpoint_sha256", "0" * 64),
    ),
)
def test_native_protocol_rejects_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="ACR_EG_NATIVE_PROTOCOL_DRIFT"):
        validate_native_protocol(frozen_args(**{field: value}))


def test_native_protocol_accepts_one_image_smoke() -> None:
    validate_native_protocol(frozen_args(smoke=True, limit=1))


def test_require_paired_path_rejects_silent_stock_fallback() -> None:
    model = Namespace(last_acr_eg_output=None)
    with pytest.raises(
        RuntimeError, match="ACR_EG_NATIVE_SILENT_STOCK_FALLBACK"
    ):
        require_paired_path(model)


def test_predict_native_batch_passes_all_five_views_to_method() -> None:
    class FakeModel:
        last_acr_eg_output = None

        def predict(self, image, **kwargs):
            self.last_acr_eg_output = object()
            self.image = image
            self.kwargs = kwargs
            return torch.zeros((1, 300, 6))

    model = FakeModel()
    batch = {
        "img": torch.zeros((1, 3, 640, 640)),
        "local_views": torch.zeros((1, 4, 3, 640, 640)),
        "source_shape": torch.tensor([[540, 960]]),
    }
    prediction = predict_native_batch(model, batch, paired=True)
    assert prediction.shape == (1, 300, 6)
    assert model.image is batch["img"]
    assert model.kwargs == {
        "local_views": batch["local_views"],
        "source_shapes": batch["source_shape"],
    }


def test_predict_native_batch_uses_only_global_view_for_baseline() -> None:
    class FakeModel:
        def predict(self, image, **kwargs):
            self.kwargs = kwargs
            return torch.zeros((1, 300, 6))

    model = FakeModel()
    prediction = predict_native_batch(
        model,
        {"img": torch.zeros((1, 3, 640, 640))},
        paired=False,
    )
    assert prediction.shape == (1, 300, 6)
    assert model.kwargs == {}


def test_run_native_arm_uses_ultralytics_validator_for_every_image() -> None:
    class FakeDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "img": torch.zeros((3, 8, 8)),
                "local_views": torch.full(
                    (4, 3, 8, 8), 255, dtype=torch.uint8
                ),
                "source_shape": torch.tensor([8, 8]),
                "index": torch.tensor(index),
            }

        @staticmethod
        def collate_fn(samples):
            return {
                key: torch.stack([sample[key] for sample in samples])
                for key in samples[0]
            }

    class FakeModel:
        last_acr_eg_output = None

        def predict(self, image, **kwargs):
            self.last_acr_eg_output = object()
            self.local_max = float(kwargs["local_views"].max())
            self.local_dtype = kwargs["local_views"].dtype
            return torch.zeros((image.shape[0], 300, 6))

    class FakeValidator:
        metrics = FakeDetMetrics()

        def __init__(self):
            self.updates = 0

        def init_metrics(self, model):
            self.model = model

        def preprocess(self, batch):
            return batch

        def postprocess(self, predictions):
            return predictions

        def update_metrics(self, predictions, batch):
            self.updates += int(batch["img"].shape[0])

        def get_stats(self):
            return {
                "metrics/precision(B)": 0.31,
                "metrics/recall(B)": 0.42,
                "metrics/mAP50(B)": 0.53,
                "metrics/mAP50-95(B)": 0.24,
            }

    validator = FakeValidator()
    model = FakeModel()
    result = run_native_arm(
        model=model,
        dataset=FakeDataset(),
        validator=validator,
        device=torch.device("cpu"),
        workers=0,
        amp=False,
        paired=True,
    )
    assert result["image_count"] == 2
    assert result["metrics"]["AP75"] == 0.29
    assert result["official_results"]["metrics/mAP50-95(B)"] == 0.24
    assert validator.updates == 2
    assert model.local_max == 1.0
    assert model.local_dtype == torch.float32


def test_build_native_result_calculates_method_minus_baseline() -> None:
    result = build_native_result(
        baseline_metrics={
            "Precision": 0.2,
            "Recall": 0.3,
            "AP50": 0.4,
            "AP75": 0.2,
            "mAP50-95": 0.25,
        },
        method_metrics={
            "Precision": 0.25,
            "Recall": 0.38,
            "AP50": 0.51,
            "AP75": 0.26,
            "mAP50-95": 0.31,
        },
        baseline={"sha256": BASELINE_SHA256},
        checkpoint={"sha256": METHOD_SHA256, "epoch": 99},
        dataset={"signature": DATASET_SIGNATURE, "image_count": 548},
        protocol={"imgsz": 640},
        source={"commit": "deadbeef"},
    )
    assert result["schema_version"] == "gcte-acr-eg-ultralytics-native/v1"
    assert result["deltas"] == pytest.approx(
        {
            "Precision": 0.05,
            "Recall": 0.08,
            "AP50": 0.11,
            "AP75": 0.06,
            "mAP50-95": 0.06,
        }
    )

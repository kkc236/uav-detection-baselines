from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src import bpdd_formal_evaluation as formal


CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


class TinyCheckpointModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))


class TinyFDRInferenceModel(nn.Module):
    instances: list["TinyFDRInferenceModel"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.loaded_strict = None
        type(self).instances.append(self)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.loaded_strict = strict
        return super().load_state_dict(state_dict, strict=strict)


@pytest.fixture(autouse=True)
def _reset_models() -> None:
    TinyFDRInferenceModel.instances.clear()


def _checkpoint(path: Path, *, epoch: int = 99) -> tuple[Path, str]:
    torch.save(
        {
            "epoch": epoch,
            "ema": TinyCheckpointModel(2.0),
            "model": TinyCheckpointModel(9.0),
            "optimizer": {"state": {}, "param_groups": []},
        },
        path,
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_exact_final_checkpoint_prefers_ema_and_strictly_loads_plain_fdr(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path / "epoch99.pt")

    loaded = formal.load_exact_final_checkpoint(
        checkpoint,
        expected_sha256=digest,
        model_factory=TinyFDRInferenceModel,
    )

    assert loaded.metadata == {
        "kind": "exact-final-ema",
        "completed_epoch": 100,
        "raw_epoch": 99,
        "sha256": digest,
        "sha256_verified": True,
        "source_field": "ema",
        "ema_state_sha256": formal.state_sha256(
            TinyCheckpointModel(2.0).state_dict()
        ),
        "strict_fdr_inference_graph": True,
    }
    assert type(loaded.model) is TinyFDRInferenceModel
    assert TinyFDRInferenceModel.instances[0].loaded_strict is True
    torch.testing.assert_close(loaded.model.weight, torch.tensor([2.0]))


@pytest.mark.parametrize(
    ("name", "epoch", "expected"),
    [
        ("epoch98.pt", 98, "epoch99"),
        ("last.pt", 99, "epoch99"),
    ],
)
def test_exact_final_checkpoint_rejects_wrong_epoch_or_filename(
    tmp_path: Path, name: str, epoch: int, expected: str
) -> None:
    checkpoint, digest = _checkpoint(tmp_path / name, epoch=epoch)

    with pytest.raises(ValueError, match=expected):
        formal.load_exact_final_checkpoint(
            checkpoint,
            expected_sha256=digest,
            model_factory=TinyFDRInferenceModel,
        )


def test_exact_final_checkpoint_rejects_sha_drift(tmp_path: Path) -> None:
    checkpoint, _ = _checkpoint(tmp_path / "epoch99.pt")
    with pytest.raises(ValueError, match="SHA256"):
        formal.load_exact_final_checkpoint(
            checkpoint,
            expected_sha256="A" * 64,
            model_factory=TinyFDRInferenceModel,
        )


def test_exact_final_checkpoint_rejects_a_stripped_training_snapshot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch99.pt"
    torch.save(
        {
            "epoch": 99,
            "ema": TinyCheckpointModel(2.0),
            "model": TinyCheckpointModel(9.0),
            "optimizer": None,
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()

    with pytest.raises(ValueError, match="optimizer|stripped"):
        formal.load_exact_final_checkpoint(
            checkpoint,
            expected_sha256=digest,
            model_factory=TinyFDRInferenceModel,
        )


def test_ema_state_sha_preserves_the_checkpoint_tensor_dtype(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch99.pt"
    ema = TinyCheckpointModel(2.0).half()
    expected_state_sha = formal.state_sha256(ema.state_dict())
    torch.save(
        {
            "epoch": 99,
            "ema": ema,
            "model": TinyCheckpointModel(9.0),
            "optimizer": {"state": {}, "param_groups": []},
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()

    loaded = formal.load_exact_final_checkpoint(
        checkpoint,
        expected_sha256=digest,
        model_factory=TinyFDRInferenceModel,
    )

    assert loaded.metadata["ema_state_sha256"] == expected_state_sha
    torch.testing.assert_close(loaded.model.weight, torch.tensor([2.0]))


@pytest.mark.parametrize(
    ("side", "bucket"),
    [
        (0.0, "tiny"),
        (15.999, "tiny"),
        (16.0, "small"),
        (31.999, "small"),
        (32.0, "medium"),
        (95.999, "medium"),
        (96.0, "large"),
    ],
)
def test_scale_boundaries_are_unambiguous(side: float, bucket: str) -> None:
    assert formal.scale_bucket_from_area(side * side) == bucket


def test_native_metrics_report_fixed_f1_and_all_ten_classes() -> None:
    all_ap = np.stack(
        [np.linspace(0.10 + index * 0.01, 0.19 + index * 0.01, 10) for index in range(10)]
    )
    box = SimpleNamespace(
        mp=0.6,
        mr=0.4,
        map50=0.5,
        map75=0.3,
        map=0.35,
        all_ap=all_ap,
        ap_class_index=np.arange(10),
    )

    summary = formal.summarize_native_box_metrics(box, CLASS_NAMES)

    assert summary["metrics"] == {
        "precision": 0.6,
        "recall": 0.4,
        "f1": pytest.approx(0.48),
        "map50": 0.5,
        "map75": 0.3,
        "map": 0.35,
    }
    assert list(summary["classes"]) == list(CLASS_NAMES)
    assert summary["classes"]["pedestrian"] == pytest.approx(float(all_ap[0].mean()))
    assert summary["class_details"]["pedestrian"] == {
        "id": 0,
        "map50": pytest.approx(all_ap[0, 0]),
        "map75": pytest.approx(all_ap[0, 5]),
        "map": pytest.approx(all_ap[0].mean()),
    }


def test_native_metrics_fail_closed_when_a_class_is_missing() -> None:
    box = SimpleNamespace(
        mp=0.6,
        mr=0.4,
        map50=0.5,
        map75=0.3,
        map=0.35,
        all_ap=np.ones((9, 10)),
        ap_class_index=np.arange(9),
    )
    with pytest.raises(ValueError, match="10 classes"):
        formal.summarize_native_box_metrics(box, CLASS_NAMES)


def test_scale_metrics_use_same_cached_predictions_and_report_gt_counts() -> None:
    predictions = [
        {
            "bboxes": torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [20.0, 20.0, 36.0, 36.0],
                    [50.0, 50.0, 82.0, 82.0],
                    [100.0, 100.0, 196.0, 196.0],
                ]
            ),
            "conf": torch.tensor([0.9, 0.9, 0.9, 0.9]),
            "cls": torch.tensor([0.0, 1.0, 2.0, 3.0]),
        }
    ]
    targets = [
        {
            "bboxes": predictions[0]["bboxes"].clone(),
            "cls": predictions[0]["cls"].clone(),
        }
    ]

    scales = formal.summarize_scale_metrics(predictions, targets, class_count=10)

    assert list(scales["scales"]) == ["tiny", "small", "medium", "large"]
    assert {name: item["gt"] for name, item in scales["scale_details"].items()} == {
        "tiny": 1,
        "small": 1,
        "medium": 1,
        "large": 1,
    }
    for name in scales["scales"]:
        assert scales["scales"][name] == pytest.approx(0.995, abs=0.01)
        assert scales["scale_details"][name]["map50"] == pytest.approx(0.995, abs=0.01)
        assert scales["scale_details"][name]["map75"] == pytest.approx(0.995, abs=0.01)


def test_create_only_json_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "formal.json"
    assert formal.write_create_only_json(output, {"status": "complete"}) == output.resolve()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        formal.write_create_only_json(output, {"status": "changed"})
    assert output.read_bytes() == before
    assert json.loads(output.read_text("utf-8")) == {"status": "complete"}

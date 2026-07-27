import importlib
import sys
from pathlib import Path
from types import ModuleType


def test_gcte_data_module_is_lazy_about_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    sys.modules.pop("src.gcte_data", None)

    module = importlib.import_module("src.gcte_data")

    assert callable(module.build_gcqf_dataset)
    assert module.GCQF_CACHE_AUGMENT is False


def test_gcqf_dataset_redirects_label_cache_off_dataset_mount(
    monkeypatch,
    tmp_path,
):
    forbidden = tmp_path / "dataset" / "labels.cache"

    class FakeBaseDataset:
        def __init__(self, **_kwargs):
            self.received_cache_path = None
            self.cache_labels(forbidden)

        def cache_labels(self, path):
            self.received_cache_path = Path(path)
            return {"labels": []}

    ultralytics = ModuleType("ultralytics")
    cfg = ModuleType("ultralytics.cfg")
    cfg.get_cfg = lambda overrides: overrides
    utils = ModuleType("ultralytics.utils")
    patches = ModuleType("ultralytics.utils.patches")
    patches.imread = lambda *_args, **_kwargs: None
    gcmv_data = ModuleType("src.gcmv_data")
    gcmv_data.GCMVRTDETRDataset = FakeBaseDataset
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", cfg)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.patches", patches)
    monkeypatch.setitem(sys.modules, "src.gcmv_data", gcmv_data)

    module = importlib.import_module("src.gcte_data")
    dataset = module.build_gcqf_dataset(
        {"train": "train.txt", "names": {0: "object"}},
        split="train",
    )

    assert dataset.received_cache_path != forbidden
    assert "gcqf-label-cache-" in dataset.received_cache_path.as_posix()
    assert not dataset.received_cache_path.exists()

import importlib
import sys


def test_gcte_data_module_is_lazy_about_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    sys.modules.pop("src.gcte_data", None)

    module = importlib.import_module("src.gcte_data")

    assert callable(module.build_gcqf_dataset)
    assert module.GCQF_CACHE_AUGMENT is False

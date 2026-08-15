from pathlib import Path

import ultralytics
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
STOCK_CONFIG = (
    Path(ultralytics.__file__).resolve().parent
    / "cfg"
    / "models"
    / "rt-detr"
    / "rtdetr-l.yaml"
)
FULL_CONFIG = CONFIG_DIR / "rtdetr-l-fdr.yaml"
FDR_CONFIGS = (
    FULL_CONFIG,
    CONFIG_DIR / "rtdetr-l-fdr-no-fgl.yaml",
    CONFIG_DIR / "rtdetr-l-fdr-no-prebox-loss.yaml",
    CONFIG_DIR / "rtdetr-l-fdr-no-cumulative.yaml",
    CONFIG_DIR / "rtdetr-l-fdr-no-prebox.yaml",
)

FULL_OPTIONS = {
    "hidden_dim": 256,
    "num_queries": 300,
    "num_decoder_layers": 6,
    "reg_max": 32,
    "reg_scale": 4.0,
    "up": 0.5,
    "cumulative": True,
    "preliminary_box": True,
    "private_seed": 10000,
}
FULL_LOSS = {
    "fgl_weight": 0.15,
    "supervise_pre_boxes": True,
}
EXPECTED_ABLATION_DIFFS = {
    CONFIG_DIR / "rtdetr-l-fdr-no-fgl.yaml": {
        ("fdr_loss", "fgl_weight"): (0.15, 0.0),
    },
    CONFIG_DIR / "rtdetr-l-fdr-no-prebox-loss.yaml": {
        ("fdr_loss", "supervise_pre_boxes"): (True, False),
    },
    CONFIG_DIR / "rtdetr-l-fdr-no-cumulative.yaml": {
        ("head", 18, 3, 2, "cumulative"): (True, False),
    },
    CONFIG_DIR / "rtdetr-l-fdr-no-prebox.yaml": {
        ("head", 18, 3, 2, "preliminary_box"): (True, False),
    },
}


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _leaf_differences(left, right, path=()):
    if type(left) is not type(right):
        return {path: (left, right)}

    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {path + ("<keys>",): (tuple(left), tuple(right))}
        differences = {}
        for key in left:
            differences.update(_leaf_differences(left[key], right[key], path + (key,)))
        return differences

    if isinstance(left, list):
        if len(left) != len(right):
            return {path + ("<length>",): (len(left), len(right))}
        differences = {}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.update(_leaf_differences(left_item, right_item, path + (index,)))
        return differences

    return {} if left == right else {path: (left, right)}


def test_all_fdr_configs_preserve_the_exact_ultralytics_8490_graph_before_decoder():
    assert ultralytics.__version__ == "8.4.90"
    stock = _load_yaml(STOCK_CONFIG)
    stock_layers = stock["backbone"] + stock["head"]

    for config_path in FDR_CONFIGS:
        config = _load_yaml(config_path)
        layers = config["backbone"] + config["head"]

        assert config["nc"] == stock["nc"], config_path
        assert config["scales"] == stock["scales"], config_path
        assert len(layers) == len(stock_layers), config_path
        assert layers[:-1] == stock_layers[:-1], config_path


def test_full_fdr_config_declares_the_exact_decoder_options_and_loss():
    config = _load_yaml(FULL_CONFIG)
    final_layer = config["head"][-1]

    assert final_layer == [
        [21, 24, 27],
        1,
        "FDRRTDETRDecoder",
        ["nc", [256, 256, 256], FULL_OPTIONS],
    ]
    assert config["fdr_loss"] == FULL_LOSS


def test_each_ablation_changes_exactly_one_intended_yaml_field():
    full = _load_yaml(FULL_CONFIG)

    for config_path, expected_differences in EXPECTED_ABLATION_DIFFS.items():
        ablation = _load_yaml(config_path)
        assert _leaf_differences(full, ablation) == expected_differences, config_path
